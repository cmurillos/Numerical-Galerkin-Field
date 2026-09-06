"""General weak Galerkin fields on user-supplied bases and simplicial domains."""

from dataclasses import dataclass
from math import prod
from numbers import Integral, Real

import numpy as np
import torch

from .forms import Coefficient, build_form, coefficient_expressions, derivative_orders, evaluate
from .geometry import SimplicialDomain, positive_integer
from .spaces import TransformedBasis

_MAX_ADAPTIVE_ORDER = 64


@dataclass
class _Table:
    points: torch.Tensor
    weights: torch.Tensor
    cells: torch.Tensor
    barycentric: torch.Tensor
    normals: torch.Tensor | None
    basis: dict[int, torch.Tensor]
    coefficients: dict[object, torch.Tensor]

    def to(self, device, dtype):
        for name in ("points", "weights", "barycentric", "normals"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device=device, dtype=dtype))
        self.cells = self.cells.to(device=device)
        self.basis = {
            order: values.to(device=device, dtype=dtype) for order, values in self.basis.items()
        }
        self.coefficients = {
            coefficient: values.to(device=device, dtype=dtype)
            for coefficient, values in self.coefficients.items()
        }


def _basis_values(geometry, basis, points, cells, barycentric, order):
    """Evaluate one fixed basis derivative and enforce the public tensor contract."""
    values = basis.evaluate(
        points,
        order=order,
        cells=cells,
        barycentric=barycentric,
    )
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=points.dtype, device=points.device)
    expected = (
        len(points),
        basis.dimension,
        *basis.value_shape,
        *((geometry.ambient_dimension,) * order),
    )
    if values.shape != expected:
        raise ValueError(
            f"Basis derivative {order} must have shape {expected}, got {tuple(values.shape)}."
        )
    if values.device != points.device or values.dtype != points.dtype:
        raise ValueError("Basis values must use the field device and dtype.")
    if order and geometry.dimension < geometry.ambient_dimension:
        projectors = points.new_tensor(geometry.tangent_projectors.copy())[cells]
        for axis in range(order):
            position = values.ndim - order + axis
            moved = values.movedim(position, -1)
            moved = torch.einsum("q...b,qab->q...a", moved, projectors)
            values = moved.movedim(-1, position)
    if not torch.isfinite(values).all():
        raise ValueError("Basis evaluation returned nonfinite values.")
    return values.detach()


def _table(
    geometry,
    basis,
    orders,
    coefficients,
    quadrature_order,
    boundary,
    region,
    rule,
    max_points,
    device,
    dtype,
):
    q = geometry.quadrature(
        quadrature_order,
        boundary=boundary,
        region=region,
        rule=rule,
        max_points=max_points,
    )
    points = torch.tensor(q.points.copy(), dtype=dtype, device=device)
    cells = torch.tensor(q.cells.copy(), device=device)
    barycentric = torch.tensor(q.barycentric.copy(), dtype=dtype, device=device)
    tables = {}
    for order in sorted(set(orders) | {0}):
        tables[order] = _basis_values(
            geometry,
            basis,
            points,
            cells,
            barycentric,
            order,
        )
    coefficient_tables = {
        coefficient: coefficient.tabulate(geometry, points, cells, barycentric)
        for coefficient in coefficients
    }
    return _Table(
        points,
        torch.tensor(q.weights.copy(), dtype=dtype, device=device),
        cells,
        barycentric,
        None if q.normals is None else torch.tensor(q.normals.copy(), dtype=dtype, device=device),
        tables,
        coefficient_tables,
    )


def _gram(table):
    values = table.basis[0].reshape(len(table.points), table.basis[0].shape[1], -1)
    return torch.einsum("qnd,qmd,q->nm", values, values, table.weights)


def _basis_contract(problem, basis):
    if not isinstance(getattr(basis, "dimension", None), int) or basis.dimension < 1:
        raise ValueError("basis.dimension must be a positive integer.")
    try:
        value_shape = tuple(basis.value_shape)
    except (AttributeError, TypeError) as exc:
        raise ValueError("basis.value_shape must be a tuple of positive integers.") from exc
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in value_shape):
        raise ValueError("basis.value_shape must contain positive integers.")
    if not callable(getattr(basis, "evaluate", None)):
        raise ValueError("A basis must implement evaluate(points, order=...).")
    if hasattr(basis, "geometry") and not problem.geometry.same_mesh(basis.geometry):
        raise ValueError("The basis belongs to a different mesh.")
    return value_shape


def _tolerance(value, dtype=torch.float64):
    if value is None:
        return 5e-5 if dtype == torch.float32 else 1e-8
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError("orthonormality tolerance must be a real number between zero and one.")
    return float(value)


def _basis_polynomial_degree(basis):
    """Return the local polynomial degree, or ``None`` when it is unknown."""
    from .spaces import ComponentBasis, FiniteElementBasis, PolynomialBasis, ProductBasis

    if isinstance(basis, FiniteElementBasis):
        return basis.degree
    if isinstance(basis, PolynomialBasis):
        return int(basis.exponents.sum(axis=1).max())
    if isinstance(basis, (TransformedBasis, ComponentBasis)):
        return _basis_polynomial_degree(basis.base)
    if isinstance(basis, ProductBasis):
        degrees = [_basis_polynomial_degree(item) for item in basis.bases]
        return None if any(degree is None for degree in degrees) else max(degrees)
    return None


def _expression_polynomial_degree(expression, basis_degree):
    """Infer a cellwise spatial degree for expressions supported by exact quadrature."""
    op = expression.op
    if op in ("u", "v"):
        return max(0, basis_degree - len(expression.data))
    if op == "x":
        return 1
    if op in ("constant", "normal"):
        return 0
    if op == "coefficient":
        return {"cell": 0, "vertex": 1}.get(expression.location)

    degrees = [
        _expression_polynomial_degree(argument, basis_degree) for argument in expression.args
    ]
    if op in ("add", "sub", "stack"):
        return None if any(degree is None for degree in degrees) else max(degrees, default=0)
    if op in ("mul", "inner", "contract", "outer"):
        return None if any(degree is None for degree in degrees) else sum(degrees)
    if op == "div":
        return degrees[0] if degrees[1] == 0 else None
    if op == "pow":
        power = expression.data
        if power == 0 or degrees[0] == 0:
            return 0
        if isinstance(power, Real) and float(power).is_integer() and power >= 0:
            return None if degrees[0] is None else int(power) * degrees[0]
        return None
    if op in ("neg", "index", "transpose", "trace"):
        return degrees[0]
    if op == "pointwise" or op in ("exp", "sin", "cos", "tanh", "log", "sqrt"):
        return 0 if all(degree == 0 for degree in degrees) else None
    return None


def _exact_quadrature_order(form, basis):
    degree = _basis_polynomial_degree(basis)
    if degree is None:
        return None
    degrees = [
        _expression_polynomial_degree(integral.integrand, degree) for integral in form.integrals
    ]
    return None if any(value is None for value in degrees) else max(degrees, default=0)


def _quadrature_specification(quadrature, dtype):
    if quadrature is None:
        return "automatic", _tolerance(None, dtype)
    if isinstance(quadrature, bool):
        raise TypeError("quadrature must be an integer order or a real tolerance in (0,1).")
    if isinstance(quadrature, Integral):
        return "fixed", positive_integer(quadrature, "quadrature", 0)
    if isinstance(quadrature, Real) and 0 < float(quadrature) < 1:
        return "adaptive", float(quadrature)
    raise ValueError(
        "quadrature must be an integer order >= 0 or a real tolerance strictly between 0 and 1."
    )


def _calibration_states(dimension, *, device, dtype):
    """Small deterministic sample of the coefficient unit ball used during preparation."""
    indices = torch.linspace(0, dimension - 1, min(dimension, 8), device=device)
    indices = torch.unique(indices.round().to(torch.long))
    axes = torch.zeros((len(indices), dimension), dtype=dtype, device=device)
    axes[torch.arange(len(indices), device=device), indices] = 1
    dense = torch.linspace(-1.0, 1.0, dimension, dtype=dtype, device=device)
    dense = dense / torch.linalg.vector_norm(dense).clamp_min(1)
    uniform = torch.full((dimension,), dimension**-0.5, dtype=dtype, device=device)
    return torch.cat(
        (torch.zeros((1, dimension), dtype=dtype, device=device), axes, dense[None], uniform[None])
    )


def _projection_exact_order(source, basis):
    basis_degree = _basis_polynomial_degree(basis)
    if basis_degree is None or not isinstance(source, Coefficient):
        return None
    source_degree = {"cell": 0, "vertex": 1}.get(source.location)
    return None if source_degree is None else basis_degree + source_degree


def _projection_error_exact_order(source, basis):
    basis_degree = _basis_polynomial_degree(basis)
    if basis_degree is None or not isinstance(source, Coefficient):
        return None
    source_degree = {"cell": 0, "vertex": 1}.get(source.location)
    return None if source_degree is None else 2 * max(basis_degree, source_degree)


class _BasisGeometry:
    """Geometry-only L2 preparation shared by Space and GalerkinProblem."""

    def __init__(self, geometry):
        self.geometry = geometry

    def orthonormalize(
        self,
        basis,
        *,
        quadrature_order=6,
        validation_order=None,
        quadrature_rule=None,
        tolerance=1e-8,
        max_quadrature_points=1_000_000,
    ):
        """Return a new fixed basis orthonormal in the numerical L2 metric."""
        _basis_contract(self, basis)
        quadrature_order = positive_integer(quadrature_order, "quadrature_order", 0)
        validation_order = (
            quadrature_order + 2
            if validation_order is None
            else positive_integer(validation_order, "validation_order", 0)
        )
        tolerance = _tolerance(tolerance)
        table = _table(
            self.geometry,
            basis,
            {0},
            set(),
            quadrature_order,
            None,
            None,
            quadrature_rule,
            max_quadrature_points,
            "cpu",
            torch.float64,
        )
        gram = _gram(table)
        try:
            factor = torch.linalg.cholesky((gram + gram.T) / 2)
        except torch.linalg.LinAlgError as exc:
            raise ValueError("The basis Gram matrix is not positive definite.") from exc
        identity = torch.eye(basis.dimension, dtype=gram.dtype)
        result = TransformedBasis(
            basis, torch.linalg.solve_triangular(factor.T, identity, upper=True)
        )
        result.family = "orthonormalized"
        result.quadrature_order = quadrature_order
        result.validation_order = validation_order
        result.orthonormality_error = self.validate_basis(
            result,
            quadrature_order=validation_order,
            quadrature_rule=quadrature_rule,
            tolerance=tolerance,
            max_quadrature_points=max_quadrature_points,
        )
        return result

    def validate_basis(
        self,
        basis,
        *,
        quadrature_order=None,
        quadrature_rule=None,
        tolerance=1e-8,
        max_quadrature_points=1_000_000,
    ):
        """Validate the basis contract and return its maximum L2 Gram error."""
        _basis_contract(self, basis)
        order = (
            getattr(basis, "validation_order", getattr(basis, "quadrature_order", 4))
            if quadrature_order is None
            else quadrature_order
        )
        order = positive_integer(order, "quadrature_order", 0)
        tolerance = _tolerance(tolerance)
        table = _table(
            self.geometry,
            basis,
            {0},
            set(),
            order,
            None,
            None,
            quadrature_rule,
            max_quadrature_points,
            "cpu",
            torch.float64,
        )
        gram = _gram(table)
        error = float(torch.max(torch.abs(gram - torch.eye(basis.dimension))).item())
        if error > tolerance:
            raise ValueError(
                "The operational basis must be L2-orthonormal: "
                f"max|M-I|={error:.3e} exceeds {tolerance:.3e}. "
                "Use problem.orthonormalize(...) or problem.basis(...)."
            )
        return error


class GalerkinProblem(_BasisGeometry):
    """A simplicial geometry plus one complete weak form.

    No boundary-condition type is part of this interface. The supplied basis defines
    the admissible finite-dimensional space; named boundary measures only integrate
    terms explicitly present in weak(u,v,dx,ds).
    """

    def __init__(
        self,
        *,
        weak,
        geometry=None,
        vertices=None,
        simplices=None,
        boundaries=None,
        regions=None,
    ):
        if not callable(weak):
            raise TypeError("weak must be callable.")
        if geometry is None:
            if vertices is None or simplices is None:
                raise TypeError("Provide geometry or both vertices and simplices.")
            self.geometry = SimplicialDomain(vertices, simplices, boundaries, regions)
        else:
            if not isinstance(geometry, SimplicialDomain):
                raise TypeError("geometry must be a SimplicialDomain.")
            if any(value is not None for value in (vertices, simplices, boundaries, regions)):
                raise ValueError(
                    "geometry cannot be combined with vertices, simplices, boundaries, or regions."
                )
            self.geometry = geometry
        self.weak = weak

    @property
    def vertices(self):
        return self.geometry.vertices

    @property
    def simplices(self):
        return self.geometry.simplices

    @property
    def boundaries(self):
        return self.geometry.boundaries

    @property
    def regions(self):
        return self.geometry.regions

    def basis(self, family="laplacian", **options):
        """Build a fixed L2-orthonormal basis adapted to this geometry."""
        from .basis_factory import build_basis

        return build_basis(self, family, options)

    def field(
        self,
        *,
        basis,
        quadrature=None,
        max_quadrature_points=1_000_000,
        max_intermediate_entries=10_000_000,
        device="cpu",
        dtype=torch.float64,
    ):
        return GalerkinField(
            self,
            basis,
            quadrature=quadrature,
            max_quadrature_points=max_quadrature_points,
            max_intermediate_entries=max_intermediate_entries,
            device=device,
            dtype=dtype,
        )


class GalerkinField:
    """The differentiable coordinate field ``G: [..., N] -> [..., N]``.

    Every leading axis is a batch axis. Geometry, basis, coefficients and quadrature
    tables are fixed at construction; differentiation is preserved only with respect
    to the state coordinates.
    """

    def __init__(
        self,
        problem,
        basis,
        *,
        quadrature=None,
        max_quadrature_points=1_000_000,
        max_intermediate_entries=10_000_000,
        device="cpu",
        dtype=torch.float64,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64.")
        value_shape = _basis_contract(problem, basis)
        max_quadrature_points = positive_integer(max_quadrature_points, "max_quadrature_points")
        self.max_quadrature_points = max_quadrature_points
        self.max_intermediate_entries = positive_integer(
            max_intermediate_entries, "max_intermediate_entries"
        )
        self.problem, self.basis = problem, basis
        self.dimension, self.value_shape = basis.dimension, value_shape
        self.form = build_form(problem.weak, value_shape, problem.geometry.ambient_dimension)
        required = derivative_orders(self.form)
        coefficients = coefficient_expressions(self.form)
        volume_orders = {"all": {0}}
        for name, orders in required["volume"].items():
            volume_orders.setdefault(name, set()).update(orders)
        self._table_requirements = (volume_orders, required["boundary"], coefficients)
        self._table_options = (max_quadrature_points, device, dtype)

        mode, target = _quadrature_specification(quadrature, dtype)
        baseline = positive_integer(
            getattr(basis, "validation_order", getattr(basis, "quadrature_order", 4)),
            "basis quadrature order",
            0,
        )
        self._basis_quadrature_order = baseline
        exact = _exact_quadrature_order(self.form, basis) if mode == "automatic" else None
        if mode == "fixed":
            self._prepare(target)
            self.quadrature_mode = "fixed"
            self.quadrature_tolerance = None
            self.quadrature_error_estimate = None
        elif exact is not None:
            self._prepare(max(baseline, exact))
            self.quadrature_mode = "automatic-exact"
            self.quadrature_tolerance = None
            self.quadrature_error_estimate = 0.0
        else:
            tolerance = target
            self._prepare_adaptive(baseline, tolerance)
            self.quadrature_mode = "automatic-adaptive" if mode == "automatic" else "adaptive"
            self.quadrature_tolerance = tolerance

        del self._table_requirements, self._table_options

    def _prepare(self, order):
        volume_orders, boundary_orders, coefficients = self._table_requirements
        max_points, device, dtype = self._table_options
        self._volumes = {
            name: _table(
                self.problem.geometry,
                self.basis,
                orders,
                coefficients["volume"].get(name, set()),
                order,
                None,
                None if name == "all" else name,
                None,
                max_points,
                device,
                dtype,
            )
            for name, orders in volume_orders.items()
        }
        self._volume = self._volumes["all"]
        self._boundary = {
            name: _table(
                self.problem.geometry,
                self.basis,
                orders,
                coefficients["boundary"].get(name, set()),
                order,
                name,
                None,
                None,
                max_points,
                device,
                dtype,
            )
            for name, orders in boundary_orders.items()
        }
        mass = _gram(self._volume)
        tolerance = _tolerance(None, dtype)
        error = torch.max(
            torch.abs(mass - torch.eye(self.dimension, dtype=dtype, device=device))
        ).item()
        if error > tolerance:
            raise ValueError(
                "The operational basis must be L2-orthonormal: "
                f"max|M-I|={error:.3e} exceeds {tolerance:.3e}. "
                "Use problem.orthonormalize(...) or problem.basis(...)."
            )
        self.mass_matrix = mass
        self.orthonormality_error = float(error)
        self.quadrature_order = order

    def _prepare_adaptive(self, initial_order, tolerance):
        if initial_order > _MAX_ADAPTIVE_ORDER:
            raise ValueError(
                f"The basis requires order {initial_order}, above the adaptive limit "
                f"{_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
            )
        probes = _calibration_states(
            self.dimension, device=self._table_options[1], dtype=self._table_options[2]
        )
        previous = None
        for order in range(initial_order, _MAX_ADAPTIVE_ORDER + 1, 2):
            try:
                self._prepare(order)
            except ValueError as exc:
                if "exceeds max_points" in str(exc):
                    raise ValueError(
                        "Adaptive quadrature exhausted max_quadrature_points before reaching "
                        f"tolerance {tolerance:.3e}. Use a fixed integer order or increase the budget."
                    ) from exc
                raise
            with torch.no_grad():
                current = self(probes)
            if previous is not None:
                scaled = torch.abs(current - previous) / (1 + torch.abs(current))
                error = float(torch.max(scaled).item())
                if error <= tolerance:
                    self.quadrature_error_estimate = error
                    return
            previous = current
        raise ValueError(
            f"Adaptive quadrature did not reach tolerance {tolerance:.3e} by order "
            f"{_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
        )

    @property
    def device(self):
        return self._volume.points.device

    @property
    def dtype(self):
        return self._volume.points.dtype

    @property
    def quadrature_size(self):
        return len(self._volume.points)

    @property
    def table_bytes(self):
        tensors = [self.mass_matrix]
        for table in (*self._volumes.values(), *self._boundary.values()):
            tensors.extend((table.points, table.weights, table.cells, table.barycentric))
            tensors.extend(table.basis.values())
            tensors.extend(table.coefficients.values())
            if table.normals is not None:
                tensors.append(table.normals)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def to(self, device=None, dtype=None):
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64.")
        for table in (*self._volumes.values(), *self._boundary.values()):
            table.to(device, dtype)
        self.mass_matrix = self.mass_matrix.to(device=device, dtype=dtype)
        return self

    def _physical_points(self, points):
        if not isinstance(points, torch.Tensor):
            raise TypeError("Physical points must be torch tensors.")
        expected = self.problem.geometry.ambient_dimension
        if points.ndim != 2 or points.shape[1] != expected:
            raise ValueError(f"Physical points must have shape [Q,{expected}].")
        if points.device != self.device or points.dtype != self.dtype:
            raise ValueError("Physical points and the field must share device and dtype.")
        if not torch.isfinite(points).all():
            raise ValueError("Physical points must be finite.")
        return points.detach()

    def _cell_coordinates(self, points, cells):
        geometry = self.problem.geometry
        corners = geometry.vertices[geometry.simplices[cells]]
        offsets = points - corners[:, 0]
        inverse = geometry.inverse_jacobians[cells]
        coordinates = np.einsum("nkp,np->nk", inverse, offsets, optimize=True)
        barycentric = np.concatenate(
            (1 - coordinates.sum(axis=1, keepdims=True), coordinates),
            axis=1,
        )

        projectors = geometry.tangent_projectors[cells]
        tangential = np.einsum("nij,nj->ni", projectors, offsets, optimize=True)
        residual = np.linalg.norm(offsets - tangential, axis=1)
        diameter = np.linalg.norm(corners[:, 1:] - corners[:, :1], axis=2).max(axis=1)
        scale = np.maximum.reduce(
            (
                np.max(np.abs(points), axis=1),
                np.max(np.abs(corners), axis=(1, 2)),
                diameter,
            )
        )
        scale = np.maximum(scale, np.finfo(np.float64).tiny)
        precision = np.float32 if self.dtype == torch.float32 else np.float64
        epsilon = np.finfo(precision).eps
        distance_tolerance = 256 * epsilon * scale
        inverse_norm = np.linalg.norm(inverse, ord=np.inf, axis=(1, 2))
        barycentric_tolerance = 256 * epsilon + distance_tolerance * inverse_norm
        valid = residual <= distance_tolerance
        valid &= np.min(barycentric, axis=1) >= -barycentric_tolerance
        valid &= np.max(barycentric, axis=1) <= 1 + barycentric_tolerance

        cleaned = np.clip(barycentric, 0.0, 1.0)
        cleaned /= cleaned.sum(axis=1, keepdims=True)
        return cleaned, valid

    def _provided_cells(self, cells, count):
        if not isinstance(cells, torch.Tensor):
            raise TypeError("cells must be a torch tensor.")
        if cells.dtype != torch.int64:
            raise ValueError("cells must use torch.int64.")
        if cells.device != self.device:
            raise ValueError("cells and the field must share device.")
        if cells.shape != (count,):
            raise ValueError(f"cells must have shape [{count}].")
        result = cells.detach().cpu().numpy()
        cell_count = len(self.problem.geometry.simplices)
        if np.any(result < 0) or np.any(result >= cell_count):
            raise ValueError("cells contains an invalid simplex index.")
        return result

    def _locate_points(self, points, cells):
        geometry = self.problem.geometry
        host_points = points.cpu().numpy().astype(np.float64, copy=False)
        count = len(host_points)
        if cells is not None:
            selected = self._provided_cells(cells, count)
            barycentric, valid = self._cell_coordinates(host_points, selected)
            if not np.all(valid):
                index = int(np.flatnonzero(~valid)[0])
                raise ValueError(
                    f"Physical point {index} does not belong to the simplex selected by cells."
                )
            return selected, barycentric, []

        simplices = geometry.simplices
        corners = geometry.vertices[simplices]
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        diameter = np.linalg.norm(corners[:, 1:] - corners[:, :1], axis=2).max(axis=1)
        scale = np.maximum(np.max(np.abs(corners), axis=(1, 2)), diameter)
        scale = np.maximum(scale, np.finfo(np.float64).tiny)
        precision = np.float32 if self.dtype == torch.float32 else np.float64
        box_tolerance = 256 * np.finfo(precision).eps * scale

        candidates = [[] for _ in range(count)]
        ambient = geometry.ambient_dimension
        pair_budget = max(1, self.max_intermediate_entries // max(1, 2 * ambient))
        point_chunk = max(1, min(count, int(pair_budget**0.5)))
        cell_chunk = max(1, pair_budget // point_chunk)
        for point_start in range(0, count, point_chunk):
            point_stop = min(count, point_start + point_chunk)
            block_points = host_points[point_start:point_stop]
            for cell_start in range(0, len(simplices), cell_chunk):
                cell_stop = min(len(simplices), cell_start + cell_chunk)
                tolerance = box_tolerance[cell_start:cell_stop]
                inside = np.all(
                    block_points[:, None, :]
                    >= lower[None, cell_start:cell_stop] - tolerance[None, :, None],
                    axis=2,
                )
                inside &= np.all(
                    block_points[:, None, :]
                    <= upper[None, cell_start:cell_stop] + tolerance[None, :, None],
                    axis=2,
                )
                local_points, local_cells = np.nonzero(inside)
                if not len(local_points):
                    continue
                global_cells = local_cells + cell_start
                barycentric, valid = self._cell_coordinates(
                    block_points[local_points],
                    global_cells,
                )
                for point, cell, bary, keep in zip(
                    local_points,
                    global_cells,
                    barycentric,
                    valid,
                ):
                    if keep:
                        candidates[point_start + int(point)].append((int(cell), bary))

        missing = [index for index, matches in enumerate(candidates) if not matches]
        if missing:
            raise ValueError(
                f"Physical point {missing[0]} does not belong to the simplicial domain."
            )
        selected = np.asarray([matches[0][0] for matches in candidates], dtype=np.int64)
        barycentric = np.stack([matches[0][1] for matches in candidates])
        alternatives = [
            (point, cell, bary)
            for point, matches in enumerate(candidates)
            for cell, bary in matches[1:]
        ]
        return selected, barycentric, alternatives

    def _basis_at_points(self, points, cells, order):
        selected, barycentric, alternatives = self._locate_points(points, cells)
        cell_tensor = torch.as_tensor(selected, dtype=torch.int64, device=self.device)
        barycentric_tensor = points.new_tensor(barycentric)
        values = _basis_values(
            self.problem.geometry,
            self.basis,
            points,
            cell_tensor,
            barycentric_tensor,
            order,
        )
        if alternatives:
            point_indices = torch.tensor(
                [item[0] for item in alternatives],
                dtype=torch.int64,
                device=self.device,
            )
            alternate_cells = torch.tensor(
                [item[1] for item in alternatives],
                dtype=torch.int64,
                device=self.device,
            )
            alternate_barycentric = points.new_tensor(np.stack([item[2] for item in alternatives]))
            alternate_values = _basis_values(
                self.problem.geometry,
                self.basis,
                points.index_select(0, point_indices),
                alternate_cells,
                alternate_barycentric,
                order,
            )
            reference = values.index_select(0, point_indices)
            tolerance = 1e-5 if self.dtype == torch.float32 else 1e-10
            equal = torch.isclose(
                alternate_values,
                reference,
                atol=tolerance,
                rtol=tolerance,
            ).reshape(len(alternatives), -1)
            ambiguous = ~torch.all(equal, dim=1)
            if torch.any(ambiguous):
                first = int(torch.nonzero(ambiguous, as_tuple=False)[0, 0].item())
                point = alternatives[first][0]
                quantity = {0: "values", 1: "gradients", 2: "Hessians"}[order]
                raise ValueError(
                    f"Physical point {point} has different {quantity} on adjacent simplices; "
                    "pass cells to select a trace."
                )
        return values

    def _spatial_reconstruction(self, z, points, cells, order):
        z, batch_shape = self._states(z)
        points = self._physical_points(points)
        if not len(points):
            if cells is not None:
                self._provided_cells(cells, 0)
            derivative_shape = (self.problem.geometry.ambient_dimension,) * order
            return z.new_empty((*batch_shape, 0, *self.value_shape, *derivative_shape))
        modes = self._basis_at_points(points, cells, order)
        values = torch.einsum("bn,qn...->bq...", z, modes)
        derivative_shape = (self.problem.geometry.ambient_dimension,) * order
        return values.reshape((*batch_shape, len(points), *self.value_shape, *derivative_shape))

    def reconstruct(self, z, points=None, *, cells=None, boundary=None):
        """Evaluate ``Phi z`` on a prepared measure or at physical points ``[Q,p]``."""
        if points is not None:
            if boundary is not None:
                raise ValueError("boundary cannot be combined with physical points.")
            return self._spatial_reconstruction(z, points, cells, 0)
        if cells is not None:
            raise ValueError("cells requires physical points.")
        z, batch_shape = self._states(z)
        table = self._volume if boundary is None else self._boundary[boundary]
        values = torch.einsum("bn,qn...->bq...", z, table.basis[0])
        return values.reshape((*batch_shape, len(table.points), *self.value_shape))

    def grad(self, z, points, *, cells=None):
        """Evaluate the elementwise tangential spatial gradient of ``Phi z``."""
        return self._spatial_reconstruction(z, points, cells, 1)

    def hessian(self, z, points, *, cells=None):
        """Evaluate the elementwise tangential spatial Hessian of ``Phi z``."""
        return self._spatial_reconstruction(z, points, cells, 2)

    def _projection_table(self, order):
        if order == self.quadrature_order:
            return self._volume
        return _table(
            self.problem.geometry,
            self.basis,
            {0},
            set(),
            order,
            None,
            None,
            None,
            self.max_quadrature_points,
            self.device,
            self.dtype,
        )

    def _source_on_table(self, source, table):
        if isinstance(source, Coefficient):
            values = source.tabulate(
                self.problem.geometry,
                table.points,
                table.cells,
                table.barycentric,
            )
        else:
            values = source(table.points)
            try:
                values = torch.as_tensor(values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "The projected function must return a real numerical tensor."
                ) from exc
            if torch.is_complex(values):
                raise ValueError("The projected function must be real-valued.")
            values = values.to(device=self.device, dtype=self.dtype)

        physical_rank = len(self.value_shape)
        point_axis = values.ndim - physical_rank - 1
        physical_shape = tuple(values.shape[-physical_rank:]) if physical_rank else ()
        if (
            point_axis < 0
            or values.shape[point_axis] != len(table.points)
            or physical_shape != self.value_shape
        ):
            expected = f"[*S,{len(table.points)},*{self.value_shape}]"
            raise ValueError(
                f"The projected function must return {expected}; got {tuple(values.shape)}."
            )
        if not torch.isfinite(values).all():
            raise ValueError("The projected function returned nonfinite values.")

        batch_shape = tuple(values.shape[:point_axis])
        batch_size = prod(batch_shape)
        physical_size = prod(self.value_shape)
        flattened = values.reshape(batch_size, len(table.points), physical_size)
        modes = table.basis[0].reshape(len(table.points), self.dimension, physical_size)
        return flattened, modes, batch_shape

    def _project_on_table(self, source, table):
        flattened, modes, batch_shape = self._source_on_table(source, table)
        result = torch.einsum("bqd,qnd,q->bn", flattened, modes, table.weights)
        return result.reshape((*batch_shape, self.dimension))

    def _projection_error_on_table(self, source, table):
        values, modes, batch_shape = self._source_on_table(source, table)
        coordinates = torch.einsum("bqd,qnd,q->bn", values, modes, table.weights)
        approximation = torch.einsum("bn,qnd->bqd", coordinates, modes)
        squared = torch.einsum(
            "bqd,bqd,q->b",
            values - approximation,
            values - approximation,
            table.weights,
        )
        return torch.sqrt(torch.clamp_min(squared, 0)).reshape(batch_shape)

    def project(self, source, *, quadrature=None):
        """Return the L2 coordinates of a callable or ``Coefficient`` in the fixed basis."""
        if not isinstance(source, Coefficient) and not callable(source):
            raise TypeError("project expects a callable function or a Coefficient.")
        mode, target = _quadrature_specification(quadrature, self.dtype)
        exact = _projection_exact_order(source, self.basis) if mode == "automatic" else None
        if mode == "fixed":
            return self._project_on_table(source, self._projection_table(target))
        if exact is not None:
            order = max(self._basis_quadrature_order, exact)
            return self._project_on_table(source, self._projection_table(order))

        if self._basis_quadrature_order > _MAX_ADAPTIVE_ORDER:
            raise ValueError(
                f"The basis requires order {self._basis_quadrature_order}, above the adaptive "
                f"limit {_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
            )
        previous = None
        for order in range(self._basis_quadrature_order, _MAX_ADAPTIVE_ORDER + 1, 2):
            try:
                current = self._project_on_table(source, self._projection_table(order))
            except ValueError as exc:
                if "exceeds max_points" in str(exc):
                    raise ValueError(
                        "Adaptive projection exhausted max_quadrature_points before reaching "
                        f"tolerance {target:.3e}. Use a fixed integer order."
                    ) from exc
                raise
            if not current.numel():
                raise ValueError(
                    "Adaptive projection cannot calibrate an empty batch; use a fixed integer order."
                )
            detached = current.detach()
            if previous is not None:
                if detached.shape != previous.shape:
                    raise ValueError(
                        "The projected function changed batch shape during adaptation."
                    )
                error = torch.max(torch.abs(detached - previous) / (1 + torch.abs(detached)))
                if float(error.item()) <= target:
                    return current
            previous = detached
        raise ValueError(
            f"Adaptive projection did not reach tolerance {target:.3e} by order "
            f"{_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
        )

    def projection_error(self, source, *, quadrature=None):
        """Estimate ``||source - P_N source||_L2`` on the fixed geometry."""
        if not isinstance(source, Coefficient) and not callable(source):
            raise TypeError("projection_error expects a callable function or a Coefficient.")
        mode, target = _quadrature_specification(quadrature, self.dtype)
        exact = _projection_error_exact_order(source, self.basis) if mode == "automatic" else None
        if mode == "fixed":
            return self._projection_error_on_table(source, self._projection_table(target))
        if exact is not None:
            order = max(self._basis_quadrature_order, exact)
            return self._projection_error_on_table(source, self._projection_table(order))

        if self._basis_quadrature_order > _MAX_ADAPTIVE_ORDER:
            raise ValueError(
                f"The basis requires order {self._basis_quadrature_order}, above the adaptive "
                f"limit {_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
            )
        previous = None
        for order in range(self._basis_quadrature_order, _MAX_ADAPTIVE_ORDER + 1, 2):
            try:
                current = self._projection_error_on_table(
                    source,
                    self._projection_table(order),
                )
            except ValueError as exc:
                if "exceeds max_points" in str(exc):
                    raise ValueError(
                        "Adaptive projection error exhausted max_quadrature_points before "
                        f"reaching tolerance {target:.3e}. Use a fixed integer order."
                    ) from exc
                raise
            if not current.numel():
                raise ValueError(
                    "Adaptive projection error cannot calibrate an empty batch; "
                    "use a fixed integer order."
                )
            detached = current.detach()
            if previous is not None:
                if detached.shape != previous.shape:
                    raise ValueError(
                        "The projected function changed batch shape during adaptation."
                    )
                error = torch.max(torch.abs(detached - previous) / (1 + torch.abs(detached)))
                if float(error.item()) <= target:
                    return current
            previous = detached
        raise ValueError(
            f"Adaptive projection error did not reach tolerance {target:.3e} by order "
            f"{_MAX_ADAPTIVE_ORDER}. Use a fixed integer order."
        )

    def quadrature_error(self, z, *, order=None):
        """Compare the reduced velocity with one assembled at a higher fixed order."""
        self._states(z)
        if not torch.isfinite(z).all():
            raise ValueError("States must be finite.")
        refined_order = self.quadrature_order + 2 if order is None else order
        refined_order = positive_integer(refined_order, "order", 0)
        if refined_order <= self.quadrature_order:
            raise ValueError("order must be greater than the field quadrature order.")
        refined = self.problem.field(
            basis=self.basis,
            quadrature=refined_order,
            max_quadrature_points=self.max_quadrature_points,
            max_intermediate_entries=self.max_intermediate_entries,
            device=self.device,
            dtype=self.dtype,
        )
        return torch.linalg.vector_norm(refined(z) - self(z), dim=-1)

    def solve(self, z0, times, *, step=None, tolerance=None):
        """Solve the autonomous Galerkin ODE at the requested times."""
        from .evolution import solve

        return solve(self, z0, times, step=step, tolerance=tolerance)

    def time_error(self, z0, times, *, step=None, tolerance=None):
        """Compare RK solutions after halving the step or adaptive tolerance."""
        from .evolution import time_error

        return time_error(self, z0, times, step=step, tolerance=tolerance)

    def _states(self, z):
        if not isinstance(z, torch.Tensor):
            raise TypeError("States must be torch tensors.")
        if z.ndim < 1 or z.shape[-1] != self.dimension:
            raise ValueError(f"Expected a state tensor with shape [...,{self.dimension}].")
        if z.device != self.device or z.dtype != self.dtype:
            raise ValueError("States and field tables must share device and dtype.")
        return z.reshape(-1, self.dimension), z.shape[:-1]

    def _action(self, z, test):
        result = z.new_zeros((len(z), test.stop - test.start))
        by_measure = {
            ("volume", None if name == "all" else name): table
            for name, table in self._volumes.items()
        }
        by_measure.update((("boundary", name), table) for name, table in self._boundary.items())
        caches = {key: {} for key in by_measure}
        for integral in self.form.integrals:
            label = integral.measure.label or (
                "all" if integral.measure.kind == "boundary" else None
            )
            key = (integral.measure.kind, label)
            table = by_measure[key]
            context = {
                "z": z,
                "test": test,
                "basis": table.basis.__getitem__,
                "coefficient": table.coefficients.__getitem__,
                "points": table.points,
                "normals": table.normals,
                "dtype": self.dtype,
                "device": self.device,
            }
            values = evaluate(integral.integrand, context, caches[key])
            expected = (len(z), test.stop - test.start, len(table.points))
            try:
                values = torch.broadcast_to(values, expected)
            except RuntimeError as exc:
                raise ValueError(
                    "A weak integrand must broadcast to [batch,test,quadrature]."
                ) from exc
            result = result + torch.einsum("bjq,q->bj", values, table.weights)
        return result

    def __call__(self, z):
        z, batch_shape = self._states(z)
        if not len(z):
            return z.clone().reshape((*batch_shape, self.dimension))
        denominator = max(1, len(z) * max(1, len(self._volume.points)))
        chunk = max(1, min(self.dimension, self.max_intermediate_entries // denominator))
        action = torch.cat(
            [
                self._action(z, slice(start, min(start + chunk, self.dimension)))
                for start in range(0, self.dimension, chunk)
            ],
            dim=1,
        )
        return action.reshape((*batch_shape, self.dimension))
