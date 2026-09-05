"""General weak Galerkin fields on user-supplied bases and simplicial domains."""

from dataclasses import dataclass

import torch

from .forms import build_form, coefficient_expressions, derivative_orders, evaluate
from .geometry import SimplicialDomain, positive_integer
from .spaces import TransformedBasis


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
    expected_prefix = (len(points), basis.dimension, *basis.value_shape)
    tables = {}
    for order in sorted(set(orders) | {0}):
        values = basis.evaluate(points, order=order, cells=cells, barycentric=barycentric)
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=dtype, device=device)
        expected = (*expected_prefix, *((geometry.ambient_dimension,) * order))
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
        tables[order] = values.detach()
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


class GalerkinProblem:
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
        quadrature_order=None,
        quadrature_rule=None,
        max_quadrature_points=1_000_000,
        max_intermediate_entries=10_000_000,
        device="cpu",
        dtype=torch.float64,
    ):
        return GalerkinField(
            self,
            basis,
            quadrature_order=quadrature_order,
            quadrature_rule=quadrature_rule,
            max_quadrature_points=max_quadrature_points,
            max_intermediate_entries=max_intermediate_entries,
            device=device,
            dtype=dtype,
        )

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
        quadrature_order=None,
        quadrature_rule=None,
        max_quadrature_points=1_000_000,
        max_intermediate_entries=10_000_000,
        device="cpu",
        dtype=torch.float64,
    ):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64.")
        value_shape = _basis_contract(problem, basis)
        if quadrature_order is None:
            quadrature_order = getattr(basis, "quadrature_order", 4)
        quadrature_order = positive_integer(quadrature_order, "quadrature_order", 0)
        max_quadrature_points = positive_integer(max_quadrature_points, "max_quadrature_points")
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
        self._volumes = {
            name: _table(
                problem.geometry,
                basis,
                orders,
                coefficients["volume"].get(name, set()),
                quadrature_order,
                None,
                None if name == "all" else name,
                quadrature_rule,
                max_quadrature_points,
                device,
                dtype,
            )
            for name, orders in volume_orders.items()
        }
        self._volume = self._volumes["all"]
        self._boundary = {
            name: _table(
                problem.geometry,
                basis,
                orders,
                coefficients["boundary"].get(name, set()),
                quadrature_order,
                name,
                None,
                quadrature_rule,
                max_quadrature_points,
                device,
                dtype,
            )
            for name, orders in required["boundary"].items()
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

    def reconstruct(self, z, *, boundary=None):
        """Reconstruct ``Phi z`` as ``[..., Q, *value_shape]`` on a prepared measure."""
        z, batch_shape = self._states(z)
        table = self._volume if boundary is None else self._boundary[boundary]
        values = torch.einsum("bn,qn...->bq...", z, table.basis[0])
        return values.reshape((*batch_shape, len(table.points), *self.value_shape))

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
            if values.shape != (len(z), test.stop - test.start, len(table.points)):
                raise ValueError("A weak integrand did not evaluate to [batch,test,quadrature].")
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
