"""Homogeneous trace, periodicity and integral constraints on nodal spans."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
import torch
from scipy.linalg import svd
from scipy.sparse import csr_matrix

from .geometry import positive_integer
from .spaces import ComponentBasis, FiniteElementBasis, ProductBasis, TransformedBasis


@dataclass(frozen=True, kw_only=True)
class ZeroTrace:
    """Require one state component to vanish on a named exterior boundary."""

    component: int
    boundary: str

    def __post_init__(self):
        object.__setattr__(self, "component", positive_integer(self.component, "component", 0))
        if not isinstance(self.boundary, str) or not self.boundary:
            raise ValueError("boundary must be a nonempty name.")

    def _validate(self, space):
        _validate_boundary(space, self.component, self.boundary, "ZeroTrace")


def _validate_boundary(space, component, boundary, kind):
    if component >= space.components:
        raise ValueError(f"{kind} component is outside the state value shape.")
    if space.regularity < 1:
        raise ValueError(f"{kind} requires regularity >= 1; an L2 state has no such trace.")
    if boundary not in space.geometry.boundaries:
        raise ValueError(f"Unknown boundary {boundary!r}.")
    if not len(space.geometry.boundaries[boundary]):
        raise ValueError(f"{kind} boundary {boundary!r} is empty.")


@dataclass(frozen=True, kw_only=True)
class Periodic:
    """Match scalar traces by a vertex bijection preserving boundary face connectivity.

    vertex_pairs are mesh indices (source, target). The map extends affinely on
    each face to all higher-degree nodes. It does not rotate state components.
    """

    component: int
    boundaries: tuple[str, str]
    vertex_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self):
        object.__setattr__(self, "component", positive_integer(self.component, "component", 0))
        if (
            not isinstance(self.boundaries, (list, tuple))
            or len(self.boundaries) != 2
            or any(not isinstance(x, str) or not x for x in self.boundaries)
            or self.boundaries[0] == self.boundaries[1]
        ):
            raise ValueError("boundaries must contain two distinct nonempty boundary names.")
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        pairs = np.asarray(self.vertex_pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or pairs.dtype.kind not in "iu":
            raise ValueError("vertex_pairs must be a nonempty integer array [pairs,2].")
        if np.any(pairs < 0):
            raise ValueError("vertex_pairs cannot contain negative vertex indices.")
        if any(len(np.unique(pairs[:, axis])) != len(pairs) for axis in (0, 1)):
            raise ValueError("vertex_pairs must define a bijection, without repeated endpoints.")
        object.__setattr__(
            self, "vertex_pairs", tuple(sorted(tuple(map(int, row)) for row in pairs))
        )

    def _validate(self, space):
        for boundary in self.boundaries:
            _validate_boundary(space, self.component, boundary, "Periodic")
        geometry = space.geometry
        source, target = [
            geometry.exterior_faces[geometry.boundaries[name]] for name in self.boundaries
        ]
        mapping = dict(self.vertex_pairs)
        if set(mapping) != set(source.ravel()) or set(mapping.values()) != set(target.ravel()):
            raise ValueError("vertex_pairs must cover exactly the vertices of both boundaries.")
        mapped_faces = {tuple(sorted(mapping[int(v)] for v in face)) for face in source}
        target_faces = {tuple(sorted(map(int, face))) for face in target}
        if mapped_faces != target_faces:
            raise ValueError(
                "Periodic boundaries must have matching face connectivity under vertex_pairs."
            )


@dataclass(frozen=True, kw_only=True)
class MeanZero:
    """Require the integral of one component over a volume region to vanish."""

    component: int
    region: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "component", positive_integer(self.component, "component", 0))
        if self.region is not None and (not isinstance(self.region, str) or not self.region):
            raise ValueError("region must be None or a nonempty region name.")
        if self.region == "all":
            object.__setattr__(self, "region", None)

    def _validate(self, space):
        if self.component >= space.components:
            raise ValueError("MeanZero component is outside the state value shape.")
        region = self.region or "all"
        if region not in space.geometry.regions:
            raise ValueError(f"Unknown region {region!r}.")
        if not len(space.geometry.regions[region]):
            raise ValueError(f"MeanZero region {region!r} is empty.")


def _budget(rows, columns, maximum):
    if rows * columns > maximum:
        raise ValueError(
            "Restriction construction exceeds max_matrix_entries; "
            "reduce the candidate space or increase the budget explicitly."
        )


def _nodal_blocks(basis, geometry, maximum):
    """Return (scalar nodal carrier, coefficient matrix) for every component.

    Only built-in representations are certified: arbitrary evaluate callbacks,
    including overrides in subclasses, cannot establish a polynomial face trace.
    """
    if type(basis) is FiniteElementBasis:
        if not geometry.same_mesh(basis.geometry):
            raise ValueError("The candidate basis belongs to a different mesh.")
        if len(basis.value_shape) > 1:
            raise ValueError("Space currently requires a vector of scalar components.")
        _budget(
            basis.ndofs,
            basis.dimension * (basis.value_shape[0] if basis.value_shape else 1),
            maximum,
        )
        if basis.coefficients is None:
            return [(basis, np.eye(basis.ndofs))]
        coefficients = basis.coefficients.reshape(basis.ndofs, basis.dimension, -1)
        return [(basis, coefficients[:, :, r]) for r in range(coefficients.shape[-1])]
    if type(basis) is TransformedBasis:
        blocks = _nodal_blocks(basis.base, geometry, maximum)
        transform = basis.transform.detach().cpu().numpy()
        _budget(sum(len(matrix) for _, matrix in blocks), basis.dimension, maximum)
        result = []
        for carrier, matrix in blocks:
            result.append((carrier, matrix @ transform))
        return result
    if type(basis) is ComponentBasis:
        blocks = _nodal_blocks(basis.base, geometry, maximum)
        if len(blocks) != 1 or len(basis.value_shape) != 1:
            raise ValueError("ComponentBasis must repeat a scalar nodal basis.")
        carrier, matrix = blocks[0]
        _budget(len(matrix), basis.dimension * basis.value_shape[0], maximum)
        result = []
        for component in range(basis.value_shape[0]):
            expanded = np.zeros((len(matrix), basis.dimension))
            start = component * basis.base.dimension
            expanded[:, start : start + basis.base.dimension] = matrix
            result.append((carrier, expanded))
        return result
    if type(basis) is ProductBasis:
        result = []
        for scalar, columns in zip(basis.bases, basis.slices):
            blocks = _nodal_blocks(scalar, geometry, maximum)
            if len(blocks) != 1:
                raise ValueError("ProductBasis must contain scalar nodal bases.")
            carrier, matrix = blocks[0]
            _budget(sum(len(item) for _, item in result) + len(matrix), basis.dimension, maximum)
            expanded = np.zeros((len(matrix), basis.dimension))
            expanded[:, columns] = matrix
            result.append((carrier, expanded))
        return result
    raise NotImplementedError(
        "Restrictions need a built-in FiniteElementBasis or its ComponentBasis, "
        "ProductBasis and TransformedBasis compositions; arbitrary callbacks "
        "do not certify the trace on an entire face."
    )


def _boundary_dofs(carrier, geometry, faces):
    """All face nodes, including edge/face-interior nodes at high degree."""
    owners = geometry._owners[faces]
    indices = np.asarray(carrier.indices)
    selected = []
    for cell, opposite in owners:
        selected.extend(carrier.element_dofs[cell, indices[:, opposite] == 0])
    return np.unique(selected).astype(np.int64)


def _periodic_dofs(carrier, geometry, restriction):
    # Integer barycentric weights identify nodes without coordinate tolerances.
    nodes = {}
    for vertices, dofs in zip(geometry.simplices, carrier.element_dofs):
        for alpha, dof in zip(carrier.indices, dofs):
            key = tuple(sorted((int(v), a) for v, a in zip(vertices, alpha) if a))
            nodes[key] = int(dof)
    mapping = dict(restriction.vertex_pairs)
    faces = geometry.boundaries[restriction.boundaries[0]]
    pairs = set()
    for cell, opposite in geometry._owners[faces]:
        vertices = geometry.simplices[cell]
        for alpha, dof in zip(carrier.indices, carrier.element_dofs[cell]):
            if alpha[opposite] == 0:
                mapped = tuple(sorted((mapping[int(v)], a) for v, a in zip(vertices, alpha) if a))
                pairs.add((int(dof), nodes[mapped]))
    return np.asarray(sorted(pairs), dtype=np.int64)


def _mean_weights(carrier, geometry, region, *, max_quadrature_points, maximum):
    q = geometry.quadrature(max(2, carrier.degree), region=region, max_points=max_quadrature_points)
    _budget(len(q.points), len(carrier.indices), maximum)
    local = carrier.local_evaluate(
        torch.tensor(q.points.copy(), dtype=torch.float64),
        cells=torch.tensor(q.cells.copy()),
        barycentric=torch.tensor(q.barycentric.copy(), dtype=torch.float64),
    ).numpy()
    weights = np.zeros(carrier.ndofs)
    np.add.at(weights, carrier.element_dofs[q.cells].ravel(), (local * q.weights[:, None]).ravel())
    # Normalize the functional, not its possibly already-zero candidate values.
    scale = np.abs(weights).sum()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MeanZero needs a finite, nonzero integration functional.")
    return weights / scale


def _kernel(constraints, tolerance, maximum):
    n = constraints.shape[1]
    _budget(len(constraints), n, maximum)
    _budget(n, n, maximum)
    if not n:
        return np.empty((0, 0)), 0, 0.0
    if not len(constraints):
        return np.eye(n), 0, 0.0
    _, singular, right = svd(constraints, full_matrices=len(constraints) < n)
    threshold = tolerance * max(1.0, float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    kernel = right[rank:].T
    error = float(np.max(np.abs(constraints @ kernel))) if kernel.size else 0.0
    if error > 10 * threshold:
        raise RuntimeError("The numerical constraint kernel failed its residual check.")
    return kernel, rank, error


def nodal_prolongations(space, carrier, *, tolerance, maximum, max_quadrature_points):
    """Sparse identifications/pinning, then integral kernels, before spectral selection."""
    prolongations, residuals = [], []
    for component in range(space.components):
        parent = np.arange(carrier.ndofs)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        restrictions = [r for r in space.restrictions if r.component == component]
        for r in restrictions:
            if type(r) is Periodic:
                for source, target in _periodic_dofs(carrier, space.geometry, r):
                    a, b = root(source), root(target)
                    parent[max(a, b)] = min(a, b)
        roots = np.asarray([root(i) for i in range(carrier.ndofs)])
        fixed = set()
        for r in restrictions:
            if type(r) is ZeroTrace:
                dofs = _boundary_dofs(
                    carrier, space.geometry, space.geometry.boundaries[r.boundary]
                )
                fixed.update(roots[dofs])
        active = np.flatnonzero(~np.isin(roots, list(fixed)))
        classes, columns = np.unique(roots[active], return_inverse=True)
        P = csr_matrix(
            (np.ones(len(active)), (active, columns)), shape=(carrier.ndofs, len(classes))
        )
        mean_restrictions = [r for r in restrictions if type(r) is MeanZero]
        _budget(len(mean_restrictions), carrier.ndofs, maximum)
        means = [
            _mean_weights(
                carrier,
                space.geometry,
                r.region,
                max_quadrature_points=max_quadrature_points,
                maximum=maximum,
            )
            for r in mean_restrictions
        ]
        error = 0.0
        if means and P.shape[1]:
            _budget(carrier.ndofs, P.shape[1], maximum)
            constraints = np.asarray([weights @ P for weights in means])
            kernel, _, error = _kernel(constraints, tolerance, maximum)
            P = csr_matrix(P @ kernel)
        prolongations.append(P)
        residuals.append(error)
    return prolongations, max(residuals, default=0.0)


def restrict_basis(
    space, basis, *, tolerance=1e-12, max_matrix_entries=10_000_000, max_quadrature_points=1_000_000
):
    """Intersect the supplied nodal span with the declared homogeneous kernels."""
    maximum = positive_integer(max_matrix_entries, "max_matrix_entries")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not 0 < tolerance < 1:
        raise ValueError("tolerance must be a real number strictly between zero and one.")
    if tuple(getattr(basis, "value_shape", ())) != space.value_shape:
        raise ValueError(
            "The candidate value_shape must equal Space.value_shape; "
            "use ComponentBasis for explicit scalar components."
        )
    if space.regularity > 1:
        raise NotImplementedError(
            "This restriction builder certifies H1 nodal conformity, not higher Sobolev orders."
        )
    blocks = _nodal_blocks(basis, space.geometry, maximum)
    if any(not np.isfinite(matrix).all() for _, matrix in blocks):
        raise ValueError("The candidate nodal representation contains nonfinite coefficients.")
    if not space.restrictions:
        return basis

    # Normalize each candidate column before rank detection: changing the units of
    # an individual mode must not make its nonzero trace disappear numerically.
    scales = np.max([np.max(np.abs(matrix), axis=0) for _, matrix in blocks], axis=0)
    if np.any(scales == 0):
        raise ValueError("The candidate basis contains an identically zero mode.")
    nodal = np.concatenate([matrix / scales for _, matrix in blocks])
    nodal_singular = svd(nodal, compute_uv=False)
    if len(nodal_singular) < basis.dimension or nodal_singular[-1] <= tolerance * max(
        1.0, float(nodal_singular[0])
    ):
        raise ValueError("The candidate nodal modes are linearly dependent within tolerance.")
    rows = []
    for component, (carrier, matrix) in enumerate(blocks):
        groups = [
            space.geometry.boundaries[r.boundary]
            for r in space.restrictions
            if r.component == component and type(r) is ZeroTrace
        ]
        if groups:
            faces = np.unique(np.concatenate(groups))
            dofs = _boundary_dofs(carrier, space.geometry, faces)
            rows.append(matrix[dofs] / scales)
        for r in space.restrictions:
            if r.component != component:
                continue
            if type(r) is Periodic:
                pairs = _periodic_dofs(carrier, space.geometry, r)
                rows.append((matrix[pairs[:, 0]] - matrix[pairs[:, 1]]) / scales)
            elif type(r) is MeanZero:
                weights = _mean_weights(
                    carrier,
                    space.geometry,
                    r.region,
                    max_quadrature_points=max_quadrature_points,
                    maximum=maximum,
                )
                rows.append((weights @ (matrix / scales))[None, :])
    n = basis.dimension
    _budget(sum(len(row) for row in rows), n, maximum)
    _budget(n, n, maximum)
    constraints = np.concatenate(rows)
    kernel, rank, error = _kernel(constraints, tolerance, maximum)
    if rank == n:
        raise ValueError(
            "The restrictions leave only the zero function in this candidate span; "
            "enrich the candidate space before selecting modes."
        )
    # Column rescaling changes neither the kernel nor the represented subspace.
    with np.errstate(over="ignore", invalid="ignore"):
        transform = kernel / scales[:, None]
    if not np.isfinite(transform).all():
        raise ValueError("The candidate scaling is too extreme; rescale the input modes.")
    transform /= np.max(np.abs(transform), axis=0)
    result = TransformedBasis(basis, transform)
    result.space = space
    result.restriction_rank = rank
    result.restriction_error = error
    return result
