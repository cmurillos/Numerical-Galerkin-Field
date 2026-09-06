"""Homogeneous trace constraints with a verifiable nodal representation."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.linalg import svd

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
        if self.component >= space.components:
            raise ValueError("ZeroTrace component is outside the state value shape.")
        if space.regularity < 1:
            raise ValueError("ZeroTrace requires regularity >= 1; an L2 state has no such trace.")
        if self.boundary not in space.geometry.boundaries:
            raise ValueError(f"Unknown boundary {self.boundary!r}.")
        if not len(space.geometry.boundaries[self.boundary]):
            raise ValueError(f"ZeroTrace boundary {self.boundary!r} is empty.")


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
        "ZeroTrace needs a built-in FiniteElementBasis or its ComponentBasis, "
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


def restrict_basis(space, basis, *, tolerance=1e-12, max_matrix_entries=10_000_000):
    """Intersect the supplied nodal span with the declared trace kernels."""
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
            if r.component == component
        ]
        if groups:
            faces = np.unique(np.concatenate(groups))
            dofs = _boundary_dofs(carrier, space.geometry, faces)
            rows.append(matrix[dofs] / scales)
    n = basis.dimension
    _budget(sum(len(row) for row in rows), n, maximum)
    _budget(n, n, maximum)
    constraints = np.concatenate(rows)
    _, singular, right = svd(constraints, full_matrices=len(constraints) < n)
    threshold = tolerance * max(1.0, float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    if rank == n:
        raise ValueError(
            "The restrictions leave only the zero function in this candidate span; "
            "enrich the candidate space before selecting modes."
        )
    kernel = right[rank:].T
    error = float(np.max(np.abs(constraints @ kernel)))
    if error > 10 * threshold:
        raise RuntimeError("The numerical trace kernel failed its residual check.")
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
