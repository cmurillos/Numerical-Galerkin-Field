"""Operational bases prepared on an admissible Space, independently of a PDE."""

from math import comb

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import block_diag

from .basis_factory import (
    FAMILIES,
    _eigenmodes,
    _finite_element_matrices,
    _metadata,
    build_basis,
)
from .galerkin import _basis_contract, _BasisGeometry, _tolerance
from .geometry import positive_integer, readonly
from .restrictions import _boundary_dofs
from .spaces import (
    ComponentBasis,
    FiniteElementBasis,
    PolynomialBasis,
    ProductBasis,
    TransformedBasis,
)


def _budget(rows, columns, maximum):
    if rows * columns > maximum:
        raise ValueError(
            "Basis preparation exceeds max_matrix_entries; "
            "reduce the space or increase the budget explicitly."
        )


def _allocation(space, size, component_sizes):
    if size is not None:
        size = positive_integer(size, "size")
    if component_sizes is None:
        return size, None
    if not isinstance(component_sizes, (list, tuple)) or len(component_sizes) != space.components:
        raise ValueError("component_sizes must contain one count per state component.")
    counts = tuple(positive_integer(n, "component_sizes entry", 0) for n in component_sizes)
    total = sum(counts)
    if total == 0 or (size is not None and total != size):
        raise ValueError("component_sizes must have a positive sum equal to size when supplied.")
    return total, counts


def _known_h1(basis):
    # Exact types: an overridden evaluator cannot inherit a conformity certificate.
    if type(basis) in (FiniteElementBasis, PolynomialBasis):
        return True
    if type(basis) in (ComponentBasis, TransformedBasis):
        return _known_h1(basis.base)
    if type(basis) is ProductBasis:
        return all(_known_h1(item) for item in basis.bases)
    return False


def _check_size(size, actual):
    if size is not None and size != actual:
        raise ValueError(
            f"size={size} differs from the full admissible dimension {actual}; "
            "this family does not truncate its span."
        )


def _free_dofs(space, raw):
    free = []
    for component in range(space.components):
        faces = [
            space.geometry.boundaries[r.boundary]
            for r in space.restrictions
            if r.component == component
        ]
        fixed = (
            _boundary_dofs(raw, space.geometry, np.unique(np.concatenate(faces)))
            if faces
            else np.empty(0, dtype=np.int64)
        )
        free.append(np.setdiff1d(np.arange(raw.ndofs), fixed))
    return free


def _nodal(
    space,
    family,
    size,
    counts,
    *,
    degree=1,
    quadrature_order=None,
    validation_order=None,
    tolerance=1e-10,
    max_dofs=100_000,
    max_quadrature_points=1_000_000,
    max_matrix_entries=20_000_000,
):
    degree = positive_integer(degree, "degree")
    tolerance = _tolerance(tolerance)
    maximum = positive_integer(max_matrix_entries, "max_matrix_entries")
    raw = FiniteElementBasis(space.geometry, degree, max_dofs=max_dofs)
    free = _free_dofs(space, raw)
    dimensions = tuple(map(len, free))
    total = sum(dimensions)
    if not total:
        raise ValueError("The admissible nodal space is zero; enrich the mesh or degree.")
    if family == "finite-element":
        _check_size(size, total)
        size = total
        if counts is not None and counts != dimensions:
            raise ValueError("component_sizes must match the full admissible nodal dimensions.")
        counts = dimensions
    elif size is None:
        raise ValueError("A laplacian basis needs size or component_sizes.")
    if size > total or (counts is not None and any(n > d for n, d in zip(counts, dimensions))):
        raise ValueError(f"Requested modes exceed admissible DOFs per component {dimensions}.")
    _budget(raw.ndofs * space.components, size, maximum)
    _budget(total, size, maximum)
    # _eigenmodes uses dense eigh for these cases; enforce the budget before allocation.
    spectral_sizes = [total] if counts is None else [d for d, n in zip(dimensions, counts) if n]
    spectral_counts = [size] if counts is None else [n for n in counts if n]
    for d, n in zip(spectral_sizes, spectral_counts):
        if family == "finite-element" or d <= 256 or n >= d - 1:
            _budget(d, d, maximum)
    order = (
        2 * degree
        if quadrature_order is None
        else positive_integer(quadrature_order, "quadrature_order", 0)
    )
    if order < 2 * degree:
        raise ValueError("Nodal basis quadrature_order must be at least 2*degree.")
    validation = (
        order + 2
        if validation_order is None
        else positive_integer(validation_order, "validation_order", 0)
    )
    if validation < 2 * degree:
        raise ValueError("Nodal basis validation_order must be at least 2*degree.")
    mass, stiffness = _finite_element_matrices(raw, order, max_quadrature_points, maximum)
    masses = [mass[dofs][:, dofs] for dofs in free]
    stiffnesses = [stiffness[dofs][:, dofs] for dofs in free]
    coefficients = np.zeros((raw.ndofs, size, space.components))
    eigenvalues = None
    if family == "laplacian" and counts is None:
        # Restriction precedes spectral selection, including the global component allocation.
        eigenvalues, vectors = _eigenmodes(
            block_diag(stiffnesses, format="csr"), block_diag(masses, format="csr"), size, tolerance
        )
        offset = 0
        for component, dofs in enumerate(free):
            coefficients[dofs, :, component] = vectors[offset : offset + len(dofs)]
            offset += len(dofs)
    else:
        offset, values = 0, []
        for component, (dofs, count) in enumerate(zip(free, counts)):
            if not count:
                continue
            if family == "laplacian":
                eigen, vectors = _eigenmodes(
                    stiffnesses[component], masses[component], count, tolerance
                )
                values.extend(eigen)
            else:
                factor = cholesky(masses[component].toarray(), lower=True)
                vectors = solve_triangular(factor.T, np.eye(count), lower=False)
            coefficients[dofs, offset : offset + count, component] = vectors
            offset += count
        if family == "laplacian":
            eigenvalues = readonly(values)
    result = FiniteElementBasis(
        space.geometry, degree, coefficients=coefficients, max_dofs=max_dofs
    )
    _metadata(result, family=family, quadrature_order=order, validation_order=validation)
    if eigenvalues is not None:
        result.eigenvalues = eigenvalues
    result.component_sizes = counts  # None: global spectrum, possibly coupled degenerate modes.
    result.admissible_dofs = dimensions
    result.restriction_rank = raw.ndofs * space.components - total
    result.restriction_error = 0.0  # eliminated nodal coefficients are exactly zero
    result.regularity_verified = True
    result.orthonormality_error = _BasisGeometry(space.geometry).validate_basis(
        result,
        quadrature_order=validation,
        tolerance=max(1e-9, 10 * tolerance),
        max_quadrature_points=max_quadrature_points,
    )
    return result


def _functional(space, family, size, counts, *, max_matrix_entries=20_000_000, **options):
    if space.restrictions:
        raise NotImplementedError(
            f"{family} does not yet implement complete ZeroTrace constraints. "
            "Use laplacian, finite-element, or a custom nodal source."
        )
    maximum = positive_integer(max_matrix_entries, "max_matrix_entries")
    if size is not None and counts is None:
        quotient, remainder = divmod(size, space.components)
        counts = tuple(quotient + (r < remainder) for r in range(space.components))
    scalar_options = dict(options)
    if counts is not None:
        scalar_dimension = max(counts)
        scalar_options["size"] = scalar_dimension
    elif family == "polynomial" and options.get("degree") is not None:
        degree = positive_integer(options["degree"], "degree", 0)
        scalar_dimension = comb(degree + space.geometry.ambient_dimension, degree)
        size = scalar_dimension * space.components
    else:
        raise ValueError("Provide size, component_sizes, or degree for a full polynomial family.")
    # Check the scalar Gram matrix and component selection before either is allocated.
    _budget(scalar_dimension, scalar_dimension, maximum)
    _budget(scalar_dimension * space.components, size, maximum)
    scalar = build_basis(_BasisGeometry(space.geometry), family, scalar_options)
    if counts is None:
        counts = (scalar.dimension,) * space.components
        size = sum(counts)
    expanded = ComponentBasis(scalar, components=space.components)
    _budget(expanded.dimension, size, maximum)
    selected = [r * scalar.dimension + j for r, count in enumerate(counts) for j in range(count)]
    transform = np.zeros((expanded.dimension, size))
    transform[selected, np.arange(size)] = 1.0
    result = TransformedBasis(expanded, transform)
    result.family = family
    result.component_sizes = counts
    result.regularity_verified = True
    # Direct sums and subsets of an orthonormal basis retain its scalar Gram bound.
    result.orthonormality_error = scalar.orthonormality_error
    return result


def _custom(
    space,
    size,
    counts,
    *,
    source,
    quadrature_order=8,
    validation_order=None,
    tolerance=1e-8,
    restriction_tolerance=1e-12,
    max_quadrature_points=1_000_000,
    max_matrix_entries=10_000_000,
):
    if counts is not None:
        raise ValueError("component_sizes is not defined for a custom, possibly coupled span.")
    context = _BasisGeometry(space.geometry)
    if _basis_contract(context, source) != space.value_shape:
        raise ValueError(f"Custom source must have value_shape={space.value_shape}.")
    maximum = positive_integer(max_matrix_entries, "max_matrix_entries")
    _budget(source.dimension, source.dimension, maximum)
    admissible = (
        space.restrict(source, tolerance=restriction_tolerance, max_matrix_entries=maximum)
        if space.restrictions
        else source
    )
    _check_size(size, admissible.dimension)
    result = context.orthonormalize(
        admissible,
        quadrature_order=quadrature_order,
        validation_order=validation_order,
        tolerance=tolerance,
        max_quadrature_points=max_quadrature_points,
    )
    result.family = "custom"
    result.component_sizes = None
    result.regularity_verified = _known_h1(source)
    for name in ("restriction_rank", "restriction_error"):
        if hasattr(admissible, name):
            setattr(result, name, getattr(admissible, name))
    return result


def build_space_basis(space, family, *, size=None, component_sizes=None, **options):
    if not isinstance(family, str):
        raise TypeError("basis family must be a string.")
    family = family.strip().lower().replace("_", "-")
    if family not in FAMILIES:
        raise ValueError(f"Unknown basis family {family!r}; choose one of: {', '.join(FAMILIES)}.")
    if space.regularity > 1:
        raise NotImplementedError("Space.basis currently supports only regularity zero or one.")
    size, counts = _allocation(space, size, component_sizes)
    if family in ("laplacian", "finite-element"):
        result = _nodal(space, family, size, counts, **options)
    elif family == "custom":
        result = _custom(space, size, counts, **options)
    else:
        result = _functional(space, family, size, counts, **options)
    result.space = space
    result.geometry = space.geometry
    return result
