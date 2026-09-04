"""Geometry-aware construction of fixed L2-orthonormal bases."""

from itertools import product
from math import pi

import numpy as np
import torch
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .geometry import positive_integer, readonly
from .spaces import CallableBasis, FiniteElementBasis, PolynomialBasis, compositions

FAMILIES = ("laplacian", "polynomial", "fourier", "finite-element", "custom")


def _positive_float(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def _metadata(basis, *, family, quadrature_order, validation_order=None, **values):
    basis.family = family
    basis.quadrature_order = int(quadrature_order)
    if validation_order is not None:
        basis.validation_order = int(validation_order)
    for name, value in values.items():
        setattr(basis, name, value)
    return basis


def _eigenmodes(stiffness, mass, count, tolerance):
    size = mass.shape[0]
    if count > size:
        raise ValueError(f"Requested {count} modes but the finite-element space has {size} DOFs.")
    if size <= 256 or count >= size - 1:
        values, vectors = eigh(stiffness.toarray(), mass.toarray(), subset_by_index=(0, count - 1))
    else:
        scale = float(np.max(stiffness.diagonal() / mass.diagonal()))
        shift = -1e-8 * max(scale, np.finfo(float).tiny)
        values, vectors = eigsh(
            stiffness,
            k=count,
            M=mass,
            sigma=shift,
            which="LM",
            tol=tolerance,
            v0=np.random.default_rng(0).normal(size=size),
        )
    factor = cholesky(vectors.T @ (mass @ vectors), lower=False)
    vectors = solve_triangular(factor.T, vectors.T, lower=True).T
    reduced = vectors.T @ (stiffness @ vectors)
    values, rotation = eigh((reduced + reduced.T) / 2)
    vectors = vectors @ rotation
    scale = max(1.0, float(np.max(np.abs(values))))
    if values.min() < -1e-9 * scale:
        raise RuntimeError("The discrete Laplacian has a significantly negative eigenvalue.")
    values = np.maximum(values, 0.0)
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(count)])
    signs[signs == 0] = 1
    vectors *= signs
    return readonly(values), readonly(vectors)


def _finite_element_matrices(basis, order, max_points, max_entries):
    geometry = basis.geometry
    quadrature = geometry.quadrature(order, max_points=max_points)
    points = torch.tensor(quadrature.points.copy(), dtype=torch.float64)
    cells = torch.tensor(quadrature.cells.copy())
    barycentric = torch.tensor(quadrature.barycentric.copy(), dtype=torch.float64)
    local_values = basis.local_evaluate(
        points, order=0, cells=cells, barycentric=barycentric
    ).numpy()
    local_gradients = basis.local_evaluate(
        points, order=1, cells=cells, barycentric=barycentric
    ).numpy()
    local_count = local_values.shape[1]
    entries = len(points) * local_count * local_count
    if entries > positive_integer(max_entries, "max_matrix_entries"):
        raise ValueError(
            "Finite-element assembly exceeds max_matrix_entries; increase the budget explicitly."
        )
    weighted_mass = np.einsum(
        "qi,qj,q->qij", local_values, local_values, quadrature.weights, optimize=True
    )
    weighted_stiffness = np.einsum(
        "qia,qja,q->qij",
        local_gradients,
        local_gradients,
        quadrature.weights,
        optimize=True,
    )
    dofs = basis.element_dofs[quadrature.cells]
    rows = np.broadcast_to(dofs[:, :, None], weighted_mass.shape)
    columns = np.broadcast_to(dofs[:, None, :], weighted_mass.shape)
    shape = (basis.ndofs, basis.ndofs)
    mass = coo_matrix((weighted_mass.ravel(), (rows.ravel(), columns.ravel())), shape=shape)
    stiffness = coo_matrix(
        (weighted_stiffness.ravel(), (rows.ravel(), columns.ravel())), shape=shape
    )
    return mass.tocsr(), stiffness.tocsr()


def _laplacian(
    problem,
    *,
    size,
    degree=1,
    quadrature_order=None,
    validation_order=None,
    tolerance=1e-10,
    max_dofs=100_000,
    max_quadrature_points=1_000_000,
    max_matrix_entries=20_000_000,
):
    size = positive_integer(size, "size")
    degree = positive_integer(degree, "degree")
    tolerance = _positive_float(tolerance, "tolerance")
    raw = FiniteElementBasis(problem.geometry, degree, max_dofs=max_dofs)
    order = (
        2 * degree
        if quadrature_order is None
        else positive_integer(quadrature_order, "quadrature_order", 0)
    )
    if order < 2 * degree:
        raise ValueError("Laplacian quadrature_order must be at least 2*degree.")
    mass, stiffness = _finite_element_matrices(
        raw, order, max_quadrature_points, max_matrix_entries
    )
    eigenvalues, vectors = _eigenmodes(stiffness, mass, size, tolerance)
    result = FiniteElementBasis(
        problem.geometry,
        degree,
        coefficients=vectors,
        max_dofs=max(max_dofs, raw.element_dofs.size),
    )
    validation = (
        order + 2
        if validation_order is None
        else positive_integer(validation_order, "validation_order", 0)
    )
    _metadata(
        result,
        family="laplacian",
        quadrature_order=order,
        validation_order=validation,
        eigenvalues=eigenvalues,
    )
    result.orthonormality_error = problem.validate_basis(
        result,
        quadrature_order=validation,
        tolerance=max(1e-9, 10 * tolerance),
        max_quadrature_points=max_quadrature_points,
    )
    return result


def _polynomial(
    problem,
    *,
    size=None,
    degree=None,
    center=None,
    scale=None,
    quadrature_order=None,
    validation_order=None,
    tolerance=1e-9,
    max_quadrature_points=1_000_000,
):
    if size is None and degree is None:
        raise ValueError("A polynomial basis needs size or degree.")
    if degree is None:
        size = positive_integer(size, "size")
        exponents, total = [], 0
        while len(exponents) < size:
            exponents.extend(compositions(total, problem.geometry.ambient_dimension))
            total += 1
        exponents = exponents[:size]
    else:
        degree = positive_integer(degree, "degree", 0)
        exponents = [
            exponent
            for total in range(degree + 1)
            for exponent in compositions(total, problem.geometry.ambient_dimension)
        ]
        if size is not None:
            size = positive_integer(size, "size")
            if size > len(exponents):
                raise ValueError("size exceeds the number of monomials allowed by degree.")
            exponents = exponents[:size]
    maximum_degree = max(map(sum, exponents))
    vertices = problem.geometry.vertices
    if center is None:
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    if scale is None:
        scale = (vertices.max(axis=0) - vertices.min(axis=0)) / 2
        scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    raw = PolynomialBasis(
        problem.geometry.ambient_dimension,
        exponents=exponents,
        degree=maximum_degree,
        center=center,
        scale=scale,
    )
    order = max(2, 2 * maximum_degree) if quadrature_order is None else quadrature_order
    result = problem.orthonormalize(
        raw,
        quadrature_order=order,
        validation_order=validation_order,
        tolerance=tolerance,
        max_quadrature_points=max_quadrature_points,
    )
    result.family = "polynomial"
    return result


def _frequency_vectors(dimension, periods, count, max_candidates):
    if count <= 0:
        return []
    candidates, radius = set(), 0
    while 2 * len(candidates) < count:
        radius += 1
        if (2 * radius + 1) ** dimension > max_candidates:
            raise ValueError("Fourier candidate budget exceeded; increase max_candidates.")
        for frequency in product(range(-radius, radius + 1), repeat=dimension):
            if not any(frequency):
                continue
            first = next(value for value in frequency if value)
            if first > 0:
                candidates.add(frequency)
    return sorted(
        candidates,
        key=lambda frequency: (
            sum((value / period) ** 2 for value, period in zip(frequency, periods)),
            frequency,
        ),
    )


def _fourier(
    problem,
    *,
    size,
    periods=None,
    origin=None,
    quadrature_order=12,
    validation_order=None,
    tolerance=1e-8,
    max_candidates=1_000_000,
    max_quadrature_points=1_000_000,
):
    size = positive_integer(size, "size")
    vertices = problem.geometry.vertices
    ranges = vertices.max(axis=0) - vertices.min(axis=0)
    if periods is None:
        active = np.flatnonzero(ranges > np.finfo(float).eps)
        if not len(active):
            raise ValueError("Cannot infer Fourier periods from a constant geometry.")
        periods = ranges[active]
    else:
        periods = np.asarray(periods, dtype=float)
        if periods.shape != (problem.geometry.ambient_dimension,):
            raise ValueError("periods must contain one positive value per ambient coordinate.")
        active = np.arange(problem.geometry.ambient_dimension)
    if not np.isfinite(periods).all() or np.any(periods <= 0):
        raise ValueError("Fourier periods must be finite and positive.")
    if origin is None:
        origin = vertices.min(axis=0)[active]
    else:
        origin = np.asarray(origin, dtype=float)
        if origin.shape == (problem.geometry.ambient_dimension,):
            origin = origin[active]
        if origin.shape != (len(active),) or not np.isfinite(origin).all():
            raise ValueError("origin must match the Fourier coordinate dimension.")
    frequencies = _frequency_vectors(
        len(active), periods, size - 1, positive_integer(max_candidates, "max_candidates")
    )
    descriptors = [("constant", (0,) * len(active))]
    for frequency in frequencies:
        descriptors.extend((("cos", frequency), ("sin", frequency)))
    descriptors = descriptors[:size]
    frequency_tensor = np.asarray([item[1] for item in descriptors], dtype=float)
    kinds = tuple(item[0] for item in descriptors)

    def values(x):
        axes = torch.as_tensor(active, dtype=torch.long, device=x.device)
        coordinates = (x.index_select(1, axes) - x.new_tensor(origin)) / x.new_tensor(periods)
        phases = 2 * pi * coordinates @ x.new_tensor(frequency_tensor).T
        columns = []
        for index, kind in enumerate(kinds):
            if kind == "constant":
                columns.append(torch.ones_like(phases[:, index]))
            elif kind == "cos":
                columns.append(torch.cos(phases[:, index]))
            else:
                columns.append(torch.sin(phases[:, index]))
        return torch.stack(columns, dim=1)

    raw = CallableBasis(values, dimension=size)
    raw.frequencies = readonly(frequency_tensor, np.int64)
    raw.kinds = kinds
    order = positive_integer(quadrature_order, "quadrature_order", 0)
    result = problem.orthonormalize(
        raw,
        quadrature_order=order,
        validation_order=validation_order,
        tolerance=tolerance,
        max_quadrature_points=max_quadrature_points,
    )
    result.family = "fourier"
    result.periods = readonly(periods)
    result.origin = readonly(origin)
    result.active_axes = readonly(active, np.int64)
    return result


def _finite_element(
    problem,
    *,
    degree=1,
    quadrature_order=None,
    validation_order=None,
    tolerance=1e-9,
    max_dofs=4_096,
    max_quadrature_points=1_000_000,
):
    raw = FiniteElementBasis(problem.geometry, degree, max_dofs=max_dofs)
    order = 2 * raw.degree if quadrature_order is None else quadrature_order
    result = problem.orthonormalize(
        raw,
        quadrature_order=order,
        validation_order=validation_order,
        tolerance=tolerance,
        max_quadrature_points=max_quadrature_points,
    )
    result.family = "finite-element"
    return result


def _custom(
    problem,
    *,
    source,
    quadrature_order=8,
    validation_order=None,
    tolerance=1e-8,
    max_quadrature_points=1_000_000,
):
    result = problem.orthonormalize(
        source,
        quadrature_order=quadrature_order,
        validation_order=validation_order,
        tolerance=tolerance,
        max_quadrature_points=max_quadrature_points,
    )
    result.family = "custom"
    return result


def build_basis(problem, family, options):
    if not isinstance(family, str):
        raise TypeError("basis family must be a string.")
    family = family.strip().lower().replace("_", "-")
    builders = {
        "laplacian": _laplacian,
        "polynomial": _polynomial,
        "fourier": _fourier,
        "finite-element": _finite_element,
        "custom": _custom,
    }
    if family not in builders:
        names = ", ".join(FAMILIES)
        raise ValueError(f"Unknown basis family {family!r}; choose one of: {names}.")
    return builders[family](problem, **options)
