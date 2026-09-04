import math

import numpy as np
import pytest
import torch

from ngfield import (
    CallableBasis,
    ComponentBasis,
    FiniteElementBasis,
    GalerkinProblem,
    PolynomialBasis,
    ProductBasis,
    grad,
    inner,
)


def interval_problem(intervals=8, weak=None):
    vertices = np.linspace(0.0, 1.0, intervals + 1)[:, None]
    simplices = np.column_stack((np.arange(intervals), np.arange(1, intervals + 1)))
    return GalerkinProblem(
        vertices=vertices,
        simplices=simplices,
        weak=weak or (lambda u, v, dx, ds: u * v * dx),
    )


def torus_mesh(n_major=6, n_minor=4):
    vertices = []
    for i in range(n_major):
        angle = 2 * math.pi * i / n_major
        for j in range(n_minor):
            minor = 2 * math.pi * j / n_minor
            radius = 2.0 + 0.5 * math.cos(minor)
            vertices.append(
                [radius * math.cos(angle), radius * math.sin(angle), 0.5 * math.sin(minor)]
            )

    def index(i, j):
        return (i % n_major) * n_minor + (j % n_minor)

    simplices = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i, j + 1), index(i + 1, j + 1)
            simplices.extend(([a, b, d], [a, d, c]))
    return np.asarray(vertices), np.asarray(simplices)


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_laplacian_factory_is_dimension_independent(dimension):
    vertices = np.vstack((np.zeros(dimension), np.eye(dimension)))
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=[np.arange(dimension + 1)],
        weak=lambda u, v, dx, ds: -inner(grad(u), grad(v)) * dx,
    )
    basis = problem.basis("laplacian", size=dimension + 1)
    field = problem.field(basis=basis)
    z = torch.linspace(0.1, 0.4, basis.dimension, dtype=field.dtype)
    torch.testing.assert_close(
        field(z),
        -torch.tensor(basis.eigenvalues, dtype=field.dtype) * z,
        atol=1e-11,
        rtol=1e-11,
    )


def test_laplacian_factory_orders_modes_and_persists_the_concrete_basis(tmp_path):
    intervals = 12
    problem = interval_problem(intervals)
    basis = problem.basis("laplacian", size=5, degree=1)
    theta = np.arange(5) * np.pi / intervals
    h = 1 / intervals
    exact = 6 * (1 - np.cos(theta)) / (h * h * (2 + np.cos(theta)))
    np.testing.assert_allclose(basis.eigenvalues, exact, rtol=1e-11, atol=1e-11)
    assert problem.validate_basis(basis) < 1e-12

    path = tmp_path / "laplacian_basis.npz"
    basis.save(path)
    restored = FiniteElementBasis.load(path)
    assert restored.family == "laplacian"
    np.testing.assert_array_equal(restored.eigenvalues, basis.eigenvalues)
    assert problem.validate_basis(restored) < 1e-12

    field = problem.field(basis=restored)
    z = torch.randn(restored.dimension, dtype=field.dtype)
    torch.testing.assert_close(field(z), z, rtol=1e-12, atol=1e-12)


def test_laplacian_factory_supports_a_closed_surface_embedded_in_r3():
    vertices, simplices = torus_mesh()
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=simplices,
        weak=lambda u, v, dx, ds: -inner(grad(u), grad(v)) * dx,
    )
    basis = problem.basis("laplacian", size=7)
    field = problem.field(basis=basis)
    z = torch.ones(7, dtype=field.dtype)
    torch.testing.assert_close(
        field(z), -torch.tensor(basis.eigenvalues, dtype=field.dtype), atol=1e-11, rtol=1e-11
    )
    assert len(problem.geometry.exterior_faces) == 0
    assert basis.eigenvalues[0] < 1e-12


def test_all_exposed_geometry_aware_families_are_orthonormal():
    problem = interval_problem(4)

    def values(x):
        return torch.stack((torch.ones_like(x[:, 0]), torch.exp(x[:, 0])), dim=1)

    bases = [
        problem.basis("polynomial", size=4),
        problem.basis("fourier", size=4),
        problem.basis("finite_element", degree=1),
        problem.basis("custom", source=CallableBasis(values, dimension=2), quadrature_order=10),
    ]
    assert [basis.family for basis in bases] == [
        "polynomial",
        "fourier",
        "finite-element",
        "custom",
    ]
    for basis in bases:
        assert problem.validate_basis(basis, tolerance=2e-8) < 2e-8
        field = problem.field(basis=basis)
        z = torch.randn(basis.dimension, dtype=field.dtype)
        torch.testing.assert_close(field(z), z, atol=2e-8, rtol=2e-8)


def test_component_and_product_bases_preserve_direct_sum_coordinates():
    problem = interval_problem(
        5,
        weak=lambda u, v, dx, ds: (u[0] * v[0] + 2 * u[1] * v[1] + 3 * u[2] * v[2]) * dx,
    )
    scalar = problem.basis("laplacian", size=2)
    repeated = ComponentBasis(scalar, components=3)
    assert problem.validate_basis(repeated) < 1e-12

    product_basis = ProductBasis(
        [
            problem.basis("laplacian", size=2),
            problem.basis("polynomial", size=3),
            problem.basis("laplacian", size=1),
        ]
    )
    assert product_basis.dimension == 6
    assert product_basis.value_shape == (3,)
    field = problem.field(basis=product_basis)
    z = torch.arange(1, 7, dtype=field.dtype)
    expected = z.clone()
    expected[product_basis.slices[1]] *= 2
    expected[product_basis.slices[2]] *= 3
    torch.testing.assert_close(field(z), expected, atol=1e-11, rtol=1e-11)


def test_independent_validation_detects_underintegrated_orthonormalization():
    problem = interval_problem(1)
    raw = PolynomialBasis(1, exponents=[(0,), (4,)], degree=4)
    with pytest.raises(ValueError, match="must be L2-orthonormal"):
        problem.orthonormalize(
            raw,
            quadrature_order=2,
            validation_order=10,
        )


def test_basis_factory_rejects_unknown_or_invalid_requests():
    problem = interval_problem(2)
    with pytest.raises(ValueError, match="Unknown basis family"):
        problem.basis("wavelet", size=3)
    with pytest.raises(ValueError, match="finite-element space has"):
        problem.basis("laplacian", size=20)
    with pytest.raises(ValueError, match="periods"):
        problem.basis("fourier", size=3, periods=[1.0, 2.0])
    with pytest.raises(TypeError, match="source"):
        problem.basis("custom")
