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
    SimplicialDomain,
    grad,
    inner,
    sin,
)


def unit_simplex(dimension):
    return np.vstack((np.zeros(dimension), np.eye(dimension))), [np.arange(dimension + 1)]


@pytest.mark.parametrize("dimension", [1, 2, 4, 6])
def test_p1_diffusion_matches_exact_matrices_in_nd(dimension):
    vertices, cells = unit_simplex(dimension)

    def weak(u, v, dx, ds):
        return -0.1 * inner(grad(u), grad(v)) * dx

    problem = GalerkinProblem(vertices=vertices, simplices=cells, weak=weak)
    basis = FiniteElementBasis(problem.geometry)
    field = problem.field(basis=basis, quadrature_order=2)
    volume = 1 / math.factorial(dimension)
    mass = np.full((dimension + 1, dimension + 1), volume / ((dimension + 1) * (dimension + 2)))
    mass[np.diag_indices_from(mass)] *= 2
    gradients = np.vstack((-np.ones(dimension), np.eye(dimension)))
    stiffness = volume * gradients @ gradients.T
    z = torch.linspace(-0.3, 0.4, dimension + 1, dtype=field.dtype, requires_grad=True)
    expected = np.linalg.solve(mass, -0.1 * stiffness @ z.detach().numpy())
    torch.testing.assert_close(field(z), torch.tensor(expected), rtol=1e-11, atol=1e-11)
    field(z).square().sum().backward()
    assert torch.isfinite(z.grad).all()


def test_complete_form_contains_volume_boundary_coordinates_and_normal():
    vertices, cells = unit_simplex(2)
    boundaries = {"edge": np.array([[1, 2]])}

    def weak(u, v, dx, ds):
        x, n = dx.x, ds.normal
        diffusion = -(1 + sin(x[0]) ** 2) * inner(grad(u), grad(v)) * dx
        robin = (2 + inner(n, n)) * u * v * ds("edge")
        return diffusion + robin

    problem = GalerkinProblem(vertices=vertices, simplices=cells, boundaries=boundaries, weak=weak)
    field = problem.field(basis=FiniteElementBasis(problem.geometry), quadrature_order=6)
    z = torch.tensor([0.2, -0.1, 0.3], dtype=field.dtype)
    assert field(z).shape == z.shape
    assert torch.isfinite(field(z)).all()
    assert not hasattr(problem, "dirichlet")


def test_nonlinear_vector_field_and_batches():
    vertices, cells = unit_simplex(2)

    def weak(u, v, dx, ds):
        reaction = (u[0] - u[0] ** 3 - u[1]) * v[0]
        reaction = reaction + 0.25 * (u[0] - u[1]) * v[1]
        return reaction * dx - 0.1 * inner(grad(u), grad(v)) * dx

    problem = GalerkinProblem(vertices=vertices, simplices=cells, weak=weak)
    scalar = PolynomialBasis(2, degree=1)
    basis = ComponentBasis(scalar, components=2)
    field = problem.field(basis=basis, quadrature_order=6)
    z = torch.randn(7, basis.dimension, dtype=field.dtype, requires_grad=True)
    result = field(z)
    assert result.shape == z.shape
    torch.autograd.grad(result.sum(), z)


def test_callable_basis_derivatives_are_automatic():
    def values(x):
        return torch.stack((torch.ones_like(x[:, 0]), torch.exp(x[:, 0])), dim=1)

    basis = CallableBasis(values, dimension=2)

    def weak(u, v, dx, ds):
        return -inner(grad(u), grad(v)) * dx

    problem = GalerkinProblem(vertices=[[0.0], [1.0]], simplices=[[0, 1]], weak=weak)
    field = problem.field(basis=basis, quadrature_order=8)
    assert torch.isfinite(field(torch.tensor([0.1, -0.2], dtype=field.dtype))).all()


def test_callable_derivatives_are_projected_tangentially_and_basis_is_fixed():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def make_basis(extension, parameter=None):
        def values(x):
            scale = 1.0 if parameter is None else parameter
            return torch.stack(
                (torch.ones_like(x[:, 0]), scale * (x[:, 0] + extension * x[:, 2])),
                dim=1,
            )

        return CallableBasis(values, dimension=2)

    def weak(u, v, dx, ds):
        return -inner(grad(u), grad(v)) * dx

    problem = GalerkinProblem(vertices=vertices, simplices=[[0, 1, 2]], weak=weak)
    flat = problem.field(basis=make_basis(0.0))
    extended = problem.field(basis=make_basis(7.0))
    z = torch.tensor([0.2, -0.3], dtype=flat.dtype, requires_grad=True)
    torch.testing.assert_close(flat(z), extended(z), atol=1e-12, rtol=1e-12)

    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    frozen = problem.field(basis=make_basis(4.0, parameter))
    before = frozen(z)
    with torch.no_grad():
        parameter.fill_(3.0)
    torch.testing.assert_close(frozen(z), before, atol=0, rtol=0)
    frozen(z).sum().backward()
    assert parameter.grad is None


def test_reusable_geometry_and_labeled_volume_measures():
    geometry = SimplicialDomain(
        [[0.0], [0.5], [1.0]],
        [[0, 1], [1, 2]],
        regions={"left": [0], "right": [1]},
    )

    def weak(u, v, dx, ds):
        return u * v * dx("left") + 2.0 * u * v * dx("right")

    problem = GalerkinProblem(geometry=geometry, weak=weak)
    assert problem.geometry is geometry
    assert set(problem.regions) == {"all", "left", "right"}
    basis = FiniteElementBasis(geometry)
    field = problem.field(basis=basis, quadrature_order=2)
    z = torch.tensor([0.2, -0.1, 0.4], dtype=field.dtype)
    left = torch.tensor([[1 / 6, 1 / 12, 0], [1 / 12, 1 / 6, 0], [0, 0, 0]], dtype=field.dtype)
    right = torch.tensor([[0, 0, 0], [0, 1 / 6, 1 / 12], [0, 1 / 12, 1 / 6]], dtype=field.dtype)
    expected = torch.linalg.solve(field.mass_matrix, (left + 2 * right) @ z)
    torch.testing.assert_close(field(z), expected, atol=1e-12, rtol=1e-12)

    with pytest.raises(ValueError, match="cannot be combined"):
        GalerkinProblem(geometry=geometry, vertices=[[0.0]], simplices=[[0, 0]], weak=weak)


def test_unknown_volume_region_is_rejected_when_field_is_built():
    problem = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: u * v * dx("missing"),
    )
    with pytest.raises(ValueError, match="Unknown region"):
        problem.field(basis=PolynomialBasis(1))


def test_numerical_orthonormalization_preserves_physical_functions():
    vertices, cells = unit_simplex(2)

    def weak(u, v, dx, ds):
        return u * v * dx

    problem = GalerkinProblem(vertices=vertices, simplices=cells, weak=weak)
    basis = PolynomialBasis(2, degree=2)
    orthonormal = problem.orthonormalize(basis, quadrature_order=5)
    field = problem.field(basis=orthonormal, quadrature_order=5)
    torch.testing.assert_close(
        field.mass_matrix,
        torch.eye(basis.dimension, dtype=field.dtype),
        atol=1e-10,
        rtol=0,
    )
    z = torch.randn(basis.dimension, dtype=field.dtype)
    torch.testing.assert_close(field(z), z, atol=1e-10, rtol=0)


def test_custom_mass_matrix_changes_coordinate_velocity():
    vertices, cells = unit_simplex(1)

    def weak(u, v, dx, ds):
        return u * v * dx

    problem = GalerkinProblem(vertices=vertices, simplices=cells, weak=weak)
    basis = PolynomialBasis(1, degree=1)
    mass = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    field = problem.field(basis=basis, mass_matrix=mass)
    z = torch.tensor([0.4, -0.7], dtype=field.dtype)
    gram = torch.tensor([[1.0, 0.5], [0.5, 1 / 3]], dtype=field.dtype)
    expected = torch.linalg.solve(mass, gram @ z)
    torch.testing.assert_close(field(z), expected, rtol=1e-12, atol=1e-12)


def test_form_rejects_nonlinearity_in_test_function_and_unknown_boundary():
    vertices, cells = unit_simplex(1)
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=cells,
        weak=lambda u, v, dx, ds: v**2 * dx,
    )
    with pytest.raises(ValueError, match="linear in v"):
        problem.field(basis=PolynomialBasis(1))
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=cells,
        weak=lambda u, v, dx, ds: u * v * ds("missing"),
    )
    with pytest.raises(ValueError, match="Unknown boundary"):
        problem.field(basis=PolynomialBasis(1))


def test_finite_element_basis_roundtrip(tmp_path):
    vertices, cells = unit_simplex(4)
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=cells,
        regions={"core": [0]},
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = FiniteElementBasis(problem.geometry, degree=2)
    path = tmp_path / "basis.npz"
    basis.save(path)
    restored = FiniteElementBasis.load(path)
    assert restored.geometry.same_mesh(problem.geometry)
    np.testing.assert_array_equal(restored.geometry.regions["core"], [0])
    np.testing.assert_array_equal(restored.element_dofs, basis.element_dofs)
