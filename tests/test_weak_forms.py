import numpy as np
import pytest
import torch

from ngfield import (
    Coefficient,
    ComponentBasis,
    FiniteElementBasis,
    GalerkinProblem,
    PolynomialBasis,
    contract,
    div,
    dot,
    grad,
    inner,
    outer,
    pointwise,
    sym_grad,
    trace,
    transpose,
)


def interval_problem(coefficient, *, vertices=None, simplices=None, boundary=False):
    vertices = [[0.0], [1.0]] if vertices is None else vertices
    simplices = [[0, 1]] if simplices is None else simplices

    def weak(u, v, dx, ds):
        measure = ds if boundary else dx
        return coefficient * u * v * measure

    problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
    basis = problem.orthonormalize(FiniteElementBasis(problem.geometry), quadrature_order=4)
    return problem.field(basis=basis, quadrature_order=4)


def test_callable_and_vertex_coefficients_match_exact_linear_field_and_are_fixed():
    slope = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    callable_coefficient = Coefficient(lambda x: 1.0 + slope * x[:, 0], shape=())
    vertex_coefficient = Coefficient.vertex([1.0, 3.0])
    callable_field = interval_problem(callable_coefficient)
    vertex_field = interval_problem(vertex_coefficient)
    z = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)

    weighted_mass = torch.tensor([[1 / 2, 1 / 3], [1 / 3, 5 / 6]], dtype=torch.float64)
    transform = vertex_field.basis.transform
    expected = transform.T @ weighted_mass @ transform @ z
    torch.testing.assert_close(callable_field(z), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(vertex_field(z), expected, atol=1e-12, rtol=1e-12)

    before = callable_field(z)
    with torch.no_grad():
        slope.fill_(7.0)
    torch.testing.assert_close(callable_field(z), before, atol=0, rtol=0)
    callable_field(z).sum().backward()
    assert slope.grad is None
    assert torch.isfinite(z.grad).all()


def test_cell_coefficient_matches_piecewise_constant_assembly():
    vertices = [[0.0], [0.5], [1.0]]
    simplices = [[0, 1], [1, 2]]
    field = interval_problem(Coefficient.cell([1.0, 2.0]), vertices=vertices, simplices=simplices)
    z = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    left = torch.tensor(
        [[1 / 6, 1 / 12, 0], [1 / 12, 1 / 6, 0], [0, 0, 0]],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[0, 0, 0], [0, 1 / 6, 1 / 12], [0, 1 / 12, 1 / 6]],
        dtype=torch.float64,
    )
    transform = field.basis.transform
    expected = transform.T @ (left + 2 * right) @ transform @ z
    torch.testing.assert_close(field(z), expected, atol=1e-12, rtol=1e-12)


def test_vertex_coefficient_interpolates_on_boundary_quadrature():
    field = interval_problem(Coefficient.vertex([2.0, 3.0]), boundary=True)
    z = torch.tensor([0.4, -0.2], dtype=torch.float64)
    boundary_action = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    transform = field.basis.transform
    expected = transform.T @ boundary_action @ transform @ z
    torch.testing.assert_close(field(z), expected, atol=1e-12, rtol=1e-12)


def test_anisotropic_tensor_coefficient_and_matmul_match_exact_stiffness():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tensor = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    diffusion = Coefficient(lambda x: tensor.expand(len(x), 2, 2), shape=(2, 2))

    def weak(u, v, dx, ds):
        return -inner(diffusion @ grad(u), grad(v)) * dx

    problem = GalerkinProblem(vertices=vertices, simplices=[[0, 1, 2]], weak=weak)
    basis = problem.orthonormalize(PolynomialBasis(2, degree=1), quadrature_order=2)
    field = problem.field(basis=basis, quadrature_order=2)
    z = torch.tensor([0.1, -0.3, 0.4], dtype=torch.float64)
    gradients = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    stiffness = 0.5 * gradients @ tensor @ gradients.T
    transform = basis.transform
    expected = -transform.T @ stiffness @ transform @ z
    torch.testing.assert_close(field(z), expected, atol=1e-12, rtol=1e-12)


def test_nd_tensor_operators_and_pointwise_preserve_batches_and_autograd():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def weak(u, v, dx, ds):
        nonlinear = pointwise(lambda y: y - y**3, u, shape=(2,))
        uu, vu = outer(u, u), outer(v, u)
        tensor_terms = inner(sym_grad(u), sym_grad(v)) + div(u) * div(v)
        tensor_terms = tensor_terms + inner(transpose(grad(u)), grad(v))
        tensor_terms = tensor_terms + contract(uu, vu, axes=2)
        algebraic = inner(nonlinear, v) + dot(u, v) + trace(outer(u, v))
        return (algebraic - 0.1 * tensor_terms) * dx

    problem = GalerkinProblem(vertices=vertices, simplices=[[0, 1, 2]], weak=weak)
    scalar = problem.basis("polynomial", size=3)
    basis = ComponentBasis(scalar, components=2)
    field = problem.field(basis=basis, quadrature_order=6)
    z = torch.randn(4, basis.dimension, dtype=torch.float64, requires_grad=True)
    result = field(z)
    assert result.shape == z.shape
    assert torch.isfinite(result).all()
    result.square().sum().backward()
    assert torch.isfinite(z.grad).all()


def test_d004_rejects_affine_test_forms_pointwise_on_v_and_interior_facets():
    affine = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: (1.0 + u) * dx,
    )
    with pytest.raises(ValueError, match="exactly linearly"):
        affine.field(basis=PolynomialBasis(1))

    invalid_pointwise = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: pointwise(lambda y: y, v) * dx,
    )
    with pytest.raises(ValueError, match="cannot receive"):
        invalid_pointwise.field(basis=PolynomialBasis(1))

    interior = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: u * v * ds.interior,
    )
    with pytest.raises(NotImplementedError, match="Interior facets"):
        interior.field(basis=PolynomialBasis(1))


def test_coefficient_contract_validates_data_count_and_callable_shape():
    with pytest.raises(ValueError, match="one value per simplex"):
        interval_problem(Coefficient.cell([1.0, 2.0]))

    bad = Coefficient(lambda x: torch.ones(len(x), 2, dtype=x.dtype), shape=())
    with pytest.raises(ValueError, match="Coefficient must return"):
        interval_problem(bad)

    assert not hasattr(Coefficient, "quadrature")
