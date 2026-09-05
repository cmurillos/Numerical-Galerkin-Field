import math

import pytest
import torch

from ngfield import ComponentBasis, GalerkinProblem, inner


def interval_field(value_shape=()):
    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=lambda u, v, dx, ds: inner(u, v) * dx if value_shape else u * v * dx,
    )
    scalar = problem.basis("polynomial", size=3)
    basis = ComponentBasis(scalar, value_shape=value_shape) if value_shape else scalar
    return problem.field(basis=basis), basis


def test_reconstruct_gradient_and_hessian_at_physical_points():
    field, _ = interval_field()
    coordinates = field.project(lambda x: x[:, 0] ** 2, quadrature=6)
    points = torch.tensor([[0.2], [0.8]], dtype=field.dtype)

    torch.testing.assert_close(
        field.reconstruct(coordinates, points),
        points[:, 0] ** 2,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        field.grad(coordinates, points),
        (2 * points).reshape(2, 1),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        field.hessian(coordinates, points),
        torch.full((2, 1, 1), 2.0, dtype=field.dtype),
        atol=1e-11,
        rtol=1e-11,
    )


def test_spatial_methods_preserve_state_batches_value_axes_and_autograd():
    field, _ = interval_field(value_shape=(2,))
    coordinates = field.project(
        lambda x: torch.stack((x[:, 0], x[:, 0] ** 2), dim=1),
        quadrature=6,
    )
    states = torch.stack((coordinates, 2 * coordinates)).reshape(2, 1, field.dimension)
    states.requires_grad_()
    points = torch.tensor([[0.25], [0.75]], dtype=field.dtype)

    values = field.reconstruct(states, points)
    gradients = field.grad(states, points)
    Hessians = field.hessian(states, points)
    assert values.shape == (2, 1, 2, 2)
    assert gradients.shape == (2, 1, 2, 2, 1)
    assert Hessians.shape == (2, 1, 2, 2, 1, 1)
    (values.sum() + gradients.sum() + Hessians.sum()).backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_embedded_derivatives_are_tangential_and_use_ambient_axes():
    problem = GalerkinProblem(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        simplices=[[0, 1, 2]],
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = problem.basis("polynomial", size=3)
    field = problem.field(basis=basis)
    coordinates = field.project(lambda x: x[:, 0] + 2 * x[:, 1], quadrature=6)
    points = torch.tensor([[0.2, 0.3, 0.0], [0.6, 0.1, 0.0]], dtype=field.dtype)

    expected = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]], dtype=field.dtype)
    torch.testing.assert_close(field.grad(coordinates, points), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        field.hessian(coordinates, points),
        torch.zeros(2, 3, 3, dtype=field.dtype),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("dimension", [1, 2, 4, 6])
def test_point_location_and_spatial_derivatives_are_dimension_independent(dimension):
    vertices = torch.cat((torch.zeros(1, dimension), torch.eye(dimension))).numpy()
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=[list(range(dimension + 1))],
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = problem.basis("polynomial", size=dimension + 1)
    field = problem.field(basis=basis)
    coordinates = field.project(lambda x: x.sum(dim=1), quadrature=4)
    points = torch.full((1, dimension), 1 / (2 * dimension), dtype=field.dtype)

    torch.testing.assert_close(
        field.reconstruct(coordinates, points),
        points.sum(dim=1),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(
        field.grad(coordinates, points),
        torch.ones(1, dimension, dtype=field.dtype),
        atol=1e-11,
        rtol=1e-11,
    )


def square_field(degree=1):
    problem = GalerkinProblem(
        vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        simplices=[[0, 1, 2], [1, 3, 2]],
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = problem.basis("finite-element", degree=degree)
    return problem.field(basis=basis, quadrature=2 * degree + 2)


def test_cells_select_the_trace_when_a_derivative_is_ambiguous():
    field = square_field()
    state = torch.tensor([0.2, -0.4, 0.8, 0.1], dtype=field.dtype)
    point = torch.tensor([[0.5, 0.5]], dtype=field.dtype)

    assert field.reconstruct(state, point).shape == (1,)
    with pytest.raises(ValueError, match="different gradients.*pass cells"):
        field.grad(state, point)
    left = field.grad(state, point, cells=torch.tensor([0], dtype=torch.int64))
    right = field.grad(state, point, cells=torch.tensor([1], dtype=torch.int64))
    assert not torch.allclose(left, right)
    torch.testing.assert_close(
        field.hessian(state, point),
        torch.zeros(1, 2, 2, dtype=field.dtype),
        atol=1e-12,
        rtol=0,
    )


def test_cells_select_the_trace_when_a_hessian_is_ambiguous():
    field = square_field(degree=2)
    state = torch.arange(field.dimension, dtype=field.dtype)
    point = torch.tensor([[0.5, 0.5]], dtype=field.dtype)

    with pytest.raises(ValueError, match="different Hessians.*pass cells"):
        field.hessian(state, point)
    left = field.hessian(state, point, cells=torch.tensor([0], dtype=torch.int64))
    right = field.hessian(state, point, cells=torch.tensor([1], dtype=torch.int64))
    assert not torch.allclose(left, right)


class PiecewiseConstantBasis:
    dimension = 2
    value_shape = ()

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        shape = (len(points), self.dimension, *((points.shape[1],) * order))
        if order:
            return points.new_zeros(shape)
        return math.sqrt(2) * torch.nn.functional.one_hot(cells, self.dimension).to(points)


def test_discontinuous_values_require_cells_on_a_shared_face():
    problem = GalerkinProblem(
        vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        simplices=[[0, 1, 2], [1, 3, 2]],
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    field = problem.field(basis=PiecewiseConstantBasis(), quadrature=2)
    state = torch.tensor([1.0, 3.0], dtype=field.dtype)
    point = torch.tensor([[0.5, 0.5]], dtype=field.dtype)

    with pytest.raises(ValueError, match="different values.*pass cells"):
        field.reconstruct(state, point)
    left = field.reconstruct(state, point, cells=torch.tensor([0], dtype=torch.int64))
    right = field.reconstruct(state, point, cells=torch.tensor([1], dtype=torch.int64))
    torch.testing.assert_close(left, torch.tensor([math.sqrt(2)], dtype=field.dtype))
    torch.testing.assert_close(right, torch.tensor([3 * math.sqrt(2)], dtype=field.dtype))


def test_invalid_physical_points_and_cells_fail_explicitly():
    field = square_field()
    state = torch.zeros(field.dimension, dtype=field.dtype)
    inside = torch.tensor([[0.1, 0.1]], dtype=field.dtype)
    outside = torch.tensor([[1.2, 0.2]], dtype=field.dtype)

    with pytest.raises(ValueError, match="does not belong to the simplicial domain"):
        field.reconstruct(state, outside)
    with pytest.raises(ValueError, match="does not belong to the simplex selected"):
        field.reconstruct(state, inside, cells=torch.tensor([1], dtype=torch.int64))
    with pytest.raises(TypeError, match="Physical points must be torch tensors"):
        field.reconstruct(state, [[0.1, 0.1]])
    with pytest.raises(ValueError, match=r"shape \[Q,2\]"):
        field.reconstruct(state, torch.tensor([0.1, 0.1], dtype=field.dtype))
    with pytest.raises(ValueError, match="share device and dtype"):
        field.reconstruct(state, inside.float())
    with pytest.raises(ValueError, match="torch.int64"):
        field.reconstruct(state, inside, cells=torch.tensor([0], dtype=torch.int32))
    with pytest.raises(ValueError, match="invalid simplex index"):
        field.reconstruct(state, inside, cells=torch.tensor([2], dtype=torch.int64))
    with pytest.raises(ValueError, match="cells requires physical points"):
        field.reconstruct(state, cells=torch.tensor([0], dtype=torch.int64))


def test_empty_physical_point_sets_preserve_all_output_axes():
    field, _ = interval_field(value_shape=(2,))
    states = torch.empty(2, 0, field.dimension, dtype=field.dtype)
    points = torch.empty(0, 1, dtype=field.dtype)
    cells = torch.empty(0, dtype=torch.int64)

    assert field.reconstruct(states, points, cells=cells).shape == (2, 0, 0, 2)
    assert field.grad(states, points, cells=cells).shape == (2, 0, 0, 2, 1)
    assert field.hessian(states, points, cells=cells).shape == (2, 0, 0, 2, 1, 1)
