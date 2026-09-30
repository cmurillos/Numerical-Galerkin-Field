import pytest
import torch

from ngfield import GalerkinProblem, multi_indices, state_derivatives


class PolynomialField(torch.nn.Module):
    dimension = 2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def _states(self, states):
        assert states.shape[-1] == self.dimension

    def forward(self, states):
        a, b = states.unbind(-1)
        return self.scale * torch.stack((a.square() * b, a * b.square()), dim=-1)


def test_multi_indices_count_mixed_partials_once():
    assert multi_indices(2, 2) == (
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
    )
    with pytest.raises(ValueError, match="order"):
        multi_indices(2, -1)
    with pytest.raises(ValueError, match="dimension"):
        multi_indices(0, 1)


def test_indexed_derivatives_match_polynomial_and_keep_parameter_gradient():
    field = PolynomialField()
    states = torch.tensor([[[0.2, -0.4], [0.3, 0.5]]], dtype=torch.float64)
    a, b = states.unbind(-1)
    values = state_derivatives(field, states, 2)
    expected = {
        (0, 0): (a.square() * b, a * b.square()),
        (1, 0): (2 * a * b, b.square()),
        (0, 1): (a.square(), 2 * a * b),
        (2, 0): (2 * b, torch.zeros_like(b)),
        (1, 1): (2 * a, 2 * b),
        (0, 2): (torch.zeros_like(a), 2 * a),
    }
    for alpha, components in expected.items():
        torch.testing.assert_close(values[alpha], torch.stack(components, dim=-1))
    values[(1, 1)].square().sum().backward()
    assert field.scale.grad is not None and torch.isfinite(field.scale.grad)
    empty = state_derivatives(field, torch.empty(2, 0, 2, dtype=torch.float64), 2)
    assert all(value.shape == (2, 0, 2) for value in empty.values())

    third = state_derivatives(field, states, 3)
    torch.testing.assert_close(
        third[(2, 1)], field.scale * torch.stack((2 * torch.ones_like(a), torch.zeros_like(b)), -1)
    )
    torch.testing.assert_close(
        third[(1, 2)], field.scale * torch.stack((torch.zeros_like(a), 2 * torch.ones_like(b)), -1)
    )


def test_numerical_galerkin_field_provides_indexed_derivatives():
    problem = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: (u - u**3) * v * dx,
    )
    field = problem.field(basis=problem.basis("polynomial", size=2), quadrature=6)
    state = torch.tensor([0.3, -0.2], dtype=field.dtype)
    indexed = field.state_derivatives(state, 2)
    jacobian = torch.func.jacrev(field)(state)
    hessian = torch.func.jacrev(torch.func.jacrev(field))(state)
    for axis in range(field.dimension):
        alpha = tuple(int(i == axis) for i in range(field.dimension))
        torch.testing.assert_close(indexed[alpha], jacobian[:, axis])
    torch.testing.assert_close(indexed[(1, 1)], hessian[:, 0, 1])
