import pytest
import torch

from ngfield import DomainExitError, integrate_field


class OutwardField:
    dimension = 1
    device = torch.device("cpu")
    dtype = torch.float64

    def __init__(self):
        self.evaluated = []

    def _states(self, state):
        assert state.shape[-1] == self.dimension

    def __call__(self, state):
        self.evaluated.append(state.detach().clone())
        return torch.ones_like(state)


@pytest.mark.parametrize("controls", [{"step": 0.1}, {"tolerance": 1e-9}])
def test_taylor_integrator_stops_at_open_boundary_without_exterior_evaluation(controls):
    field = OutwardField()
    state = torch.tensor([0.9], dtype=field.dtype)
    times = torch.tensor([0.0, 0.3], dtype=field.dtype)
    with pytest.raises(DomainExitError) as error:
        integrate_field(field, state, times, radius=1.0, order=3, **controls)
    assert error.value.time >= 0
    assert torch.linalg.vector_norm(error.value.last_state) < 1
    assert all(torch.all(torch.abs(point) < 1) for point in field.evaluated)


def test_taylor_integrator_accepts_orders_independently_of_a_sobolev_loss():
    field = OutwardField()
    state = torch.tensor([0.1], dtype=field.dtype)
    times = torch.tensor([0.0, 0.2], dtype=field.dtype)
    for order in (1, 2, 5):
        trajectory = integrate_field(field, state, times, radius=1.0, step=0.1, order=order)
        torch.testing.assert_close(trajectory[-1], torch.tensor([0.3], dtype=field.dtype))
    with pytest.raises(ValueError, match="initial state"):
        integrate_field(field, torch.tensor([1.0], dtype=field.dtype), times, radius=1.0)
