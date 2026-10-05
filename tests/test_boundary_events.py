import pytest
import torch

from ngfield import DomainExitError, integrate_field


class InteriorConstant:
    dimension = 1
    dtype = torch.float64
    device = torch.device("cpu")

    def _states(self, state):
        assert state.shape[-1] == self.dimension

    def __call__(self, state):
        assert bool((torch.linalg.vector_norm(state, dim=-1) < 1).all())
        return torch.ones_like(state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("direction", [1, -1])
def test_contact_localized_and_excluded_from_path_in_either_time_direction(
    adaptive, direction, dtype
):
    field = InteriorConstant()
    field.dtype = dtype
    initial = torch.tensor([0.2 * direction], dtype=field.dtype)
    times = torch.tensor([3.0, 3.0 + 0.25 * direction, 3.0 + 2 * direction], dtype=field.dtype)
    options = {"tolerance": 1e-10} if adaptive else {"step": 2.0}
    with pytest.raises(DomainExitError) as caught:
        integrate_field(field, initial, times, order=3, radius=1.0, **options)
    event = caught.value
    assert event.exit_time == pytest.approx(3.0 + 0.8 * direction)
    torch.testing.assert_close(event.exit_state, torch.tensor([direction], dtype=field.dtype))
    assert bool(event.exit_mask)
    assert torch.linalg.vector_norm(event.last_state) < 1
    assert len(event.completed_times) == 2
    torch.testing.assert_close(event.completed_states[-1], initial + 0.25 * direction)


@pytest.mark.parametrize("adaptive", [False, True])
def test_first_contact_detected_even_when_step_endpoint_reenters(adaptive):
    class Parabola(InteriorConstant):
        dimension = 2

        def __call__(self, z):
            assert bool((torch.linalg.vector_norm(z, dim=-1) < 1).all())
            return torch.stack((torch.ones_like(z[..., 0]), -10 * z[..., 0]), dim=-1)

    field = Parabola()
    initial = torch.tensor([-0.5, 0.3], dtype=field.dtype)
    times = torch.tensor([0.0, 1.0], dtype=field.dtype)
    options = {"tolerance": 1e-10} if adaptive else {"step": 1.0}
    with pytest.raises(DomainExitError) as caught:
        integrate_field(field, initial, times, order=2, radius=1.0, **options)
    event = caught.value
    assert 0 < event.exit_time < 0.5
    torch.testing.assert_close(
        torch.linalg.vector_norm(event.exit_state), torch.tensor(1.0, dtype=field.dtype)
    )
    assert len(event.completed_times) == 1


def test_batched_solve_marks_first_contact_and_preserves_batch_shape():
    initial = torch.tensor([[[0.2], [0.7]], [[0.1], [0.0]]], dtype=torch.float64)
    times = torch.tensor([0.0, 2.0], dtype=initial.dtype)
    with pytest.raises(DomainExitError) as caught:
        integrate_field(InteriorConstant(), initial, times, step=2.0, order=1, radius=1.0)
    event = caught.value
    assert event.exit_time == pytest.approx(0.3)
    assert event.exit_state.shape == initial.shape
    assert event.exit_mask.shape == initial.shape[:-1]
    assert event.exit_mask.tolist() == [[False, True], [False, False]]


def test_grazing_contact_stops_before_reentry():
    class Grazing(InteriorConstant):
        dimension = 2

        def __call__(self, z):
            assert bool((torch.linalg.vector_norm(z, dim=-1) < 1).all())
            return torch.stack((torch.ones_like(z[..., 0]), -8 * z[..., 0]), dim=-1)

    field = Grazing()
    with pytest.raises(DomainExitError) as caught:
        integrate_field(
            field,
            torch.tensor([-0.5, 0.0], dtype=field.dtype),
            torch.tensor([0.0, 1.0], dtype=field.dtype),
            step=1.0,
            order=2,
            radius=1.0,
        )
    assert caught.value.exit_time == pytest.approx(0.5, abs=1e-7)
