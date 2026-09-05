import math

import pytest
import torch

from ngfield import GalerkinProblem


def decay_field(dtype=torch.float64):
    problem = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: -u * v * dx,
    )
    basis = problem.basis("polynomial", size=3)
    return problem.field(basis=basis, quadrature=6, dtype=dtype)


def test_fixed_rk4_solves_batched_decay_at_nonuniform_output_times():
    field = decay_field()
    z0 = torch.randn(2, 3, field.dimension, dtype=field.dtype)
    times = torch.tensor([0.0, 0.13, 0.4, 1.0], dtype=field.dtype)
    states = field.solve(z0, times, step=0.01)
    expected = torch.exp(-times).reshape(-1, 1, 1, 1) * z0

    assert states.shape == (len(times), *z0.shape)
    torch.testing.assert_close(states, expected, atol=2e-10, rtol=2e-10)


def test_adaptive_rk45_is_the_default_and_supports_backward_time():
    field = decay_field()
    z0 = torch.tensor([0.4, -0.2, 0.7], dtype=field.dtype)
    forward = torch.tensor([0.0, 0.2, 1.0], dtype=field.dtype)
    backward = torch.tensor([1.0, 0.6, 0.0], dtype=field.dtype)

    forward_states = field.solve(z0, forward, tolerance=1e-10)
    backward_states = field.solve(z0, backward, tolerance=1e-10)

    torch.testing.assert_close(
        forward_states,
        torch.exp(-forward).reshape(-1, 1) * z0,
        atol=2e-9,
        rtol=2e-9,
    )
    torch.testing.assert_close(
        backward_states,
        torch.exp(1 - backward).reshape(-1, 1) * z0,
        atol=2e-9,
        rtol=2e-9,
    )


def test_default_adaptive_solver_supports_float32():
    field = decay_field(torch.float32)
    z0 = torch.tensor([0.4, -0.2, 0.7], dtype=field.dtype)
    times = torch.tensor([0.0, 0.3, 1.0], dtype=field.dtype)
    states = field.solve(z0, times)
    expected = torch.exp(-times).reshape(-1, 1) * z0
    torch.testing.assert_close(states, expected, atol=2e-4, rtol=2e-4)


def test_solve_preserves_autograd_and_reconstructs_the_complete_trajectory():
    field = decay_field()
    z0 = torch.tensor([0.2, -0.1, 0.3], dtype=field.dtype, requires_grad=True)
    times = torch.linspace(0, 1, 11, dtype=field.dtype)
    states = field.solve(z0, times, step=0.01)
    points = torch.tensor([[0.25], [0.75]], dtype=field.dtype)
    values = field.reconstruct(states, points)
    states[-1].sum().backward()

    assert values.shape == (len(times), len(points))
    torch.testing.assert_close(
        z0.grad,
        torch.full_like(z0, math.exp(-1)),
        atol=2e-10,
        rtol=2e-10,
    )


def test_solve_supports_single_time_and_empty_batches():
    field = decay_field()
    single = torch.tensor([2.0], dtype=field.dtype)
    z0 = torch.empty(2, 0, field.dimension, dtype=field.dtype)
    empty_times = torch.tensor([0.0, 0.5, 1.0], dtype=field.dtype)

    torch.testing.assert_close(field.solve(z0, single), z0.unsqueeze(0))
    assert field.solve(z0, empty_times).shape == (3, 2, 0, field.dimension)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 1.0], "torch tensor"),
        (torch.empty(0, dtype=torch.float64), r"T >= 1"),
        (torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64), "strictly monotone"),
        (torch.tensor([0.0, 0.0], dtype=torch.float64), "strictly monotone"),
    ],
)
def test_solve_rejects_invalid_time_grids(times, message):
    field = decay_field()
    z0 = torch.zeros(field.dimension, dtype=field.dtype)
    with pytest.raises((TypeError, ValueError), match=message):
        field.solve(z0, times)


@pytest.mark.parametrize("step", [True, 0.0, -0.1, float("inf")])
def test_solve_rejects_invalid_fixed_steps(step):
    field = decay_field()
    z0 = torch.zeros(field.dimension, dtype=field.dtype)
    times = torch.tensor([0.0, 1.0], dtype=field.dtype)
    with pytest.raises((TypeError, ValueError), match="step"):
        field.solve(z0, times, step=step)


def test_solve_rejects_ambiguous_controls_and_nonfinite_states():
    field = decay_field()
    times = torch.tensor([0.0, 1.0], dtype=field.dtype)
    with pytest.raises(ValueError, match="cannot be combined"):
        field.solve(
            torch.zeros(field.dimension, dtype=field.dtype), times, step=0.1, tolerance=1e-6
        )
    with pytest.raises(ValueError, match="initial state"):
        field.solve(torch.full((field.dimension,), torch.inf, dtype=field.dtype), times)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_solve_runs_on_cuda():
    field = decay_field().to(device="cuda")
    z0 = torch.ones(4, field.dimension, dtype=field.dtype, device=field.device)
    times = torch.linspace(0, 1, 5, dtype=field.dtype, device=field.device)
    states = field.solve(z0, times, step=0.05)
    assert states.shape == (5, 4, field.dimension)
    assert states.device.type == "cuda"
