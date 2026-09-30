"""Function-valued workflow, coordinate derivatives and stopped trajectories."""

import pytest
import torch

from ngfield import Function, Geometry, Space, State, System


def decay_system():
    geometry = Geometry(vertices=[[0.0], [1.0]], simplices=[[0, 1]])
    space = Space(geometry=geometry, components=1)
    basis = space.basis("polynomial", degree=1)
    return System(basis=basis, weak=lambda u, v, dx, ds: -u[0] * v[0] * dx)


def test_functional_state_evolution_and_independent_spatial_derivatives():
    system = decay_system()
    initial = system.state(lambda x: torch.ones_like(x))
    times = torch.tensor([0.0, 0.1, 0.25], dtype=system.dtype)
    solution = system.evolve(initial, times, order=3, step=0.01)
    state = solution.at(times[-1])
    points = torch.tensor([[0.2], [0.8]], dtype=system.dtype)

    assert state.basis is system.basis and state.space is system.space
    assert state.geometry is system.geometry
    torch.testing.assert_close(
        state.coefficients(), torch.exp(-times[-1]) * initial.coefficients(), atol=1e-8, rtol=1e-8
    )
    assert state.values(points).shape == (2, 1)
    assert state.gradient(points).shape == (2, 1, 1)
    assert state.hessian(points).shape == (2, 1, 1, 1)
    torch.testing.assert_close(state.norm_L2(), state.coefficients().norm())
    torch.testing.assert_close(
        state.integral(), state.values(points).mean(dim=0), atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(state.velocity().coefficients(), -state.coefficients())
    assert not solution.exit_status()["exited"]
    with pytest.raises(ValueError, match="output times"):
        solution.at(0.15)


def test_indexed_derivatives_are_functions_and_metrics_use_same_field():
    system = decay_system()
    state = system.from_coefficients(torch.tensor([0.3, -0.1], dtype=system.dtype))
    derivatives = state.indexed_derivatives(order=2)
    assert tuple(derivatives) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    assert all(isinstance(value, Function) for value in derivatives.values())
    torch.testing.assert_close(derivatives[(0, 0)].coefficients(), -state.coefficients())
    torch.testing.assert_close(
        derivatives[(1, 0)].coefficients(), state.coefficients().new_tensor([-1, 0])
    )
    torch.testing.assert_close(
        derivatives[(0, 1)].coefficients(), state.coefficients().new_tensor([0, -1])
    )
    for alpha in ((2, 0), (1, 1), (0, 2)):
        torch.testing.assert_close(
            derivatives[alpha].coefficients(), torch.zeros(2, dtype=system.dtype)
        )
    torch.testing.assert_close(system.metrics.integral_rate(state), -state.integral())
    torch.testing.assert_close(system.metrics.radial_rate(state), -state.norm_L2().square())
    torch.testing.assert_close(
        system.metrics.sampled_lipschitz([state]), state.norm_L2().new_tensor(1.0)
    )
    assert system.metrics.projection(lambda x: torch.ones_like(x)) < 1e-10
    assert system.metrics.quadrature(state) < 1e-10
    times = torch.tensor([0.0, 0.2], dtype=system.dtype)
    assert system.metrics.time_refinement(state, times, step=0.1).shape == (2,)


def test_workflow_preserves_batches_and_reports_partial_exit_without_evaluating_outside():
    system = decay_system()
    batch = torch.tensor([[[0.2, 0.0]], [[0.3, 0.0]]], dtype=system.dtype)
    state = system.from_coefficients(batch)
    times = torch.tensor([0.0, -0.1, -0.3, -2.0], dtype=system.dtype)
    solution = system.evolve(state, times, order=2, step=0.02, radius=0.5)
    status = solution.exit_status()
    assert status["exited"]
    assert isinstance(status["last_accepted_state"], State)
    assert status["last_accepted_time"] <= -0.3
    assert solution.coefficients().shape == (3, 2, 1, 2)
    assert solution.times.shape == (3,)
    assert torch.all(solution.at(-0.3).norm_L2() < 0.5)
    with pytest.raises(ValueError, match="this system"):
        decay_system().indexed_derivatives(state, 1)
