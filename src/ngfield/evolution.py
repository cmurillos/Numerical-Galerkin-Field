"""Shared Taylor integration of autonomous reduced coordinate fields."""

from math import ceil, factorial, isfinite
from numbers import Integral, Real

import torch

_MAX_INTERNAL_STEPS = 1_000_000


class DomainExitError(RuntimeError):
    """An integration step reaches the boundary of its open coordinate ball.

    ``time`` and ``last_state`` describe the last accepted interior state. The
    exact exit time is not certified by a numerical integrator.
    """

    def __init__(self, time, last_state):
        self.time = float(time)
        self.last_state = last_state.detach().clone()
        super().__init__(f"The trajectory reaches the open-ball boundary after t={time:g}.")


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive real number.")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return result


def _tolerance(value, dtype):
    if value is None:
        return 5e-5 if dtype == torch.float32 else 1e-8
    result = _positive_real(value, "tolerance")
    if result >= 1:
        raise ValueError("tolerance must be strictly smaller than one.")
    return result


def _order(value):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("order must be a positive integer.")
    if value < 1:
        raise ValueError("order must be a positive integer.")
    return int(value)


def _inside(state, radius):
    return radius is None or bool(torch.all(torch.linalg.vector_norm(state, dim=-1) < radius))


def _inputs(field, z0, times, radius):
    field._states(z0)
    if not torch.isfinite(z0).all():
        raise ValueError("The initial state must be finite.")
    if not _inside(z0, radius):
        raise ValueError("The initial state must lie strictly inside the coordinate ball.")
    if not isinstance(times, torch.Tensor):
        raise TypeError("times must be a torch tensor.")
    if times.ndim != 1 or len(times) < 1:
        raise ValueError("times must have shape [T] with T >= 1.")
    if times.device != field.device or times.dtype != field.dtype:
        raise ValueError("times and the field must share device and dtype.")
    if not torch.isfinite(times).all():
        raise ValueError("times must be finite.")
    if len(times) > 1:
        increments = times[1:] - times[:-1]
        if not bool(torch.all(increments > 0)) and not bool(torch.all(increments < 0)):
            raise ValueError("times must be strictly monotone.")


def _taylor_step(field, state, step, order, *, estimate_error):
    """Use J_1=f, J_(r+1)=D J_r f; evaluate f only at the interior state."""
    result = state
    jet = field
    last_term = None
    highest = order + int(estimate_error)
    velocity = field(state)
    for degree in range(1, highest + 1):
        if degree == 1:
            value = velocity
        else:
            previous = jet
            # The outer directional derivative always uses the velocity at
            # the same accepted state. Reuse it, while keeping field(z)
            # variable inside each jet so higher derivatives remain correct.
            value = torch.func.jvp(previous, (state,), (velocity,))[1]

            if degree < highest:

                def next_jet(z, previous=previous):
                    return torch.func.jvp(previous, (z,), (field(z),))[1]

                jet = next_jet
        if value.shape != state.shape:
            raise ValueError("A Taylor jet changed the field's batch or coordinate shape.")
        if not torch.isfinite(value).all():
            raise FloatingPointError("The field returned a nonfinite Taylor jet.")
        term = (step**degree / factorial(degree)) * value
        if degree <= order:
            result = result + term
        else:
            last_term = term
    if not torch.isfinite(result).all():
        raise FloatingPointError("Taylor integration produced a nonfinite state.")
    return result, last_term


def _fixed_interval(field, state, start, stop, maximum_step, order, radius, budget):
    count = max(1, ceil(abs(stop - start) / maximum_step))
    if count > budget:
        raise RuntimeError("Time integration exceeded its internal step budget.")
    step = (stop - start) / count
    for index in range(count):
        candidate, _ = _taylor_step(field, state, step, order, estimate_error=False)
        if not _inside(candidate, radius):
            raise DomainExitError(start + index * step, state)
        state = candidate
    return state, count


def _adaptive_interval(field, state, start, stop, tolerance, order, radius, budget):
    direction = 1.0 if stop > start else -1.0
    current = start
    step_size = abs(stop - start)
    attempts = 0
    epsilon = torch.finfo(field.dtype).eps
    while direction * (stop - current) > 0:
        if attempts >= budget:
            raise RuntimeError("Time integration exceeded its internal step budget.")
        remaining = abs(stop - current)
        step_size = min(step_size, remaining)
        signed_step = direction * step_size
        candidate, term = _taylor_step(field, state, signed_step, order, estimate_error=True)
        attempts += 1
        minimum = 16 * epsilon * max(1.0, abs(current), abs(stop))
        if not _inside(candidate, radius):
            if step_size <= minimum:
                raise DomainExitError(current, state)
            step_size *= 0.5
            continue
        scale = tolerance * (1 + torch.maximum(torch.abs(state), torch.abs(candidate)))
        error = float(torch.max(torch.abs(term) / scale).detach().item())
        if not isfinite(error):
            raise FloatingPointError("Taylor integration produced a nonfinite error estimate.")
        if error <= 1:
            state = candidate
            current += signed_step
            if abs(stop - current) <= minimum:
                current = stop
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** (-1 / (order + 1))))
            step_size *= factor
        else:
            if step_size <= minimum:
                raise RuntimeError(
                    "Taylor integration cannot satisfy the tolerance at machine precision."
                )
            step_size *= max(0.2, 0.9 * error ** (-1 / (order + 1)))
    return state, attempts


def integrate_field(field, z0, times, *, step=None, tolerance=None, order=4, radius=None):
    """Integrate one autonomous field by Taylor jets in the original time variable.

    A fixed maximum ``step`` or an adaptive ``tolerance`` selects the time-step
    rule. With ``radius`` the field is never evaluated outside the open ball;
    reaching its boundary raises ``DomainExitError``. Use identical controls
    on the Galerkin and learned fields for temporal comparisons.
    """
    order = _order(order)
    radius = None if radius is None else _positive_real(radius, "radius")
    _inputs(field, z0, times, radius)
    if step is not None and tolerance is not None:
        raise ValueError("step and tolerance cannot be combined.")
    fixed_step = None if step is None else _positive_real(step, "step")
    adaptive_tolerance = None if fixed_step is not None else _tolerance(tolerance, field.dtype)
    if not z0.numel() or len(times) == 1:
        return torch.stack([z0] * len(times), dim=0)
    host_times = times.detach().cpu().tolist()
    states = [z0]
    state = z0
    used_steps = 0
    for start, stop in zip(host_times[:-1], host_times[1:]):
        budget = _MAX_INTERNAL_STEPS - used_steps
        try:
            if fixed_step is None:
                state, count = _adaptive_interval(
                    field, state, start, stop, adaptive_tolerance, order, radius, budget
                )
            else:
                state, count = _fixed_interval(
                    field, state, start, stop, fixed_step, order, radius, budget
                )
        except DomainExitError as error:
            # Keep the existing exception API, and make the already completed
            # requested outputs available to function-valued workflows.
            error.completed_times = times[: len(states)].detach().clone()
            error.completed_states = torch.stack(states, dim=0)
            raise
        used_steps += count
        states.append(state)
    return torch.stack(states, dim=0)


def solve(field, z0, times, *, step=None, tolerance=None, order=4, radius=None):
    """Compatibility entry point for ``GalerkinField.solve``."""
    return integrate_field(
        field, z0, times, step=step, tolerance=tolerance, order=order, radius=radius
    )


def time_error(field, z0, times, *, step=None, tolerance=None, order=4, radius=None):
    """Return the absolute temporal refinement indicator at requested times."""
    if step is not None and tolerance is not None:
        raise ValueError("step and tolerance cannot be combined.")
    if step is not None:
        coarse_step = _positive_real(step, "step")
        coarse = solve(field, z0, times, step=coarse_step, order=order, radius=radius)
        refined = solve(field, z0, times, step=coarse_step / 2, order=order, radius=radius)
    else:
        coarse_tolerance = _tolerance(tolerance, field.dtype)
        coarse = solve(field, z0, times, tolerance=coarse_tolerance, order=order, radius=radius)
        refined = solve(
            field, z0, times, tolerance=coarse_tolerance / 2, order=order, radius=radius
        )
    return torch.linalg.vector_norm(refined - coarse, dim=-1)
