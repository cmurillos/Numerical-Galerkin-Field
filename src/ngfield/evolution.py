"""Time integration for autonomous Galerkin coordinate fields."""

from math import ceil, isfinite
from numbers import Real

import torch

_MAX_INTERNAL_STEPS = 1_000_000


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


def _inputs(field, z0, times):
    field._states(z0)
    if not torch.isfinite(z0).all():
        raise ValueError("The initial state must be finite.")
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
        increasing = bool(torch.all(increments > 0).item())
        decreasing = bool(torch.all(increments < 0).item())
        if not increasing and not decreasing:
            raise ValueError("times must be strictly monotone.")


def _checked_field(field, state):
    value = field(state)
    if value.shape != state.shape:
        raise ValueError("The Galerkin field changed the state shape during integration.")
    if not torch.isfinite(value).all():
        raise FloatingPointError("The Galerkin field returned a nonfinite velocity.")
    return value


def _rk4_step(field, state, step):
    k1 = _checked_field(field, state)
    k2 = _checked_field(field, state + (step / 2) * k1)
    k3 = _checked_field(field, state + (step / 2) * k2)
    k4 = _checked_field(field, state + step * k3)
    result = state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not torch.isfinite(result).all():
        raise FloatingPointError("RK4 produced a nonfinite state.")
    return result


def _rk45_step(field, state, step):
    k1 = _checked_field(field, state)
    k2 = _checked_field(field, state + step * (k1 / 5))
    k3 = _checked_field(field, state + step * (3 * k1 / 40 + 9 * k2 / 40))
    k4 = _checked_field(
        field,
        state + step * (44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9),
    )
    k5 = _checked_field(
        field,
        state + step * (19372 * k1 / 6561 - 25360 * k2 / 2187 + 64448 * k3 / 6561 - 212 * k4 / 729),
    )
    k6 = _checked_field(
        field,
        state
        + step
        * (
            9017 * k1 / 3168 - 355 * k2 / 33 + 46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656
        ),
    )
    fifth = state + step * (
        35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84
    )
    k7 = _checked_field(field, fifth)
    fourth = state + step * (
        5179 * k1 / 57600
        + 7571 * k3 / 16695
        + 393 * k4 / 640
        - 92097 * k5 / 339200
        + 187 * k6 / 2100
        + k7 / 40
    )
    if not torch.isfinite(fifth).all() or not torch.isfinite(fourth).all():
        raise FloatingPointError("RK45 produced a nonfinite state.")
    return fifth, fifth - fourth


def _fixed_interval(field, state, start, stop, maximum_step, budget):
    count = max(1, ceil(abs(stop - start) / maximum_step))
    if count > budget:
        raise RuntimeError("Time integration exceeded its internal step budget.")
    step = (stop - start) / count
    for _ in range(count):
        state = _rk4_step(field, state, step)
    return state, count


def _adaptive_interval(field, state, start, stop, tolerance, budget):
    direction = 1.0 if stop > start else -1.0
    current = start
    step_size = abs(stop - start)
    count = 0
    epsilon = torch.finfo(field.dtype).eps

    while direction * (stop - current) > 0:
        if count >= budget:
            raise RuntimeError("Time integration exceeded its internal step budget.")
        remaining = abs(stop - current)
        step_size = min(step_size, remaining)
        signed_step = direction * step_size
        candidate, difference = _rk45_step(field, state, signed_step)
        scale = tolerance * (1 + torch.maximum(torch.abs(state), torch.abs(candidate)))
        error = float(torch.max(torch.abs(difference) / scale).detach().item())
        count += 1

        minimum = 16 * epsilon * max(1.0, abs(current), abs(stop))
        if error <= 1:
            state = candidate
            current += signed_step
            if abs(stop - current) <= minimum:
                current = stop
            factor = 5.0 if error == 0 else min(5.0, max(0.2, 0.9 * error ** (-0.2)))
            step_size *= factor
        else:
            if step_size <= minimum:
                raise RuntimeError(
                    "RK45 cannot satisfy the requested tolerance at machine precision."
                )
            step_size *= max(0.2, 0.9 * error ** (-0.2))

    return state, count


def solve(field, z0, times, *, step=None, tolerance=None):
    """Integrate ``z' = field(z)`` and return states at the requested times."""
    _inputs(field, z0, times)
    if step is not None and tolerance is not None:
        raise ValueError("step and tolerance select different solvers and cannot be combined.")
    fixed_step = None if step is None else _positive_real(step, "step")
    adaptive_tolerance = None if fixed_step is not None else _tolerance(tolerance, field.dtype)

    if not z0.numel() or len(times) == 1:
        return torch.stack([z0] * len(times), dim=0)

    host_times = times.detach().cpu().tolist()
    states = [z0]
    state = z0
    used_steps = 0
    for start, stop in zip(host_times[:-1], host_times[1:]):
        remaining_budget = _MAX_INTERNAL_STEPS - used_steps
        if fixed_step is None:
            state, count = _adaptive_interval(
                field,
                state,
                start,
                stop,
                adaptive_tolerance,
                remaining_budget,
            )
        else:
            state, count = _fixed_interval(
                field,
                state,
                start,
                stop,
                fixed_step,
                remaining_budget,
            )
        used_steps += count
        states.append(state)
    return torch.stack(states, dim=0)


def time_error(field, z0, times, *, step=None, tolerance=None):
    """Return the L2 time-refinement indicator at every requested time."""
    if step is not None and tolerance is not None:
        raise ValueError("step and tolerance cannot be combined.")
    if step is not None:
        coarse_step = _positive_real(step, "step")
        coarse = solve(field, z0, times, step=coarse_step)
        refined = solve(field, z0, times, step=coarse_step / 2)
    else:
        coarse_tolerance = _tolerance(tolerance, field.dtype)
        coarse = solve(field, z0, times, tolerance=coarse_tolerance)
        refined = solve(field, z0, times, tolerance=coarse_tolerance / 2)
    return torch.linalg.vector_norm(refined - coarse, dim=-1)
