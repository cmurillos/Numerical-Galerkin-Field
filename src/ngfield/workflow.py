"""Function-valued workflow over the fixed numerical Galerkin field.

The coordinate vector is an implementation of a function in the operational
orthonormal basis.  Spatial differentiation and differentiation of the reduced
velocity with respect to its coordinates are deliberately separate operations.
"""

from math import isfinite
from numbers import Real

import torch

from .evolution import DomainExitError, integrate_field, time_error
from .galerkin import GalerkinField
from .state_derivatives import state_derivatives


def _coordinates(owner, coefficients):
    owner._field._states(coefficients)
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("Function coefficients must be finite.")
    return coefficients


class Function:
    """An element of the fixed operational space, with arbitrary batch axes."""

    def __init__(self, owner, coefficients):
        self._owner = owner
        self._coefficients = _coordinates(owner, coefficients).clone()

    @property
    def basis(self):
        return self._owner.basis

    @property
    def space(self):
        return self._owner.space

    @property
    def geometry(self):
        return self._owner.geometry

    def coefficients(self):
        """Return reduced coordinates ``[...,N]`` without exposing internal storage."""
        return self._coefficients.clone()

    def values(self, points=None, *, cells=None, boundary=None):
        return self._owner._field.reconstruct(
            self._coefficients, points, cells=cells, boundary=boundary
        )

    def gradient(self, points, *, cells=None):
        """Elementwise spatial gradient; ambiguous facet traces require ``cells``."""
        return self._owner._field.grad(self._coefficients, points, cells=cells)

    def hessian(self, points, *, cells=None):
        """Elementwise spatial Hessian; this does not assert global H² regularity."""
        return self._owner._field.hessian(self._coefficients, points, cells=cells)

    def norm_L2(self):
        """Discrete L² norm from the operational orthonormal basis."""
        return torch.linalg.vector_norm(self._coefficients, dim=-1)

    def integral(self):
        """Quadrature integral by physical component, with all batch axes preserved."""
        return torch.tensordot(
            self._coefficients, self._owner._field.integral_weights(), dims=([-1], [0])
        )


class State(Function):
    """A phase point at which the autonomous reduced velocity is defined."""

    def __init__(self, owner, coefficients):
        super().__init__(owner, coefficients)
        owner._validate_state(self._coefficients)

    def velocity(self):
        """The function in V_N representing the temporal derivative at this point."""
        return Function(self._owner, self._owner._dynamics(self._coefficients))

    def indexed_derivatives(self, order):
        """Map each multi-index to the function ∂_a^α velocity(a), |α|≤order."""
        return self._owner.indexed_derivatives(self, order)


class Solution:
    """A function-valued numerical trajectory sampled at its requested times.

    If the open ball is reached, only completed output times are retained.
    ``exit_status`` also records the last accepted interior integration state,
    which need not be one of the requested output times.
    """

    def __init__(self, owner, times, coefficients, *, exit_error=None):
        self._owner = owner
        self._times = times.detach().clone()
        self._coefficients = coefficients.clone()
        self._exit_error = exit_error

    def at(self, time):
        """Return a State at an output time; no temporal interpolation is implied."""
        if isinstance(time, torch.Tensor):
            if (
                time.ndim != 0
                or time.device != self._times.device
                or time.dtype != self._times.dtype
            ):
                raise TypeError("time must be a scalar on the solution device and dtype.")
            time = float(time)
        if isinstance(time, bool) or not isinstance(time, Real) or not isfinite(float(time)):
            raise TypeError("time must be a finite real number.")
        query = self._times.new_tensor(float(time))
        indices = torch.nonzero(self._times == query).reshape(-1)
        if not len(indices):
            raise ValueError("time must be one of the completed output times.")
        return State(self._owner, self._coefficients[int(indices[0])])

    def coefficients(self):
        """Return the sampled path ``[T,*batch,N]``."""
        return self._coefficients.clone()

    @property
    def times(self):
        return self._times.clone()

    def exit_status(self):
        """Numerical open-ball status; the last accepted time is not an exact exit."""
        if self._exit_error is None:
            time, state = float(self._times[-1]), self._coefficients[-1]
        else:
            time, state = self._exit_error.time, self._exit_error.last_state
        return {
            "exited": self._exit_error is not None,
            "last_accepted_time": time,
            "last_accepted_state": State(self._owner, state),
        }


class Metrics:
    """Absolute numerical indicators and observables of a function-valued flow."""

    def __init__(self, owner):
        self._owner = owner

    def projection(self, source, *, quadrature=None):
        return self._owner._field.projection_error(source, quadrature=quadrature)

    def quadrature(self, state, *, order=None):
        """Reference Galerkin field quadrature refinement indicator."""
        z = self._owner._state_coordinates(state)
        return self._owner._field.quadrature_error(z, order=order)

    def time_refinement(self, initial, times, *, step=None, tolerance=None, order=4, radius=None):
        z = self._owner._state_coordinates(initial)
        return time_error(
            self._owner._dynamics,
            z,
            times,
            step=step,
            tolerance=tolerance,
            order=order,
            radius=self._owner._integration_radius(radius),
        )

    def integral_rate(self, state):
        self._owner._state_coordinates(state)
        return state.velocity().integral()

    def radial_rate(self, state):
        """Half the time derivative of the squared discrete L² norm."""
        z = self._owner._state_coordinates(state)
        return (z * self._owner._dynamics(z)).sum(dim=-1)

    def sampled_lipschitz(self, states):
        """Maximum sampled Jacobian operator norm; not a global Lipschitz bound."""
        z = self._owner._sample_coordinates(states)
        if not len(z):
            raise ValueError("states must contain at least one point.")
        derivatives = state_derivatives(self._owner._dynamics, z, 1)
        axes = [tuple(int(i == j) for i in range(z.shape[-1])) for j in range(z.shape[-1])]
        matrix = torch.stack([derivatives[alpha] for alpha in axes], dim=-1)
        return torch.linalg.matrix_norm(matrix, ord=2).max()


class System:
    """Fixed Galerkin system whose flow acts on functions in its operational space."""

    def __init__(
        self,
        *,
        basis,
        weak,
        quadrature=None,
        device="cpu",
        dtype=torch.float64,
        max_quadrature_points=1_000_000,
        max_intermediate_entries=10_000_000,
    ):
        self._field = GalerkinField(
            basis=basis,
            weak=weak,
            quadrature=quadrature,
            device=device,
            dtype=dtype,
            max_quadrature_points=max_quadrature_points,
            max_intermediate_entries=max_intermediate_entries,
        )
        self._dynamics = self._field
        self.metrics = Metrics(self)

    @property
    def basis(self):
        return self._field.basis

    @property
    def space(self):
        return self._field.space

    @property
    def geometry(self):
        return self._field.geometry

    @property
    def dimension(self):
        return self._field.dimension

    @property
    def device(self):
        return self._field.device

    @property
    def dtype(self):
        return self._field.dtype

    def _validate_state(self, z):
        _coordinates(self, z)

    def _owns(self, state):
        return isinstance(state, State) and state._owner is self

    def _state_coordinates(self, state):
        if not self._owns(state):
            raise ValueError("state must belong to this system.")
        return state._coefficients

    def _sample_coordinates(self, states):
        if isinstance(states, State):
            return self._state_coordinates(states).reshape(-1, self.dimension)
        if isinstance(states, (tuple, list)) and all(isinstance(s, State) for s in states):
            if not states:
                raise ValueError("states must contain at least one point.")
            return torch.stack([self._state_coordinates(s) for s in states]).reshape(
                -1, self.dimension
            )
        self._validate_state(states)
        return states.reshape(-1, self.dimension)

    @staticmethod
    def _integration_radius(radius):
        return radius

    def state(self, source, *, projection_quadrature=None):
        """Project a physical function onto the fixed basis to obtain a phase point."""
        return State(self, self._field.project(source, quadrature=projection_quadrature))

    def from_coefficients(self, coefficients):
        """Wrap reduced coordinates explicitly, preserving their batch axes."""
        return State(self, coefficients)

    def indexed_derivatives(self, state, order):
        """All ∂_a^α of the reduced velocity, reconstructed as functions in V_N."""
        z = self._state_coordinates(state)
        return {
            alpha: Function(self, value)
            for alpha, value in state_derivatives(self._dynamics, z, order).items()
        }

    def evolve(self, initial, times, *, step=None, tolerance=None, order=4, radius=None):
        """Integrate from the first output time using the shared Taylor integrator."""
        z = self._state_coordinates(initial)
        radius = self._integration_radius(radius)
        try:
            path = integrate_field(
                self._dynamics, z, times, step=step, tolerance=tolerance, order=order, radius=radius
            )
        except DomainExitError as error:
            if not hasattr(error, "completed_states"):
                raise
            return Solution(self, error.completed_times, error.completed_states, exit_error=error)
        return Solution(self, times, path)
