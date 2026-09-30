"""Indexed derivatives of a reduced field with respect to its state coordinates.

These derivatives are distinct from the spatial ``grad`` and ``hessian`` of a
Galerkin reconstruction. Every multi-index occurs exactly once, as in the
standard integer-order Sobolev norm on a coordinate ball.
"""

from functools import lru_cache
from itertools import combinations_with_replacement
from numbers import Integral

import torch


@lru_cache(maxsize=128)
def _indices(dimension, order):
    result = []
    for degree in range(order + 1):
        for axes in combinations_with_replacement(range(dimension), degree):
            result.append(tuple(axes.count(axis) for axis in range(dimension)))
    return tuple(result)


def multi_indices(dimension, order):
    """Return all ``alpha in N_0^dimension`` with ``|alpha| <= order``.

    The zero index is first; subsequent indices are grouped by total order.
    No factorial weights or multiplicities are attached to mixed derivatives.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("order must be a nonnegative integer.")
    return _indices(int(dimension), int(order))


def state_derivatives(field, states, order):
    """Evaluate ``partial_z^alpha field(states)`` for all ``|alpha| <= order``.

    The field acts independently on arbitrary leading batch axes and maps
    ``[..., N]`` to ``[..., N]``. Differentiation uses directional automatic
    differentiation along coordinate axes, without building a full Jacobian or
    repeating mixed partials. Returned tensors retain their parameter graph.
    """
    field._states(states)
    multi_indices(field.dimension, order)
    directions = torch.eye(field.dimension, dtype=states.dtype, device=states.device)
    functions = {(): field}
    result = {(0,) * field.dimension: field(states)}
    if result[(0,) * field.dimension].shape != states.shape:
        raise ValueError("A state derivative changed the field's batch or coordinate shape.")

    for degree in range(1, order + 1):
        for axes in combinations_with_replacement(range(field.dimension), degree - 1):
            parent = functions[axes]
            first = axes[-1] if axes else 0
            indices = range(first, field.dimension)

            def differentiate(direction):
                return torch.func.jvp(parent, (states,), (direction.expand_as(states),))[1]

            # Group the distinct mixed derivatives with the same parent. This
            # removes one Python/AD dispatch per coordinate without forming a
            # full Hessian with repeated mixed entries.
            chunk_size = max(1, min(len(indices), 1_000_000 // max(1, states.numel())))
            try:
                values = torch.vmap(differentiate, chunk_size=chunk_size)(directions[first:])
            except RuntimeError:
                # A user-supplied differentiable field need not support vmap.
                # The original directional-JVP path remains valid in that case.
                values = torch.stack([differentiate(directions[axis]) for axis in indices])

            for axis, value in zip(indices, values):
                if value.shape != states.shape:
                    raise ValueError(
                        "A state derivative changed the field's batch or coordinate shape."
                    )
                alpha = tuple((axes + (axis,)).count(i) for i in range(field.dimension))
                result[alpha] = value
                direction = directions[axis]

                def partial(z, parent=parent, direction=direction):
                    return torch.func.jvp(parent, (z,), (direction.expand_as(z),))[1]

                functions[axes + (axis,)] = partial
    return result
