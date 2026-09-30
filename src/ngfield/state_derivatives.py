"""Indexed derivatives of a reduced field with respect to its state coordinates.

These derivatives are distinct from the spatial ``grad`` and ``hessian`` of a
Galerkin reconstruction. Every multi-index occurs exactly once, as in the
standard integer-order Sobolev norm on a coordinate ball.
"""

from itertools import combinations_with_replacement
from numbers import Integral

import torch


def multi_indices(dimension, order):
    """Return all ``alpha in N_0^dimension`` with ``|alpha| <= order``.

    The zero index is first; subsequent indices are grouped by total order.
    No factorial weights or multiplicities are attached to mixed derivatives.
    """
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 0:
        raise ValueError("order must be a nonnegative integer.")
    result = []
    for degree in range(order + 1):
        for axes in combinations_with_replacement(range(dimension), degree):
            result.append(tuple(axes.count(axis) for axis in range(dimension)))
    return tuple(result)


def state_derivatives(field, states, order):
    """Evaluate ``partial_z^alpha field(states)`` for all ``|alpha| <= order``.

    The field acts independently on arbitrary leading batch axes and maps
    ``[..., N]`` to ``[..., N]``. Differentiation uses directional automatic
    differentiation along coordinate axes, without building a full Jacobian or
    repeating mixed partials. Returned tensors retain their parameter graph.
    """
    field._states(states)
    indices = multi_indices(field.dimension, order)
    directions = torch.eye(field.dimension, dtype=states.dtype, device=states.device)
    functions = {(): field}
    result = {}

    for alpha in indices:
        axes = tuple(axis for axis, multiplicity in enumerate(alpha) for _ in range(multiplicity))
        if axes:
            parent = functions[axes[:-1]]
            direction = directions[axes[-1]]

            def partial(z, parent=parent, direction=direction):
                return torch.func.jvp(parent, (z,), (direction.expand_as(z),))[1]

            functions[axes] = partial
        value = functions[axes](states)
        if value.shape != states.shape:
            raise ValueError("A state derivative changed the field's batch or coordinate shape.")
        result[alpha] = value
    return result
