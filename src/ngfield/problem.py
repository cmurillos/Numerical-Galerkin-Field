"""The user-supplied local weak action; no spatial discretization lives here."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Problem:
    """An autonomous weak field on L2(Omega; R**components).

    volume(x, u, grad_u) returns (f0, f1) for the integrand f0.v + f1:grad(v).
    x: [Q,m], u: [B,d,Q], grad_u: [B,d,Q,m]. Return these last two shapes,
    or tensors/scalars that broadcast to them. Coupling between components is allowed.
    A callback must act independently on the leading batch axis.

    dirichlet[r] lists homogeneous essential boundary tags for component r.
    Unlisted exterior boundaries are natural. boundary[tag](x,u,grad_u,normal)
    returns an additive boundary load [B,d,Q]; normal has shape [Q,m].
    """

    components: int
    volume: Callable
    dirichlet: tuple[tuple[str, ...], ...] | None = None
    boundary: Mapping[str, Callable] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.components, bool) or not isinstance(self.components, int):
            raise ValueError("components must be a positive integer.")
        if self.components < 1 or not callable(self.volume):
            raise ValueError("Provide positive components and a callable volume action.")
        entries = self.dirichlet
        if entries is None:
            entries = ((),) * self.components
        if len(entries) != self.components or any(isinstance(e, str) for e in entries):
            raise ValueError("dirichlet must contain one tuple of boundary names per component.")
        normalized = []
        for entry in entries:
            if any(not isinstance(n, str) or not n for n in entry):
                raise ValueError("Dirichlet tags must be nonempty strings.")
            normalized.append(tuple(sorted(set(entry))))
        object.__setattr__(self, "dirichlet", tuple(normalized))
        callbacks = dict(self.boundary)
        if any(not isinstance(k, str) or not k or not callable(v) for k, v in callbacks.items()):
            raise ValueError("boundary must map names to callable weak boundary loads.")
        object.__setattr__(self, "boundary", MappingProxyType(callbacks))
