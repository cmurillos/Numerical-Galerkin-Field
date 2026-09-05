"""Descriptions of admissible function spaces on fixed simplicial geometries."""

from dataclasses import dataclass

from .geometry import SimplicialDomain, positive_integer


@dataclass(frozen=True, kw_only=True)
class Space:
    """Declare a real function space before selecting a finite-dimensional basis.

    ``components`` is the number of state components, independent of the intrinsic
    and ambient geometry dimensions. Even one component has ``value_shape=(1,)``.
    ``regularity`` is the nonnegative integer Sobolev order requested for every
    component; zero denotes L2 and one denotes H1. This declaration does not
    certify a basis or provide a conforming discretization of higher order.

    The description is frozen and retains the supplied SimplicialDomain by
    reference; do not mutate that geometry. The ambient state inner product is
    L2 with its induced measure, summed over components.

    In D-013 part 2, only empty restrictions are supported. Nonempty restrictions
    are rejected rather than ignored. Restriction construction, basis selection
    and direct field construction belong to subsequent parts of the contract.
    """

    geometry: SimplicialDomain
    components: int
    regularity: int = 1
    restrictions: tuple[object, ...] = ()

    def __post_init__(self):
        if not isinstance(self.geometry, SimplicialDomain):
            raise TypeError("geometry must be a SimplicialDomain.")
        object.__setattr__(self, "components", positive_integer(self.components, "components"))
        object.__setattr__(self, "regularity", positive_integer(self.regularity, "regularity", 0))
        if not isinstance(self.restrictions, (list, tuple)):
            raise TypeError("restrictions must be a list or tuple.")
        if self.restrictions:
            raise NotImplementedError(
                "Restriction construction is not implemented yet. "
                "The existing API accepts an explicitly admissible basis."
            )
        object.__setattr__(self, "restrictions", ())

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Physical value shape; this is not the reduced coordinate dimension."""
        return (self.components,)
