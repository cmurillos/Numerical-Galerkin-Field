"""Descriptions of admissible function spaces on fixed simplicial geometries."""

from dataclasses import dataclass

from .geometry import SimplicialDomain, positive_integer
from .restrictions import ZeroTrace, restrict_basis


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

    D-013 part 3 supports ZeroTrace restrictions on named exterior boundaries.
    ``restrict`` constructs their nodal kernel before basis selection and L2
    orthonormalization. Basis selection and direct field construction belong to
    subsequent parts of the contract.
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
        for restriction in self.restrictions:
            if type(restriction) is not ZeroTrace:
                raise TypeError("Only ZeroTrace restrictions are implemented in this stage.")
            restriction._validate(self)
        object.__setattr__(self, "restrictions", tuple(dict.fromkeys(self.restrictions)))

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Physical value shape; this is not the reduced coordinate dimension."""
        return (self.components,)

    def restrict(self, basis, *, tolerance=1e-12, max_matrix_entries=10_000_000):
        """Return the candidate span satisfying the declared nodal trace constraints.

        The candidate must have value_shape=(components,). Built-in nodal FEM
        bases and their component/product/linear combinations are supported.
        The result is not L2-orthonormalized and no number of modes is selected.
        ``tolerance`` controls numerical rank after candidate-column scaling;
        ``max_matrix_entries`` bounds dense algebraic preparation.
        """
        return restrict_basis(
            self, basis, tolerance=tolerance, max_matrix_entries=max_matrix_entries
        )
