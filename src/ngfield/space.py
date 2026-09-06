"""Descriptions of admissible function spaces on fixed simplicial geometries."""

from dataclasses import dataclass

from .geometry import SimplicialDomain, positive_integer
from .restrictions import MeanZero, Periodic, ZeroTrace, restrict_basis


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

    ``basis`` constructs a fixed L2-orthonormal basis in this space. ZeroTrace, Periodic and
    MeanZero constraints are supported for nodal FEM representations; unsupported family
    combinations are rejected. ``restrict`` also exposes the unnormalized nodal
    kernel. Direct field construction belongs to a subsequent part of D-013.
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
            if type(restriction) not in (ZeroTrace, Periodic, MeanZero):
                raise TypeError(
                    "Only ZeroTrace, Periodic and MeanZero restrictions are implemented."
                )
            restriction._validate(self)
        object.__setattr__(self, "restrictions", tuple(dict.fromkeys(self.restrictions)))

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Physical value shape; this is not the reduced coordinate dimension."""
        return (self.components,)

    def restrict(
        self,
        basis,
        *,
        tolerance=1e-12,
        max_matrix_entries=10_000_000,
        max_quadrature_points=1_000_000,
    ):
        """Return the candidate span satisfying the declared homogeneous nodal constraints.

        The candidate must have value_shape=(components,). Built-in nodal FEM
        bases and their component/product/linear combinations are supported.
        The result is not L2-orthonormalized and no number of modes is selected.
        ``tolerance`` controls numerical rank after candidate-column scaling;
        ``max_matrix_entries`` bounds dense algebraic preparation;
        ``max_quadrature_points`` bounds MeanZero integration.
        """
        return restrict_basis(
            self,
            basis,
            tolerance=tolerance,
            max_matrix_entries=max_matrix_entries,
            max_quadrature_points=max_quadrature_points,
        )

    def basis(self, family="laplacian", *, size=None, component_sizes=None, **options):
        """Build an admissible, numerically L2-orthonormal operational basis.

        ``size`` always counts total modes. Laplacian modes minimize the discrete
        spectrum over the constrained space; ``component_sizes`` optionally fixes
        their allocation. Polynomial/Fourier modes use a balanced allocation by
        default. Finite-element/custom families preserve their full admissible
        span; optional ``size`` checks its dimension, without truncation.

        Restrictions require a built-in nodal representation. Custom sources
        must have value_shape=(components,); unknown source regularity remains the
        user's declaration, recorded by ``regularity_verified=False``. This stage
        implements only regularity zero or one. See D-013 for family limitations.
        """
        from .space_bases import build_space_basis

        return build_space_basis(
            self, family, size=size, component_sizes=component_sizes, **options
        )
