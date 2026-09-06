"""Numerical Galerkin Field: general weak fields on fixed function bases."""

from .basis import GalerkinBasis
from .domain import Domain
from .fem import FEMSpace
from .field import GalerkinField as LegacyGalerkinField
from .forms import (
    Coefficient,
    contract,
    cos,
    div,
    dot,
    exp,
    grad,
    inner,
    log,
    outer,
    pointwise,
    sin,
    sqrt,
    stack,
    sym_grad,
    tanh,
    trace,
    transpose,
)
from .galerkin import GalerkinField as GeneralGalerkinField
from .galerkin import GalerkinProblem
from .geometry import SimplicialDomain
from .io import load_basis, save_basis
from .problem import Problem
from .restrictions import MeanZero, Periodic, ZeroTrace
from .space import Space
from .spaces import (
    Basis,
    CallableBasis,
    ComponentBasis,
    FiniteElementBasis,
    PolynomialBasis,
    ProductBasis,
    TransformedBasis,
)

__version__ = "0.9.0"
__all__ = [
    "Basis",
    "CallableBasis",
    "Coefficient",
    "ComponentBasis",
    "Domain",
    "FEMSpace",
    "FiniteElementBasis",
    "GalerkinBasis",
    "GalerkinField",
    "GeneralGalerkinField",
    "GalerkinProblem",
    "MeanZero",
    "Periodic",
    "PolynomialBasis",
    "ProductBasis",
    "Problem",
    "LegacyGalerkinField",
    "SimplicialDomain",
    "Space",
    "TransformedBasis",
    "ZeroTrace",
    "contract",
    "cos",
    "div",
    "dot",
    "exp",
    "grad",
    "inner",
    "load_basis",
    "log",
    "outer",
    "pointwise",
    "save_basis",
    "sin",
    "sqrt",
    "stack",
    "sym_grad",
    "tanh",
    "trace",
    "transpose",
]


def GalerkinField(*args, **kwargs):
    """Construct from basis/weak, an explicit GalerkinProblem, or the legacy API.

    The direct route is ``GalerkinField(basis=V.basis(...), weak=weak, ...)``.
    Existing ``GalerkinField(problem, basis)`` and legacy ``(basis, problem)``
    calls retain their argument order; keyword-based versions are also supported.
    """
    general_problem = (args and isinstance(args[0], GalerkinProblem)) or isinstance(
        kwargs.get("problem"), GalerkinProblem
    )
    direct = "weak" in kwargs or (not args and "problem" not in kwargs)
    if general_problem or direct:
        return GeneralGalerkinField(*args, **kwargs)
    return LegacyGalerkinField(*args, **kwargs)
