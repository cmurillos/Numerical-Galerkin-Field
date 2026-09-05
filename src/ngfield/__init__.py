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
from .spaces import (
    Basis,
    CallableBasis,
    ComponentBasis,
    FiniteElementBasis,
    PolynomialBasis,
    ProductBasis,
    TransformedBasis,
)

__version__ = "0.3.0"
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
    "PolynomialBasis",
    "ProductBasis",
    "Problem",
    "LegacyGalerkinField",
    "SimplicialDomain",
    "TransformedBasis",
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
    """Construct the general field, with transparent compatibility for version 0.1."""
    if args and isinstance(args[0], GalerkinProblem):
        return GeneralGalerkinField(*args, **kwargs)
    return LegacyGalerkinField(*args, **kwargs)
