"""Numerical Galerkin Field: general weak fields on fixed function bases."""

from .basis import GalerkinBasis
from .domain import Domain
from .fem import FEMSpace
from .field import GalerkinField as LegacyGalerkinField
from .forms import cos, exp, grad, inner, log, sin, sqrt, stack, tanh
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
    TransformedBasis,
)

__version__ = "0.2.0"
__all__ = [
    "Basis",
    "CallableBasis",
    "ComponentBasis",
    "Domain",
    "FEMSpace",
    "FiniteElementBasis",
    "GalerkinBasis",
    "GalerkinField",
    "GeneralGalerkinField",
    "GalerkinProblem",
    "PolynomialBasis",
    "Problem",
    "LegacyGalerkinField",
    "SimplicialDomain",
    "TransformedBasis",
    "cos",
    "exp",
    "grad",
    "inner",
    "load_basis",
    "log",
    "save_basis",
    "sin",
    "sqrt",
    "stack",
    "tanh",
]


def GalerkinField(*args, **kwargs):
    """Construct the general field, with transparent compatibility for version 0.1."""
    if args and isinstance(args[0], GalerkinProblem):
        return GeneralGalerkinField(*args, **kwargs)
    return LegacyGalerkinField(*args, **kwargs)
