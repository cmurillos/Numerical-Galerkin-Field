"""Numerical Galerkin Field: geometry, spectral bases and weak vector fields."""

from .basis import GalerkinBasis
from .domain import Domain
from .fem import FEMSpace
from .io import load_basis, save_basis
from .problem import Problem

__version__ = "0.1.0"
__all__ = [
    "Domain",
    "FEMSpace",
    "GalerkinBasis",
    "GalerkinField",
    "Problem",
    "load_basis",
    "save_basis",
]


def __getattr__(name):
    # Preparing and inspecting a basis does not initialize PyTorch.
    if name == "GalerkinField":
        from .field import GalerkinField

        return GalerkinField
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
