"""Finite-element preprocessing and reduced-mode tabulation."""

from dataclasses import dataclass

import numpy as np
from skfem import (
    Basis,
    ElementLineP1,
    ElementLineP2,
    ElementTetP1,
    ElementTetP2,
    ElementTriP1,
    ElementTriP2,
    FacetBasis,
    asm,
)
from skfem.models.poisson import laplace, mass

from .domain import Domain


@dataclass(frozen=True)
class Quadrature:
    """CPU float64 arrays with flattened element/quadrature axis Q."""

    x: np.ndarray  # [Q,m]
    weights: np.ndarray  # [Q]
    values: np.ndarray  # [N,Q]
    gradients: np.ndarray  # [N,Q,m]
    normals: np.ndarray | None = None  # [Q,m] for exterior facets


class FEMSpace:
    """Scalar continuous P1/P2 FEM on an affine simplicial mesh.

    All components use this mesh and element family; essential boundaries may differ.
    mass_order controls the metric once, independently of field quadrature refinement.
    """

    def __init__(self, domain: Domain, degree: int = 1, mass_order: int | None = None):
        if isinstance(degree, bool) or degree not in (1, 2):
            raise ValueError("degree must be 1 or 2.")
        order = 2 * degree if mass_order is None else mass_order
        if isinstance(order, bool) or not isinstance(order, int) or order < 2 * degree:
            raise ValueError("mass_order must be an integer >= 2*degree.")
        elements = {
            (1, 1): ElementLineP1,
            (1, 2): ElementLineP2,
            (2, 1): ElementTriP1,
            (2, 2): ElementTriP2,
            (3, 1): ElementTetP1,
            (3, 2): ElementTetP2,
        }
        self.domain = domain
        self.degree = degree
        self.mass_order = order
        self.element = elements[(domain.dimension, degree)]()
        self.basis = Basis(domain.mesh, self.element, intorder=order)
        self.mass = asm(mass, self.basis).tocsr()
        self.stiffness = asm(laplace, self.basis).tocsr()

    @property
    def ndofs(self) -> int:
        return int(self.basis.N)

    def essential_dofs(self, names) -> np.ndarray:
        facets = self.domain.facets(names)
        if not facets.size:
            return np.empty(0, dtype=np.int32)
        return self.basis.get_dofs(facets=facets).all()

    def tabulate(self, coefficients: np.ndarray, order: int, boundary=None) -> Quadrature:
        """Tabulate scalar modes C[ndofs,N] at physical volume or facet points.

        Preparation uses contractions over all modes; evaluation needs no FEM assembly.
        Signed high-order quadrature weights are accepted (no square-root weighting).
        """
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("Quadrature order must be a positive integer.")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[0] != self.ndofs:
            raise ValueError("coefficients must have shape [ndofs, N].")
        if boundary is None:
            basis = Basis(self.domain.mesh, self.element, intorder=order)
            normals = None
        else:
            basis = FacetBasis(
                self.domain.mesh,
                self.element,
                facets=self.domain.facets(boundary),
                intorder=order,
            )
            normals = np.asarray(basis.normals).reshape(self.domain.dimension, -1).T
        local = coefficients[basis.element_dofs]  # [local,element,mode]
        shape_values = np.stack([np.asarray(b[0]) for b in basis.basis])
        shape_gradients = np.stack([np.asarray(b[0].grad) for b in basis.basis])
        values = np.einsum("len,leq->neq", local, shape_values, optimize=True)
        gradients = np.einsum("len,lmeq->neqm", local, shape_gradients, optimize=True)
        n = coefficients.shape[1]
        return Quadrature(
            x=np.ascontiguousarray(
                np.asarray(basis.global_coordinates()).reshape(self.domain.dimension, -1).T
            ),
            weights=np.ascontiguousarray(basis.dx.reshape(-1)),
            values=np.ascontiguousarray(values.reshape(n, -1)),
            gradients=np.ascontiguousarray(gradients.reshape(n, -1, self.domain.dimension)),
            normals=None if normals is None else np.ascontiguousarray(normals),
        )
