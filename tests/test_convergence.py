import numpy as np
import torch
from skfem import MeshLine

from ngfield import Domain, FEMSpace, GalerkinBasis, GalerkinField, Problem


def test_field_converges_to_continuous_heat_mode():
    errors = []
    problem = Problem(1, lambda x, u, grad: (0, -grad), (("all",),))
    for intervals in (8, 16, 32):
        fem = FEMSpace(Domain(MeshLine(np.linspace(0, 1, intervals + 1))))
        basis = GalerkinBasis.build(fem, problem, 1)
        value = GalerkinField(basis, problem)(torch.ones(1, dtype=torch.float64)).item()
        errors.append(abs(value + np.pi**2))
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_quadrature_refinement_changes_action_without_changing_basis():
    problem = Problem(1, lambda x, u, grad: (-(u**3), 0))
    basis = GalerkinBasis.build(FEMSpace(Domain(MeshLine(np.linspace(0, 1, 7)))), problem, 3)
    z = torch.tensor([[0.3, 0.7, -0.4]], dtype=torch.float64)
    low, sufficient, reference = [
        GalerkinField(basis, problem, quadrature_order=order)(z) for order in (2, 4, 8)
    ]
    assert torch.linalg.vector_norm(low - reference).item() > 1e-5
    torch.testing.assert_close(sufficient, reference, rtol=1e-11, atol=1e-11)
