import numpy as np
import pytest
import torch
from skfem import Basis, LinearForm, MeshLine, MeshTet, MeshTri, asm
from skfem.helpers import dot

from ngfield import Domain, FEMSpace, GalerkinBasis, GalerkinField, Problem, load_basis, save_basis


def diffusion(x, u, grad_u):
    return 0, -0.2 * grad_u


def coupled_action(x, u, grad_u):
    f0 = torch.stack((u[:, 0] - u[:, 0] ** 3 + 0.2 * u[:, 1], -2 * u[:, 1] + 0.1 * u[:, 0]), dim=1)
    diffusivities = u.new_tensor([0.1, 0.3]).reshape(1, 2, 1, 1)
    return f0, -diffusivities * grad_u


def test_diffusion_field_matches_reference_spectrum_and_jacobian():
    fem = FEMSpace(Domain(MeshTri.init_lshaped().refined(2)))
    problem = Problem(1, diffusion, (("all",),))
    basis = GalerkinBasis.build(fem, problem, 4)
    field = GalerkinField(basis, problem)
    z = torch.randn(7, 4, dtype=torch.float64)
    eigenvalues = torch.tensor(basis.eigenvalues[0].copy())
    torch.testing.assert_close(field(z), -0.2 * z * eigenvalues, rtol=1e-10, atol=1e-10)
    jacobian = torch.autograd.functional.jacobian(field, z[0])
    torch.testing.assert_close(jacobian, torch.diag(-0.2 * eigenvalues), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(field(z), torch.stack([field(row) for row in z]))
    assert field(z[:0]).shape == (0, 4)
    assert field.table_bytes > 0


def test_coupled_nonlinear_field_against_independent_fem_assembly():
    fem = FEMSpace(Domain(MeshTri.init_lshaped().refined(2)), degree=2)
    problem = Problem(2, coupled_action, (("all",), ()))
    basis = GalerkinBasis.build(fem, problem, modes=(3, 2))
    field = GalerkinField(basis, problem, quadrature_order=8)
    z = torch.randn(4, basis.dimension, dtype=torch.float64) * 0.2
    actual = field(z).detach().numpy()
    reference = []
    integration = Basis(fem.domain.mesh, fem.element, intorder=8)

    @LinearForm
    def first(v, w):
        return (w.a - w.a**3 + 0.2 * w.b) * v - 0.1 * dot(w.a.grad, v.grad)

    @LinearForm
    def second(v, w):
        return (-2 * w.b + 0.1 * w.a) * v - 0.3 * dot(w.b.grad, v.grad)

    for row in z.numpy():
        states = [
            integration.interpolate(c @ row[s]) for c, s in zip(basis.coefficients, basis.slices)
        ]
        actions = [asm(form, integration, a=states[0], b=states[1]) for form in (first, second)]
        reference.append(np.concatenate([c.T @ r for c, r in zip(basis.coefficients, actions)]))
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
    permutation = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(field(z[permutation]), field(z)[permutation])
    assert torch.autograd.gradcheck(field, (z[:1].clone().requires_grad_(),))


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_boundary_normals_and_weights_by_divergence_theorem(dimension):
    meshes = {
        1: lambda: MeshLine(np.linspace(0, 1, 5)),
        2: lambda: MeshTri().refined(2),
        3: lambda: MeshTet().refined(1),
    }
    problem = Problem(
        1,
        lambda x, u, grad: (0, 0),
        boundary={"all": lambda x, u, grad, normal: (x * normal).sum(-1)},
    )
    fem = FEMSpace(Domain(meshes[dimension]()))
    basis = GalerkinBasis.build(fem, problem, 1)
    field = GalerkinField(basis, problem)
    z = torch.zeros(2, 1, dtype=torch.float64)
    torch.testing.assert_close(field(z), torch.full_like(z, float(dimension)), rtol=1e-9, atol=1e-9)


def test_robin_boundary_and_named_flux():
    domain = Domain(MeshLine(np.linspace(0, 1, 12))).with_boundaries(
        right=lambda x: np.isclose(x[0], 1)
    )
    problem = Problem(
        1,
        lambda x, u, grad: (0, 0),
        boundary={
            "all": lambda x, u, grad, normal: -3 * u,
            "right": lambda x, u, grad, normal: 2,
        },
    )
    basis = GalerkinBasis.build(FEMSpace(domain), problem, 1)
    field = GalerkinField(basis, problem)
    z = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    torch.testing.assert_close(field(z), -6 * z + 2, rtol=1e-9, atol=1e-9)


def test_persisted_basis_produces_identical_field(tmp_path):
    fem = FEMSpace(Domain(MeshTri.init_lshaped().refined(2)))
    problem = Problem(2, coupled_action, (("all",), ()))
    basis = GalerkinBasis.build(fem, problem, (3, 2))
    path = tmp_path / "basis.npz"
    save_basis(path, basis)
    z = torch.randn(3, basis.dimension, dtype=torch.float64)
    torch.testing.assert_close(
        GalerkinField(basis, problem)(z),
        GalerkinField(load_basis(path), problem)(z),
        rtol=0,
        atol=0,
    )


def test_dtype_shapes_and_problem_mismatch():
    problem = Problem(1, diffusion)
    basis = GalerkinBasis.build(FEMSpace(Domain(MeshLine(np.linspace(0, 1, 10)))), problem, 3)
    field = GalerkinField(basis, problem)
    with pytest.raises(ValueError, match="same device and dtype"):
        field(torch.zeros(1, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match="Expected states"):
        field(torch.zeros(2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="must match"):
        GalerkinField(basis, Problem(1, diffusion, (("all",),)))
    with pytest.raises(ValueError, match="broadcast"):
        GalerkinField(basis, Problem(1, lambda x, u, grad: (torch.zeros(7, dtype=u.dtype), 0)))(
            torch.zeros(2, 3, dtype=torch.float64)
        )
    assert field.to(dtype=torch.float32)(torch.zeros(1, 3)).dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_cpu():
    problem = Problem(2, coupled_action)
    basis = GalerkinBasis.build(FEMSpace(Domain(MeshTri.init_lshaped().refined(2))), problem, 3)
    field = GalerkinField(basis, problem)
    z = torch.randn(8, 6, dtype=torch.float64)
    reference = field(z)
    torch.testing.assert_close(field.to("cuda")(z.cuda()).cpu(), reference, rtol=1e-9, atol=1e-9)
