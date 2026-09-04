import numpy as np
import pytest
from skfem import MeshLine, MeshTet, MeshTri

from ngfield import Domain, FEMSpace, GalerkinBasis, Problem, load_basis, save_basis


def zero_action(x, u, grad_u):
    return 0, 0


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_mass_metric_eigenproblem_and_component_boundaries(dimension, degree):
    meshes = {
        1: lambda: MeshLine(np.linspace(0, 1, 12)),
        2: lambda: MeshTri.init_lshaped().refined(2),
        3: lambda: MeshTet().refined(2),
    }
    fem = FEMSpace(Domain(meshes[dimension]()), degree=degree)
    problem = Problem(2, zero_action, dirichlet=(("all",), ()))
    basis = GalerkinBasis.build(fem, problem, modes=(2, 3))
    assert basis.dimension == 5
    assert basis.modes == (2, 3)
    for report in basis.diagnostics()["components"]:
        assert report["mass_error"] < 1e-10
        assert report["eigen_residual"] < 1e-8
        assert report["essential_error"] == 0
    assert basis.eigenvalues[1][0] < 1e-8
    assert basis.eigenvalues[0][0] > 0
    q = fem.tabulate(basis.coefficients[0], order=2 * degree)
    np.testing.assert_allclose((q.values * q.weights) @ q.values.T, np.eye(2), atol=1e-10)


def test_uniform_p1_spectrum_against_closed_form():
    intervals = 20
    h = 1 / intervals
    fem = FEMSpace(Domain(MeshLine(np.linspace(0, 1, intervals + 1))))
    basis = GalerkinBasis.build(fem, Problem(1, zero_action, (("all",),)), modes=4)
    theta = np.arange(1, 5) * np.pi / intervals
    exact_discrete = 6 * (1 - np.cos(theta)) / (h * h * (2 + np.cos(theta)))
    np.testing.assert_allclose(basis.eigenvalues[0], exact_discrete, rtol=1e-11)


def test_sparse_neumann_solver_keeps_zero_mode():
    fem = FEMSpace(Domain(MeshLine(np.linspace(0, 1, 301))))
    basis = GalerkinBasis.build(fem, Problem(1, zero_action), modes=3)
    np.testing.assert_allclose(basis.coefficients[0][:, 0], 1, atol=1e-7)
    np.testing.assert_allclose(basis.eigenvalues[0], (np.arange(3) * np.pi) ** 2, atol=0.002)
    assert basis.diagnostics()["components"][0]["eigen_residual"] < 1e-8


def test_named_mixed_boundary_and_basis_roundtrip(tmp_path):
    domain = Domain(MeshTri.init_lshaped().refined(2)).with_boundaries(
        left=lambda x: np.isclose(x[0], -1)
    )
    fem = FEMSpace(domain, degree=2)
    problem = Problem(2, zero_action, (("left",), ()))
    basis = GalerkinBasis.build(fem, problem, modes=(3, 2))
    path = tmp_path / "basis.npz"
    save_basis(path, basis)
    restored = load_basis(path)
    assert restored.dirichlet == basis.dirichlet
    assert restored.modes == basis.modes
    for a, b in zip(restored.coefficients, basis.coefficients):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(restored.eigenvalues, basis.eigenvalues):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored.fem.basis.element_dofs, fem.basis.element_dofs)
    with pytest.raises(FileExistsError):
        save_basis(path, basis)


def test_mesh_import(tmp_path):
    domain = Domain(MeshTri.init_lshaped())
    path = tmp_path / "domain.vtu"
    domain.mesh.save(path)
    restored = Domain.from_file(path)
    np.testing.assert_allclose(restored.mesh.p, domain.mesh.p)
    assert restored.mesh.t.shape == domain.mesh.t.shape
    assert restored.facets("all").size == domain.facets("all").size


def test_invalid_inputs_fail_explicitly():
    domain = Domain(MeshLine(np.linspace(0, 1, 5)))
    fem = FEMSpace(domain)
    with pytest.raises(ValueError, match="Unknown boundary"):
        GalerkinBasis.build(fem, Problem(1, zero_action, (("typo",),)), 1)
    with pytest.raises(ValueError, match="unconstrained DOFs"):
        GalerkinBasis.build(fem, Problem(1, zero_action, (("all",),)), 10)
    with pytest.raises(ValueError, match="positive mode"):
        GalerkinBasis.build(fem, Problem(1, zero_action), (0,))
    with pytest.raises(ValueError, match="mass_order"):
        FEMSpace(domain, degree=2, mass_order=1)
    with pytest.raises(ValueError, match="reserved"):
        domain.with_boundaries(all=lambda x: x[0] == 0)
    with pytest.raises(ValueError, match="degenerate"):
        Domain.from_arrays([[0, 0]], [[0], [1]])
