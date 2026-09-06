import numpy as np
import pytest
import torch
from skfem import MeshLine
from test_basis_contract import torus_mesh
from test_periodic_mean_contract import square
from test_space_basis_contract import interval

from ngfield import (
    CallableBasis,
    Coefficient,
    ComponentBasis,
    Domain,
    FEMSpace,
    FiniteElementBasis,
    GalerkinBasis,
    GalerkinField,
    GalerkinProblem,
    GeneralGalerkinField,
    LegacyGalerkinField,
    MeanZero,
    Periodic,
    Problem,
    SimplicialDomain,
    Space,
    TransformedBasis,
    ZeroTrace,
    grad,
    inner,
    tanh,
)


def heat(u, v, dx, ds):
    return -inner(grad(u[0]), grad(v[0])) * dx


@pytest.mark.parametrize("route", ["direct", "alias", "positional", "keyword", "method"])
def test_construction_routes_keep_the_same_constrained_coordinates(route):
    geometry = interval(2)
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis(size=1)
    before = basis.coefficients.copy()
    problem = GalerkinProblem(geometry=geometry, weak=heat)
    builders = {
        "direct": lambda: GalerkinField(basis=basis, weak=heat),
        "alias": lambda: GeneralGalerkinField(basis=basis, weak=heat),
        "positional": lambda: GalerkinField(problem, basis),
        "keyword": lambda: GalerkinField(problem=problem, basis=basis),
        "method": lambda: problem.field(basis=basis),
    }
    G = builders[route]()
    z = torch.tensor([[0.2], [-0.4]], dtype=G.dtype)
    # Only the midpoint P1 hat survives the full Dirichlet constraint; lambda=12.
    torch.testing.assert_close(G(z), -12 * z, atol=1e-12, rtol=1e-12)
    assert G.basis is basis and G.space is V and G.geometry is geometry
    assert G.value_shape == (1,) and G.dimension == 1
    assert G.problem.weak is heat
    np.testing.assert_array_equal(basis.coefficients, before)
    points = torch.tensor([[0.0], [0.5], [1.0]], dtype=G.dtype)
    values = G.reconstruct(z, points)
    torch.testing.assert_close(values[:, [0, 2]], torch.zeros(2, 2, 1, dtype=G.dtype))
    torch.testing.assert_close(values[:, 1, 0], np.sqrt(3) * z[:, 0])


def nonlinear_constants():
    geometry = interval(2)
    basis = Space(geometry=geometry, components=2).basis(component_sizes=(1, 1))

    def weak(u, v, dx, ds):
        return ((u[0] - u[0] ** 3 + 2 * u[1]) * v[0] + (u[0] - 3 * u[1]) * v[1]) * dx

    return GalerkinField(basis=basis, weak=weak)


def exact_reaction(z):
    a, b = z[..., 0], z[..., 1]
    return torch.stack((a - a**3 + 2 * b, a - 3 * b), dim=-1)


@pytest.mark.parametrize("batch", [(), (4,), (2, 3), (2, 0, 3)])
def test_direct_field_batches_and_coupling_match_analytical_reaction(batch):
    G = nonlinear_constants()
    z = torch.linspace(-0.3, 0.4, max(1, int(np.prod(batch))) * 2, dtype=G.dtype)
    z = z[: int(np.prod(batch)) * 2].reshape(*batch, 2)
    torch.testing.assert_close(G(z), exact_reaction(z), atol=2e-12, rtol=2e-12)
    assert G.reconstruct(z).shape == (*batch, G.quadrature_size, 2)


def test_direct_field_derivatives_match_analytic_jacobian_and_hessian():
    G = nonlinear_constants()
    z = torch.tensor([0.2, -0.1], dtype=G.dtype)
    J = torch.tensor([[1 - 3 * 0.2**2, 2], [1, -3]], dtype=G.dtype)
    direction = torch.tensor([0.7, -0.3], dtype=G.dtype)
    torch.testing.assert_close(torch.func.jacrev(G)(z), J, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.func.jvp(G, (z,), (direction,))[1], J @ direction)
    _, pullback = torch.func.vjp(G, z)
    torch.testing.assert_close(pullback(direction)[0], J.T @ direction)
    hessian = torch.zeros(2, 2, 2, dtype=G.dtype)
    hessian[0, 0, 0] = -6 * z[0]
    torch.testing.assert_close(torch.func.jacrev(torch.func.jacrev(G))(z), hessian)


def test_explicit_basis_rotation_preserves_space_and_transforms_coordinates_only_as_requested():
    geometry = interval(5)
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis(size=3)
    Q, _ = np.linalg.qr(np.random.default_rng(29).normal(size=(3, 3)))
    rotated = TransformedBasis(basis, Q)

    def weak(u, v, dx, ds):
        return -inner(grad(u[0]), grad(v[0])) * dx - u[0] ** 3 * v[0] * dx

    original = GalerkinField(basis=basis, weak=weak)
    G = GalerkinField(basis=rotated, weak=weak)
    assert G.space is V and G.basis is rotated and rotated.regularity_verified
    transform = torch.tensor(Q, dtype=G.dtype)
    Z = torch.tensor([[0.1, -0.2, 0.3], [0.3, 0.2, -0.1]], dtype=G.dtype)
    torch.testing.assert_close(G(Z), original(Z @ transform.T) @ transform, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(G.reconstruct(Z), original.reconstruct(Z @ transform.T))
    before = rotated.transform.clone()
    G.to(dtype=torch.float32)
    torch.testing.assert_close(
        G(Z.float()), (original(Z @ transform.T) @ transform).float(), atol=5e-5, rtol=5e-5
    )
    torch.testing.assert_close(rotated.transform, before)
    bad = TransformedBasis(basis, 2 * np.eye(3))
    with pytest.raises(ValueError, match="L2-orthonormal"):
        GalerkinField(basis=bad, weak=weak)
    torch.testing.assert_close(bad.transform, 2 * torch.eye(3, dtype=torch.float64))


def test_direct_periodic_mean_zero_field_projects_and_integrates_in_the_same_space():
    geometry = interval(8)
    V = Space(
        geometry=geometry,
        components=1,
        restrictions=[
            Periodic(component=0, boundaries=("left", "right"), vertex_pairs=[(0, 8)]),
            MeanZero(component=0),
        ],
    )
    basis = V.basis(size=3, degree=2)
    G = GalerkinField(basis=basis, weak=heat)
    z0 = torch.tensor([0.2, -0.1, 0.1], dtype=G.dtype)
    projected = G.project(lambda x: G.reconstruct(z0, x))
    torch.testing.assert_close(projected, z0, atol=1e-12, rtol=1e-12)
    times = torch.linspace(0, 0.01, 4, dtype=G.dtype)
    Z = G.solve(projected, times, tolerance=1e-10)
    expected = z0 * torch.exp(-times[:, None] * torch.tensor(basis.eigenvalues))
    torch.testing.assert_close(Z, expected, atol=1e-9, rtol=1e-8)
    ends = G.reconstruct(Z, torch.tensor([[0.0], [1.0]], dtype=G.dtype))
    torch.testing.assert_close(ends[:, 0], ends[:, 1], atol=1e-12, rtol=1e-12)


def test_direct_torus_uses_the_surface_geometry_and_spectrum():
    geometry = SimplicialDomain(*torus_mesh())
    V = Space(geometry=geometry, components=1, restrictions=[MeanZero(component=0)])
    basis = V.basis(size=4)
    G = GalerkinField(basis=basis, weak=heat)
    z = torch.arange(4, dtype=G.dtype) / 4
    assert G.geometry.dimension == 2 and G.geometry.ambient_dimension == 3
    torch.testing.assert_close(G(z), -z * torch.tensor(basis.eigenvalues), atol=1e-11, rtol=1e-11)


def test_direct_static_lift_and_region_and_boundary_measures_keep_physical_equilibrium():
    geometry = square()
    basis = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="left")]
    ).basis("finite-element", degree=2)

    def weak(w, v, dx, ds):
        x, y = dx.x[0], dx.x[1]
        temperature = w[0] + 1 + x + y**2
        return (
            -inner(grad(temperature), grad(v[0])) * dx
            - 2 * v[0] * dx("lower")
            - 2 * v[0] * dx("upper")
            + v[0] * ds("right")
            - (temperature - (4 + x)) * v[0] * ds("top")
        )

    G = GalerkinField(basis=basis, weak=weak)
    z = torch.zeros(G.dimension, dtype=G.dtype)
    torch.testing.assert_close(G(z), z, atol=1e-11, rtol=0)
    J = torch.func.jacrev(G)(z)
    torch.testing.assert_close(J, J.T, atol=1e-11, rtol=1e-11)
    assert torch.linalg.eigvalsh(J).max() < 0


@pytest.mark.parametrize("quadrature", [None, 6, 1e-8])
def test_numerical_options_and_fixed_callbacks_are_shared_by_both_routes(quadrature):
    geometry = interval(2)
    basis = Space(geometry=geometry, components=1).basis(size=2)
    data = {"value": 2.0, "calls": 0}

    def coefficient(x):
        data["calls"] += 1
        return torch.full((len(x),), data["value"], dtype=x.dtype, device=x.device)

    fixed = Coefficient(coefficient)

    def weak(u, v, dx, ds):
        return (fixed * tanh(u[0])) * v[0] * dx

    G = GalerkinField(basis=basis, weak=weak, quadrature=quadrature, max_intermediate_entries=1)
    H = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis, quadrature=quadrature)
    z = torch.tensor([[0.1, -0.1], [0.2, 0.05]], dtype=G.dtype)
    expected = H(z)
    calls = data["calls"]
    data["value"] = 100
    torch.testing.assert_close(G(z), expected)
    torch.func.jacrev(G)(z[0])
    assert data["calls"] == calls
    assert G.quadrature_mode == H.quadrature_mode and G.quadrature_order == H.quadrature_order


def test_legacy_positional_and_keyword_calls_remain_legacy_fields():
    fem = FEMSpace(Domain(MeshLine(np.linspace(0, 1, 7))))
    problem = Problem(1, lambda x, u, du: (0, -du), (("all",),))
    basis = GalerkinBasis.build(fem, problem, 2)
    first = GalerkinField(basis, problem)
    second = GalerkinField(basis=basis, problem=problem)
    assert isinstance(first, LegacyGalerkinField) and isinstance(second, LegacyGalerkinField)
    z = torch.tensor([0.2, -0.3], dtype=torch.float64)
    torch.testing.assert_close(first(z), -z * torch.tensor(basis.eigenvalues[0].copy()))
    torch.testing.assert_close(second(z), first(z))


def test_existing_unbound_scalar_and_saved_bases_use_the_explicit_problem_route(tmp_path):
    geometry = interval(3)
    problem = GalerkinProblem(geometry=geometry, weak=lambda u, v, dx, ds: u * v * dx)
    scalar = problem.basis(size=2)
    G = GalerkinField(problem=problem, basis=scalar)
    assert G.space is None and G.value_shape == ()
    torch.testing.assert_close(G(torch.ones(2, dtype=G.dtype)), torch.ones(2, dtype=G.dtype))
    bound = Space(geometry=geometry, components=1).basis(size=2)
    path = tmp_path / "basis.npz"
    bound.save(path)
    restored = FiniteElementBasis.load(path)
    with pytest.raises(ValueError, match="associated with Space"):
        GalerkinField(basis=restored, weak=heat)
    explicit = GalerkinProblem(geometry=geometry, weak=heat).field(basis=restored)
    z = torch.tensor([0.2, 0.3], dtype=explicit.dtype)
    torch.testing.assert_close(explicit(z), GalerkinField(basis=bound, weak=heat)(z))


@pytest.mark.parametrize("builder", [GalerkinField, GeneralGalerkinField])
def test_invalid_or_ambiguous_construction_has_actionable_errors(builder):
    geometry = interval(3)
    basis = Space(geometry=geometry, components=1).basis(size=2)
    problem = GalerkinProblem(geometry=geometry, weak=heat)
    with pytest.raises(TypeError, match="either weak"):
        builder(problem=problem, basis=basis, weak=heat)
    with pytest.raises(TypeError, match="either weak"):
        builder(basis, weak=heat, basis=basis)
    with pytest.raises(TypeError, match="weak must be callable"):
        builder(basis=basis, weak=None)
    with pytest.raises(TypeError, match="Provide weak"):
        builder(basis=basis)
    with pytest.raises(TypeError, match="Provide basis"):
        builder(weak=heat)
    with pytest.raises(ValueError, match="associated with Space"):
        builder(basis=ComponentBasis(FiniteElementBasis(geometry), components=1), weak=heat)
    with pytest.raises(TypeError, match="geometry"):
        builder(basis=basis, weak=heat, geometry=geometry)


def test_space_shape_mesh_and_labels_are_checked_without_changing_metadata():
    geometry = interval(3)
    V = Space(geometry=geometry, components=1)
    basis = V.basis(size=2)
    same = interval(3)
    G = GalerkinField(problem=GalerkinProblem(geometry=same, weak=heat), basis=basis)
    assert G.space is V and G.geometry is same
    relabelled = SimplicialDomain(
        geometry.vertices, geometry.simplices, boundaries={"left": [[3]], "right": [[0]]}
    )
    with pytest.raises(ValueError, match="boundaries must match"):
        GalerkinProblem(geometry=relabelled, weak=heat).field(basis=basis)
    other_regions = SimplicialDomain(
        geometry.vertices,
        geometry.simplices,
        boundaries={"left": [[0]], "right": [[3]]},
        regions={"new": [0]},
    )
    with pytest.raises(ValueError, match="regions must match"):
        GalerkinProblem(geometry=other_regions, weak=heat).field(basis=basis)
    for wrong_space, message in [
        (Space(geometry=geometry, components=2), "components"),
        (Space(geometry=interval(4), components=1), "different mesh"),
        ("not a Space", "basis.space"),
    ]:
        basis.space = wrong_space
        with pytest.raises((TypeError, ValueError), match=message):
            GalerkinField(basis=basis, weak=heat)
    basis.space = V
    torch.testing.assert_close(G(torch.zeros(2, dtype=G.dtype)), torch.zeros(2, dtype=G.dtype))


def test_manual_restriction_and_normalization_retains_space_but_declared_callbacks_stay_declared():
    geometry = interval(4)
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    raw = V.restrict(ComponentBasis(FiniteElementBasis(geometry), components=1))
    normalized = GalerkinProblem(geometry=geometry, weak=heat).orthonormalize(raw)
    assert normalized.space is V and GalerkinField(basis=normalized, weak=heat).space is V
    source = ComponentBasis(
        CallableBasis(lambda x: torch.sin(torch.pi * x), dimension=1), components=1
    )
    declared = Space(geometry=geometry, components=1).basis("custom", source=source)
    G = GalerkinField(basis=declared, weak=heat)
    assert declared.regularity_verified is False
    torch.testing.assert_close(
        G(torch.ones(1, dtype=G.dtype)), -(torch.pi**2) * torch.ones(1, dtype=G.dtype)
    )


def test_direct_form_validation_and_quadrature_budget_still_apply():
    basis = Space(geometry=interval(2), components=1).basis(size=2)
    with pytest.raises(ValueError, match="linear"):
        GalerkinField(basis=basis, weak=lambda u, v, dx, ds: u[0] * v[0] ** 2 * dx)
    with pytest.raises((IndexError, ValueError)):
        GalerkinField(basis=basis, weak=lambda u, v, dx, ds: u[1] * v[0] * dx)
    with pytest.raises(ValueError, match="Unknown boundary"):
        GalerkinField(basis=basis, weak=lambda u, v, dx, ds: u[0] * v[0] * ds("missing"))
    with pytest.raises(ValueError, match="max_points"):
        GalerkinField(basis=basis, weak=heat, max_quadrature_points=1)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_direct_cuda_construction_and_derivatives():
    basis = Space(geometry=interval(2), components=1).basis(size=2)
    G = GalerkinField(basis=basis, weak=heat, device="cuda", dtype=torch.float64)
    z = torch.tensor([0.2, -0.3], device="cuda", dtype=G.dtype)
    eigenvalues = torch.tensor(basis.eigenvalues, device="cuda", dtype=G.dtype)
    torch.testing.assert_close(G(z), -eigenvalues * z)
    torch.testing.assert_close(torch.func.jacrev(G)(z), -torch.diag(eigenvalues))
