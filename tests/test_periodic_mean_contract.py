from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from test_basis_contract import torus_mesh
from test_space_basis_contract import check_gram, heat_field, interval, table

from ngfield import (
    ComponentBasis,
    FiniteElementBasis,
    GalerkinProblem,
    MeanZero,
    Periodic,
    SimplicialDomain,
    Space,
    TransformedBasis,
    ZeroTrace,
    grad,
    inner,
)


def periodic_interval(cells):
    return Periodic(component=0, boundaries=("left", "right"), vertex_pairs=[(0, cells)])


def means(basis, region=None):
    q = basis.geometry.quadrature(basis.validation_order + 2, region=region)
    values = basis.evaluate(
        torch.tensor(q.points.copy(), dtype=torch.float64),
        cells=torch.tensor(q.cells.copy()),
        barycentric=torch.tensor(q.barycentric.copy(), dtype=torch.float64),
    ).numpy()
    return np.einsum("qic,q->ic", values, q.weights) / q.weights.sum()


def square(embedded=False):
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    if embedded:
        vertices = np.column_stack((vertices, vertices[:, 0] + 2 * vertices[:, 1]))
    return SimplicialDomain(
        vertices,
        [[0, 1, 2], [0, 2, 3]],
        boundaries={
            "left": [[0, 3]],
            "right": [[1, 2]],
            "bottom": [[0, 1]],
            "top": [[3, 2]],
        },
        regions={"lower": [0], "upper": [1]},
    )


def periodic_x(component=0):
    return Periodic(
        component=component, boundaries=("left", "right"), vertex_pairs=[(0, 1), (3, 2)]
    )


def periodic_y(component=0):
    return Periodic(
        component=component, boundaries=("bottom", "top"), vertex_pairs=[(0, 3), (1, 2)]
    )


@pytest.mark.parametrize("cells,zero_mean", [(16, False), (16, True), (260, True)])
def test_periodic_spectrum_with_optional_mean_kernel(cells, zero_mean):
    geometry = interval(cells)
    restrictions = [periodic_interval(cells)] + ([MeanZero(component=0)] if zero_mean else [])
    basis = Space(geometry=geometry, components=1, restrictions=restrictions).basis(size=5)
    angles = 2 * np.pi * np.arange(cells) / cells
    spectrum = np.sort(6 * cells**2 * (1 - np.cos(angles)) / (2 + np.cos(angles)))
    expected = spectrum[1:6] if zero_mean else spectrum[:5]
    np.testing.assert_allclose(basis.eigenvalues, expected, rtol=1e-10, atol=2e-9)
    assert basis.admissible_dofs == (cells - int(zero_mean),)
    np.testing.assert_allclose(table(basis, "left")[0], table(basis, "right")[0], atol=1e-12)
    if zero_mean:
        np.testing.assert_allclose(means(basis), 0, atol=2e-12)
    check_gram(basis)
    G = heat_field(geometry, basis)
    z = torch.linspace(-1, 1, 5, dtype=G.dtype)
    torch.testing.assert_close(G(z), -z * torch.tensor(expected), rtol=1e-10, atol=2e-9)
    np.testing.assert_allclose(
        torch.func.jacrev(G)(z).numpy(), -np.diag(expected), rtol=1e-10, atol=2e-9
    )


@pytest.mark.parametrize("family", ["laplacian", "finite-element", "custom"])
def test_periodic_corner_cycles_and_high_order_traces_on_embedded_faces(family):
    geometry = square(embedded=True)
    V = Space(
        geometry=geometry,
        components=2,
        restrictions=[
            periodic_x(),
            periodic_y(),
            MeanZero(component=0),
            ZeroTrace(component=1, boundary="left"),
        ],
    )
    options = {"degree": 3}
    if family == "laplacian":
        options["component_sizes"] = (4, 3)
    elif family == "custom":
        options = {"source": ComponentBasis(FiniteElementBasis(geometry, degree=3), components=2)}
    basis = V.basis(family, **options)
    assert basis.dimension == (7 if family == "laplacian" else 20)
    check_gram(basis)
    np.testing.assert_allclose(means(basis)[:, 0], 0, atol=1e-12)
    G = heat_field(geometry, basis)
    y = np.linspace(0, 1, 13)
    left = np.column_stack((np.zeros_like(y), y, 2 * y))
    right = np.column_stack((np.ones_like(y), y, 1 + 2 * y))
    bottom = np.column_stack((y, np.zeros_like(y), y))
    top = np.column_stack((y, np.ones_like(y), y + 2))
    z = torch.eye(basis.dimension, dtype=G.dtype)
    uleft, uright, ubottom, utop = [
        G.reconstruct(z, torch.tensor(p)) for p in (left, right, bottom, top)
    ]
    torch.testing.assert_close(uleft[..., 0], uright[..., 0], atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(ubottom[..., 0], utop[..., 0], atol=2e-12, rtol=2e-12)
    assert uleft[..., 1].abs().max().item() < 2e-12


def test_periodic_identification_propagates_zero_trace_to_the_entire_class():
    geometry = square()
    restrictions = [periodic_x(), periodic_y(), ZeroTrace(component=0, boundary="left")]
    V = Space(geometry=geometry, components=1, restrictions=restrictions)
    basis = V.basis("finite-element", degree=2)
    assert basis.dimension == 2
    assert table(basis, "left")[0].abs().max().item() == 0
    assert table(basis, "right")[0].abs().max().item() == 0
    check_gram(basis)


def test_multiple_region_means_combine_with_trace_and_redundant_constraints():
    geometry = square()
    restrictions = [
        MeanZero(component=0, region="lower"),
        MeanZero(component=0, region="upper"),
        MeanZero(component=0),
        ZeroTrace(component=0, boundary="left"),
    ]
    V = Space(geometry=geometry, components=1, restrictions=restrictions)
    basis = V.basis("finite-element", degree=3)
    assert basis.dimension == 10 and basis.restriction_rank == 6
    for region in (None, "lower", "upper"):
        np.testing.assert_allclose(means(basis, region), 0, atol=1e-12)
    check_gram(basis)
    reverse = Space(geometry=geometry, components=1, restrictions=list(reversed(restrictions)))
    custom = reverse.basis(
        "custom", source=ComponentBasis(FiniteElementBasis(geometry, degree=3), components=1)
    )
    a, weights = table(basis, order=10)
    b, _ = table(custom, order=10)
    overlap = np.einsum("qic,qjc,q->ij", a.numpy(), b.numpy(), weights)
    np.testing.assert_allclose(overlap @ overlap.T, np.eye(10), atol=2e-12)


@pytest.mark.parametrize("length", [1e-6, 1.0, 1e6])
def test_means_are_not_silently_lost_when_geometry_or_candidate_units_change(length):
    geometry = SimplicialDomain(
        np.linspace(0, length, 5)[:, None], [[0, 1], [1, 2], [2, 3], [3, 4]]
    )
    V = Space(geometry=geometry, components=1, restrictions=[MeanZero(component=0)])
    raw = ComponentBasis(FiniteElementBasis(geometry), components=1)
    source = TransformedBasis(raw, np.diag([1e-12, 1e8, 1e-3, 1e12, 1.0]))
    basis = V.basis("custom", source=source)
    assert basis.dimension == 4
    np.testing.assert_allclose(means(basis) * length**0.5, 0, atol=1e-12)
    again = V.basis("custom", source=basis)
    assert again.dimension == 4 and again.restriction_rank == 0
    check_gram(again)


def test_global_mean_does_not_remove_all_constants_on_disconnected_domains():
    geometry = SimplicialDomain([[0.0], [1.0], [2.0], [3.0]], [[0, 1], [2, 3]])
    V = Space(geometry=geometry, components=1, restrictions=[MeanZero(component=0)])
    basis = V.basis(size=3)
    np.testing.assert_allclose(basis.eigenvalues, [0, 12, 12], atol=1e-12)
    np.testing.assert_allclose(means(basis), 0, atol=1e-12)
    assert basis.dimension == 3


def test_closed_torus_mean_zero_uses_surface_measure_and_removes_one_constant():
    geometry = SimplicialDomain(*torus_mesh())
    unrestricted = Space(geometry=geometry, components=1).basis(size=5)
    V = Space(geometry=geometry, components=1, restrictions=[MeanZero(component=0)])
    basis = V.basis(size=4)
    np.testing.assert_allclose(
        basis.eigenvalues, unrestricted.eigenvalues[1:], atol=1e-11, rtol=1e-11
    )
    np.testing.assert_allclose(means(basis), 0, atol=1e-12)
    check_gram(basis)


def test_coupled_custom_modes_satisfy_component_specific_combinations():
    geometry = interval(4)
    V = Space(
        geometry=geometry,
        components=2,
        restrictions=[
            periodic_interval(4),
            MeanZero(component=0),
            MeanZero(component=1),
            ZeroTrace(component=1, boundary="left"),
        ],
    )
    raw = ComponentBasis(FiniteElementBasis(geometry, degree=2), components=2)
    source = TransformedBasis(
        raw, np.eye(18) + 0.03 * np.random.default_rng(8).normal(size=(18, 18))
    )
    basis = V.basis("custom", source=source)
    assert basis.dimension == 14 and basis.restriction_rank == 4
    left, right = table(basis, "left")[0], table(basis, "right")[0]
    torch.testing.assert_close(left[..., 0], right[..., 0], atol=1e-12, rtol=1e-12)
    assert left[..., 1].abs().max().item() < 1e-12
    np.testing.assert_allclose(means(basis), 0, atol=1e-12)
    check_gram(basis)


def test_periodic_and_mean_declarations_copy_and_freeze_their_inputs():
    pairs = np.array([[0, 4]])
    boundaries = ["left", "right"]
    p = Periodic(component=0, boundaries=boundaries, vertex_pairs=pairs)
    pairs[0, 0] = 2
    boundaries[0] = "changed"
    assert p.vertex_pairs == ((0, 4),) and p.boundaries == ("left", "right")
    with pytest.raises(FrozenInstanceError):
        p.component = 1
    V = Space(
        geometry=interval(4),
        components=1,
        restrictions=[p, p, MeanZero(component=0), MeanZero(component=0, region="all")],
    )
    assert len(V.restrictions) == 2
    assert (
        Space(
            geometry=interval(4), components=1, regularity=0, restrictions=[MeanZero(component=0)]
        )
        .basis(size=1)
        .dimension
        == 1
    )


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [[0]],
        [[0, 1, 2]],
        [[0.0, 4.0]],
        [[True, False]],
        [[-1, 4]],
        [[0, 4], [0, 3]],
        [[0, 4], [1, 4]],
    ],
)
def test_invalid_vertex_pairs_are_rejected(pairs):
    with pytest.raises((TypeError, ValueError)):
        Periodic(component=0, boundaries=("left", "right"), vertex_pairs=pairs)


def test_periodicity_validates_whole_boundary_connectivity_not_just_vertices():
    geometry = SimplicialDomain(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
            [0, 0, 3],
        ],
        [[0, 1, 2, 4], [0, 2, 3, 4], [5, 6, 8, 9], [6, 7, 8, 9]],
        boundaries={"first": [[0, 1, 2], [0, 2, 3]], "second": [[5, 6, 8], [6, 7, 8]]},
    )
    restriction = Periodic(
        component=0, boundaries=("first", "second"), vertex_pairs=[(i, i + 5) for i in range(4)]
    )
    with pytest.raises(ValueError, match="face connectivity"):
        Space(geometry=geometry, components=1, restrictions=[restriction])


def test_unknown_incomplete_and_l2_periodic_boundaries_fail_explicitly():
    geometry = square()
    with pytest.raises(ValueError, match="cover exactly"):
        Space(
            geometry=geometry,
            components=1,
            restrictions=[
                Periodic(component=0, boundaries=("left", "right"), vertex_pairs=[(0, 1)])
            ],
        )
    with pytest.raises(ValueError, match="Unknown boundary"):
        Space(
            geometry=geometry,
            components=1,
            restrictions=[
                Periodic(component=0, boundaries=("missing", "right"), vertex_pairs=[(0, 1)])
            ],
        )
    with pytest.raises(ValueError, match="regularity"):
        Space(geometry=geometry, components=1, regularity=0, restrictions=[periodic_x()])
    with pytest.raises(ValueError, match="Unknown region"):
        Space(geometry=geometry, components=1, restrictions=[MeanZero(component=0, region="left")])
    with pytest.raises(ValueError, match="outside"):
        Space(geometry=geometry, components=1, restrictions=[MeanZero(component=1)])


def test_zero_space_and_dense_mean_budget_are_explicit_errors():
    geometry = square()
    V = Space(
        geometry=geometry,
        components=1,
        restrictions=[periodic_x(), periodic_y(), MeanZero(component=0)],
    )
    with pytest.raises(ValueError, match="space is zero"):
        V.basis(size=1, degree=1)
    with pytest.raises(ValueError, match="zero function"):
        V.restrict(ComponentBasis(FiniteElementBasis(geometry), components=1))
    with pytest.raises(ValueError, match="max_matrix_entries"):
        Space(geometry=interval(40), components=1, restrictions=[MeanZero(component=0)]).basis(
            size=1, max_matrix_entries=500
        )


def test_fixed_nonzero_dirichlet_lift_preserves_autonomous_heat_and_reconstruction():
    geometry = interval(8)
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis(size=3)

    def weak(w, v, dx, ds):
        temperature = w[0] + 1 + dx.x[0]
        return -inner(grad(temperature), grad(v[0])) * dx

    G = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis)
    z0 = torch.tensor([0.2, -0.1, 0.3], dtype=G.dtype)

    def initial(x):
        return 1 + x + G.reconstruct(z0, x)

    projected = G.project(lambda x: initial(x) - (1 + x))
    torch.testing.assert_close(projected, z0, atol=1e-12, rtol=1e-12)
    times = torch.linspace(0, 0.02, 5, dtype=G.dtype)
    Z = G.solve(z0, times, tolerance=1e-10)
    expected = z0 * torch.exp(-times[:, None] * torch.tensor(basis.eigenvalues))
    torch.testing.assert_close(Z, expected, atol=1e-9, rtol=1e-8)
    points = torch.linspace(0, 1, 17, dtype=G.dtype)[:, None]
    temperature = 1 + points + G.reconstruct(Z, points)
    torch.testing.assert_close(temperature[:, 0, 0], torch.ones(5, dtype=G.dtype))
    torch.testing.assert_close(temperature[:, -1, 0], 2 * torch.ones(5, dtype=G.dtype))


def test_fixed_dirichlet_neumann_and_robin_with_spatially_varying_data():
    geometry = square()
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="left")]
    )
    basis = V.basis("finite-element", degree=2)

    def weak(w, v, dx, ds):
        x, y = dx.x[0], dx.x[1]
        temperature = w[0] + 1 + x + y**2
        return (
            -inner(grad(temperature), grad(v[0])) * dx
            - 2 * v[0] * dx
            + v[0] * ds("right")
            - (temperature - (4 + x)) * v[0] * ds("top")
        )

    G = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis)
    # ell is the exact equilibrium for f=-2, right outward flux=-1,
    # bottom flux=0, top Robin alpha=1 and ambient=4+x.
    z = torch.zeros(basis.dimension, dtype=G.dtype)
    torch.testing.assert_close(G(z), z, atol=1e-11, rtol=0)
    J = torch.func.jacrev(G)(z)
    torch.testing.assert_close(J, J.T, atol=1e-11, rtol=1e-11)
    assert torch.linalg.eigvalsh(J).max().item() < 0
    with pytest.raises(TypeError):
        ZeroTrace(component=0, boundary="left", value=1)
