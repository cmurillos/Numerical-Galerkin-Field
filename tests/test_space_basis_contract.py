import numpy as np
import pytest
import torch
from test_basis_contract import torus_mesh

from ngfield import (
    CallableBasis,
    ComponentBasis,
    FiniteElementBasis,
    GalerkinProblem,
    ProductBasis,
    SimplicialDomain,
    Space,
    TransformedBasis,
    ZeroTrace,
    grad,
    inner,
)


def interval(cells=8):
    return SimplicialDomain(
        np.linspace(0, 1, cells + 1)[:, None],
        np.column_stack((np.arange(cells), np.arange(1, cells + 1))),
        boundaries={"left": [[0]], "right": [[cells]]},
    )


def table(basis, boundary=None, order=8):
    q = basis.geometry.quadrature(order, boundary=boundary)
    values = basis.evaluate(
        torch.tensor(q.points.copy(), dtype=torch.float64),
        cells=torch.tensor(q.cells.copy()),
        barycentric=torch.tensor(q.barycentric.copy(), dtype=torch.float64),
    )
    return values, q.weights


def check_gram(basis):
    values, weights = table(basis, order=basis.validation_order + 2)
    gram = np.einsum("qic,qjc,q->ij", values.numpy(), values.numpy(), weights)
    np.testing.assert_allclose(gram, np.eye(basis.dimension), atol=3e-11, rtol=3e-11)


def heat_field(geometry, basis):
    return GalerkinProblem(
        geometry=geometry, weak=lambda u, v, dx, ds: -inner(grad(u), grad(v)) * dx
    ).field(basis=basis)


@pytest.mark.parametrize("cells", [12, 300])
def test_dirichlet_modes_are_selected_in_full_constrained_space(cells):
    geometry = interval(cells)
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis("laplacian", size=4)
    angles = np.arange(1, 5) * np.pi / cells
    exact_fem = 6 * cells**2 * (1 - np.cos(angles)) / (2 + np.cos(angles))
    np.testing.assert_allclose(basis.eigenvalues, exact_fem, atol=1e-9, rtol=1e-10)
    assert basis.dimension == 4 and basis.value_shape == (1,)
    assert basis.space is V and basis.geometry is geometry
    assert basis.regularity_verified
    assert basis.restriction_rank == 2
    assert table(basis, "all")[0].abs().max().item() == 0
    check_gram(basis)
    G = heat_field(geometry, basis)
    z = torch.arange(8, dtype=G.dtype).reshape(2, 4) / 8
    torch.testing.assert_close(G(z), -z * torch.tensor(exact_fem), atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(
        torch.func.jacrev(G)(z[0]).numpy(), -np.diag(exact_fem), atol=1e-9, rtol=1e-10
    )


def test_total_size_selects_global_spectrum_with_distinct_component_constraints():
    geometry = interval()
    V = Space(
        geometry=geometry, components=2, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis(size=5)
    angles = np.arange(1, 3) * np.pi / 8
    positive = 6 * 8**2 * (1 - np.cos(angles)) / (2 + np.cos(angles))
    expected = np.r_[0, np.repeat(positive, 2)]
    np.testing.assert_allclose(basis.eigenvalues, expected, atol=1e-11, rtol=1e-11)
    assert basis.dimension == 5 and basis.value_shape == (2,)
    assert basis.component_sizes is None
    values = table(basis, "all")[0]
    assert values[..., 0].abs().max().item() == 0
    assert values[..., 1].abs().max().item() > 0.5
    check_gram(basis)


def test_explicit_component_allocation_preserves_grouping_and_total():
    geometry = interval()
    V = Space(
        geometry=geometry, components=2, restrictions=[ZeroTrace(component=0, boundary="left")]
    )
    basis = V.basis(size=5, component_sizes=(3, 2))
    first = Space(geometry=geometry, components=1, restrictions=V.restrictions).basis(size=3)
    second = Space(geometry=geometry, components=1).basis(size=2)
    np.testing.assert_allclose(
        basis.eigenvalues, np.r_[first.eigenvalues, second.eigenvalues], atol=1e-11
    )
    assert basis.component_sizes == (3, 2)
    values = table(basis)[0]
    assert values[:, :3, 1].abs().max().item() == 0
    assert values[:, 3:, 0].abs().max().item() == 0
    check_gram(basis)


@pytest.mark.parametrize("family", ["laplacian", "finite-element"])
def test_zero_dimensional_component_does_not_erase_other_components(family):
    geometry = interval(1)
    V = Space(
        geometry=geometry, components=2, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    basis = V.basis(family, size=2)
    assert basis.admissible_dofs == (0, 2)
    assert table(basis)[0][..., 0].abs().max().item() == 0
    check_gram(basis)


@pytest.mark.parametrize("family", ["laplacian", "finite-element"])
def test_high_degree_embedded_face_trace_and_l2(family):
    geometry = SimplicialDomain(
        [[0, 0, 0], [1, 0, 1], [0, 2, 0]], [[0, 1, 2]], boundaries={"wall": [[1, 2]]}
    )
    V = Space(
        geometry=geometry, components=2, restrictions=[ZeroTrace(component=0, boundary="wall")]
    )
    basis = V.basis(family, degree=3, **({"size": 5} if family == "laplacian" else {}))
    assert basis.admissible_dofs == (6, 10)
    assert basis.dimension == (5 if family == "laplacian" else 16)
    assert table(basis, "wall", order=11)[0][..., 0].abs().max().item() < 1e-12
    check_gram(basis)


def test_torus_operational_basis_keeps_constant_and_heat_dissipation():
    vertices, simplices = torus_mesh()
    geometry = SimplicialDomain(vertices, simplices)
    V = Space(geometry=geometry, components=1)
    basis = V.basis(size=7)
    assert len(geometry.exterior_faces) == 0
    values, weights = table(basis)
    assert np.ptp(values[:, 0, 0].numpy()) < 1e-12
    assert abs(values[0, 0, 0].item()) == pytest.approx(weights.sum() ** -0.5)
    assert basis.eigenvalues[0] < 1e-12
    check_gram(basis)
    G = heat_field(geometry, basis)
    z = torch.linspace(-1, 1, 7, dtype=G.dtype)
    torch.testing.assert_close(G(z), -z * torch.tensor(basis.eigenvalues), atol=1e-11, rtol=1e-11)
    assert torch.dot(z, G(z)).item() < 0


def test_custom_coupled_product_is_restricted_before_l2_normalization():
    geometry = interval(2)
    V = Space(
        geometry=geometry,
        components=2,
        restrictions=[
            ZeroTrace(component=0, boundary="left"),
            ZeroTrace(component=1, boundary="right"),
        ],
    )
    raw = ProductBasis([FiniteElementBasis(geometry, degree=d) for d in (1, 2)])
    transform = np.eye(8) + 0.1 * np.random.default_rng(22).normal(size=(8, 8))
    source = TransformedBasis(raw, transform)
    basis = V.basis("custom", source=source, size=6)
    assert not hasattr(source, "space")
    assert basis.restriction_rank == 2 and basis.regularity_verified
    assert table(basis, "left")[0][..., 0].abs().max().item() < 1e-12
    assert table(basis, "right")[0][..., 1].abs().max().item() < 1e-12
    check_gram(basis)
    with pytest.raises(ValueError, match="does not truncate"):
        V.basis("custom", source=source, size=5)


def test_custom_callable_regularities_are_declared_and_trace_cannot_be_sampled():
    geometry = interval()
    source = ComponentBasis(
        CallableBasis(lambda x: torch.sin(torch.pi * x), dimension=1), components=1
    )
    V = Space(geometry=geometry, components=1)
    basis = V.basis("custom", source=source)
    assert basis.regularity_verified is False
    check_gram(basis)
    fixed = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    with pytest.raises(NotImplementedError, match="entire face"):
        fixed.basis("custom", source=source)


@pytest.mark.parametrize("family", ["polynomial", "fourier"])
@pytest.mark.parametrize("counts", [None, (1, 3), (0, 4)])
def test_functional_families_respect_total_size_and_allocation(family, counts):
    V = Space(geometry=interval(), components=2)
    basis = V.basis(family, size=4, component_sizes=counts)
    assert basis.dimension == 4 and basis.value_shape == (2,)
    assert basis.component_sizes == (counts or (2, 2))
    assert basis.regularity_verified and basis.space is V
    check_gram(basis)


def test_polynomial_degree_and_nodal_mesh_determine_full_dimension():
    V = Space(geometry=interval(2), components=2)
    polynomial = V.basis("polynomial", degree=2)
    finite_element = V.basis("finite-element", degree=2)
    assert polynomial.dimension == 6
    assert finite_element.dimension == 10
    assert finite_element.component_sizes == (5, 5)
    check_gram(polynomial)
    check_gram(finite_element)
    with pytest.raises(ValueError, match="does not truncate"):
        V.basis("finite-element", degree=2, size=6)


@pytest.mark.parametrize("family", ["polynomial", "fourier"])
def test_unsupported_functional_trace_is_rejected_before_mode_selection(family):
    V = Space(
        geometry=interval(), components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    with pytest.raises(NotImplementedError, match="complete ZeroTrace"):
        V.basis(family, size=4)


@pytest.mark.parametrize(
    "options",
    [
        {"size": True},
        {"size": 0},
        {"component_sizes": (1,)},
        {"component_sizes": (0, 0)},
        {"component_sizes": (2, -1)},
        {"size": 4, "component_sizes": (1, 2)},
        {"size": 20},
        {"size": 5, "component_sizes": (0, 5)},
        {"size": 1, "quadrature_order": 1},
        {"size": 1, "validation_order": 1},
        {"size": 1, "max_matrix_entries": 1},
    ],
)
def test_invalid_size_allocation_and_preparation_controls(options):
    V = Space(geometry=interval(2), components=2)
    with pytest.raises((TypeError, ValueError)):
        V.basis(**options)


def test_custom_shape_mesh_higher_regularity_and_null_space_errors():
    geometry = interval(1)
    V = Space(geometry=geometry, components=1)
    with pytest.raises(ValueError, match="value_shape"):
        V.basis("custom", source=FiniteElementBasis(geometry))
    with pytest.raises(ValueError, match="different mesh"):
        V.basis("custom", source=ComponentBasis(FiniteElementBasis(interval(2)), components=1))
    with pytest.raises(NotImplementedError, match="regularity"):
        Space(geometry=geometry, components=1, regularity=2).basis(size=1)
    with pytest.raises(ValueError, match="space is zero"):
        Space(
            geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
        ).basis(size=1)


def test_named_measures_and_fixed_robin_data_remain_usable_with_the_new_basis():
    geometry = SimplicialDomain(
        [[0], [0.5], [1]],
        [[0, 1], [1, 2]],
        boundaries={"fixed": [[0]], "exchange": [[2]]},
        regions={"heated": [0]},
    )
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="fixed")]
    )
    basis = V.basis(size=2)

    def weak(u, v, dx, ds):
        return (
            -inner(grad(u[0]), grad(v[0])) * dx
            + v[0] * dx("heated")
            - 2 * (u[0] - 3) * v[0] * ds("exchange")
        )

    G = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis)
    z = torch.zeros(2, dtype=G.dtype)
    # Analytical nodal load: integral on [0,.5] gives .25 at midpoint;
    # Robin ambient temperature adds 6 at the right endpoint.
    nodal = basis.coefficients[..., 0]
    middle = np.argmin(abs(basis.dof_points[:, 0] - 0.5))
    right = np.argmax(basis.dof_points[:, 0])
    np.testing.assert_allclose(G(z).numpy(), 0.25 * nodal[middle] + 6 * nodal[right], atol=1e-12)
    expected = -np.diag(basis.eigenvalues) - 2 * np.outer(nodal[right], nodal[right])
    np.testing.assert_allclose(torch.func.jacrev(G)(z).numpy(), expected, atol=1e-12)
