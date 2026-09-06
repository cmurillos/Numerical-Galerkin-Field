from dataclasses import FrozenInstanceError
from math import comb

import numpy as np
import pytest
import torch

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


def interval():
    return SimplicialDomain(
        [[0.0], [0.5], [1.0]],
        [[0, 1], [1, 2]],
        boundaries={"left": [[0]], "right": [[2]], "empty": np.empty((0, 1), dtype=int)},
    )


def boundary_values(basis, geometry, boundary, order=9):
    q = geometry.quadrature(order, boundary=boundary)
    return basis.evaluate(
        torch.tensor(q.points.copy(), dtype=torch.float64),
        cells=torch.tensor(q.cells.copy()),
        barycentric=torch.tensor(q.barycentric.copy(), dtype=torch.float64),
    )


@pytest.mark.parametrize("dimension,ambient,degree", [(1, 3, 3), (2, 3, 3), (3, 3, 3), (4, 5, 2)])
def test_trace_vanishes_on_entire_high_order_face(dimension, ambient, degree):
    vertices = np.vstack((np.zeros(ambient), np.eye(ambient)[:dimension]))
    geometry = SimplicialDomain(
        vertices,
        [np.arange(dimension + 1)],
        boundaries={"wall": [np.arange(1, dimension + 1)]},
    )
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="wall")]
    )
    raw = ComponentBasis(FiniteElementBasis(geometry, degree), components=1)
    admissible = V.restrict(raw)
    face_nodes = comb(degree + dimension - 1, dimension - 1)
    assert admissible.dimension == comb(degree + dimension, dimension) - face_nodes
    assert admissible.restriction_rank == face_nodes
    assert admissible.space is V
    values = boundary_values(admissible, geometry, "wall", order=2 * degree + 3)
    assert values.abs().max().item() < 2e-13


def test_duplicate_and_overlapping_restrictions_preserve_other_components():
    geometry = interval()
    restrictions = [
        ZeroTrace(component=0, boundary="all"),
        ZeroTrace(component=0, boundary="left"),
        ZeroTrace(component=0, boundary="all"),
    ]
    V = Space(geometry=geometry, components=2, restrictions=restrictions)
    restrictions.clear()
    assert len(V.restrictions) == 2
    raw = ComponentBasis(FiniteElementBasis(geometry, degree=2), components=2)
    result = V.restrict(raw)
    assert result.dimension == raw.dimension - 2
    traces = boundary_values(result, geometry, "all")
    assert traces[..., 0].abs().max().item() < 1e-13
    assert traces[..., 1].abs().max().item() > 0.5
    with pytest.raises(FrozenInstanceError):
        V.restrictions[0].component = 1


def test_product_basis_can_use_different_degrees_and_boundary_conditions():
    geometry = interval()
    V = Space(
        geometry=geometry,
        components=2,
        restrictions=[
            ZeroTrace(component=0, boundary="left"),
            ZeroTrace(component=1, boundary="right"),
        ],
    )
    raw = ProductBasis([FiniteElementBasis(geometry, degree=d) for d in (2, 3)])
    result = V.restrict(raw)
    assert result.dimension == raw.dimension - 2
    left = boundary_values(result, geometry, "left")
    right = boundary_values(result, geometry, "right")
    assert left[..., 0].abs().max().item() < 1e-13
    assert right[..., 1].abs().max().item() < 1e-13
    assert left[..., 1].abs().max().item() > 0.5
    assert right[..., 0].abs().max().item() > 0.5


@pytest.mark.parametrize("representation", ["transformed", "coefficients"])
def test_coupled_candidates_restrict_before_orthonormalization(representation):
    geometry = interval()
    V = Space(
        geometry=geometry, components=2, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    raw = ComponentBasis(FiniteElementBasis(geometry), components=2)
    transform = np.eye(6) + 0.1 * np.random.default_rng(12).normal(size=(6, 6))
    if representation == "transformed":
        candidate = TransformedBasis(raw, transform)
    else:
        coefficients = np.stack((transform[:3], transform[3:]), axis=-1)
        candidate = FiniteElementBasis(geometry, coefficients=coefficients)
    original = transform.copy()
    admissible = V.restrict(candidate)
    assert admissible.dimension == 4
    problem = GalerkinProblem(geometry=geometry, weak=lambda u, v, dx, ds: inner(u, v) * dx)
    operational = problem.orthonormalize(admissible)
    G = problem.field(basis=operational)
    z = torch.arange(12, dtype=G.dtype).reshape(3, 4) / 7
    torch.testing.assert_close(G(z), z, atol=1e-12, rtol=1e-12)
    assert boundary_values(operational, geometry, "all")[..., 0].abs().max().item() < 1e-12
    np.testing.assert_array_equal(transform, original)


def test_fixed_end_heat_field_has_the_expected_one_mode_dynamics():
    geometry = interval()
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    raw = ComponentBasis(FiniteElementBasis(geometry), components=1)
    candidate = V.restrict(raw)
    problem = GalerkinProblem(
        geometry=geometry,
        weak=lambda u, v, dx, ds: -inner(grad(u[0]), grad(v[0])) * dx,
    )
    basis = problem.orthonormalize(candidate)
    G = problem.field(basis=basis)
    z = torch.tensor([[0.3], [-0.7]], dtype=G.dtype, requires_grad=True)
    torch.testing.assert_close(G(z), -12 * z, atol=1e-12, rtol=1e-12)
    derivative = torch.autograd.grad(G(z).sum(), z)[0]
    torch.testing.assert_close(derivative, torch.full_like(z, -12), atol=1e-12, rtol=1e-12)


def test_rank_detection_respects_independently_scaled_candidate_modes():
    geometry = interval()
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    raw = ComponentBasis(FiniteElementBasis(geometry), components=1)
    scaled = TransformedBasis(raw, np.diag([1e-8, 1.0, 1e8]))
    result = V.restrict(scaled)
    assert result.dimension == 1
    assert result.restriction_rank == 2
    assert boundary_values(result, geometry, "all").abs().max().item() < 1e-13
    assert V.restrict(result).dimension == 1


def test_zero_span_and_unverified_candidates_are_rejected():
    geometry = SimplicialDomain([[0.0], [1.0]], [[0, 1]])
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    with pytest.raises(ValueError, match="only the zero function"):
        V.restrict(ComponentBasis(FiniteElementBasis(geometry), components=1))
    callback = CallableBasis(lambda x: x[:, None, :], dimension=1, value_shape=(1,))
    with pytest.raises(NotImplementedError, match="arbitrary callbacks"):
        V.restrict(callback)


def test_dependent_modes_do_not_create_spurious_admissible_coordinates():
    geometry = interval()
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="left")]
    )
    dependent = FiniteElementBasis(
        geometry, coefficients=np.tile(np.array([1.0, 2.0])[None, :, None], (3, 1, 1))
    )
    with pytest.raises(ValueError, match="linearly dependent"):
        V.restrict(dependent)


def test_a_closed_surface_keeps_its_space_and_rejects_empty_trace_targets():
    geometry = SimplicialDomain(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
    )
    V = Space(geometry=geometry, components=1)
    raw = ComponentBasis(FiniteElementBasis(geometry), components=1)
    assert V.restrict(raw) is raw
    with pytest.raises(ValueError, match="empty"):
        Space(
            geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
        )


def test_invalid_space_or_basis_cannot_silently_weaken_a_condition():
    geometry = interval()
    for restriction in (
        ZeroTrace(component=1, boundary="all"),
        ZeroTrace(component=0, boundary="missing"),
        ZeroTrace(component=0, boundary="empty"),
    ):
        with pytest.raises(ValueError):
            Space(geometry=geometry, components=1, restrictions=[restriction])
    with pytest.raises(ValueError, match="regularity"):
        Space(
            geometry=geometry,
            components=1,
            regularity=0,
            restrictions=[ZeroTrace(component=0, boundary="all")],
        )
    V = Space(
        geometry=geometry, components=1, restrictions=[ZeroTrace(component=0, boundary="all")]
    )
    with pytest.raises(ValueError, match="value_shape"):
        V.restrict(FiniteElementBasis(geometry))
    different = SimplicialDomain([[0.0], [2.0]], [[0, 1]])
    with pytest.raises(ValueError, match="different mesh"):
        V.restrict(ComponentBasis(FiniteElementBasis(different), components=1))
    raw = ComponentBasis(FiniteElementBasis(geometry), components=1)
    with pytest.raises(ValueError, match="max_matrix_entries"):
        V.restrict(raw, max_matrix_entries=4)
    with pytest.raises(NotImplementedError, match="higher Sobolev"):
        Space(geometry=geometry, components=1, regularity=2).restrict(raw)


@pytest.mark.parametrize("options", [{"component": -1}, {"component": True}, {"boundary": ""}])
def test_zero_trace_declaration_validates_its_arguments(options):
    with pytest.raises(ValueError):
        ZeroTrace(**({"component": 0, "boundary": "all"} | options))
