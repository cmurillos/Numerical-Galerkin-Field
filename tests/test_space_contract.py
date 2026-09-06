from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from ngfield import ComponentBasis, GalerkinProblem, SimplicialDomain, Space, grad, inner


def labelled_triangle():
    return SimplicialDomain(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        [[0, 1, 2]],
        regions={"heated": [0]},
        boundaries={"fixed": [[0, 1]]},
    )


def test_state_components_are_independent_of_geometry_and_labels_do_not_restrict():
    geometry = labelled_triangle()
    V = Space(geometry=geometry, components=1, regularity=1)
    assert V.geometry is geometry
    assert (V.geometry.intrinsic_dimension, V.geometry.ambient_dimension) == (2, 3)
    assert V.value_shape == (1,)
    assert V.restrictions == ()
    assert V.geometry.quadrature(region="heated").weights.sum() == pytest.approx(3.0)
    assert V.geometry.quadrature(boundary="fixed").weights.sum() == pytest.approx(2.0)

    problem = GalerkinProblem(
        geometry=V.geometry, weak=lambda u, v, dx, ds: -inner(grad(u), grad(v)) * dx
    )
    basis = problem.basis("laplacian", size=2)
    assert basis.eigenvalues[0] < 1e-12  # The label "fixed" has not removed constants.


@pytest.mark.parametrize("components", [1, 2])
def test_space_metadata_can_feed_the_existing_vector_field_path(components):
    geometry = SimplicialDomain([[0.0], [0.5], [1.0]], [[0, 1], [1, 2]])
    V = Space(geometry=geometry, components=components, regularity=1)

    def weak(u, v, dx, ds):
        return sum((r + 1) * u[r] * v[r] * dx for r in range(V.components))

    problem = GalerkinProblem(geometry=V.geometry, weak=weak)
    scalar = problem.basis("laplacian", size=2)
    basis = ComponentBasis(scalar, components=V.components)
    assert basis.value_shape == V.value_shape
    field = problem.field(basis=basis)
    z = torch.linspace(-1.0, 1.0, 3 * basis.dimension, dtype=field.dtype).reshape(3, -1)
    rates = torch.arange(1, components + 1, dtype=field.dtype).repeat_interleave(2)
    torch.testing.assert_close(field(z), z * rates, atol=1e-12, rtol=1e-12)


def test_description_is_frozen_and_does_not_retain_a_mutable_restrictions_list():
    restrictions = []
    V = Space(geometry=labelled_triangle(), components=2, restrictions=restrictions)
    restrictions.append("unimplemented restriction")
    assert V.restrictions == ()
    with pytest.raises(FrozenInstanceError):
        V.components = 3
    with pytest.raises(FrozenInstanceError):
        V.geometry = labelled_triangle()


@pytest.mark.parametrize("restrictions", [["fixed"], (object(),)])
def test_unimplemented_restrictions_are_never_silently_accepted(restrictions):
    with pytest.raises(NotImplementedError, match="Restriction construction"):
        Space(geometry=labelled_triangle(), components=1, restrictions=restrictions)


@pytest.mark.parametrize("restrictions", ["fixed", None, {}])
def test_restrictions_require_an_explicit_sequence(restrictions):
    with pytest.raises(TypeError, match="list or tuple"):
        Space(geometry=labelled_triangle(), components=1, restrictions=restrictions)


@pytest.mark.parametrize(
    "options", [{"components": 0}, {"components": True}, {"regularity": -1}, {"regularity": 0.5}]
)
def test_invalid_component_counts_and_sobolev_orders_are_rejected(options):
    with pytest.raises(ValueError):
        Space(geometry=labelled_triangle(), **({"components": 1} | options))


def test_space_requires_a_validated_geometry_object():
    with pytest.raises(TypeError, match="SimplicialDomain"):
        Space(geometry={"vertices": [[0.0], [1.0]], "simplices": [[0, 1]]}, components=1)


@pytest.mark.parametrize("regularity", [0, 1, 2])
def test_regularity_is_a_requirement_not_a_basis_construction(regularity):
    V = Space(geometry=labelled_triangle(), components=np.int64(2), regularity=regularity)
    assert V.components == 2
    assert V.regularity == regularity
    assert V.value_shape == (2,)
    assert not hasattr(V, "dimension")  # No finite reduced dimension has been chosen.
