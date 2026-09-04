import math

import numpy as np
import pytest

from ngfield.geometry import SimplicialDomain, reference_quadrature


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 6])
def test_unit_simplex_volume_and_boundary(dimension):
    vertices = np.vstack((np.zeros(dimension), np.eye(dimension)))
    domain = SimplicialDomain(vertices, [np.arange(dimension + 1)])
    volume = domain.quadrature(order=4)
    assert volume.weights.sum() == pytest.approx(1 / math.factorial(dimension))
    boundary = domain.quadrature(order=4, boundary="all")
    expected = (
        2.0
        if dimension == 1
        else (dimension + math.sqrt(dimension)) / math.factorial(dimension - 1)
    )
    assert boundary.weights.sum() == pytest.approx(expected)
    np.testing.assert_allclose(np.linalg.norm(boundary.normals, axis=1), 1)


def test_reference_rule_integrates_monomials():
    dimension, degree = 5, 4
    barycentric, weights = reference_quadrature(dimension, degree)
    exponents = np.array([2, 0, 1, 0, 0, 1])
    actual = np.sum(weights * np.prod(barycentric**exponents, axis=1))
    expected = np.prod([math.factorial(int(a)) for a in exponents]) / math.factorial(
        dimension + int(exponents.sum())
    )
    assert actual == pytest.approx(expected, abs=2e-16)


def test_embedded_simplex_uses_induced_measure_and_conormals():
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    domain = SimplicialDomain(vertices, [[0, 1, 2]])
    assert domain.dimension == 2
    assert domain.ambient_dimension == 3
    assert domain.quadrature().weights.sum() == pytest.approx(3.0)
    q = domain.quadrature(boundary="all")
    np.testing.assert_allclose(q.normals[:, 2], 0)


def test_boundary_arrays_and_predicates_are_equivalent():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    by_face = SimplicialDomain(vertices, [[0, 1, 2]], {"bottom": [[0, 1]]})
    by_predicate = SimplicialDomain(
        vertices, [[0, 1, 2]], {"bottom": lambda x: np.isclose(x[:, 1], 0)}
    )
    np.testing.assert_array_equal(by_face.boundaries["bottom"], by_predicate.boundaries["bottom"])


@pytest.mark.parametrize(
    "vertices,cells",
    [
        ([[0.0, 0.0], [1.0, 0.0]], [[0, 1, 1]]),
        ([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[0, 1, 2]]),
        ([[0.0], [1.0]], [[0, 2]]),
    ],
)
def test_invalid_simplex_inputs(vertices, cells):
    with pytest.raises(ValueError):
        SimplicialDomain(vertices, cells)


def test_nonmanifold_face_is_rejected():
    vertices = np.array([[0, 0], [1, 0], [0, 1], [0, -1], [0, 2]], dtype=float)
    with pytest.raises(ValueError, match="more than two"):
        SimplicialDomain(vertices, [[0, 1, 2], [0, 1, 3], [0, 1, 4]])


def test_custom_quadrature_validation_and_budget():
    domain = SimplicialDomain([[0.0], [1.0]], [[0, 1]])
    with pytest.raises(ValueError, match="max_points"):
        reference_quadrature(10, 8, max_points=10)
    with pytest.raises(ValueError, match="Invalid simplex quadrature"):
        domain.quadrature(rule=lambda d, p: (np.array([[0.2, 0.2]]), np.ones(1)))
