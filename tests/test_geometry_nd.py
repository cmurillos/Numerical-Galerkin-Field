import math

import numpy as np
import pytest

from ngfield.geometry import SimplicialDomain, reference_quadrature


def triangulated_torus(major_radius=2.0, minor_radius=0.5, n_major=16, n_minor=12):
    vertices = []
    for i in range(n_major):
        u = 2 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2 * math.pi * j / n_minor
            radius = major_radius + minor_radius * math.cos(v)
            vertices.append(
                [radius * math.cos(u), radius * math.sin(u), minor_radius * math.sin(v)]
            )
    cells = []

    def index(i, j):
        return (i % n_major) * n_minor + (j % n_minor)

    for i in range(n_major):
        for j in range(n_minor):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i, j + 1), index(i + 1, j + 1)
            cells.extend(([a, b, d], [a, d, c]))
    return np.asarray(vertices), np.asarray(cells)


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


def test_closed_torus_is_a_boundaryless_two_complex_in_r3():
    vertices, cells = triangulated_torus()
    domain = SimplicialDomain(vertices, cells)
    assert domain.intrinsic_dimension == 2
    assert domain.ambient_dimension == 3
    assert domain.exterior_faces.shape == (0, 2)
    assert domain.quadrature(boundary="all").weights.size == 0
    np.testing.assert_allclose(
        np.trace(domain.tangent_projectors, axis1=1, axis2=2),
        2.0,
        atol=1e-12,
    )
    exact_area = 4 * math.pi**2 * 2.0 * 0.5
    assert domain.quadrature(order=2).weights.sum() == pytest.approx(exact_area, rel=0.06)


def test_boundary_arrays_and_predicates_are_equivalent():
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    by_face = SimplicialDomain(vertices, [[0, 1, 2]], {"bottom": [[0, 1]]})
    by_predicate = SimplicialDomain(
        vertices, [[0, 1, 2]], {"bottom": lambda x: np.isclose(x[:, 1], 0)}
    )
    np.testing.assert_array_equal(by_face.boundaries["bottom"], by_predicate.boundaries["bottom"])


def test_regions_accept_indices_masks_cells_and_predicates():
    vertices = np.array([[0.0], [0.5], [1.0]])
    cells = np.array([[0, 1], [1, 2]])
    domain = SimplicialDomain(
        vertices,
        cells,
        regions={
            "left": [0],
            "right": lambda x: x[:, 0] > 0.5,
            "masked": np.array([True, False]),
            "by_cell": np.array([[1, 0]]),
        },
    )
    for name in ("left", "masked", "by_cell"):
        np.testing.assert_array_equal(domain.regions[name], [0])
    np.testing.assert_array_equal(domain.regions["right"], [1])
    assert domain.quadrature(region="left").weights.sum() == pytest.approx(0.5)
    assert domain.quadrature(region="right").weights.sum() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Unknown region"):
        domain.quadrature(region="missing")
    with pytest.raises(ValueError, match="either a boundary or a volume region"):
        domain.quadrature(boundary="all", region="left")


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
