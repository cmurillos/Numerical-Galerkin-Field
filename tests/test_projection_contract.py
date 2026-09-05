import pytest
import torch

from ngfield import Coefficient, ComponentBasis, GalerkinProblem, inner


def scalar_field(size=3):
    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = problem.basis("polynomial", size=size)
    return problem.field(basis=basis), basis


def test_project_is_the_inverse_of_synthesis_on_the_fixed_space():
    field, basis = scalar_field()
    expected = torch.tensor([0.2, -0.3, 0.4], dtype=field.dtype)

    def function(points):
        modes = basis.evaluate(points)
        return torch.einsum("n,qn->q", expected, modes)

    coordinates = field.project(function, quadrature=10)
    torch.testing.assert_close(coordinates, expected, atol=1e-12, rtol=1e-12)


def test_project_preserves_arbitrary_batch_axes():
    field, basis = scalar_field()
    expected = torch.randn(2, 3, basis.dimension, dtype=field.dtype)

    def functions(points):
        return torch.einsum("...n,qn->...q", expected, basis.evaluate(points))

    coordinates = field.project(functions, quadrature=10)
    assert coordinates.shape == expected.shape
    torch.testing.assert_close(coordinates, expected, atol=1e-12, rtol=1e-12)


def test_project_supports_tensor_valued_spaces():
    scalar_field_, scalar_basis = scalar_field(size=2)
    problem = GalerkinProblem(
        geometry=scalar_field_.problem.geometry,
        weak=lambda u, v, dx, ds: inner(u, v) * dx,
    )
    basis = ComponentBasis(scalar_basis, value_shape=(2, 2))
    field = problem.field(basis=basis)
    expected = torch.randn(4, basis.dimension, dtype=field.dtype)

    def functions(points):
        return torch.einsum("bn,qn...->bq...", expected, basis.evaluate(points))

    coordinates = field.project(functions, quadrature=8)
    assert coordinates.shape == expected.shape
    torch.testing.assert_close(coordinates, expected, atol=1e-12, rtol=1e-12)


def test_project_accepts_vertex_and_cell_coefficients():
    field, _ = scalar_field(size=2)
    vertex = field.project(Coefficient.vertex([0.0, 0.5, 1.0]))
    callable_ = field.project(lambda x: x[:, 0], quadrature=8)
    cell = field.project(Coefficient.cell([1.0, 2.0]))

    torch.testing.assert_close(vertex, callable_, atol=1e-12, rtol=1e-12)
    assert cell.shape == (field.dimension,)
    assert torch.isfinite(cell).all()


def test_project_accepts_a_callable_coefficient():
    field, _ = scalar_field(size=3)
    coefficient = Coefficient(lambda x: x[:, 0] ** 2)
    from_coefficient = field.project(coefficient)
    from_function = field.project(lambda x: x[:, 0] ** 2, quadrature=10)
    torch.testing.assert_close(from_coefficient, from_function, atol=1e-12, rtol=1e-12)


def test_callable_projection_adapts_and_matches_a_refined_fixed_rule():
    field, _ = scalar_field()
    metadata = (field.quadrature_order, field.quadrature_size, field.table_bytes)
    state = torch.tensor([0.1, -0.2, 0.3], dtype=field.dtype)
    reconstruction = field.reconstruct(state).clone()

    def source(x):
        return torch.exp(x[:, 0])

    automatic = field.project(source)
    reference = field.project(source, quadrature=20)
    low = field.project(source, quadrature=0)

    assert torch.linalg.vector_norm(low - reference) > 1e-3
    torch.testing.assert_close(automatic, reference, atol=2e-8, rtol=2e-8)
    assert (field.quadrature_order, field.quadrature_size, field.table_bytes) == metadata
    torch.testing.assert_close(field.reconstruct(state), reconstruction, atol=0, rtol=0)


def test_callable_projection_preserves_autograd():
    field, _ = scalar_field(size=2)
    amplitude = torch.tensor(2.0, dtype=field.dtype, requires_grad=True)
    coordinates = field.project(lambda x: amplitude * x[:, 0], quadrature=6)
    coordinates.square().sum().backward()

    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)
    assert amplitude.grad != 0


def test_fixed_projection_supports_empty_batches():
    field, _ = scalar_field(size=2)
    coordinates = field.project(
        lambda x: torch.empty(2, 0, len(x), dtype=x.dtype, device=x.device),
        quadrature=6,
    )
    assert coordinates.shape == (2, 0, field.dimension)


def test_projection_rejects_raw_values_and_incompatible_shapes():
    field, _ = scalar_field(size=2)
    with pytest.raises(TypeError, match="callable function or a Coefficient"):
        field.project(torch.ones(field.quadrature_size))
    with pytest.raises(ValueError, match="must return"):
        field.project(lambda x: torch.ones(len(x), 2), quadrature=6)
    with pytest.raises(ValueError, match="real-valued"):
        field.project(lambda x: torch.exp(1j * x[:, 0]), quadrature=6)
    with pytest.raises(ValueError, match="nonfinite"):
        field.project(lambda x: torch.full_like(x[:, 0], torch.inf), quadrature=6)


@pytest.mark.parametrize("quadrature", [True, 1.0, 0.0, "adaptive"])
def test_projection_reuses_the_unambiguous_quadrature_contract(quadrature):
    field, _ = scalar_field(size=2)
    with pytest.raises((TypeError, ValueError), match="quadrature"):
        field.project(lambda x: x[:, 0], quadrature=quadrature)
