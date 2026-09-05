import torch

from ngfield import GalerkinProblem, exp


def projection_field(size):
    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=lambda u, v, dx, ds: -u * v * dx,
    )
    basis = problem.basis("polynomial", size=size)
    return problem.field(basis=basis, quadrature=8), basis


def test_projection_error_vanishes_on_the_fixed_space_and_preserves_batches():
    field, basis = projection_field(3)
    coordinates = torch.tensor(
        [[0.2, -0.3, 0.4], [-0.1, 0.5, 0.2]],
        dtype=field.dtype,
    )

    def source(points):
        return torch.einsum("bn,qn->bq", coordinates, basis.evaluate(points))

    error = field.projection_error(source, quadrature=12)
    assert error.shape == (2,)
    torch.testing.assert_close(error, torch.zeros_like(error), atol=2e-12, rtol=0)


def test_projection_error_decreases_when_the_polynomial_space_is_enriched():
    coarse, _ = projection_field(2)
    fine, _ = projection_field(5)

    def source(x):
        return torch.exp(x[:, 0])

    coarse_error = coarse.projection_error(source, quadrature=20)
    fine_error = fine.projection_error(source, quadrature=20)

    assert fine_error < coarse_error
    assert fine_error < 1e-3 * coarse_error


def test_time_error_is_an_l2_step_refinement_indicator():
    field, _ = projection_field(2)
    z0 = torch.tensor([0.4, -0.2], dtype=field.dtype)
    times = torch.tensor([0.0, 0.5, 1.0], dtype=field.dtype)
    coarse = field.time_error(z0, times, step=0.2)
    fine = field.time_error(z0, times, step=0.1)

    assert coarse.shape == (len(times),)
    assert coarse[0] == 0
    assert 0 < fine[-1] < coarse[-1]


def test_adaptive_time_error_preserves_arbitrary_batch_axes():
    field, _ = projection_field(2)
    z0 = torch.randn(2, 3, field.dimension, dtype=field.dtype)
    times = torch.tensor([0.0, 0.2], dtype=field.dtype)
    error = field.time_error(z0, times, tolerance=1e-6)
    assert error.shape == (2, 2, 3)
    assert torch.isfinite(error).all()


def test_quadrature_error_compares_reduced_velocities_in_l2():
    problem = GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: exp(3 * u) * v * dx,
    )
    basis = problem.basis("polynomial", size=2)
    coarse = problem.field(basis=basis, quadrature=2)
    refined = problem.field(basis=basis, quadrature=10)
    z = torch.tensor(
        [[[0.6, -0.7], [0.4, 0.8]], [[-0.5, 0.9], [0.7, 0.3]]],
        dtype=coarse.dtype,
    )
    metadata = (
        coarse.quadrature_order,
        coarse.quadrature_size,
        coarse.table_bytes,
        coarse.mass_matrix.clone(),
    )

    coarse_error = coarse.quadrature_error(z, order=12)
    refined_error = refined.quadrature_error(z, order=12)

    assert coarse_error.shape == z.shape[:-1]
    assert torch.max(coarse_error) > 1e-7
    assert torch.max(refined_error) < torch.max(coarse_error)
    assert metadata[:3] == (
        coarse.quadrature_order,
        coarse.quadrature_size,
        coarse.table_bytes,
    )
    torch.testing.assert_close(coarse.mass_matrix, metadata[3], atol=0, rtol=0)


def test_quadrature_error_requires_a_strictly_higher_order():
    field, _ = projection_field(2)
    z = torch.zeros(field.dimension, dtype=field.dtype)
    for order in (0, field.quadrature_order):
        try:
            field.quadrature_error(z, order=order)
        except ValueError as exc:
            assert "greater" in str(exc)
        else:
            raise AssertionError("A non-refined quadrature order must be rejected.")
