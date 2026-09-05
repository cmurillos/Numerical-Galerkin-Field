import numpy as np
import torch

from ngfield import Coefficient, GalerkinProblem


def projected_step(cell_count):
    vertices = np.linspace(0.0, 1.0, cell_count + 1)[:, None]
    simplices = np.column_stack((np.arange(cell_count), np.arange(1, cell_count + 1)))
    centers = vertices[simplices].mean(axis=(1, 2))
    source = Coefficient.cell((centers < 0.5).astype(float))
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=simplices,
        weak=lambda u, v, dx, ds: u * v * dx,
    )
    basis = problem.basis("finite-element", degree=1)
    field = problem.field(basis=basis)
    return field, field.project(source)


def test_a_discontinuous_field_is_projected_to_one_continuous_trace():
    field, coordinates = projected_step(8)
    point = torch.tensor([[0.5]], dtype=field.dtype)
    left_cell = torch.tensor([3], dtype=torch.int64)
    right_cell = torch.tensor([4], dtype=torch.int64)

    automatic = field.reconstruct(coordinates, point)
    left = field.reconstruct(coordinates, point, cells=left_cell)
    right = field.reconstruct(coordinates, point, cells=right_cell)
    torch.testing.assert_close(automatic, left, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        automatic,
        torch.tensor([0.5], dtype=field.dtype),
        atol=1e-12,
        rtol=1e-12,
    )


def test_refinement_reduces_the_l2_error_of_the_continuous_approximation():
    coarse_field, coarse = projected_step(4)
    fine_field, fine = projected_step(16)
    source_norm_squared = 0.5
    coarse_error = source_norm_squared - torch.dot(coarse, coarse)
    fine_error = source_norm_squared - torch.dot(fine, fine)

    assert coarse_field.dimension < fine_field.dimension
    assert coarse_error > 0
    assert fine_error > 0
    assert fine_error < coarse_error


def test_projection_remains_differentiable_with_respect_to_scaling_parameters():
    field, _ = projected_step(8)
    amplitude = torch.tensor(2.0, dtype=field.dtype, requires_grad=True)
    coordinates = field.project(
        lambda x: amplitude * (x[:, 0] < 0.5).to(field.dtype),
        quadrature=20,
    )
    coordinates.square().sum().backward()

    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)
    assert amplitude.grad > 0
