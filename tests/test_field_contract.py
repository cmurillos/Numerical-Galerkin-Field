import pytest
import torch

from ngfield import ComponentBasis, GalerkinProblem, inner


def nonlinear_field():
    def weak(u, v, dx, ds):
        return (u - u**3) * v * dx

    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=weak,
    )
    basis = problem.basis("laplacian", size=3)
    return problem.field(basis=basis, quadrature_order=6)


@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3), (2, 1, 3)])
def test_field_preserves_arbitrary_leading_batch_dimensions(batch_shape):
    field = nonlinear_field()
    z = torch.randn((*batch_shape, field.dimension), dtype=field.dtype)
    result = field(z)
    reference = torch.stack([field(state) for state in z.reshape(-1, field.dimension)])
    torch.testing.assert_close(result, reference.reshape(z.shape))
    assert result.shape == z.shape


def test_reconstruction_preserves_batch_dimensions_and_removes_fake_singleton():
    field = nonlinear_field()
    single = torch.randn(field.dimension, dtype=field.dtype)
    batch = torch.randn(2, 3, field.dimension, dtype=field.dtype)
    assert field.reconstruct(single).shape == (field.quadrature_size,)
    assert field.reconstruct(batch).shape == (2, 3, field.quadrature_size)
    reference = torch.stack(
        [field.reconstruct(state) for state in batch.reshape(-1, field.dimension)]
    )
    torch.testing.assert_close(
        field.reconstruct(batch), reference.reshape(2, 3, field.quadrature_size)
    )


def test_empty_batches_preserve_their_complete_shape():
    field = nonlinear_field()
    z = torch.empty(2, 0, 4, field.dimension, dtype=field.dtype)
    assert field(z).shape == z.shape
    assert field.reconstruct(z).shape == (2, 0, 4, field.quadrature_size)


def test_vector_valued_reconstruction_preserves_all_physical_axes():
    def weak(u, v, dx, ds):
        return inner(u, v) * dx

    problem = GalerkinProblem(vertices=[[0.0], [1.0]], simplices=[[0, 1]], weak=weak)
    scalar = problem.basis("laplacian", size=2)
    basis = ComponentBasis(scalar, components=2)
    field = problem.field(basis=basis)
    z = torch.randn(2, 3, basis.dimension, dtype=field.dtype)
    torch.testing.assert_close(field(z), z)
    assert field.reconstruct(z).shape == (2, 3, field.quadrature_size, 2)


def test_torch_func_jacobian_jvp_vjp_and_second_derivatives():
    field = nonlinear_field()
    z = torch.tensor([0.2, -0.1, 0.3], dtype=field.dtype)
    direction = torch.tensor([-0.4, 0.5, 0.1], dtype=field.dtype)
    cotangent = torch.tensor([0.3, -0.2, 0.7], dtype=field.dtype)

    jacobian = torch.func.jacrev(field)(z)
    _, jvp = torch.func.jvp(field, (z,), (direction,))
    _, pullback = torch.func.vjp(field, z)
    (vjp,) = pullback(cotangent)
    hessian = torch.func.jacrev(torch.func.jacrev(field))(z)

    torch.testing.assert_close(jvp, jacobian @ direction)
    torch.testing.assert_close(vjp, jacobian.T @ cotangent)
    assert jacobian.shape == (field.dimension, field.dimension)
    assert hessian.shape == (field.dimension, field.dimension, field.dimension)
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(hessian).all()


def test_batched_jacobian_has_no_cross_state_coupling():
    field = nonlinear_field()
    z = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.2, 0.1]], dtype=field.dtype)
    jacobian = torch.func.jacrev(field)(z)
    assert jacobian.shape == (2, field.dimension, 2, field.dimension)
    torch.testing.assert_close(jacobian[0, :, 0, :], torch.func.jacrev(field)(z[0]))
    torch.testing.assert_close(jacobian[1, :, 1, :], torch.func.jacrev(field)(z[1]))
    torch.testing.assert_close(jacobian[0, :, 1, :], torch.zeros_like(jacobian[0, :, 1, :]))
    torch.testing.assert_close(jacobian[1, :, 0, :], torch.zeros_like(jacobian[1, :, 0, :]))


def test_vectorized_call_equals_vmap_and_input_contract_is_strict():
    field = nonlinear_field()
    z = torch.randn(4, field.dimension, dtype=field.dtype)
    torch.testing.assert_close(field(z), torch.vmap(field)(z))
    with pytest.raises(TypeError, match="torch tensors"):
        field([0.0] * field.dimension)
    with pytest.raises(ValueError, match=r"\[\.\.\.,3\]"):
        field(torch.zeros(2, 4, dtype=field.dtype))
    with pytest.raises(ValueError, match="device and dtype"):
        field(z.float())


def test_field_preserves_general_batches_after_a_dtype_move():
    field = nonlinear_field().to(dtype=torch.float32)
    z = torch.randn(2, 3, field.dimension, dtype=torch.float32)
    result = field(z)
    assert result.shape == z.shape
    assert result.dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_general_batch_contract_on_cuda():
    field = nonlinear_field().to(device="cuda")
    z = torch.randn(2, 3, field.dimension, dtype=field.dtype, device="cuda")
    result = field(z)
    assert result.shape == z.shape
    assert result.device.type == "cuda"
