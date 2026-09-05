import pytest
import torch

from ngfield import GalerkinProblem, exp, sin


def polynomial_problem():
    return GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: u**3 * v * dx,
    )


def nonpolynomial_problem():
    return GalerkinProblem(
        vertices=[[0.0], [1.0]],
        simplices=[[0, 1]],
        weak=lambda u, v, dx, ds: exp(u) * v * dx,
    )


def test_automatic_mode_infers_an_exact_polynomial_order():
    problem = polynomial_problem()
    basis = problem.basis("polynomial", size=3)
    field = problem.field(basis=basis)
    reference = problem.field(basis=basis, quadrature=14)
    z = torch.tensor([0.2, -0.4, 0.3], dtype=field.dtype)

    assert field.quadrature_mode == "automatic-exact"
    assert field.quadrature_order >= 8
    assert field.quadrature_error_estimate == 0.0
    torch.testing.assert_close(field(z), reference(z), atol=1e-12, rtol=1e-12)


def test_automatic_mode_adapts_nonpolynomial_integrands_then_freezes():
    problem = nonpolynomial_problem()
    basis = problem.basis("polynomial", size=2)
    field = problem.field(basis=basis)
    reference = problem.field(basis=basis, quadrature=20)
    order = field.quadrature_order
    size = field.quadrature_size
    z = torch.tensor([0.4, -0.2], dtype=field.dtype)

    assert field.quadrature_mode == "automatic-adaptive"
    assert field.quadrature_error_estimate <= field.quadrature_tolerance
    torch.testing.assert_close(field(z), reference(z), atol=2e-8, rtol=2e-8)
    field(2 * z)
    assert field.quadrature_order == order
    assert field.quadrature_size == size


def test_a_float_requests_adaptive_quadrature_and_an_integer_fixes_the_order():
    problem = nonpolynomial_problem()
    basis = problem.basis("polynomial", size=2)
    adaptive = problem.field(basis=basis, quadrature=1e-6)
    fixed = problem.field(basis=basis, quadrature=12)

    assert adaptive.quadrature_mode == "adaptive"
    assert adaptive.quadrature_tolerance == 1e-6
    assert adaptive.quadrature_error_estimate <= 1e-6
    assert fixed.quadrature_mode == "fixed"
    assert fixed.quadrature_order == 12
    assert fixed.quadrature_tolerance is None
    assert fixed.quadrature_error_estimate is None


def test_adaptation_includes_named_boundary_integrals():
    problem = GalerkinProblem(
        vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        simplices=[[0, 1, 2]],
        boundaries={"edge": [[1, 2]]},
        weak=lambda u, v, dx, ds: sin(3 * ds.x[0]) * v * ds("edge"),
    )
    basis = problem.basis("polynomial", size=3)
    adaptive = problem.field(basis=basis, quadrature=1e-8)
    reference = problem.field(basis=basis, quadrature=24)
    z = torch.zeros(basis.dimension, dtype=adaptive.dtype)

    assert adaptive.quadrature_mode == "adaptive"
    assert adaptive.quadrature_error_estimate <= 1e-8
    torch.testing.assert_close(adaptive(z), reference(z), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("value", [True, False, 1.0, 0.0, -1e-4, "auto"])
def test_quadrature_contract_rejects_ambiguous_values(value):
    problem = polynomial_problem()
    basis = problem.basis("polynomial", size=2)
    with pytest.raises((TypeError, ValueError), match="quadrature"):
        problem.field(basis=basis, quadrature=value)


def test_old_field_quadrature_keywords_are_not_part_of_the_general_api():
    problem = polynomial_problem()
    basis = problem.basis("polynomial", size=2)
    with pytest.raises(TypeError, match="quadrature_order"):
        problem.field(basis=basis, quadrature_order=6)


def test_adaptation_fails_explicitly_when_the_point_budget_is_exhausted():
    problem = nonpolynomial_problem()
    basis = problem.basis("polynomial", size=2)
    with pytest.raises(ValueError, match="exhausted max_quadrature_points"):
        problem.field(basis=basis, quadrature=1e-8, max_quadrature_points=1)
