"""A nonlinear field with a fixed heterogeneous coefficient on an interval."""

import json

import torch

from ngfield import Coefficient, GalerkinProblem, grad, inner, pointwise


def main():
    diffusivity = Coefficient.cell([0.1, 0.4])

    def weak(u, v, dx, ds):
        reaction = pointwise(lambda value: value - value**3, u)
        return reaction * v * dx - diffusivity * inner(grad(u), grad(v)) * dx

    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=weak,
    )
    basis = problem.basis("laplacian", size=3)
    field = problem.field(basis=basis, quadrature=6)
    z = torch.tensor([0.1, -0.2, 0.3], dtype=field.dtype, requires_grad=True)
    result = field(z)
    result.square().sum().backward()
    if not torch.isfinite(result).all() or not torch.isfinite(z.grad).all():
        raise RuntimeError("The heterogeneous field or its state derivative is nonfinite.")
    print(
        json.dumps(
            {
                "coefficient": "piecewise constant",
                "input_shape": list(z.shape),
                "output_shape": list(result.shape),
                "field_norm": torch.linalg.vector_norm(result).item(),
                "state_gradient_norm": torch.linalg.vector_norm(z.grad).item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
