"""Project, evolve and diagnose one Galerkin trajectory."""

import json

import torch

from ngfield import GalerkinProblem


def weak(u, v, dx, ds):
    return -u * v * dx


def main():
    problem = GalerkinProblem(
        vertices=[[0.0], [0.5], [1.0]],
        simplices=[[0, 1], [1, 2]],
        weak=weak,
    )
    basis = problem.basis("polynomial", size=4)
    field = problem.field(basis=basis, quadrature=8)

    def initial(x):
        return torch.exp(x[:, 0])

    z0 = field.project(initial, quadrature=16)
    times = torch.linspace(0, 1, 21, dtype=field.dtype, device=field.device)
    states = field.solve(z0, times)
    points = torch.tensor([[0.25], [0.75]], dtype=field.dtype, device=field.device)
    values = field.reconstruct(states, points)

    diagnostics = {
        "projection_error": field.projection_error(initial, quadrature=16).item(),
        "time_error": field.time_error(z0, times, step=0.05)[-1].item(),
        "quadrature_error": field.quadrature_error(states, order=10).max().item(),
    }
    if not torch.isfinite(values).all() or not all(value >= 0 for value in diagnostics.values()):
        raise RuntimeError("The evolution example produced invalid numerical values.")
    print(
        json.dumps(
            {
                "state_shape": list(states.shape),
                "value_shape": list(values.shape),
                **diagnostics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
