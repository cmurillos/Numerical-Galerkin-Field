"""Approximate a discontinuous field in one continuous Galerkin space."""

import json

import numpy as np
import torch

from ngfield import Coefficient, GalerkinProblem


def main():
    cell_count = 16
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
    coordinates = field.project(source)

    points = torch.tensor([[0.49], [0.50], [0.51]], dtype=field.dtype)
    values = field.reconstruct(coordinates, points)
    error_squared = 0.5 - torch.dot(coordinates, coordinates)
    print(
        json.dumps(
            {
                "basis": "continuous P1",
                "modes": field.dimension,
                "points": points[:, 0].tolist(),
                "projected_values": values.tolist(),
                "l2_projection_error": torch.sqrt(error_squared).item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
