"""The compact general interface on a single four-dimensional simplex."""

import json

import numpy as np
import torch

from ngfield import GalerkinProblem, grad, inner


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx - 2.0 * u * v * ds("wall")


def main():
    vertices = np.vstack((np.zeros(4), np.eye(4)))
    simplices = np.array([[0, 1, 2, 3, 4]])
    boundaries = {"wall": np.array([[1, 2, 3, 4]])}
    problem = GalerkinProblem(
        vertices=vertices,
        simplices=simplices,
        boundaries=boundaries,
        weak=weak,
    )
    basis = problem.basis("laplacian", size=5, degree=1)
    field = problem.field(basis=basis, quadrature=4)
    z = field.project(lambda x: x.sum(dim=1), quadrature=4)
    result = field(z)
    print(
        json.dumps(
            {
                "spatial_dimension": problem.geometry.dimension,
                "modes": basis.dimension,
                "input_shape": list(z.shape),
                "output_shape": list(result.shape),
                "field_norm": torch.linalg.vector_norm(result).item(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
