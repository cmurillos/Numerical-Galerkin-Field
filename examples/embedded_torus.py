"""A closed triangulated torus embedded in R3."""

import json
import math

import numpy as np
import torch

from ngfield import GalerkinProblem, grad, inner


def torus_mesh(major_radius=2.0, minor_radius=0.5, n_major=12, n_minor=8):
    vertices = []
    for i in range(n_major):
        u = 2 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2 * math.pi * j / n_minor
            radius = major_radius + minor_radius * math.cos(v)
            vertices.append(
                [radius * math.cos(u), radius * math.sin(u), minor_radius * math.sin(v)]
            )

    def index(i, j):
        return (i % n_major) * n_minor + (j % n_minor)

    simplices = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i, j + 1), index(i + 1, j + 1)
            simplices.extend(([a, b, d], [a, d, c]))
    return np.asarray(vertices), np.asarray(simplices)


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx


def main():
    vertices, simplices = torus_mesh()
    problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
    basis = problem.basis("laplacian", size=16)
    field = problem.field(basis=basis)
    z = torch.sin(torch.linspace(0, 2 * math.pi, basis.dimension, dtype=field.dtype))
    result = field(z)
    point = torch.tensor(vertices[simplices[0]].mean(axis=0)[None, :], dtype=field.dtype)
    cell = torch.tensor([0], dtype=torch.int64)
    value = field.reconstruct(z, point, cells=cell)
    gradient = field.grad(z, point, cells=cell)
    Hessian = field.hessian(z, point, cells=cell)
    print(
        json.dumps(
            {
                "intrinsic_dimension": problem.geometry.intrinsic_dimension,
                "ambient_dimension": problem.geometry.ambient_dimension,
                "boundary_facets": len(problem.geometry.exterior_faces),
                "vertices": len(vertices),
                "triangles": len(simplices),
                "modes": basis.dimension,
                "first_eigenvalues": basis.eigenvalues[:5].tolist(),
                "field_norm": torch.linalg.vector_norm(result).item(),
                "value_at_cell_center": value.tolist(),
                "gradient_at_cell_center": gradient.tolist(),
                "hessian_at_cell_center": Hessian.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
