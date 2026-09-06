"""D-013/7: surface heat, conserved mean and dissipation on a triangulated torus."""

import json

import numpy as np
import torch

from ngfield import GalerkinField, SimplicialDomain, Space, grad, inner

KAPPA = 0.1


def build_field():
    n_major, n_minor = 12, 8
    u, v = np.meshgrid(
        2 * np.pi * np.arange(n_major) / n_major,
        2 * np.pi * np.arange(n_minor) / n_minor,
        indexing="ij",
    )
    vertices = np.stack(
        ((2 + 0.5 * np.cos(v)) * np.cos(u), (2 + 0.5 * np.cos(v)) * np.sin(u), 0.5 * np.sin(v)),
        axis=-1,
    ).reshape(-1, 3)

    def index(i, j):
        return (i % n_major) * n_minor + j % n_minor

    simplices = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i, j + 1), index(i + 1, j + 1)
            simplices.extend(((a, b, d), (a, d, c)))

    geometry = SimplicialDomain(vertices=vertices, simplices=simplices)
    V = Space(geometry=geometry, components=1, regularity=1)
    basis = V.basis("laplacian", size=12, degree=1)

    def weak(u, v, dx, ds):
        return -KAPPA * inner(grad(u[0]), grad(v[0])) * dx

    return GalerkinField(basis=basis, weak=weak)


def run():
    G = build_field()
    geometry = G.geometry
    assert geometry.dimension == 2 and geometry.ambient_dimension == 3
    assert len(geometry.exterior_faces) == 0
    z0 = G.project(lambda x: 2 + 0.2 * x[:, :1] + 0.3 * x[:, 2:3], quadrature=6)
    times = torch.linspace(0, 0.5, 6, dtype=G.dtype)
    Z = G.solve(z0, times, tolerance=1e-10)

    # Exact trajectory of the semidiscrete spectral problem on this same mesh.
    eigenvalues = torch.tensor(G.basis.eigenvalues, dtype=G.dtype)
    reference = z0 * torch.exp(-KAPPA * times[:, None] * eigenvalues)
    torch.testing.assert_close(Z, reference, atol=2e-9, rtol=2e-9)

    # Independent spatial quadrature on the triangles, not on the smooth torus.
    q = geometry.quadrature(6)
    points, cells = torch.tensor(q.points.copy()), torch.tensor(q.cells.copy())
    weights = torch.tensor(q.weights.copy())
    values = G.reconstruct(Z, points, cells=cells)[..., 0]
    means = (values @ weights) / weights.sum()
    gradients = G.grad(Z, points, cells=cells)
    dissipation = -KAPPA * (gradients.square().sum(dim=(-1, -2)) @ weights)
    energy_rate = (Z * G(Z)).sum(dim=-1)
    torch.testing.assert_close(means, torch.full_like(means, 2), atol=1e-11, rtol=0)
    torch.testing.assert_close(energy_rate, dissipation, atol=1e-11, rtol=1e-10)
    energy = 0.5 * ((values - means[:, None]).square() @ weights)
    assert torch.all(torch.diff(energy) < 0)

    return {
        "example": "torus",
        "dimension": G.dimension,
        "triangles": len(geometry.simplices),
        "surface_area": weights.sum().item(),
        "time_error_vs_semidiscrete": (Z - reference).abs().max().item(),
        "mean_error": (means - 2).abs().max().item(),
        "energy_balance_error": (energy_rate - dissipation).abs().max().item(),
        "final_energy_fraction": (energy[-1] / energy[0]).item(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
