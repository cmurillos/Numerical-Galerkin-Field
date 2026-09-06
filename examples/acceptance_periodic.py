"""D-013/7: periodic mean-zero heat, exact PDE solution and spatial refinement."""

import json

import numpy as np
import torch

from ngfield import GalerkinField, MeanZero, Periodic, SimplicialDomain, Space, grad, inner

KAPPA = 0.05


def build_field(cells):
    vertices = np.linspace(0, 1, cells + 1)[:, None]
    simplices = np.column_stack((np.arange(cells), np.arange(1, cells + 1)))
    geometry = SimplicialDomain(
        vertices=vertices,
        simplices=simplices,
        boundaries={"left": [[0]], "right": [[cells]]},
    )
    V = Space(
        geometry=geometry,
        components=1,
        regularity=1,
        restrictions=[
            Periodic(component=0, boundaries=("left", "right"), vertex_pairs=[(0, cells)]),
            MeanZero(component=0),
        ],
    )
    basis = V.basis("laplacian", size=4, degree=1)

    def weak(u, v, dx, ds):
        return -KAPPA * inner(grad(u[0]), grad(v[0])) * dx

    return GalerkinField(basis=basis, weak=weak)


def exact(times, points):
    t, x = times[:, None], points[None, :, 0]
    return torch.exp(-KAPPA * (2 * torch.pi) ** 2 * t) * torch.sin(
        2 * torch.pi * x
    ) + 0.25 * torch.exp(-KAPPA * (4 * torch.pi) ** 2 * t) * torch.cos(4 * torch.pi * x)


def run():
    errors, time_errors, mean_errors, trace_errors = [], [], [], []
    for cells in (16, 32):
        G = build_field(cells)
        times = torch.linspace(0, 0.3, 7, dtype=G.dtype)
        z0 = G.project(lambda x: exact(times[:1], x).T, quadrature=8)
        Z = G.solve(z0, times, tolerance=1e-11)
        eigenvalues = torch.tensor(G.basis.eigenvalues, dtype=G.dtype)
        reference = z0 * torch.exp(-KAPPA * times[:, None] * eigenvalues)
        torch.testing.assert_close(Z, reference, atol=2e-9, rtol=2e-9)

        q = G.geometry.quadrature(8)
        points, weights = torch.tensor(q.points.copy()), torch.tensor(q.weights.copy())
        U = G.reconstruct(Z, points, cells=torch.tensor(q.cells.copy()))[..., 0]
        spatial_error = torch.sqrt((U - exact(times, points)).square() @ weights)
        means = (U @ weights) / weights.sum()
        ends = G.reconstruct(Z, torch.tensor([[0.0], [1.0]], dtype=G.dtype))
        torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-12, rtol=0)
        torch.testing.assert_close(ends[:, 0], ends[:, 1], atol=1e-12, rtol=0)
        errors.append(spatial_error.max().item())
        time_errors.append((Z - reference).abs().max().item())
        mean_errors.append(means.abs().max().item())
        trace_errors.append((ends[:, 0] - ends[:, 1]).abs().max().item())

    # P1 geometry/bases are refined; the four physical Fourier modes stay fixed.
    assert errors[1] < 0.003
    assert 0 < errors[1] / errors[0] < 0.35
    return {
        "example": "periodic",
        "dimension": 4,
        "cells": [16, 32],
        "max_L2_error_vs_PDE": errors,
        "refinement_ratio": errors[1] / errors[0],
        "time_error_vs_semidiscrete": max(time_errors),
        "mean_error": max(mean_errors),
        "periodic_trace_error": max(trace_errors),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
