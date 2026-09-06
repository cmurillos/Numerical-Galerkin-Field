"""D-013/7: fixed Dirichlet lift, Robin/Neumann exchange and a localized source."""

import json

import numpy as np
import torch
from scipy.linalg import expm

from ngfield import GalerkinField, SimplicialDomain, Space, ZeroTrace, grad, inner


def lift(points):
    return 1 + points[:, 1:2]


def equilibrium(points):
    x = points[:, :1]
    return lift(points) + x - torch.clamp(x - 0.5, min=0).square()


def build_field():
    vertices = np.array([[x, y] for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)])
    simplices = []
    for j in range(2):
        for i in range(2):
            a = 3 * j + i
            simplices.extend(((a, a + 1, a + 4), (a, a + 4, a + 3)))
    simplices = np.array(simplices)
    heated = np.flatnonzero(vertices[simplices].mean(axis=1)[:, 0] > 0.5)
    geometry = SimplicialDomain(
        vertices=vertices,
        simplices=simplices,
        regions={"heated": heated},
        boundaries={
            "left": [[0, 3], [3, 6]],
            "right": [[2, 5], [5, 8]],
            "bottom": [[0, 1], [1, 2]],
            "top": [[6, 7], [7, 8]],
        },
    )
    V = Space(
        geometry=geometry,
        components=1,
        regularity=1,
        restrictions=[ZeroTrace(component=0, boundary="left")],
    )
    basis = V.basis("finite-element", degree=2)

    def weak(w, v, dx, ds):
        y = dx.x[1]
        temperature = w[0] + 1 + y
        return (
            -inner(grad(temperature), grad(v[0])) * dx
            + 2 * v[0] * dx("heated")
            - v[0] * ds("bottom")
            + v[0] * ds("top")
            - (1 + y) * (temperature - (1.75 + y)) * v[0] * ds("right")
        )

    return GalerkinField(basis=basis, weak=weak, quadrature=6)


def run():
    G = build_field()
    steady = G.project(lambda x: equilibrium(x) - lift(x), quadrature=6)
    residual = G(steady)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10, rtol=0)
    initial_temperature = lift
    z0 = G.project(lambda x: initial_temperature(x) - lift(x), quadrature=6)
    times = torch.linspace(0, 0.3, 7, dtype=G.dtype)
    Z = G.solve(z0, times, tolerance=1e-10)

    # This reference checks temporal integration of the assembled affine ODE only.
    A = torch.func.jacrev(G)(steady)
    reference = torch.stack(
        [
            steady + torch.tensor(expm(t.item() * A.numpy()), dtype=G.dtype) @ (z0 - steady)
            for t in times
        ]
    )
    torch.testing.assert_close(Z, reference, atol=3e-8, rtol=3e-8)

    q = G.geometry.quadrature(6)
    points, weights = torch.tensor(q.points.copy()), torch.tensor(q.weights.copy())
    cells = torch.tensor(q.cells.copy())
    recovered = lift(points) + G.reconstruct(steady, points, cells=cells)
    torch.testing.assert_close(recovered, equilibrium(points), atol=1e-11, rtol=0)
    delta = Z - steady
    error_gradients = G.grad(delta, points, cells=cells)
    dissipation = -(error_gradients.square().sum(dim=(-1, -2)) @ weights)
    qr = G.geometry.quadrature(6, boundary="right")
    right = torch.tensor(qr.points.copy())
    error_right = G.reconstruct(delta, right, cells=torch.tensor(qr.cells.copy()))[..., 0]
    dissipation -= error_right.square() @ ((1 + right[:, 1]) * torch.tensor(qr.weights.copy()))
    energy_rate = (delta * G(Z)).sum(dim=-1)
    torch.testing.assert_close(energy_rate, dissipation, atol=1e-10, rtol=1e-10)
    energy = 0.5 * delta.square().sum(dim=-1)
    assert torch.all(torch.diff(energy) < 0)

    left = torch.stack(
        (torch.zeros(11, dtype=G.dtype), torch.linspace(0, 1, 11, dtype=G.dtype)), dim=1
    )
    boundary_temperature = lift(left) + G.reconstruct(Z, left)
    trace_error = (boundary_temperature - lift(left)).abs().max().item()
    assert trace_error < 1e-12
    source_power = 2 * G.geometry.quadrature(6, region="heated").weights.sum()
    np.testing.assert_allclose(source_power, 1, atol=1e-12, rtol=0)
    return {
        "example": "mixed_plate",
        "dimension": G.dimension,
        "source_power": float(source_power),
        "equilibrium_residual": residual.abs().max().item(),
        "equilibrium_value_error": (recovered - equilibrium(points)).abs().max().item(),
        "time_error_vs_matrix_exponential": (Z - reference).abs().max().item(),
        "dirichlet_trace_error": trace_error,
        "energy_balance_error": (energy_rate - dissipation).abs().max().item(),
        "final_energy_fraction": (energy[-1] / energy[0]).item(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
