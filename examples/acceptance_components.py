"""D-013/7: nonlinear coupled diffusion with different essential boundaries."""

import json

import numpy as np
import torch

from ngfield import GalerkinField, SimplicialDomain, Space, ZeroTrace, grad, inner

KAPPA = (0.2, 0.1)
BETA = 0.3


def equilibrium(points):
    x = points[:, 0]
    return torch.stack((x * (2 - x), 1 - x**2), dim=-1)


def build_field():
    vertices = np.linspace(0, 1, 5)[:, None]
    simplices = np.column_stack((np.arange(4), np.arange(1, 5)))
    geometry = SimplicialDomain(
        vertices=vertices,
        simplices=simplices,
        boundaries={"left": [[0]], "right": [[4]]},
    )
    V = Space(
        geometry=geometry,
        components=2,
        regularity=1,
        restrictions=[
            ZeroTrace(component=0, boundary="left"),
            ZeroTrace(component=1, boundary="right"),
        ],
    )
    basis = V.basis("finite-element", degree=2)

    def weak(u, v, dx, ds):
        x = dx.x[0]
        a, b = x * (2 - x), 1 - x**2
        f0 = 2 * KAPPA[0] - BETA * (b - a) + a**3
        f1 = 2 * KAPPA[1] - BETA * (a - b) + b**3
        return (
            -KAPPA[0] * inner(grad(u[0]), grad(v[0]))
            - KAPPA[1] * inner(grad(u[1]), grad(v[1]))
            + (BETA * (u[1] - u[0]) - u[0] ** 3 + f0) * v[0]
            + (BETA * (u[0] - u[1]) - u[1] ** 3 + f1) * v[1]
        ) * dx

    return GalerkinField(basis=basis, weak=weak, quadrature=8)


def run():
    G = build_field()
    steady = G.project(equilibrium, quadrature=8)
    residual = G(steady)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-10, rtol=0)
    z0 = G.project(lambda x: equilibrium(x) * x.new_tensor([0.7, 1.2]), quadrature=8)
    times = torch.linspace(0, 0.6, 7, dtype=G.dtype)
    Z = G.solve(z0, times, tolerance=1e-10)
    delta = Z - steady

    q = G.geometry.quadrature(8)
    points, weights = torch.tensor(q.points.copy()), torch.tensor(q.weights.copy())
    cells = torch.tensor(q.cells.copy())
    exact_steady = equilibrium(points)
    recovered = G.reconstruct(steady, points, cells=cells)
    torch.testing.assert_close(recovered, exact_steady, atol=1e-11, rtol=0)
    U = G.reconstruct(Z, points, cells=cells)
    error = U - exact_steady
    error_gradients = G.grad(delta, points, cells=cells)[..., 0]
    diffusion = (error_gradients.square() * points.new_tensor(KAPPA)).sum(dim=-1)
    exchange = BETA * (error[..., 0] - error[..., 1]).square()
    reaction = (error * (U**3 - exact_steady**3)).sum(dim=-1)
    dissipation = -(diffusion + exchange + reaction) @ weights
    energy_rate = (delta * G(Z)).sum(dim=-1)
    torch.testing.assert_close(energy_rate, dissipation, atol=1e-10, rtol=1e-10)
    energy = 0.5 * delta.square().sum(dim=-1)
    assert torch.all(torch.diff(energy) < 0)

    ends = G.reconstruct(Z, torch.tensor([[0.0], [1.0]], dtype=G.dtype))
    trace_error = torch.stack((ends[:, 0, 0], ends[:, 1, 1])).abs().max().item()
    assert trace_error < 1e-12
    # The other component remains free at each endpoint.
    assert ends[0, 0, 1] > 1 and ends[0, 1, 0] > 0.5
    J = torch.func.jacrev(G)(steady)
    torch.testing.assert_close(J, J.T, atol=1e-10, rtol=1e-10)
    assert torch.linalg.eigvalsh(J).max() < 0
    split = G.basis.component_sizes[0]
    coupling_norm = torch.linalg.matrix_norm(J[:split, split:]).item()
    assert coupling_norm > 0.1
    return {
        "example": "components",
        "dimension": G.dimension,
        "component_sizes": list(G.basis.component_sizes),
        "equilibrium_residual": residual.abs().max().item(),
        "equilibrium_value_error": (recovered - exact_steady).abs().max().item(),
        "dirichlet_trace_error": trace_error,
        "energy_balance_error": (energy_rate - dissipation).abs().max().item(),
        "final_energy_fraction": (energy[-1] / energy[0]).item(),
        "jacobian_coupling_norm": coupling_norm,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
