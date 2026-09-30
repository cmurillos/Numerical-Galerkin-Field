"""Evolve a physical function and inspect its numerical Galerkin flow."""

import torch

from ngfield import Geometry, Space, System


def main():
    geometry = Geometry(vertices=[[0.0], [1.0]], simplices=[[0, 1]])
    V = Space(geometry=geometry, components=1)
    basis = V.basis("polynomial", degree=1)

    def weak(u, v, dx, ds):
        return -u[0] * v[0] * dx

    system = System(basis=basis, weak=weak)
    initial = system.state(lambda x: torch.ones_like(x))
    times = torch.tensor([0.0, 0.1, 0.2], dtype=system.dtype, device=system.device)
    path = system.evolve(initial, times, order=4, step=0.02)
    u_t = path.at(times[-1])
    points = torch.tensor([[0.2], [0.8]], dtype=system.dtype, device=system.device)
    print("u(t) at points:", u_t.values(points))
    print("u'(t) in V_N:", u_t.velocity().values(points))
    print("indexed state derivatives:", tuple(u_t.indexed_derivatives(2)))
    print("L2 norm:", u_t.norm_L2())
    print("time refinement indicator:", system.metrics.time_refinement(initial, times, step=0.02))


if __name__ == "__main__":
    main()
