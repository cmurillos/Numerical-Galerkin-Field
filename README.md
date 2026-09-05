# Numerical Galerkin Field

[![Numerical verification](https://github.com/cmurillos/Numerical-Galerkin-Field/actions/workflows/ci.yml/badge.svg)](https://github.com/cmurillos/Numerical-Galerkin-Field/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/numerical-galerkin-field.svg)](https://pypi.org/project/numerical-galerkin-field/)
[![Python](https://img.shields.io/pypi/pyversions/numerical-galerkin-field.svg)](https://pypi.org/project/numerical-galerkin-field/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Numerical Galerkin Field builds finite-dimensional vector fields directly from complete
weak formulations on simplicial geometries. It keeps the geometry, basis, coefficients
and quadrature fixed, while exposing the reduced autonomous field

```text
G: R^N -> R^N,    G_i(z) = a(Phi_N z; phi_i).
```

The implementation is batch-first, differentiable with respect to the Galerkin
coordinates, compatible with CPU and CUDA, and supports domains of arbitrary intrinsic
and ambient dimension.

## Installation

```bash
pip install numerical-galerkin-field
```

Python 3.11 or newer is required.

## Quick start

```python
import torch

from ngfield import GalerkinProblem, grad, inner


vertices = [[i / 16] for i in range(17)]
simplices = [[i, i + 1] for i in range(16)]


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx


problem = GalerkinProblem(
    vertices=vertices,
    simplices=simplices,
    weak=weak,
)
basis = problem.basis("laplacian", size=8, degree=1)
G = problem.field(basis=basis)


def u0(x):
    return torch.sin(torch.pi * x[:, 0])


z0 = G.project(u0)

times = torch.linspace(0, 0.1, 11, dtype=G.dtype, device=G.device)
Z = G.solve(z0, times)

points = torch.linspace(0, 1, 101, dtype=G.dtype, device=G.device).reshape(-1, 1)
U = G.reconstruct(Z, points)
```

Here `Z` has shape `[time, N]` and `U` has shape `[time, point]`. Arbitrary batch axes
may precede `N`.

## Core interface

| Operation | Purpose |
| --- | --- |
| `GalerkinProblem(...)` | Combine a simplicial geometry with one complete weak form. |
| `problem.basis(...)` | Build a fixed real `L2`-orthonormal basis. |
| `problem.field(...)` | Assemble the reusable coordinate field `G`. |
| `G(z)` | Evaluate `G: [...,N] -> [...,N]`. |
| `G.project(u)` | Project a physical function onto the fixed basis. |
| `G.solve(z0, times)` | Integrate with adaptive Dormand--Prince RK45. |
| `G.solve(z0, times, step=h)` | Integrate with fixed-step RK4. |
| `G.reconstruct(z, points)` | Evaluate the reconstructed field at physical points. |
| `G.grad(z, points)` | Evaluate elementwise tangential gradients. |
| `G.hessian(z, points)` | Evaluate elementwise tangential Hessians. |
| `G.projection_error(u)` | Estimate the spatial `L2` projection error. |
| `G.time_error(...)` | Compare temporally refined trajectories. |
| `G.quadrature_error(z)` | Compare fields assembled with refined quadrature. |

## Geometry and weak forms

The geometry is an affine simplicial complex with vertices `[M,p]` and simplices
`[E,k+1]`, where `1 <= k <= p`. This includes ordinary domains in `R^k`, embedded
curves and surfaces, and triangulated closed manifolds such as a torus in `R^3`.

A weak form is written as a single Python function:

```python
def weak(u, v, dx, ds):
    volume = -inner(K @ grad(u), grad(v)) * dx("material")
    boundary = beta * u * v * ds("wall")
    return volume + boundary
```

The expression must be scalar and linear in `v`; it may be nonlinear in `u`.
Boundary labels restrict integration only. Essential conditions and other admissibility
constraints belong to the selected basis rather than to special boundary-condition
classes.

## Bases and differentiation

Available basis families include geometry-adapted Laplacian modes, total-degree
polynomials, real Fourier modes, full finite-element bases and custom callable bases.
The operational basis is always fixed and numerically orthonormal in `L2`.

The field preserves PyTorch automatic differentiation:

```python
J = torch.func.jacrev(G)(z)
_, Jw = torch.func.jvp(G, (z,), (w,))
```

The geometry, basis and fixed spatial coefficients remain outside the differentiation
graph.

## Documentation and examples

- [Usage guide](docs/usage.md)
- [Mathematical contract](docs/mathematics.md)
- [Accepted design decisions D-001--D-012](docs/design-contract.md)
- [General ND problem](examples/general_problem.py)
- [Embedded torus](examples/embedded_torus.py)
- [Discontinuous-data projection](examples/discontinuous_projection.py)
- [Time evolution and diagnostics](examples/time_evolution.py)

## Current scope

- Simplices are affine; curved geometry is represented by a simplicial approximation.
- Discontinuous data are currently projected into a continuous fixed space in `L2`.
  Interior-facet DG operators are not yet implemented.
- RK4 and RK45 are explicit. Stiff diffusive systems may require small steps; implicit
  and IMEX integrators remain a future extension.
- The error methods are refinement indicators, not certified a posteriori bounds.

These limitations are explicit: the package does not silently assign mathematical
meaning to unsupported operations.

## Development

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
python -m build
```

The continuous-integration workflow verifies Python 3.11 and 3.12, all numerical
contracts, examples, a small benchmark and both distribution formats.

## Citation

If this software contributes to academic work, cite the repository metadata in
[`CITATION.cff`](CITATION.cff).

## License

Copyright © 2026 Carlos Andrés Murillo. Distributed under the
[BSD 3-Clause License](LICENSE).
