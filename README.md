# Numerical Galerkin Field

[![Numerical verification](https://github.com/cmurillos/Numerical-Galerkin-Field/actions/workflows/ci.yml/badge.svg)](https://github.com/cmurillos/Numerical-Galerkin-Field/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/numerical-galerkin-field.svg)](https://pypi.org/project/numerical-galerkin-field/)
[![Python](https://img.shields.io/pypi/pyversions/numerical-galerkin-field.svg)](https://pypi.org/project/numerical-galerkin-field/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Numerical Galerkin Field builds finite-dimensional vector fields directly from complete
weak formulations on simplicial geometries. It keeps the geometry, basis, coefficients
and quadrature fixed, while exposing the reduced autonomous field

```text
G: R^N -> R^N,    G_i(z) = a(Phi z; phi_i).
```

The implementation is batch-first, differentiable with respect to the Galerkin
coordinates, compatible with CPU and CUDA, and supports domains of arbitrary intrinsic
and ambient dimension.

This guide documents the D-013 source API: geometry, admissible space, operational
basis and field. The [migration guide](docs/migration.md) explains how existing
0.9.0 code remains usable and how to move to this workflow.

## Installation from this checkout

```bash
python -m pip install -e .
```

Run this command in the repository directory. Python 3.11 or newer is required.
The distribution is named `numerical-galerkin-field`; its Python import is `ngfield`.
The D-013 additions are not part of the original 0.9.0 release. Installing that
release from PyPI does not provide this source API; see the migration guide for
the distinction between installed versions and source revisions.

## Quick start

```python
import torch

from ngfield import GalerkinField, SimplicialDomain, Space, ZeroTrace, grad, inner


vertices = [[i / 16] for i in range(17)]
simplices = [[i, i + 1] for i in range(16)]
geometry = SimplicialDomain(vertices=vertices, simplices=simplices)
V = Space(
    geometry=geometry,
    components=1,
    regularity=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
basis = V.basis("laplacian", size=8, degree=1)


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u[0]), grad(v[0])) * dx


G = GalerkinField(basis=basis, weak=weak)
```

This constructs the heat field with diffusivity 0.1 on `[0,1]` and zero temperature
at both endpoints. The basis enforces that essential condition before the spectral
problem is solved. `size=8` is the total coordinate dimension. No initial state or
time interval is needed to build `G`.

To obtain a trajectory using the objects above:

```python
def u0(x):
    return torch.sin(torch.pi * x[:, :1])


z0 = G.project(u0)

times = torch.linspace(0, 0.1, 11, dtype=G.dtype, device=G.device)
Z = G.solve(z0, times)

points = torch.linspace(0, 1, 101, dtype=G.dtype, device=G.device).reshape(-1, 1)
U = G.reconstruct(Z, points)
```

Here `Z` has shape `[time,8]` and `U` has shape `[time,point,1]`. The physical component
axis remains explicit even for a scalar problem. Arbitrary batch axes may precede
the final coordinate axis of `G`.

## Core interface

| Operation | Purpose |
| --- | --- |
| `SimplicialDomain(...)` | Receive vertices, simplices and named spatial subsets. |
| `Space(geometry=..., components=..., restrictions=...)` | Declare regularity and homogeneous admissibility conditions. |
| `V.basis(family, ...)` | Prepare a fixed real `L2`-orthonormal basis in that space. |
| `GalerkinField(basis=basis, weak=weak)` | Assemble the reusable autonomous field `G`. |
| `G.space`, `G.geometry`, `G.basis` | Inspect the associated objects. |
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

`GalerkinProblem` and the original `Problem`/`GalerkinBasis` interface remain
available. Existing scalar bases keep their scalar value shape; adopting `Space`
requires explicit components. [Migration and compatibility](docs/migration.md)
explain the corresponding calls and coordinate transfers.

## Geometry and weak forms

The geometry is an affine simplicial complex with vertices `[M,p]` and simplices
`[E,k+1]`, where `1 <= k <= p`. This includes ordinary domains in `R^k`, embedded
curves and surfaces, and triangulated closed manifolds such as a torus in `R^3`.

A weak form is written as a single Python function. This fragment assumes a scalar
space, the `material` and `wall` labels, and fixed coefficients `K`, `alpha` and `g`:

```python
def weak(u, v, dx, ds):
    volume = -inner(K @ grad(u[0]), grad(v[0])) * dx("material")
    boundary = -alpha * (u[0] - g) * v[0] * ds("wall")
    return volume + boundary
```

The expression must be scalar and linear in `v`; it may be nonlinear in `u`.
Boundary labels restrict integration only. Essential conditions and other admissibility
constraints must be satisfied by the selected basis. The API declares
supported constraints through `Space` before preparing that basis.

## Bases and differentiation

Available basis families include geometry-adapted Laplacian modes, total-degree
polynomials, real Fourier modes, full finite-element bases and custom callable bases.
The operational basis is always fixed and numerically orthonormal in `L2`, summed
over components. Nodal families support `ZeroTrace`, `Periodic` and `MeanZero`;
arbitrary callbacks, polynomials and Fourier families do not automatically provide
these restriction constructors. See the [family support table](docs/usage.md#bases-del-espacio-admisible).

The field preserves PyTorch automatic differentiation. Continuing the quick start:

```python
z = z0
w = torch.ones_like(z)
J = torch.func.jacrev(G)(z)
_, Jw = torch.func.jvp(G, (z,), (w,))
```

The geometry, basis and fixed spatial coefficients remain outside the differentiation
graph.

## Documentation and examples

- [Complete Space-to-field acceptance examples](docs/acceptance-examples.md): torus,
  mixed plate, coupled components and periodic mean-zero diffusion, with executable
  numerical checks and explicit distinctions between PDE and discrete references.
- [Usage guide](docs/usage.md)
- [Migration from 0.9.0 and compatibility](docs/migration.md)
- [Mathematical contract](docs/mathematics.md)
- [Accepted design decisions](docs/design-contract.md)
- [Periodic and fixed-boundary recipes](docs/periodic-and-fixed-boundaries.md)
- Existing-interface examples: [general ND problem](examples/general_problem.py),
  [embedded torus](examples/embedded_torus.py), [discontinuous-data projection](examples/discontinuous_projection.py),
  [time evolution and diagnostics](examples/time_evolution.py).

## Current scope

- Simplices are affine; curved geometry is represented by a simplicial approximation.
- Geometry and operator data remain fixed in time. Nonhomogeneous fixed Dirichlet
  data use an explicit stationary lift; there is no automatic affine-space object.
- `Space.basis` supports requested regularity 0 or 1. Available elementwise higher
  derivatives do not certify global H2 conformity. Arbitrary tensor value shapes
  remain available through an explicit `GalerkinProblem`.
- Discontinuous data are currently projected into a continuous fixed space in `L2`.
  Interior-facet DG operators are not yet implemented.
- RK4 and RK45 are explicit. Stiff diffusive systems may require small steps; implicit
  and IMEX integrators remain a future extension.
- The error methods are refinement indicators, not certified a posteriori bounds.

The [usage guide](docs/usage.md) documents validation, memory limits and persistence.

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
