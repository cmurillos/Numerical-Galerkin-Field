# Changelog

All notable changes to Numerical Galerkin Field are documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `Space`: a frozen description of geometry, state components and requested Sobolev
  regularity under the approved D-013 contract.
- `ZeroTrace` and `Space.restrict`: component-wise homogeneous trace constraints on
  built-in nodal finite-element bases and their linear, component and product
  combinations. Constraints cover whole faces at arbitrary polynomial degree;
  unsupported representations and zero or dependent candidate spaces are rejected.
  The result is prepared for the existing L2 orthonormalization and field API.

- `Space.basis`: geometry-only preparation of fixed, component-explicit L2 bases.
  Laplacian modes are selected after full nodal constraint elimination; `size` is
  always the total dimension and optional `component_sizes` specifies allocation.
  Full finite-element and custom admissible spans are preserved without truncation.
  Polynomial/Fourier families work without extra restrictions; unsupported trace
  combinations are rejected. Custom source regularity distinguishes known
  conformity from user declarations. Existing field and basis APIs remain available.

## [0.9.0] - 2026-09-05

First public beta release.

### Added

- Affine simplicial geometries in arbitrary intrinsic and ambient dimensions.
- Named volume regions and exterior boundaries, including embedded manifolds.
- One complete weak-form language with scalar, vector and tensor operations.
- Fixed functional, cellwise and nodal spatial coefficients.
- Fixed real `L2`-orthonormal Laplacian, polynomial, Fourier, finite-element and custom
  bases.
- Batched differentiable Galerkin fields `G: [...,N] -> [...,N]` with CPU and CUDA
  support.
- Automatic, fixed and adaptive quadrature.
- Projection, reconstruction, gradient and Hessian evaluation at physical points.
- Continuous `L2` approximation of discontinuous data.
- Fixed RK4 and adaptive Dormand--Prince RK45 time integration.
- Spatial, temporal and quadrature refinement indicators.

### Known limitations

- Interior-facet discontinuous Galerkin operators are not implemented.
- Curved elements, implicit solvers and IMEX solvers remain future extensions.
- Numerical error methods are refinement indicators rather than certified bounds.

[0.9.0]: https://github.com/cmurillos/Numerical-Galerkin-Field/releases/tag/v0.9.0
