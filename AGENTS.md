# AGENTS.md

## Purpose and scope

This file gives coding agents the repository-specific context needed to use, explain,
test, and modify Numerical Galerkin Field correctly. It applies to the entire repository.
It is intentionally more prescriptive than the README: the library represents mathematical
objects whose meaning can be changed by a seemingly harmless shape conversion, basis
normalization, sign change, or quadrature shortcut.

The repository is the numerical layer of a two-package project:

- **Numerical Galerkin Field (`ngfield`)** owns fixed geometry, admissible spaces, bases,
  weak forms, numerical Galerkin fields, projection/reconstruction, explicit time
  integration, and numerical refinement indicators.
- **Galerkin Neural Semigroup (`galerkin_neural_semigroup`)** is a downstream consumer. It
  owns neural approximation, sampling, training, neural checkpoints, and learned flows.

Do not add neural networks, optimizers, training losses, or data samplers here. Do not move
Galerkin assembly into the neural package. A cross-repository change must preserve this
boundary and be verified against the exact downstream dependency revision.

## Instruction and source-of-truth order

Follow, in order:

1. The user's explicit request.
2. Accepted decisions in `docs/design-contract.md`.
3. Executable contracts in `tests/` and acceptance programs in `examples/acceptance_*.py`.
4. The mathematical interpretation in `docs/mathematics.md`.
5. The public workflow in `README.md` and `docs/usage.md`.
6. Existing implementation details.

If these disagree, do not silently choose the most convenient interpretation. Identify the
inconsistency, determine whether the task is a bug fix or a contract change, and keep code,
tests, documentation, examples, and changelog aligned. Accepted D-001 through D-013
decisions do not change incidentally as part of refactoring.

The source tree currently contains implemented D-013 additions while the declared package
version remains `0.9.0`. The PyPI 0.9.0 release does not contain all source-tree APIs. Never
claim that source-only behavior is available from PyPI until the release metadata and
changelog say so.

## First actions for every task

Before editing:

1. Read `README.md` and the sections of `docs/design-contract.md` relevant to the task.
2. Read the corresponding implementation and tests; search with `rg` rather than guessing
   names or signatures.
3. Inspect `git status --short` and preserve unrelated user changes.
4. State the mathematical object being changed, its domain/codomain, tensor shapes, fixed
   data, and properties that must remain invariant.
5. Prefer the modern `Space -> basis -> GalerkinField` route for new user code. Use an older
   route only for a compatibility task.
6. Reproduce a reported defect with the smallest relevant test or script before changing
   behavior.

Ask a question only when a missing choice changes the mathematical problem—for example,
the weak sign convention, boundary condition, component allocation, basis family, or
meaning of a physical coefficient. Do not ask users to choose internal helper classes,
quadrature tables, mass-matrix plumbing, or implementation details that the public API
already fixes.

## Core mathematical contract

The central object is a fixed autonomous reduced field

```text
Phi(z) = sum_j z_j phi_j,
G: R^N -> R^N,
G_i(z) = a(Phi(z); phi_i).
```

The operational basis is real, ordered, fixed, and numerically orthonormal in the physical
`L2` inner product (summed over state components). Consequently, reduced Euclidean norms
represent discrete `L2` norms and no post-hoc mass matrix is applied to `G`. Do not insert
`M^-1`, change coordinates, reorder modes, or renormalize a basis inside field evaluation.
Any coordinate transform must be explicit, produce a new basis, and be validated.

The geometry, basis, spatial coefficients, weak form, and quadrature are fixed when `G` is
constructed. Calling `G(z)` evaluates the already prepared field. It must not remesh,
resample, adapt quadrature, rebuild a basis, mutate coefficients, or advance time.

The library discretizes the supplied weak form; it does not infer the intended PDE. A sign
in `weak` is part of the model. For example, heat written as `u_t = kappa Delta u` normally
uses `-kappa * inner(grad(u), grad(v)) * dx` after integration by parts under the package's
field convention.

## Canonical public workflow

Use this route in new examples and user-facing answers:

```python
import torch

from ngfield import GalerkinField, SimplicialDomain, Space, ZeroTrace, grad, inner

geometry = SimplicialDomain(
    vertices=vertices,  # [number_of_vertices, ambient_dimension]
    simplices=simplices,  # [number_of_cells, intrinsic_dimension + 1]
    regions=regions,  # optional named cell subsets
    boundaries=boundaries,  # optional named exterior-face subsets
)

V = Space(
    geometry=geometry,
    components=1,
    regularity=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)

basis = V.basis("laplacian", size=N, degree=1)


def weak(u, v, dx, ds):
    return -kappa * inner(grad(u[0]), grad(v[0])) * dx


G = GalerkinField(basis=basis, weak=weak)

z0 = G.project(initial_state)
times = torch.linspace(t0, t1, steps, dtype=G.dtype, device=G.device)
Z = G.solve(z0, times)
U = G.reconstruct(Z, points)
```

In the `Space` route, physical components are explicit even for scalar problems:
`components=1`, `basis.value_shape == (1,)`, and weak forms use `u[0]` and `v[0]`.
An initial-state callable for a scalar `Space` therefore returns `[Q,1]`, not `[Q]`.

The public `ngfield.GalerkinField` name is a dispatching constructor, not a single class. It
routes modern/general calls to `ngfield.galerkin.GalerkinField` and original legacy calls to
`ngfield.field.GalerkinField`. Do not rely on `isinstance(x, ngfield.GalerkinField)`. Preserve
all dispatch paths when changing constructors.

## Geometry contract

`SimplicialDomain` represents an affine simplicial complex:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `vertices` | `[M,p]` | Vertices in ambient `R^p`. |
| `simplices` | `[E,k+1]` | Intrinsic `k`-simplices, with `1 <= k <= p`. |
| `regions` | named cell subsets | Selectors for `dx("name")`. |
| `boundaries` | named exterior-face subsets | Selectors for `ds("name")`. |

Intrinsic and ambient dimensions are inferred; never request them redundantly. Embedded
curves and surfaces are supported, including `k=2, p=3` triangulated surfaces. Their measure
is induced by the embedding, spatial derivatives are elementwise tangential derivatives
expressed in ambient coordinates, and `ds.normal` is the outward conormal inside the parent
simplex. The core is unoriented and does not provide an oriented ambient surface normal.

Boundary and region labels only select integration sets. A label named `fixed`, `wall`, or
`periodic` does not impose a condition. The reserved label `all` denotes the complete set.
Exterior boundary facets are derived from connectivity; interior facets are not currently a
weak-form measure. Do not fake curved elements, interior traces, or manifold orientation with
ambient-coordinate heuristics.

## Spaces, restrictions, and bases

`Space` is a frozen declaration of geometry, physical components, requested Sobolev
regularity, and homogeneous restrictions. It has no reduced dimension until a basis is
chosen. `size=N` always means the **total** reduced dimension, not modes per component.
Use `component_sizes=(n0, ..., nc_minus_1)` only when an explicit allocation is required.

Supported restrictions are:

- `ZeroTrace(component=r, boundary=name)`: homogeneous trace on a nonempty exterior label;
  requires `regularity >= 1`.
- `Periodic(component=r, boundaries=(a,b), vertex_pairs=pairs)`: complete nodal trace
  identification under a boundary-vertex bijection preserving face connectivity. It does
  not rotate components, add phase shifts, or match incompatible meshes.
- `MeanZero(component=r, region=name_or_none)`: zero induced-measure integral over a named
  volume region or the whole domain. It is not a nodal arithmetic mean.

Restrictions are enforced before spectral mode selection. Never build unrestricted modes
and then zero, identify, or discard selected coordinates as a substitute. `V.restrict(raw)`
constructs an admissible nodal span but does not select a mode count or orthonormalize it.

Basis-family expectations:

| Family | Intended behavior | Restriction support |
| --- | --- | --- |
| `laplacian` | Lowest geometry-adapted FEM modes. | Built-in nodal restrictions. |
| `finite-element` | Complete admissible Lagrange space. | Built-in nodal restrictions. |
| `polynomial` | Total-degree global monomials, then L2 orthonormalization. | No declarative trace/periodic/mean restriction. |
| `fourier` | Real sine/cosine family, then L2 orthonormalization. | No implicit periodicity or declarative restriction. |
| `custom` | Supplied family, restricted when its nodal representation is certifiable, then orthonormalized. | Only when the representation supports certification. |

Do not infer global `H2` conformity from elementwise Hessians. Current `Space.basis` supports
requested regularity zero or one. Callback regularity may be user-declared but is not thereby
verified. Preserve metadata that distinguishes verified properties from declarations.

Repeated eigenvalues permit rotations of eigenspaces. For reproducibility, reuse or persist
the concrete ordered basis rather than rebuilding it and assuming identical coordinates.

## Weak-form language

A weak form has the exact signature

```python
def weak(u, v, dx, ds):
    return ...
```

It returns a scalar sum of volume and exterior-boundary integrals. Every nonzero integrand
must depend **exactly linearly** on the test function `v`; dependence on `u` may be nonlinear.
Keep the expression in the package language instead of evaluating tensors manually.

Use:

- `dx` or `dx("region")` for volume integration;
- `ds` or `ds("boundary")` for exterior-boundary integration;
- `dx.x` / `ds.x` for physical coordinates and `ds.normal` for the conormal;
- `grad`, `inner`, `contract`, `dot`, `outer`, `transpose`, `trace`, `div`, `sym_grad`,
  `stack`, and exported unary functions for supported expressions;
- `pointwise(function, *values, shape=...)` only for vectorized PyTorch operations that do
  not depend on `v`.

`Coefficient(function, shape=...)`, `Coefficient.cell(values)`, and
`Coefficient.vertex(values)` are fixed spatial operator data. They are tabulated and detached
at field construction. Do not make them trainable or mutate a constructed field by changing
captured source data. Spatial differentiation of an external `Coefficient` or opaque
`pointwise` callback is not automatic; express the derivative separately when the model
needs it.

`ds.interior`, jumps, averages, two-sided traces, numerical fluxes, and DG penalties are not
implemented. Do not emulate them with exterior `ds` or an arbitrary choice of parent cell.

Neumann and Robin data belong in the weak form. Homogeneous Dirichlet data belong in the
admissible basis. Fixed nonhomogeneous Dirichlet data require an explicit stationary lift
`physical_state = lift + Phi(z)`: project `initial_state - lift`, use the lifted state wherever
the weak form requires the physical state, and add the lift after reconstruction. There is no
automatic affine-space or lift object. Time-dependent boundary data are outside the current
autonomous contract.

## Tensor, dtype, and device contract

Use the following shapes without implicit squeezing:

| Operation | Input | Output |
| --- | --- | --- |
| `G(z)` | `[*S,N]` | `[*S,N]` |
| `G.project(source)` | source values `[*S,Q,*value_shape]` | `[*S,N]` |
| `G.reconstruct(z, points)` | `[*S,N]`, points `[Q,p]` | `[*S,Q,*value_shape]` |
| `G.grad(z, points)` | same | `[*S,Q,*value_shape,p]` |
| `G.hessian(z, points)` | same | `[*S,Q,*value_shape,p,p]` |
| `G.solve(z0, times)` | `z0:[*S,N]`, `times:[T]` | `[T,*S,N]` |

All axes before the final coordinate axis are batch axes and must be preserved, including
multiple leading axes and zero-sized batches. Do not special-case `[B,N]` in a way that
breaks `[N]`, `[T,B,N]`, or empty batches.

Core field operations require PyTorch tensors with the same real `float32` or `float64`
dtype and device as the field. Do not add silent list/NumPy conversion, device transfer, or
precision coercion to strict numerical paths. Use `G.to(device=..., dtype=...)` to move all
fixed tables together. Construction may deliberately use CPU/float64 for robust algebra;
evaluation must preserve the selected device and dtype.

Legacy scalar bases have `value_shape=()`. The modern `Space(components=1)` route has
`value_shape=(1,)`. Preserve both; never silently promote or squeeze one into the other.

## Quadrature contract

One argument has three unambiguous meanings:

- `quadrature=None`: infer an exact piecewise-polynomial order when possible; otherwise
  prepare adaptively.
- `quadrature=q` where `q` is a nonnegative integer: fixed polynomial exactness order.
- `quadrature=tol` where `0 < tol < 1` is real: adaptive preparation tolerance.

Booleans, strings, and reals greater than or equal to one are invalid. Adaptive preparation
occurs only while constructing a field or performing the requested projection/error
operation. `G(z)` never adapts. It refines quadrature order, not the mesh, and its result is a
deterministic numerical estimate rather than a uniform analytic bound. Preserve point and
intermediate-entry budgets and fail explicitly when exhausted.

## Projection, reconstruction, and differentiation

Because the basis is L2-orthonormal, `G.project(u)` computes coordinates by inner products;
it does not solve a new mass system. Accept callables and `Coefficient` representations, not
unlabelled raw quadrature arrays. A callable may preserve autograd with respect to its own
PyTorch parameters; fixed `Coefficient` data remain detached.

`reconstruct`, `grad`, and `hessian` evaluate `Phi(z)`, not `G` in physical space. On a shared
facet, values or derivatives can have multiple element traces. If they disagree, require an
explicit `cells` selection rather than choosing a side by mesh numbering. Derivatives are
elementwise and tangential on embedded meshes.

`G` must remain differentiable with respect to `z` through ordinary autograd and
`torch.func.jacrev`, `jvp`, and `vjp`. Geometry, basis tables, quadrature, and fixed
coefficients are intentionally outside that graph. Avoid in-place state operations, Python
data-dependent branching on differentiable values, and callbacks with side effects.

For the Sobolev coordinate objective, call `G.state_derivatives(z, k)` or the
generic `ngfield.state_derivatives(f,z,k)`. Keys are all multi-indices of total
degree at most `k`, with each mixed partial included once. This operation is
about state coordinates and never implies spatial `H2` conformity.

The optional D-016 function-valued route is `System(basis,weak).state(u0)`
followed by `system.evolve(initial,times)`. `State.velocity()` and
`State.indexed_derivatives(k)` reconstruct functions in the fixed basis;
the latter differentiates the reduced field with respect to state coordinates.
`Function.gradient/hessian` remain spatial, elementwise operations.

## Time integration and numerical indicators

`G.solve(z0, times, order=p)` uses adaptive Taylor jets (default `p=4`).
`G.solve(..., step=h, order=p)` uses a fixed maximum Taylor step. `step` and
`tolerance` cannot be combined. Times may increase or decrease and match the
field's dtype/device. `ngfield.integrate_field` is the same engine used by the
downstream neural package: use the same order and time controls for comparisons.
The optional `radius=R` stops trajectories at the open-ball boundary with
`DomainExitError`, including the last accepted interior state/time. The numerical
exit time is approximate; no exterior field evaluation occurs.

Both methods are explicit. Do not describe them as suitable for stiff diffusion merely
because they return a result. If a run is unstable or expensive, expose the stiffness/time
step issue rather than weakening finite checks or silently clipping the state.

The methods `projection_error`, `time_error`, and `quadrature_error` are absolute refinement
**indicators**, not certified a posteriori error bounds. Preserve that wording. There is no
universal energy, conservation law, positivity condition, or invariant-ball certificate for
an arbitrary weak form; problem-specific observables require problem-specific checks.

## Compatibility routes

Maintain all three construction families unless an explicitly approved breaking release
removes one:

1. Modern: `GalerkinField(basis=V.basis(...), weak=weak)`.
2. General: `problem = GalerkinProblem(...); problem.field(basis=basis)` or
   `GalerkinField(problem, basis)`.
3. Original legacy: `GalerkinField(legacy_basis, legacy_problem)` using `GalerkinBasis` and
   `Problem`.

New tutorials should use route 1. Route 2 is appropriate for general tensor-valued or custom
bases beyond the component-vector `Space` workflow. Route 3 is compatibility-only. When
editing dispatch, projection, persistence, or shape validation, add regression coverage for
every affected route.

## Repository map

- `src/ngfield/geometry.py`: ND affine simplicial geometry, labels, quadrature, tangents,
  conormals, and point location data.
- `src/ngfield/space.py`: immutable admissible-space declaration.
- `src/ngfield/restrictions.py`: certified nodal `ZeroTrace`, `Periodic`, and `MeanZero`
  kernels.
- `src/ngfield/spaces.py`: basis protocol and callable, polynomial, component, product,
  transformed, and finite-element bases.
- `src/ngfield/space_bases.py`: `Space.basis` preparation and component allocation.
- `src/ngfield/basis_factory.py`: geometry-based basis factories for the general route.
- `src/ngfield/forms.py`: symbolic weak-expression language and numerical evaluation.
- `src/ngfield/galerkin.py`: modern/general field construction, fixed tables, field calls,
  projection, spatial evaluation, diagnostics, and public coordinate behavior.
- `src/ngfield/evolution.py`: shared Taylor integration and temporal refinement indicator.
- `src/ngfield/state_derivatives.py`: multi-indices and state-coordinate derivatives.
- `src/ngfield/workflow.py`: function-valued System, State, Solution and grouped metrics.
- `src/ngfield/basis.py`, `domain.py`, `fem.py`, `field.py`, `problem.py`, `io.py`: original
  API and compatibility machinery; do not delete as dead code.
- `src/ngfield/__init__.py`: public exports and `GalerkinField` dispatch.
- `docs/design-contract.md`: accepted design decisions D-001 through D-013.
- `docs/usage.md`: detailed public recipes and support tables.
- `docs/mathematics.md`: mathematical identities versus numerical checks.
- `docs/migration.md`: source/PyPI and modern/general/legacy migration rules.
- `docs/acceptance-examples.md` and `examples/acceptance_*.py`: end-to-end scientific
  acceptance cases.

## Task playbooks

### Write a user example or notebook

1. Define the PDE and weak sign convention in one or two equations.
2. Identify intrinsic/ambient dimension, components, fixed geometry, and named subsets.
3. Put only homogeneous admissibility constraints in `Space`.
4. Choose a basis family, total size, degree, and quadrature for a stated reason.
5. Build `G` through the modern public API; never import private modules.
6. Project the initial state, integrate, and reconstruct with explicit shapes.
7. Include at least one independent mathematical or refinement check.
8. Keep visualization code outside the numerical core and do not reimplement library
   operators in the notebook.

### Add or change a feature

1. Locate the governing D-number. If none exists, write or revise the design contract before
   presenting the behavior as stable.
2. Add failing tests for mathematical semantics, shapes, invalid inputs, dtype/device, and
   immutability as applicable.
3. Implement through the narrowest owning module; avoid parallel implementations for modern
   and general routes.
4. Update exports only for deliberately public names.
5. Update README/usage/mathematics/migration/changelog wherever the public claim changes.
6. Run targeted tests, then the complete verification sequence.

### Diagnose a numerical mismatch

Separate, in this order:

1. Modeling/weak-form sign and boundary terms.
2. Geometry and labels.
3. Admissible space and basis restrictions.
4. Basis orthonormality and coordinate ordering.
5. Projection error.
6. Quadrature error.
7. Reduced ODE time-integration error.
8. Difference between the discrete Galerkin model and the continuous PDE reference.

Never call a Galerkin-vs-PDE discrepancy a quadrature or time error without isolating it.
Warnings and nonfinite values should be traced to their cause, not hidden globally.

### Optimize performance

Profile field construction separately from repeated `G(z)` calls. Preserve exact output,
batch semantics, autograd, validation, and budgets. Avoid device synchronizations and Python
loops over batch states in hot evaluation paths. A dense representation is an implementation
detail, but replacing it requires parity tests and memory measurements. Do not trade away
exact spectral/basis semantics for speed without an explicit contract change.

## Testing and verification

Install from the repository root with Python 3.11 or newer:

```bash
python -m pip install -e ".[dev]"
```

Use targeted tests while iterating:

| Change area | Minimum focused coverage |
| --- | --- |
| Geometry, labels, embedded meshes | `tests/test_geometry_nd.py` |
| Weak forms and coefficients | `tests/test_weak_forms.py` |
| Bases and orthonormality | `tests/test_basis_contract.py`, `tests/test_space_basis_contract.py` |
| Restrictions | `tests/test_restriction_contract.py`, `tests/test_periodic_mean_contract.py` |
| Field shapes/autograd/dispatch | `tests/test_field_contract.py`, `tests/test_unified_field_contract.py` |
| Projection/spatial evaluation | `tests/test_projection_contract.py`, `tests/test_spatial_evaluation_contract.py` |
| Quadrature | `tests/test_quadrature_contract.py` |
| Evolution/errors | `tests/test_evolution_contract.py`, `tests/test_error_contract.py` |
| End-to-end scientific behavior | `tests/test_acceptance_examples.py`, `tests/test_convergence.py` |

Before declaring a repository change complete, run the CI-equivalent checks:

```bash
ruff check .
ruff format --check .
python -m pytest
python examples/general_problem.py
python examples/embedded_torus.py
python examples/discontinuous_projection.py
python examples/time_evolution.py
python examples/heterogeneous_coefficient.py
python examples/diffusion.py
python examples/reaction_diffusion.py
python -m build
```

For performance-sensitive changes, also run the small benchmark command used by CI. CUDA
behavior must be covered when a compatible device is available, but CPU-only development
must not weaken CUDA invariants or mark untested CUDA behavior as verified.

Do not update tolerances merely to make a regression pass. Explain why the tolerance follows
from discretization, quadrature, conditioning, dtype, or integrator order. Prefer analytic
small cases and independent assembly checks over self-comparison against the same code path.

## Public API and documentation discipline

- Public names live in `src/ngfield/__init__.py`; underscore/private helpers are not user API.
- Add public parameters only when the user must make that mathematical choice. Infer or hide
  mechanical details when the contract already determines them.
- Validate invalid and ambiguous inputs early with informative errors. Never silently weaken
  a restriction, drop a component, change a basis, or choose an arbitrary trace.
- Preserve the language and style of the edited document: README and API prose are currently
  English; detailed contracts/guides are currently Spanish. Do not translate unrelated text.
- Examples must be executable and use public imports. Include them in tests when they make a
  scientific acceptance claim.
- Distinguish exact identities, discrete identities, quadrature verification, refinement
  evidence, and continuous-PDE convergence claims.
- Add an `Unreleased` changelog entry for user-visible behavior. Do not bump versions,
  publish packages, create releases, or push changes unless the user's request authorizes it.

## Current non-goals and prohibited shortcuts

Do not claim or silently implement any of the following under the existing contract:

- time-dependent geometry, coefficients, weak forms, or boundary data;
- curved/isoparametric elements rather than affine simplicial approximations;
- automatic nonhomogeneous Dirichlet lifts;
- interior-facet DG operators;
- automatic certification of higher Sobolev conformity;
- implicit or IMEX time integrators;
- certified a posteriori bounds;
- automatic positivity, energy conservation, global existence, or invariant-ball guarantees;
- neural approximation or learning logic;
- raw quadrature-value inputs without their geometric meaning;
- implicit mass-matrix corrections, dtype/device coercions, or component squeezing.

A request for one of these is a design-extension task, not a small patch. Explain the missing
mathematical contract, propose the minimal extension, and wait for approval when it would
change the package's declared scope.

## Code review rules

Flag any change that:

- alters `G_i(z)=a(Phi(z);phi_i)` or the L2 coordinate metric;
- changes mode order, basis normalization, or restriction order without migration;
- makes field evaluation stateful or adaptive;
- loses arbitrary leading batch axes, empty batches, dtype/device checks, or autograd;
- treats boundary labels as boundary conditions;
- lets a weak form be nonlinear in `v`;
- exposes unsupported continuous-PDE guarantees;
- duplicates modern/general/legacy numerical machinery;
- adds a public API without documentation and negative tests;
- weakens finite checks, memory budgets, or numerical tolerances without evidence.

## Definition of done

A task is complete only when:

- the implementation matches the governing mathematical and design contract;
- focused regression tests cover the changed behavior and failure modes;
- the full relevant suite passes without hiding warnings;
- public examples use exact current signatures and tensor shapes;
- documentation and changelog match the implementation;
- compatibility and downstream GNS implications have been considered;
- the final report states what changed, what was run, and any unverified limitation.
