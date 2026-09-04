"""L2-orthonormal Laplacian modes, separated by physical component."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.sparse.linalg import eigsh

from .fem import FEMSpace
from .problem import Problem


def _eigenmodes(stiffness, mass, count, tolerance):
    size = mass.shape[0]
    if count < 1 or count > size:
        raise ValueError(f"Requested {count} modes but only {size} unconstrained DOFs exist.")
    if size <= 256 or count >= size - 1:
        values, vectors = eigh(stiffness.toarray(), mass.toarray(), subset_by_index=(0, count - 1))
    else:
        # A negative shift also handles Neumann zero modes without inverting K itself.
        scale = float(np.max(stiffness.diagonal() / mass.diagonal()))
        shift = -1e-8 * max(scale, np.finfo(float).tiny)
        values, vectors = eigsh(
            stiffness,
            k=count,
            M=mass,
            sigma=shift,
            which="LM",
            tol=tolerance,
            v0=np.random.default_rng(0).normal(size=size),
        )
    # Restore the mass metric and diagonalize K in that same subspace.
    factor = cholesky(vectors.T @ (mass @ vectors), lower=False)
    vectors = solve_triangular(factor.T, vectors.T, lower=True).T
    reduced = vectors.T @ (stiffness @ vectors)
    values, rotation = eigh((reduced + reduced.T) / 2)
    vectors = vectors @ rotation
    scale = max(1.0, float(np.max(np.abs(values))))
    if values.min() < -1e-9 * scale:
        raise RuntimeError("The reference Laplacian has a significantly negative eigenvalue.")
    values = np.maximum(values, 0.0)
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(count)])
    vectors *= signs
    return values, vectors


@dataclass(frozen=True)
class GalerkinBasis:
    """Numerical modes in the FEM mass metric, stored once for reproducibility.

    coefficients[r] has shape [scalar_ndofs, modes[r]]. Conceptually the full
    vector-valued synthesis matrix is block diagonal, not a scalar concatenation.
    Modes are ordered by component, then increasing reference eigenvalue.
    """

    fem: FEMSpace
    coefficients: tuple[np.ndarray, ...]
    eigenvalues: tuple[np.ndarray, ...]
    dirichlet: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        if not self.coefficients or not (
            len(self.coefficients) == len(self.eigenvalues) == len(self.dirichlet)
        ):
            raise ValueError("Provide coefficients, eigenvalues and boundaries for each component.")
        columns, eigenvalues = [], []
        boundaries = tuple(tuple(sorted(set(b))) for b in self.dirichlet)
        for c, values, names in zip(self.coefficients, self.eigenvalues, boundaries):
            c = np.array(c, dtype=np.float64, copy=True)
            values = np.array(values, dtype=np.float64, copy=True)
            if c.ndim != 2 or c.shape[0] != self.fem.ndofs or c.shape[1] == 0:
                raise ValueError("Each coefficient matrix must have shape [scalar_ndofs, modes].")
            if values.shape != (c.shape[1],) or not np.isfinite(c).all():
                raise ValueError("Invalid mode coefficients or eigenvalue shape.")
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values) < 0):
                raise ValueError("Reference eigenvalues must be finite, nonnegative and ordered.")
            gram = c.T @ (self.fem.mass @ c)
            if not np.allclose(gram, np.eye(c.shape[1]), rtol=0, atol=1e-8):
                raise ValueError("Modes are not orthonormal in the FEM mass metric.")
            essential = self.fem.essential_dofs(names)
            if essential.size and np.max(np.abs(c[essential])) > 1e-10:
                raise ValueError("Modes violate the homogeneous essential boundary condition.")
            c.setflags(write=False)
            values.setflags(write=False)
            columns.append(c)
            eigenvalues.append(values)
        object.__setattr__(self, "coefficients", tuple(columns))
        object.__setattr__(self, "eigenvalues", tuple(eigenvalues))
        object.__setattr__(self, "dirichlet", boundaries)

    @classmethod
    def build(cls, fem: FEMSpace, problem: Problem, modes, tolerance=1e-10) -> "GalerkinBasis":
        """Build modes per component; an integer requests that many for every component.

        Dirichlet DOFs are eliminated exactly. The remaining boundary has the natural
        Neumann condition for this auxiliary eigenproblem, irrespective of the real
        operator's natural boundary load. Repeated essential DOF sets share the solve.
        """
        if isinstance(modes, Integral) and not isinstance(modes, bool):
            counts = (int(modes),) * problem.components
        else:
            try:
                counts = tuple(modes)
            except TypeError as exc:
                raise ValueError(
                    "modes must be a positive integer or one count per component."
                ) from exc
        if len(counts) != problem.components or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in counts
        ):
            raise ValueError("Provide one positive mode count per component.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        keys = [tuple(fem.essential_dofs(names)) for names in problem.dirichlet]
        requests = {}
        for key, count in zip(keys, counts):
            requests[key] = max(requests.get(key, 0), count)
        solved = {}
        for key, count in requests.items():
            free = np.setdiff1d(np.arange(fem.ndofs), key)
            if not free.size:
                raise ValueError("There are no unconstrained DOFs; refine the mesh.")
            k = fem.stiffness[free][:, free]
            m = fem.mass[free][:, free]
            values, vectors = _eigenmodes(k, m, count, tolerance)
            c = np.zeros((fem.ndofs, count))
            c[free] = vectors
            solved[key] = (values, c)
        coefficients, eigenvalues = [], []
        for key, count in zip(keys, counts):
            values, c = solved[key]
            coefficients.append(c[:, :count])
            eigenvalues.append(values[:count])
        return cls(fem, tuple(coefficients), tuple(eigenvalues), problem.dirichlet)

    @property
    def components(self) -> int:
        return len(self.coefficients)

    @property
    def modes(self) -> tuple[int, ...]:
        return tuple(c.shape[1] for c in self.coefficients)

    @property
    def dimension(self) -> int:
        """Total reduced dimension N, distinct from scalar mesh DOFs and components."""
        return sum(self.modes)

    @property
    def slices(self) -> tuple[slice, ...]:
        offsets = np.cumsum((0,) + self.modes)
        return tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))

    def diagnostics(self) -> dict:
        """Report mass orthogonality and backward eigen-residuals on unconstrained DOFs."""
        reports = []
        for c, values, names in zip(self.coefficients, self.eigenvalues, self.dirichlet):
            essential = self.fem.essential_dofs(names)
            free = np.setdiff1d(np.arange(self.fem.ndofs), essential)
            k = self.fem.stiffness[free][:, free]
            m = self.fem.mass[free][:, free]
            v = c[free]
            residual = k @ v - (m @ v) * values
            knorm = float(np.max(np.asarray(np.abs(k).sum(axis=1))))
            mnorm = float(np.max(np.asarray(np.abs(m).sum(axis=1))))
            denominator = (knorm + np.abs(values) * mnorm) * np.linalg.norm(v, axis=0)
            relative = np.linalg.norm(residual, axis=0) / np.maximum(
                denominator, np.finfo(float).tiny
            )
            reports.append(
                {
                    "mass_error": float(
                        np.max(np.abs(c.T @ (self.fem.mass @ c) - np.eye(c.shape[1])))
                    ),
                    "eigen_residual": float(relative.max()),
                    "essential_error": float(np.max(np.abs(c[essential])))
                    if essential.size
                    else 0.0,
                }
            )
        return {"dimension": self.dimension, "modes": self.modes, "components": reports}
