"""Affine simplicial geometry and positive quadrature in arbitrary dimension."""

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from types import MappingProxyType

import numpy as np
from scipy.special import roots_jacobi


def positive_integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@lru_cache(maxsize=32)
def reference_quadrature(dimension, order, max_points=1_000_000):
    """Barycentric points and weights on the unit simplex (volume 1/d!).

    A Duffy map with Gauss-Jacobi weights integrates total-degree ``order``
    polynomials exactly in exact arithmetic. No dimension-specific rule tables.
    Dimension zero uses counting measure, needed for endpoints in 1D.
    """
    d = positive_integer(dimension, "dimension", 0)
    order = positive_integer(order, "order", 0)
    max_points = positive_integer(max_points, "max_points")
    count = (order + 2) // 2
    if count**d > max_points:
        raise ValueError("Quadrature exceeds max_points; lower order or supply a custom rule.")
    if d == 0:
        return readonly([[1.0]]), readonly([1.0])
    axes, weights = [], []
    for i in range(d):
        alpha = d - 1 - i
        x, w = roots_jacobi(count, alpha, 0)
        axes.append((x + 1) / 2)
        weights.append(w / 2 ** (alpha + 1))
    t = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    w = np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1), axis=-1)
    remaining = np.ones(len(t))
    bary = np.empty((len(t), d + 1))
    for i in range(d):
        bary[:, i + 1] = remaining * t[:, i]
        remaining *= 1 - t[:, i]
    bary[:, 0] = remaining
    return readonly(bary), readonly(w.reshape(-1))


@dataclass(frozen=True)
class SimplexQuadrature:
    points: np.ndarray  # [Q, ambient_dimension]
    weights: np.ndarray  # [Q], physical volume/surface measure
    cells: np.ndarray  # [Q], parent cells, also on the boundary
    barycentric: np.ndarray  # [Q, intrinsic_dimension+1], in parent cells
    normals: np.ndarray | None  # outward normals / element conormals


class SimplicialDomain:
    """Vertices [M,p], simplices [E,k+1], 1 <= k <= p.

    Supports full-dimensional domains and embedded piecewise affine manifolds.
    Inputs must form a conforming complex. Structural and local incidence checks
    are performed; arbitrary intersections between distant cells are not detected.
    Boundary labels contain vertex-index faces [F,k], or barycenter predicates.
    Region labels contain cell indices, masks, cell rows, or barycenter predicates.
    """

    def __init__(self, vertices, simplices, boundaries=None, regions=None):
        p = np.asarray(vertices)
        t = np.asarray(simplices)
        if p.dtype.kind not in "fiu" or p.ndim != 2 or not p.size:
            raise ValueError("vertices must be a nonempty real matrix [M,p].")
        if t.dtype.kind not in "iu" or t.ndim != 2 or not len(t):
            raise ValueError("simplices must be a nonempty integer matrix [E,k+1].")
        k = t.shape[1] - 1
        if not 1 <= k <= p.shape[1]:
            raise ValueError("Require 1 <= intrinsic dimension <= ambient dimension.")
        if not np.isfinite(p).all() or np.any(t < 0) or np.any(t >= len(p)):
            raise ValueError("Nonfinite vertices or invalid vertex indices.")
        canonical = np.sort(t, axis=1)
        if np.any(np.diff(canonical, axis=1) == 0):
            raise ValueError("A simplex repeats a vertex.")
        if len(np.unique(canonical, axis=0)) != len(t):
            raise ValueError("Duplicate simplices are not allowed.")
        self.vertices = readonly(p, np.float64)
        self.simplices = readonly(t, np.int64)
        self.dimension = k
        self.intrinsic_dimension = k
        self.ambient_dimension = p.shape[1]
        corners = self.vertices[t]
        edges = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        singular = np.linalg.svd(edges, compute_uv=False)
        if np.any(singular[:, -1] <= np.finfo(float).eps * max(edges.shape[1:]) * singular[:, 0]):
            raise ValueError("Degenerate or numerically rank-deficient simplex.")
        q, r = np.linalg.qr(edges, mode="reduced")
        self.jacobians = readonly(np.abs(np.linalg.det(r)))
        self.inverse_jacobians = readonly(np.linalg.solve(r, np.swapaxes(q, 1, 2)))
        self.tangent_projectors = readonly(np.einsum("eik,ejk->eij", q, q))
        gradients = np.concatenate(
            (-self.inverse_jacobians.sum(axis=1, keepdims=True), self.inverse_jacobians),
            axis=1,
        )
        self.barycentric_gradients = readonly(gradients)
        incidence = {}
        for cell, vertices_ in enumerate(t):
            for opposite in range(k + 1):
                face = tuple(sorted(np.delete(vertices_, opposite).tolist()))
                incidence.setdefault(face, []).append((cell, opposite))
        if any(len(owners) > 2 for owners in incidence.values()):
            raise ValueError("A face belongs to more than two cells (nonmanifold incidence).")
        if k == self.ambient_dimension:
            for owners in incidence.values():
                if len(owners) == 2:
                    (a, i), (b, j) = owners
                    if np.dot(gradients[a, i], gradients[b, j]) >= 0:
                        raise ValueError("Adjacent cells lie on the same side of a shared face.")
        exterior = [(face, owners[0]) for face, owners in incidence.items() if len(owners) == 1]
        self.exterior_faces = readonly([face for face, _ in exterior], np.int64).reshape(-1, k)
        self._owners = readonly([owner for _, owner in exterior], np.int64).reshape(-1, 2)
        labels = {"all": readonly(np.arange(len(exterior)), np.int64)}
        lookup = {face: i for i, (face, _) in enumerate(exterior)}
        for name, faces in (boundaries or {}).items():
            if not isinstance(name, str) or not name or name == "all":
                raise ValueError("Boundary names must be nonempty; 'all' is reserved.")
            if callable(faces):
                midpoints = self.vertices[self.exterior_faces].mean(axis=1)
                mask = np.asarray(faces(midpoints))
                if mask.shape != (len(exterior),) or mask.dtype.kind != "b":
                    raise ValueError("A boundary predicate must return a boolean [faces] array.")
                indices = np.flatnonzero(mask)
            else:
                faces = np.asarray(faces)
                if faces.ndim != 2 or faces.shape[1] != k or faces.dtype.kind not in "iu":
                    raise ValueError("Boundary faces must be integer vertex indices [F,k].")
                try:
                    indices = [lookup[tuple(sorted(face.tolist()))] for face in faces]
                except KeyError as exc:
                    raise ValueError("A boundary tag contains a non-exterior face.") from exc
            labels[name] = readonly(np.unique(indices), np.int64)
        self.boundaries = MappingProxyType(labels)

        cell_labels = {"all": readonly(np.arange(len(t)), np.int64)}
        cell_lookup = {tuple(row): i for i, row in enumerate(canonical)}
        centers = corners.mean(axis=1)
        for name, cells in (regions or {}).items():
            if not isinstance(name, str) or not name or name == "all":
                raise ValueError("Region names must be nonempty; 'all' is reserved.")
            if callable(cells):
                mask = np.asarray(cells(centers))
                if mask.shape != (len(t),) or mask.dtype.kind != "b":
                    raise ValueError("A region predicate must return a boolean [cells] array.")
                indices = np.flatnonzero(mask)
            else:
                cells = np.asarray(cells)
                if cells.dtype.kind == "b" and cells.shape == (len(t),):
                    indices = np.flatnonzero(cells)
                elif cells.dtype.kind in "iu" and cells.ndim == 1:
                    if np.any(cells < 0) or np.any(cells >= len(t)):
                        raise ValueError("A region contains an invalid cell index.")
                    indices = cells
                elif cells.dtype.kind in "iu" and cells.ndim == 2 and cells.shape[1] == k + 1:
                    try:
                        indices = [cell_lookup[tuple(sorted(cell.tolist()))] for cell in cells]
                    except KeyError as exc:
                        raise ValueError("A region contains a cell outside the domain.") from exc
                else:
                    raise ValueError(
                        "Regions must be cell indices, a boolean [cells] mask, "
                        "integer cells [R,k+1], or a predicate."
                    )
            cell_labels[name] = readonly(np.unique(indices), np.int64)
        self.regions = MappingProxyType(cell_labels)

    def same_mesh(self, other):
        return np.array_equal(self.vertices, other.vertices) and np.array_equal(
            self.simplices, other.simplices
        )

    def quadrature(
        self,
        order=4,
        *,
        boundary=None,
        region=None,
        rule=None,
        max_points=1_000_000,
    ):
        """Custom rule(dimension, order) returns reference barycentric points, weights.

        Weights integrate the reference simplex, not a probability distribution.
        Boundary quadrature carries coordinates in its parent volume cell.
        """
        if boundary is not None and region is not None:
            raise ValueError("Choose either a boundary or a volume region, not both.")
        order = positive_integer(order, "order", 0)
        max_points = positive_integer(max_points, "max_points")
        dim = self.dimension - (boundary is not None)
        bary, weights = (
            reference_quadrature(dim, order, max_points) if rule is None else rule(dim, order)
        )
        bary, weights = np.asarray(bary, dtype=float), np.asarray(weights, dtype=float)
        if (
            bary.ndim != 2
            or bary.shape[1] != dim + 1
            or weights.shape != (len(bary),)
            or not len(bary)
            or not np.isfinite(bary).all()
            or not np.isfinite(weights).all()
            or not np.allclose(bary.sum(axis=1), 1)
            or np.any(bary < -1e-12)
        ):
            raise ValueError("Invalid simplex quadrature rule.")
        nq = len(bary)
        if boundary is None:
            label = "all" if region is None else region
            if label not in self.regions:
                raise ValueError(f"Unknown region {label!r}.")
            cells = self.regions[label]
            corners = self.vertices[self.simplices[cells]]
            cell_bary = np.broadcast_to(bary, (len(cells), *bary.shape))
            factors = self.jacobians[cells]
            normals = None
        else:
            if boundary not in self.boundaries:
                raise ValueError(f"Unknown boundary {boundary!r}.")
            ids = self.boundaries[boundary]
            faces = self.exterior_faces[ids]
            cells, opposite = self._owners[ids].T
            corners = self.vertices[faces]
            cell_bary = np.zeros((len(cells), nq, self.dimension + 1))
            for i, (face, cell) in enumerate(zip(faces, cells)):
                positions = np.argmax(face[:, None] == self.simplices[cell][None, :], axis=1)
                cell_bary[i][:, positions] = bary
            gradient = self.barycentric_gradients[cells, opposite]
            norms = np.linalg.norm(gradient, axis=-1)
            factors = self.jacobians[cells] * norms
            normals = np.repeat(-gradient / norms[:, None], nq, axis=0)
        if len(cells) * nq > max_points:
            raise ValueError(
                "Physical quadrature exceeds max_points; adjust order, rule or budget."
            )
        return SimplexQuadrature(
            points=readonly(
                np.einsum("qi,eip->eqp", bary, corners).reshape(-1, self.ambient_dimension)
            ),
            weights=readonly((factors[:, None] * weights).reshape(-1)),
            cells=readonly(np.repeat(cells, nq), np.int64),
            barycentric=readonly(cell_bary.reshape(-1, self.dimension + 1)),
            normals=None if normals is None else readonly(normals),
        )
