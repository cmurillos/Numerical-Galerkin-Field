"""Affine simplicial domains and named exterior boundaries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skfem import MeshLine, MeshTet, MeshTri


@dataclass(frozen=True)
class Domain:
    """A meshed bounded domain in 1D, 2D or 3D.

    The reserved boundary name ``all`` denotes every exterior facet.
    Geometry is piecewise affine; curved geometry is represented by the mesh.
    Do not mutate the mesh after constructing a FEMSpace or a GalerkinBasis.
    """

    mesh: object

    def __post_init__(self):
        mesh = self.mesh
        m = mesh.dim()
        if m not in (1, 2, 3) or not mesh.affine or mesh.t.shape[0] != m + 1:
            raise ValueError("Use an affine line, triangle or tetrahedron mesh in 1D–3D.")
        if mesh.p.shape[0] != m or mesh.t.shape[1] == 0:
            raise ValueError("The mesh must have cells and matching coordinate dimension.")
        if not np.isfinite(mesh.p).all():
            raise ValueError("Mesh coordinates must be finite.")
        if np.any(mesh.t < 0) or np.any(mesh.t >= mesh.p.shape[1]):
            raise ValueError("Cell connectivity contains an invalid vertex index.")
        vertices = mesh.p[:, mesh.t]
        edges = vertices[:, 1:, :] - vertices[:, :1, :]
        determinants = np.linalg.det(np.moveaxis(edges, -1, 0))
        if np.any(determinants == 0) or not np.isfinite(determinants).all():
            raise ValueError("The mesh contains a degenerate simplex.")
        if np.unique(mesh.t).size != mesh.p.shape[1]:
            raise ValueError("Remove unused mesh vertices before constructing the domain.")
        exterior = mesh.boundary_facets()
        for name, facets in (mesh.boundaries or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("Boundary names must be nonempty strings.")
            if not np.isin(facets, exterior).all():
                raise ValueError(f"Boundary {name!r} includes a non-exterior facet.")
        object.__setattr__(self, "mesh", mesh.with_boundaries({"all": exterior}))

    @property
    def dimension(self) -> int:
        return self.mesh.dim()

    @classmethod
    def from_file(cls, path: str | Path) -> "Domain":
        """Import a mesh through meshio, including supported physical boundary tags."""
        import meshio
        from skfem.io import from_meshio

        return cls(from_meshio(meshio.read(path)))

    @classmethod
    def from_arrays(cls, points, cells, boundaries=None) -> "Domain":
        """Construct from points[m, vertices] and cells[m+1, elements]."""
        points = np.asarray(points, dtype=np.float64)
        cells = np.asarray(cells)
        if points.ndim != 2 or cells.ndim != 2 or cells.dtype.kind not in "iu":
            raise ValueError("Expected a coordinate matrix and an integer connectivity matrix.")
        types = {1: MeshLine, 2: MeshTri, 3: MeshTet}
        if points.shape[0] not in types or cells.shape[0] != points.shape[0] + 1:
            raise ValueError("Expected affine simplices in 1D–3D.")
        if np.any(cells < 0) or np.any(cells >= points.shape[1]):
            raise ValueError("Cell connectivity contains an invalid vertex index.")
        mesh = types[points.shape[0]](points, cells.astype(np.int32))
        if boundaries:
            mesh = mesh.with_boundaries(boundaries)
        return cls(mesh)

    def with_boundaries(self, **boundaries) -> "Domain":
        """Return a domain with labels selected by facet indices or midpoint predicates."""
        if "all" in boundaries:
            raise ValueError("The boundary name 'all' is reserved.")
        return Domain(self.mesh.with_boundaries(boundaries))

    def facets(self, names) -> np.ndarray:
        """Return the union of named exterior facets; unknown/empty tags are errors."""
        if isinstance(names, str):
            names = (names,)
        groups = []
        for name in names:
            if name not in self.mesh.boundaries:
                raise ValueError(f"Unknown boundary {name!r}.")
            facets = np.asarray(self.mesh.boundaries[name], dtype=np.int32)
            if not facets.size:
                raise ValueError(f"Boundary {name!r} is empty.")
            groups.append(facets)
        return np.unique(np.concatenate(groups)) if groups else np.empty(0, dtype=np.int32)
