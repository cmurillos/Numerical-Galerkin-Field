"""Versioned numerical basis files; user callbacks are never serialized."""

import json
from pathlib import Path

import numpy as np
import scipy
import skfem

from .basis import GalerkinBasis
from .domain import Domain
from .fem import FEMSpace

_SCHEMA = 1


def save_basis(path: str | Path, basis: GalerkinBasis, *, overwrite=False) -> None:
    """Store mesh, boundary labels and exact computed coefficients in a compressed NPZ.

    The parent directory must exist. Existing files require explicit overwrite=True.
    No operator code, pickle objects or PyTorch device state is stored.
    """
    boundaries = basis.fem.domain.mesh.boundaries
    metadata = {
        "schema": _SCHEMA,
        "ngfield_version": "0.4.0",
        "skfem_version": skfem.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "degree": basis.fem.degree,
        "mass_order": basis.fem.mass_order,
        "components": basis.components,
        "dirichlet": basis.dirichlet,
        "boundary_names": sorted(boundaries),
    }
    arrays = {
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
        "points": basis.fem.domain.mesh.p,
        "cells": basis.fem.domain.mesh.t,
        "dof_locations": basis.fem.basis.doflocs,
        "element_dofs": basis.fem.basis.element_dofs,
    }
    for i, name in enumerate(metadata["boundary_names"]):
        arrays[f"boundary_{i}"] = boundaries[name]
    for r, (c, eigenvalues) in enumerate(zip(basis.coefficients, basis.eigenvalues)):
        arrays[f"coefficients_{r}"] = c
        arrays[f"eigenvalues_{r}"] = eigenvalues
    with Path(path).open("wb" if overwrite else "xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_basis(path: str | Path) -> GalerkinBasis:
    """Restore the basis without re-solving eigenvectors; verify its mass metric and DOFs.

    Require the same scikit-fem version to avoid silent numbering changes.
    Attach a Problem explicitly when constructing a new GalerkinField.
    """
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("schema") != _SCHEMA:
            raise ValueError("Unsupported basis-file schema.")
        if metadata["skfem_version"] != skfem.__version__:
            raise ValueError("Use the scikit-fem version recorded in the basis file.")
        boundaries = {
            name: data[f"boundary_{i}"] for i, name in enumerate(metadata["boundary_names"])
        }
        domain = Domain.from_arrays(data["points"], data["cells"], boundaries)
        fem = FEMSpace(domain, metadata["degree"], metadata["mass_order"])
        if not np.array_equal(fem.basis.doflocs, data["dof_locations"]) or not np.array_equal(
            fem.basis.element_dofs, data["element_dofs"]
        ):
            raise ValueError("Stored and reconstructed finite-element DOF numbering differ.")
        coefficients = tuple(data[f"coefficients_{r}"] for r in range(metadata["components"]))
        eigenvalues = tuple(data[f"eigenvalues_{r}"] for r in range(metadata["components"]))
        return GalerkinBasis(
            fem, coefficients, eigenvalues, tuple(tuple(b) for b in metadata["dirichlet"])
        )
