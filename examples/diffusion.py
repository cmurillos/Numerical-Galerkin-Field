"""Evaluate a heat-equation Galerkin field on an L-shaped domain."""

import argparse
import json

import torch
from skfem import MeshTri

from ngfield import Domain, FEMSpace, GalerkinBasis, GalerkinField, Problem


def weak_diffusion(x, u, grad_u):
    return 0, -0.1 * grad_u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    domain = Domain(MeshTri.init_lshaped().refined(3))
    problem = Problem(components=1, volume=weak_diffusion, dirichlet=(("all",),))
    fem = FEMSpace(domain, degree=1)
    basis = GalerkinBasis.build(fem, problem, modes=12)
    field = GalerkinField(basis, problem, quadrature_order=4, device=args.device)
    generator = torch.Generator().manual_seed(4)
    z = torch.randn(32, basis.dimension, generator=generator, dtype=field.dtype).to(field.device)
    result = field(z)
    eigenvalues = torch.tensor(basis.eigenvalues[0].copy(), dtype=field.dtype, device=field.device)
    reference = -0.1 * z * eigenvalues
    relative_error = (
        torch.linalg.vector_norm(result - reference) / torch.linalg.vector_norm(reference)
    ).item()
    if relative_error > 1e-8:
        raise RuntimeError("Diffusion failed the reference-eigenvalue check.")
    print(
        json.dumps(
            {
                "domain": "L-shaped",
                "components": problem.components,
                "reduced_dimension": basis.dimension,
                "scalar_fem_dofs": fem.ndofs,
                "batch_size": len(z),
                "quadrature_points": field.quadrature_size,
                "relative_field_error": relative_error,
                "basis": basis.diagnostics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
