"""Evaluate two coupled nonlinear fields on a triangulated disk."""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from skfem import MeshTri

from ngfield import Domain, FEMSpace, GalerkinBasis, GalerkinField, Problem, load_basis, save_basis


def weak_reaction_diffusion(x, u, grad_u):
    a, b = u[:, 0], u[:, 1]
    reaction = torch.stack((a - a**3 - b, 0.25 * (a - b)), dim=1)
    diffusivity = u.new_tensor([0.1, 0.04]).reshape(1, 2, 1, 1)
    return reaction, -diffusivity * grad_u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    domain = Domain(MeshTri.init_circle(nrefs=3))
    # Homogeneous Dirichlet for the first field; natural Neumann for the second.
    problem = Problem(2, weak_reaction_diffusion, dirichlet=(("all",), ()))
    basis = GalerkinBasis.build(FEMSpace(domain, degree=2), problem, modes=(8, 6))
    field = GalerkinField(basis, problem, quadrature_order=8, device=args.device)
    generator = torch.Generator().manual_seed(9)
    z = (0.1 * torch.randn(16, basis.dimension, generator=generator, dtype=field.dtype)).to(
        field.device
    )
    result = field(z)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "basis.npz"
        save_basis(path, basis)
        restored = GalerkinField(load_basis(path), problem, quadrature_order=8, device=args.device)
        torch.testing.assert_close(restored(z), result, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "domain": "triangulated disk",
                "components": 2,
                "modes": basis.modes,
                "input_shape": list(z.shape),
                "output_shape": list(result.shape),
                "quadrature_points": field.quadrature_size,
                "field_norm": torch.linalg.vector_norm(result).item(),
                "basis_roundtrip": "identical",
                "basis": basis.diagnostics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
