"""Measure general-field preparation, batch latency and resident table memory."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch

from ngfield import FiniteElementBasis, GalerkinProblem, grad, inner


def weak(u, v, dx, ds):
    return (u - u**3) * v * dx - 0.1 * inner(grad(u), grad(v)) * dx


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--modes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--quadrature", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_sizes + args.modes + [args.repeats]) < 1:
        parser.error("Batch sizes, mode counts and repeats must be positive.")
    device = torch.device(args.device)
    segments = max(args.modes) + 1
    vertices = np.linspace(0, 1, segments + 1)[:, None]
    simplices = np.column_stack((np.arange(segments), np.arange(1, segments + 1)))
    problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
    start = time.perf_counter()
    nodal = FiniteElementBasis(problem.geometry)
    geometry_seconds = time.perf_counter() - start
    results = []
    for modes in args.modes:
        basis = problem.basis("laplacian", size=modes, degree=1)
        synchronize(device)
        start = time.perf_counter()
        field = problem.field(basis=basis, quadrature=args.quadrature, device=device)
        synchronize(device)
        preparation = time.perf_counter() - start
        for batch_size in args.batch_sizes:
            z = torch.randn(batch_size, modes, dtype=field.dtype, device=device) * 0.1
            with torch.no_grad():
                for _ in range(2):
                    field(z)
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    field(z)
                synchronize(device)
            elapsed = (time.perf_counter() - start) / args.repeats
            results.append(
                {
                    "N": modes,
                    "B": batch_size,
                    "Q": field.quadrature_size,
                    "field_preparation_seconds": preparation,
                    "seconds_per_batch": elapsed,
                    "states_per_second": batch_size / elapsed,
                    "table_bytes": field.table_bytes,
                    "cuda_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                }
            )
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": "float64",
        "autograd": False,
        "geometry_seconds": geometry_seconds,
        "nodal_dofs": nodal.ndofs,
        "repeats": args.repeats,
        "results": results,
    }
    content = json.dumps(report, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
