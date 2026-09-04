"""Batch-first weak Galerkin evaluation on precomputed reduced-mode tables."""

from dataclasses import dataclass

import numpy as np
import torch

from .basis import GalerkinBasis
from .problem import Problem


@dataclass
class _Table:
    x: torch.Tensor
    weights: torch.Tensor
    values: torch.Tensor
    gradients: torch.Tensor
    normals: torch.Tensor | None

    def to(self, device, dtype):
        for name in ("x", "weights", "values", "gradients", "normals"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device=device, dtype=dtype))


class GalerkinField:
    """A callable G: [B,N] -> [B,N], with optional single-state [N] convenience.

    CPU/GPU tables are prepared once. No loops over states or modes occur in evaluation;
    loops over physical components and named boundary pieces only assemble contractions.
    The callable preserves PyTorch autograd. It never integrates an evolution in time.
    """

    def __init__(
        self,
        basis: GalerkinBasis,
        problem: Problem,
        quadrature_order: int | None = None,
        *,
        device="cpu",
        dtype=torch.float64,
    ):
        if problem.components != basis.components or problem.dirichlet != basis.dirichlet:
            raise ValueError(
                "Problem components and essential boundaries must match the saved basis."
            )
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64 for real fields.")
        self.basis = basis
        self.problem = problem
        self.quadrature_order = (
            max(4, 2 * basis.fem.degree) if quadrature_order is None else quadrature_order
        )
        self._slices = basis.slices
        combined = np.concatenate(basis.coefficients, axis=1)

        def prepare(boundary=None):
            q = basis.fem.tabulate(combined, self.quadrature_order, boundary=boundary)
            return _Table(
                **{
                    name: None
                    if getattr(q, name) is None
                    else torch.tensor(getattr(q, name), device=device, dtype=dtype)
                    for name in ("x", "weights", "values", "gradients", "normals")
                }
            )

        self._volume = prepare()
        self._boundary = {name: prepare(name) for name in problem.boundary}

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def device(self) -> torch.device:
        return self._volume.x.device

    @property
    def dtype(self) -> torch.dtype:
        return self._volume.x.dtype

    @property
    def quadrature_size(self) -> int:
        return self._volume.weights.numel()

    @property
    def table_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for table in (self._volume, *self._boundary.values())
            for tensor in vars(table).values()
            if tensor is not None
        )

    def to(self, device=None, dtype=None) -> "GalerkinField":
        """Move numerical tables in place; user callback parameters remain user-owned."""
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64.")
        for table in (self._volume, *self._boundary.values()):
            table.to(device, dtype)
        return self

    def _states(self, z):
        if not isinstance(z, torch.Tensor):
            raise TypeError("States must be a torch.Tensor.")
        single = z.ndim == 1
        if single:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape[1] != self.dimension:
            raise ValueError(
                f"Expected states with shape [B,{self.dimension}] or [{self.dimension}]."
            )
        if z.device != self.device or z.dtype != self.dtype:
            raise ValueError("States and field tables must have the same device and dtype.")
        return z, single

    def _reconstruct(self, z, table):
        values = torch.stack([z[:, s] @ table.values[s] for s in self._slices], dim=1)
        gradients = torch.stack(
            [torch.einsum("bn,nqm->bqm", z[:, s], table.gradients[s]) for s in self._slices], dim=1
        )
        return values, gradients

    def _coerce(self, value, shape, name):
        if isinstance(value, torch.Tensor):
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(f"{name} returned a tensor with the wrong device or dtype.")
        else:
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        try:
            return torch.broadcast_to(value, shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must broadcast to {tuple(shape)}, got {tuple(value.shape)}."
            ) from exc

    def _project(self, f0, f1, table):
        result = []
        for r, s in enumerate(self._slices):
            term = torch.einsum("bq,nq,q->bn", f0[:, r], table.values[s], table.weights)
            if f1 is not None:
                term = term + torch.einsum(
                    "bqm,nqm,q->bn", f1[:, r], table.gradients[s], table.weights
                )
            result.append(term)
        return torch.cat(result, dim=1)

    def reconstruct(self, z):
        """Return values [B,d,Q] and gradients [B,d,Q,m] at volume quadrature.

        The leading batch axis is kept, even for a single input state.
        """
        z, _ = self._states(z)
        return self._reconstruct(z, self._volume)

    def quadrature_points(self) -> torch.Tensor:
        """Return a copy of physical volume quadrature points [Q,m]."""
        return self._volume.x.clone()

    def __call__(self, z):
        z, single = self._states(z)
        if z.shape[0] == 0:
            return z.clone()
        u, grad_u = self._reconstruct(z, self._volume)
        output = self.problem.volume(self._volume.x, u, grad_u)
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise ValueError("volume must return (f0, f1).")
        f0 = self._coerce(output[0], u.shape, "volume f0")
        f1 = self._coerce(output[1], grad_u.shape, "volume f1")
        result = self._project(f0, f1, self._volume)
        for name, table in self._boundary.items():
            ub, grad_ub = self._reconstruct(z, table)
            load = self.problem.boundary[name](table.x, ub, grad_ub, table.normals)
            load = self._coerce(load, ub.shape, f"boundary {name!r}")
            result = result + self._project(load, None, table)
        return result[0] if single else result
