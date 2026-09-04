"""User-defined and simplicial function bases, independent of boundary-condition types."""

import json
from math import comb, prod
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.func import jacrev, vmap

from .geometry import SimplicialDomain, positive_integer, readonly


class Basis(Protocol):
    """Fixed basis contract; derivatives use ambient Cartesian axes.

    evaluate returns [Q,N,*value_shape,*([ambient_dim]*order)]. On embedded
    simplices, the field projects derivative axes onto the element tangent space.
    cells and barycentric refer to the problem geometry and may be ignored by
    globally evaluable bases. Basis parameters are fixed when a field is prepared.
    """

    dimension: int
    value_shape: tuple[int, ...]

    def evaluate(self, points, *, order=0, cells=None, barycentric=None): ...


def _shape(value):
    return tuple(positive_integer(i, "value_shape entry") for i in value)


def compositions(total, length):
    if length == 1:
        yield (total,)
    else:
        for first in range(total, -1, -1):
            for rest in compositions(total - first, length - 1):
                yield (first, *rest)


def differentiate_values(function, points, order):
    if order == 0:
        return function(points)

    def single(x):
        return function(x.unsqueeze(0))[0]

    derivative = single
    for _ in range(order):
        derivative = jacrev(derivative)
    return vmap(derivative)(points)


class CallableBasis:
    """Wrap values(points)->[Q,N,*S], optionally derivatives={order: callable}.

    Missing derivatives are computed using torch.func, so values must preserve
    PyTorch differentiation and act independently at each point. Providing explicit
    derivatives also supports external numerical evaluators.
    """

    def __init__(self, values, *, dimension, value_shape=(), derivatives=None):
        if not callable(values):
            raise TypeError("values must be callable.")
        self.values = values
        self.dimension = positive_integer(dimension, "dimension")
        self.value_shape = _shape(value_shape)
        self.derivatives = dict(derivatives or {})
        for order, callback in self.derivatives.items():
            positive_integer(order, "derivative order")
            if not callable(callback):
                raise TypeError("Each derivative must be callable.")

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        if order in self.derivatives:
            return self.derivatives[order](points)
        return differentiate_values(self.values, points, order)


class PolynomialBasis(CallableBasis):
    """Total-degree monomials or explicit exponents in any ambient dimension.

    Monomials are not assumed orthonormal. The field solves their Gram system;
    orthonormalize() can prepare better-conditioned coordinates. Scaling x is useful.
    """

    def __init__(self, dimension, degree=1, *, exponents=None, center=None, scale=None):
        n = positive_integer(dimension, "spatial dimension")
        degree = positive_integer(degree, "degree", 0)
        if exponents is None:
            exponents = [a for p in range(degree + 1) for a in compositions(p, n)]
        a = np.asarray(exponents)
        if a.ndim != 2 or a.shape[1] != n or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("exponents must be nonnegative integer rows [N,n].")
        if not len(a) or len(np.unique(a, axis=0)) != len(a):
            raise ValueError("Provide nonempty, distinct polynomial exponents.")
        self.exponents = readonly(a, np.int64)
        self.center = readonly(np.broadcast_to(0.0 if center is None else center, (n,)))
        self.scale = readonly(np.broadcast_to(1.0 if scale is None else scale, (n,)))
        if (
            not np.isfinite(self.center).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale == 0)
        ):
            raise ValueError(
                "Polynomial coordinates require finite center and nonzero finite scale."
            )

        def values(x):
            if x.shape[-1] != n:
                raise ValueError("Polynomial and geometry dimensions differ.")
            y = (x - x.new_tensor(self.center.copy())) / x.new_tensor(self.scale.copy())
            powers = torch.tensor(self.exponents.copy(), device=x.device)
            return (y[:, None, :] ** powers[None, :, :]).prod(dim=-1)

        super().__init__(values, dimension=len(a))


class ComponentBasis:
    """Expand a scalar basis into scalar copies, or any tensor value shape.

    Modes are ordered by flattened component, then scalar basis mode. Linear
    combinations can subsequently couple components or impose linear constraints.
    """

    def __init__(self, scalar_basis, components=None, *, value_shape=None):
        if tuple(scalar_basis.value_shape):
            raise ValueError("ComponentBasis requires a scalar basis.")
        if (components is None) == (value_shape is None):
            raise ValueError("Provide components or value_shape.")
        self.base = scalar_basis
        self.value_shape = _shape((components,) if value_shape is None else value_shape)
        self.dimension = scalar_basis.dimension * prod(self.value_shape)
        if hasattr(scalar_basis, "geometry"):
            self.geometry = scalar_basis.geometry

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        a = self.base.evaluate(points, order=order, cells=cells, barycentric=barycentric)
        c = prod(self.value_shape)
        eye = torch.eye(c, dtype=points.dtype, device=points.device)
        # [Q,component,mode,value_component,*derivative_axes]
        result = torch.einsum("qn...,ab->qanb...", a, eye)
        return result.reshape(len(points), self.dimension, *self.value_shape, *a.shape[2:])


class TransformedBasis:
    """phi_new[j] = sum_i phi_old[i] * transform[i,j]."""

    def __init__(self, basis, transform):
        t = torch.as_tensor(transform).detach().cpu().to(torch.float64).clone()
        if (
            t.ndim != 2
            or t.shape[0] != basis.dimension
            or not t.shape[1]
            or not torch.isfinite(t).all()
        ):
            raise ValueError("transform must be a finite matrix [old_modes,new_modes].")
        self.base, self.transform = basis, t
        self.dimension, self.value_shape = int(t.shape[1]), tuple(basis.value_shape)
        if hasattr(basis, "geometry"):
            self.geometry = basis.geometry

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        a = self.base.evaluate(points, order=order, cells=cells, barycentric=barycentric)
        return torch.einsum("qi...,ij->qj...", a, self.transform.to(points))


class FiniteElementBasis:
    """Continuous nodal Lagrange functions on k-simplices, for any k and degree >= 1.

    Optional coefficients[global_dofs,N,*S] define arbitrary coupled modes.
    Matching barycentric nodes share a global DOF regardless of vertex ordering.
    Higher derivatives are elementwise; global Sobolev conformity is the caller's
    responsibility (continuous P1/P2, for example, are generally not H2-conforming).
    """

    def __init__(self, geometry, degree=1, *, coefficients=None, max_dofs=100_000):
        self.geometry = geometry
        self.degree = positive_integer(degree, "degree")
        k = geometry.dimension
        local_count = comb(self.degree + k, k)
        if local_count * len(geometry.simplices) > positive_integer(max_dofs, "max_dofs"):
            raise ValueError("Local DOF budget exceeded; increase max_dofs explicitly.")
        self.indices = tuple(compositions(self.degree, k + 1))
        nodes, dofs, coordinates = {}, [], []
        for vertices in geometry.simplices:
            row = []
            for alpha in self.indices:
                key = tuple(sorted((int(v), a) for v, a in zip(vertices, alpha) if a))
                if key not in nodes:
                    nodes[key] = len(nodes)
                    coordinates.append(
                        np.asarray(alpha) @ geometry.vertices[vertices] / self.degree
                    )
                row.append(nodes[key])
            dofs.append(row)
        self.element_dofs = readonly(dofs, np.int64)
        self.dof_points = readonly(coordinates)
        self.ndofs = len(nodes)
        if coefficients is None:
            self.coefficients = None
            self.dimension, self.value_shape = self.ndofs, ()
        else:
            c = np.asarray(coefficients)
            if (
                c.dtype.kind not in "fiu"
                or c.ndim < 2
                or c.shape[0] != self.ndofs
                or not c.shape[1]
                or not np.isfinite(c).all()
            ):
                raise ValueError("coefficients must be finite and shaped [ndofs,N,*value_shape].")
            self.coefficients = readonly(c, np.float64)
            self.dimension = c.shape[1]
            self.value_shape = _shape(c.shape[2:])

    def _local(self, xi):
        bary = torch.cat((1 - xi.sum().reshape(1), xi))
        values = []
        for alpha in self.indices:
            value = xi.new_ones(())
            for i, count in enumerate(alpha):
                for j in range(count):
                    value = value * (self.degree * bary[i] - j) / (j + 1)
            values.append(value)
        return torch.stack(values)

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        if cells is None or barycentric is None:
            raise ValueError(
                "FiniteElementBasis requires parent cells and barycentric coordinates."
            )
        derivative = self._local
        for _ in range(order):
            derivative = jacrev(derivative)
        local = vmap(derivative)(barycentric[:, 1:])
        if order:
            inverse = points.new_tensor(self.geometry.inverse_jacobians.copy())[cells]
            args = [local, [0, 1, *range(2, order + 2)]]
            for i in range(order):
                args.extend([inverse, [0, 2 + i, 2 + order + i]])
            args.append([0, 1, *range(2 + order, 2 + 2 * order)])
            local = torch.einsum(*args)
        dofs = torch.tensor(self.element_dofs.copy(), device=points.device)[cells]
        if self.coefficients is None:
            c = torch.nn.functional.one_hot(dofs, self.dimension).to(points.dtype).unsqueeze(-1)
        else:
            c = points.new_tensor(self.coefficients.copy())[dofs].reshape(
                len(points), len(self.indices), self.dimension, -1
            )
        result = torch.einsum("ql...,qlnc->qnc...", local, c)
        return result.reshape(
            len(points), self.dimension, *self.value_shape, *([points.shape[1]] * order)
        )

    def save(self, path, *, overwrite=False):
        """Persist numerical modes and geometry without serializing executable callbacks."""
        names = [name for name in self.geometry.boundaries if name != "all"]
        arrays = {
            "metadata": np.array(
                json.dumps({"schema": 1, "degree": self.degree, "boundaries": names})
            ),
            "vertices": self.geometry.vertices,
            "simplices": self.geometry.simplices,
            "element_dofs": self.element_dofs,
        }
        if self.coefficients is not None:
            arrays["coefficients"] = self.coefficients
        for i, name in enumerate(names):
            arrays[f"boundary_{i}"] = self.geometry.exterior_faces[self.geometry.boundaries[name]]
        with Path(path).open("wb" if overwrite else "xb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["schema"] != 1:
                raise ValueError("Unsupported finite-element basis schema.")
            geometry = SimplicialDomain(
                data["vertices"],
                data["simplices"],
                {name: data[f"boundary_{i}"] for i, name in enumerate(metadata["boundaries"])},
            )
            result = cls(
                geometry,
                metadata["degree"],
                coefficients=data["coefficients"] if "coefficients" in data else None,
                max_dofs=max(100_000, data["element_dofs"].size),
            )
            if not np.array_equal(result.element_dofs, data["element_dofs"]):
                raise ValueError("Stored and reconstructed DOF numbering differ.")
            return result
