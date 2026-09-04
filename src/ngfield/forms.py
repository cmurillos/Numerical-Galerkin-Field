"""A small differentiable language for complete weak forms in Python."""

from dataclasses import dataclass

import numpy as np
import torch


def _degrees(value):
    if isinstance(value, Expr):
        return value.test_degrees
    array = (
        np.asarray(value) if not isinstance(value, torch.Tensor) else value.detach().cpu().numpy()
    )
    return frozenset() if not np.any(array) else frozenset({0})


def _combine_degrees(a, b, operation):
    if operation == "add":
        return a | b
    if not a or not b:
        return frozenset()
    return frozenset(i + j for i in a for j in b)


def _broadcast_shape(a, b):
    try:
        return np.broadcast_shapes(a, b)
    except ValueError as exc:
        raise ValueError(f"Incompatible value shapes {a} and {b}.") from exc


class Expr:
    __array_priority__ = 1000

    def __init__(self, op, args=(), *, shape=(), spatial_dimension=None, data=None, degrees=None):
        self.op, self.args, self.shape, self.spatial_dimension = (
            op,
            tuple(args),
            tuple(shape),
            spatial_dimension,
        )
        self.data = data
        self.test_degrees = frozenset({0}) if degrees is None else frozenset(degrees)

    def _binary(self, other, op):
        other = as_expr(other, self.spatial_dimension)
        dimension = (
            self.spatial_dimension
            if self.spatial_dimension is not None
            else other.spatial_dimension
        )
        if self.spatial_dimension not in (None, dimension) or other.spatial_dimension not in (
            None,
            dimension,
        ):
            raise ValueError("Expressions use different spatial dimensions.")
        degrees = _combine_degrees(
            self.test_degrees, other.test_degrees, "add" if op in ("add", "sub") else "mul"
        )
        return Expr(
            op,
            (self, other),
            shape=_broadcast_shape(self.shape, other.shape),
            spatial_dimension=dimension,
            degrees=degrees,
        )

    def __add__(self, other):
        if isinstance(other, (Integral, Form)):
            return other.__radd__(self)
        return self._binary(other, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return as_expr(other, self.spatial_dimension)._binary(self, "sub")

    def __mul__(self, other):
        if isinstance(other, Measure):
            return Integral(self, other)
        if isinstance(other, (Integral, Form)):
            return other.__rmul__(self)
        return self._binary(other, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_expr(other, self.spatial_dimension)
        if any(d != 0 for d in other.test_degrees):
            raise ValueError("A weak form cannot divide by an expression containing v.")
        result = self._binary(other, "div")
        result.test_degrees = self.test_degrees
        return result

    def __rtruediv__(self, other):
        return as_expr(other, self.spatial_dimension).__truediv__(self)

    def __pow__(self, power):
        if not isinstance(power, (int, float)) or isinstance(power, bool):
            raise TypeError("Powers in weak forms currently require a real scalar exponent.")
        if any(d != 0 for d in self.test_degrees) and power != 1:
            raise ValueError("The weak form must remain linear in v.")
        degrees = frozenset(d * power for d in self.test_degrees)
        return Expr(
            "pow",
            (self,),
            shape=self.shape,
            spatial_dimension=self.spatial_dimension,
            data=power,
            degrees=degrees,
        )

    def __neg__(self):
        return Expr(
            "neg",
            (self,),
            shape=self.shape,
            spatial_dimension=self.spatial_dimension,
            degrees=self.test_degrees,
        )

    def __getitem__(self, item):
        shape = np.empty(self.shape)[item].shape
        return Expr(
            "index",
            (self,),
            shape=shape,
            spatial_dimension=self.spatial_dimension,
            data=(item, len(self.shape)),
            degrees=self.test_degrees,
        )

    def diff(self, axis):
        if self.spatial_dimension is None:
            raise ValueError("Cannot differentiate an expression without a spatial dimension.")
        if not 0 <= axis < self.spatial_dimension:
            raise IndexError("Spatial derivative axis is out of range.")
        if self.op in ("u", "v"):
            return Expr(
                self.op,
                shape=self.shape,
                spatial_dimension=self.spatial_dimension,
                data=(*self.data, axis),
                degrees=self.test_degrees,
            )
        if self.op == "x":
            value = np.zeros(self.spatial_dimension)
            value[axis] = 1
            return as_expr(value, self.spatial_dimension)
        if self.op in ("constant", "normal"):
            return zero(self.shape, self.spatial_dimension)
        if self.op in ("add", "sub"):
            return self.args[0].diff(axis)._binary(self.args[1].diff(axis), self.op)
        if self.op == "mul":
            return self.args[0].diff(axis) * self.args[1] + self.args[0] * self.args[1].diff(axis)
        if self.op == "div":
            a, b = self.args
            return (a.diff(axis) * b - a * b.diff(axis)) / (b**2)
        if self.op == "pow":
            return self.data * (self.args[0] ** (self.data - 1)) * self.args[0].diff(axis)
        if self.op == "neg":
            return -self.args[0].diff(axis)
        if self.op == "index":
            item, source_rank = self.data
            return Expr(
                "index",
                (self.args[0].diff(axis),),
                shape=self.shape,
                spatial_dimension=self.spatial_dimension,
                data=(item, source_rank),
                degrees=self.test_degrees,
            )
        if self.op == "stack":
            return stack([arg.diff(axis) for arg in self.args], axis=self.data)
        if self.op == "inner":
            return inner(self.args[0].diff(axis), self.args[1]) + inner(
                self.args[0], self.args[1].diff(axis)
            )
        if self.op in _UNARY_DERIVATIVES:
            return _UNARY_DERIVATIVES[self.op](self.args[0]) * self.args[0].diff(axis)
        raise NotImplementedError(f"Spatial differentiation is unavailable for {self.op!r}.")


def as_expr(value, spatial_dimension=None):
    if isinstance(value, Expr):
        return value
    if not isinstance(value, (torch.Tensor, np.ndarray, int, float, list, tuple)):
        raise TypeError(f"Cannot use {type(value).__name__} in a weak expression.")
    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else np.asarray(value).shape
    return Expr(
        "constant",
        shape=shape,
        spatial_dimension=spatial_dimension,
        data=value,
        degrees=_degrees(value),
    )


def zero(shape, spatial_dimension):
    return Expr(
        "constant",
        shape=shape,
        spatial_dimension=spatial_dimension,
        data=np.zeros(shape),
        degrees=frozenset(),
    )


def grad(value):
    value = as_expr(value)
    return stack([value.diff(i) for i in range(value.spatial_dimension)], axis=-1)


def inner(a, b):
    a, b = as_expr(a), as_expr(b)
    if a.shape != b.shape:
        raise ValueError(f"inner requires equal value shapes, got {a.shape} and {b.shape}.")
    dimension = a.spatial_dimension if a.spatial_dimension is not None else b.spatial_dimension
    degrees = _combine_degrees(a.test_degrees, b.test_degrees, "mul")
    return Expr("inner", (a, b), spatial_dimension=dimension, degrees=degrees)


def stack(values, axis=0):
    values = tuple(as_expr(value) for value in values)
    if not values or any(value.shape != values[0].shape for value in values):
        raise ValueError("stack requires nonempty expressions with one value shape.")
    axis = axis if axis >= 0 else len(values[0].shape) + 1 + axis
    if not 0 <= axis <= len(values[0].shape):
        raise IndexError("stack axis is out of range.")
    shape = list(values[0].shape)
    shape.insert(axis, len(values))
    degrees = frozenset().union(*(value.test_degrees for value in values))
    return Expr(
        "stack",
        values,
        shape=shape,
        spatial_dimension=values[0].spatial_dimension,
        data=axis,
        degrees=degrees,
    )


def _unary(name, derivative):
    def operation(value):
        value = as_expr(value)
        if any(d != 0 for d in value.test_degrees):
            raise ValueError(f"{name} cannot be applied to v in a linear weak form.")
        return Expr(
            name,
            (value,),
            shape=value.shape,
            spatial_dimension=value.spatial_dimension,
            degrees=value.test_degrees,
        )

    _UNARY_DERIVATIVES[name] = derivative
    return operation


_UNARY_DERIVATIVES = {}
exp = _unary("exp", lambda x: exp(x))
sin = _unary("sin", lambda x: cos(x))
cos = _unary("cos", lambda x: -sin(x))
tanh = _unary("tanh", lambda x: 1 - tanh(x) ** 2)
log = _unary("log", lambda x: 1 / x)
sqrt = _unary("sqrt", lambda x: 0.5 / sqrt(x))


@dataclass(frozen=True)
class Measure:
    kind: str
    label: str | None = None
    spatial_dimension: int | None = None

    def __call__(self, label):
        if not isinstance(label, str) or not label:
            raise ValueError("Measure labels must be nonempty strings.")
        return Measure(self.kind, label, self.spatial_dimension)

    @property
    def x(self):
        return Expr("x", shape=(self.spatial_dimension,), spatial_dimension=self.spatial_dimension)

    @property
    def normal(self):
        if self.kind != "boundary":
            raise ValueError("Normals are defined only for boundary measures.")
        return Expr(
            "normal", shape=(self.spatial_dimension,), spatial_dimension=self.spatial_dimension
        )

    def __rmul__(self, value):
        return Integral(as_expr(value, self.spatial_dimension), self)


@dataclass(frozen=True)
class Integral:
    integrand: Expr
    measure: Measure

    def __post_init__(self):
        if self.integrand.shape:
            raise ValueError("Each weak integrand must be scalar-valued; use inner or indexing.")

    def __add__(self, other):
        return Form((self,)) + other

    __radd__ = __add__

    def __sub__(self, other):
        return Form((self,)) - other

    def __neg__(self):
        return Integral(-self.integrand, self.measure)

    def __rmul__(self, value):
        value = as_expr(value, self.integrand.spatial_dimension)
        if value.shape or any(d != 0 for d in value.test_degrees):
            raise ValueError("A form can only be scaled by a scalar independent of v.")
        return Integral(value * self.integrand, self.measure)


@dataclass(frozen=True)
class Form:
    integrals: tuple[Integral, ...]

    def __add__(self, other):
        if isinstance(other, Integral):
            return Form((*self.integrals, other))
        if isinstance(other, Form):
            return Form((*self.integrals, *other.integrals))
        if other == 0:
            return self
        raise TypeError("Weak forms are sums of integrals.")

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Form(tuple(-integral for integral in self.integrals))

    def __rmul__(self, value):
        return Form(tuple(value * integral for integral in self.integrals))


def build_form(callback, value_shape, spatial_dimension):
    u = Expr("u", shape=value_shape, spatial_dimension=spatial_dimension, data=(), degrees={0})
    v = Expr("v", shape=value_shape, spatial_dimension=spatial_dimension, data=(), degrees={1})
    dx = Measure("volume", spatial_dimension=spatial_dimension)
    ds = Measure("boundary", spatial_dimension=spatial_dimension)
    result = callback(u, v, dx, ds)
    form = (
        result
        if isinstance(result, Form)
        else Form((result,))
        if isinstance(result, Integral)
        else None
    )
    if form is None or not form.integrals:
        raise TypeError("weak(u,v,dx,ds) must return one or more integrals.")
    for integral in form.integrals:
        if integral.integrand.test_degrees not in (frozenset(), frozenset({1})):
            raise ValueError("Every integrand must be linear in the test function v.")
    return form


def _pad(value, shape, rank):
    return value.reshape(*value.shape[:3], *((1,) * (rank - len(shape))), *shape)


def evaluate(expression, context, cache=None):
    cache = {} if cache is None else cache
    if expression in cache:
        return cache[expression]
    op = expression.op
    if op in ("u", "v"):
        table = context["basis"](len(expression.data))
        table = table[
            (slice(None), slice(None))
            + (slice(None),) * len(expression.shape)
            + tuple(expression.data)
        ]
        if op == "u":
            value = torch.einsum("bn,qn...->bq...", context["z"], table).unsqueeze(1)
        else:
            value = table[:, context["test"]].movedim(0, 1).unsqueeze(0)
    elif op == "constant":
        value = torch.as_tensor(expression.data, dtype=context["dtype"], device=context["device"])
        value = value.reshape(1, 1, 1, *expression.shape)
    elif op == "x":
        value = context["points"].reshape(1, 1, len(context["points"]), *expression.shape)
    elif op == "normal":
        if context["normals"] is None:
            raise ValueError("A boundary normal was used in a volume integral.")
        value = context["normals"].reshape(1, 1, len(context["normals"]), *expression.shape)
    elif op in ("add", "sub", "mul", "div"):
        a, b = expression.args
        rank = max(len(a.shape), len(b.shape))
        av, bv = (
            _pad(evaluate(a, context, cache), a.shape, rank),
            _pad(evaluate(b, context, cache), b.shape, rank),
        )
        value = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](av, bv)
    elif op == "pow":
        value = evaluate(expression.args[0], context, cache) ** expression.data
    elif op == "neg":
        value = -evaluate(expression.args[0], context, cache)
    elif op == "index":
        argument = expression.args[0]
        item, source_rank = expression.data
        item = item if isinstance(item, tuple) else (item,)
        remainder = len(argument.shape) - source_rank
        value = evaluate(argument, context, cache)[
            (slice(None),) * 3 + item + (slice(None),) * remainder
        ]
    elif op == "stack":
        value = torch.stack(
            [evaluate(arg, context, cache) for arg in expression.args], dim=3 + expression.data
        )
    elif op == "inner":
        value = evaluate(expression.args[0], context, cache) * evaluate(
            expression.args[1], context, cache
        )
        if expression.args[0].shape:
            value = value.sum(dim=tuple(range(3, 3 + len(expression.args[0].shape))))
    elif op in _UNARY_DERIVATIVES:
        value = getattr(torch, op)(evaluate(expression.args[0], context, cache))
    else:
        raise RuntimeError(f"Unknown expression operation {op!r}.")
    cache[expression] = value
    return value


def derivative_orders(form):
    result = {"volume": {}, "boundary": {}}
    for integral in form.integrals:
        key = integral.measure.label or "all"
        target = result[integral.measure.kind].setdefault(key, set())
        stack_ = [integral.integrand]
        while stack_:
            expression = stack_.pop()
            if expression.op in ("u", "v"):
                target.add(len(expression.data))
            stack_.extend(expression.args)
    return result
