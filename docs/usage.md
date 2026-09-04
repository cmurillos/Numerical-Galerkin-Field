# Interfaz general de Numerical Galerkin Field

## Problema mínimo

La interfaz central sólo recibe la geometría simplicial y una forma débil completa:

```python
import numpy as np
from ngfield import FiniteElementBasis, GalerkinProblem, grad, inner, sin


vertices = np.array([[0.0], [0.5], [1.0]])
simplices = np.array([[0, 1], [1, 2]])


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx


problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
basis = FiniteElementBasis(problem.geometry, degree=1)
G = problem.field(basis=basis)
```

No se declara un tipo de condición de frontera. La base suministrada por el usuario
define el espacio admisible. Las medidas de frontera sólo añaden los términos escritos
en `weak`.

## Geometría simplicial

`vertices` tiene forma `[M,p]` y `simplices` forma `[E,k+1]`, con `1 <= k <= p`.
Se admiten dominios de dimensión intrínseca `k` en un ambiente euclídeo de dimensión
`p`, incluidos simplejos de dimensión arbitraria y complejos afines embebidos.

Los vértices de una cara exterior pueden agruparse por índices:

```python
boundaries = {"wall": np.array([[1, 2, 3, 4]])}
```

o mediante un predicado aplicado a los puntos medios de todas las caras exteriores:

```python
boundaries = {"left": lambda x: x[:, 0] < 0}
```

La etiqueta reservada `"all"` existe siempre. El objeto valida rango local,
incidencias, caras y adyacencias de dominios de dimensión completa. El usuario debe
suministrar un complejo conforme; detectar todas las intersecciones globales entre
simplejos distantes queda fuera del contrato.

La cuadratura usa un mapa de Duffy y reglas de Gauss-Jacobi en el simplejo de referencia.
No contiene tablas específicas para 1D, 2D o 3D. Se puede reemplazar con
`quadrature_rule(dimension, order)`, que devuelve coordenadas baricéntricas y pesos.
`max_quadrature_points` exige que los costes grandes se autoricen explícitamente.

## Una forma débil completa

`weak(u,v,dx,ds)` se ejecuta una vez para construir una expresión. Cada integrando debe
ser escalar y lineal en `v`; la dependencia en `u` puede ser no lineal.

```python
def weak(u, v, dx, ds):
    x = dx.x
    normal = ds.normal
    volume = -(1 + sin(x[0]) ** 2) * inner(grad(u), grad(v)) * dx
    boundary = (2 + inner(normal, normal)) * u * v * ds("wall")
    return volume + boundary
```

`dx` integra en todos los simplejos. `ds("name")` integra en las caras etiquetadas y
`ds` sin etiqueta equivale a `ds("all")`. `dx.x` son las coordenadas espaciales y
`ds.normal` es la normal exterior; en un complejo embebido es la conormal unitaria
dentro del simplejo padre.

Se proporcionan `grad`, `inner`, `stack`, `sin`, `cos`, `exp`, `log`, `sqrt` y `tanh`,
además de aritmética, potencias escalares e indexación. `grad` puede componerse para
solicitar derivadas superiores de la base. La conformidad Sobolev necesaria para una
forma concreta sigue siendo una propiedad de la base elegida.

## Bases suministradas por el usuario

Una base implementa:

```python
class MyBasis:
    dimension = N
    value_shape = ()

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        # [Q,N,*value_shape,*([ambient_dimension] * order)]
        ...
```

El paquete incluye cuatro construcciones:

- `CallableBasis`: funciones Python diferenciables; deriva automáticamente con
  `torch.func` o acepta derivadas explícitas.
- `PolynomialBasis`: monomios de grado total o exponentes elegidos, en cualquier
  dimensión.
- `FiniteElementBasis`: Lagrange nodal continuo de grado arbitrario sobre simplejos.
- `ComponentBasis`: copias escalares para campos vectoriales o tensoriales.

`TransformedBasis` aplica una combinación lineal a cualquier base. Por ejemplo,
`problem.orthonormalize(basis)` devuelve coordenadas ortonormales numéricamente en L2.

Para una base Lagrange se pueden entregar coeficientes
`[grados_globales,N,*value_shape]`. Sus columnas pueden codificar condiciones de
frontera, periodicidad, restricciones de divergencia, acoplamiento entre componentes
o cualquier otro subespacio lineal construido externamente.

## Significado de G

Para `z` de forma `[N]` o `[B,N]`, la base define

```text
u_z = sum_j z_j phi_j.
```

Si la base no es ortonormal, el campo resuelve

```text
M G(z) = (a(u_z; phi_1), ..., a(u_z; phi_N)),
M_ij = (phi_j, phi_i)_L2.
```

Por ello `G(z)` contiene la velocidad de los coeficientes de `u_z`. El paquete evalúa
este campo y conserva autograd respecto de `z`; no integra una evolución temporal.
`mass_matrix` permite suministrar otra matriz de Gram simétrica positiva definida.

## Campos vectoriales y tensoriales

```python
scalar = PolynomialBasis(dimension=p, degree=3)
basis = ComponentBasis(scalar, components=2)


def weak(u, v, dx, ds):
    reaction = (u[0] - u[0] ** 3 - u[1]) * v[0]
    reaction += 0.25 * (u[0] - u[1]) * v[1]
    return reaction * dx - 0.1 * inner(grad(u), grad(v)) * dx
```

`value_shape` puede tener cualquier cantidad de ejes. `inner` contrae todos los ejes
del valor físico y deja libres lote, modo de prueba y cuadratura.

## Precisión, memoria y dispositivo

```python
G = problem.field(
    basis=basis,
    quadrature_order=8,
    device="cuda",
    dtype=torch.float64,
    max_quadrature_points=2_000_000,
)
```

La cuadratura define también la matriz de Gram cuando no se suministra `mass_matrix`.
Una matriz no positiva definida produce un error: suele indicar dependencia lineal,
cuadratura insuficiente o una base que no pertenece a la geometría. La evaluación
divide automáticamente los modos de prueba para respetar `max_intermediate_entries`.

`G.to(device=..., dtype=...)` mueve las tablas. Los tensores externos capturados por
la forma son responsabilidad del usuario. Se admiten `float32` y `float64`.

## Persistencia

`FiniteElementBasis.save(path)` conserva geometría, etiquetas, numeración y coeficientes
en NPZ sin pickle. `FiniteElementBasis.load(path)` restaura la misma base. Una
`CallableBasis` contiene código Python y no se serializa como datos ejecutables.

La API 0.1 basada en `Domain`, `FEMSpace`, `Problem` y `GalerkinBasis` se conserva por
compatibilidad. Los nuevos desarrollos deben usar `GalerkinProblem` y una base explícita.
