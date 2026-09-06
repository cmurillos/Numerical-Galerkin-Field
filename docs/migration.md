# Migración a Space y compatibilidad — D-013, parte 8

El recorrido público actual es **geometría → espacio admisible → base → campo G**.
Los ejemplos del [README](../README.md) y de la [guía de aceptación](acceptance-examples.md)
usan ese recorrido. Esta guía explica cómo adoptarlo conservando el significado
matemático de los programas anteriores.

## Qué versión y qué código se está usando

D-013 amplía el código original de 0.9.0. La presencia de estas funciones depende
del checkout instalado; el número `ngfield.__version__` permanece en 0.9.0 hasta
preparar otra versión y, por sí solo, no distingue estas revisiones del código.
Desde un checkout que contenga D-013:

```bash
python -m pip install -e .
```

Para localizar el paquete que está importando el intérprete:

```python
import ngfield
from ngfield import GalerkinField, Space

print(ngfield.__file__)
```

En Colab o Jupyter, reiniciar el entorno después de cambiar la instalación evita
seguir utilizando módulos de una importación anterior. Una instalación fijada a
un commit permite repetir el mismo experimento. No es necesario cambiar de
versión para continuar usando un programa de la interfaz anterior que ya funciona.

## Qué permanece compatible

| Código existente | Situación actual |
|---|---|
| `problem.field(basis=basis)` con `GalerkinProblem` | Sigue disponible. |
| `GalerkinField(problem, basis)` con `GalerkinProblem` primero | Sigue disponible; también admite ambos argumentos nombrados. |
| `GalerkinField(legacy_basis, legacy_problem)` con `GalerkinBasis` y `Problem` | Sigue disponible, con ese orden; también admite argumentos nombrados. |
| `GalerkinField(basis=basis, weak=weak)` | Entrada desde una base asociada a Space. |
| `GeneralGalerkinField(...)` | Clase general expuesta, con entrada directa o problema general explícito. |
| Bases escalares anteriores con `value_shape=()` | Conservan valores escalares y reconstrucción `[...,Q]`. |
| `G(z)`, `G.project`, `G.reconstruct`, `G.solve`, derivadas y diagnósticos | Conservan sus contratos de lote, precisión y dispositivo. |

El nombre público `GalerkinField` es una función que selecciona la construcción,
no una clase nueva para usar con `isinstance`. `weak` y `problem` no se entregan a
la vez. La entrada directa obtiene la geometría desde la base y no recibe otra
geometría independiente.

## Cambios al adoptar el recorrido actual

| Decisión | Forma actual | Efecto matemático |
|---|---|---|
| Geometría | `SimplicialDomain(vertices=..., simplices=..., boundaries=..., regions=...)` | Las mismas medidas inducidas y etiquetas de integración. |
| Espacio | `Space(geometry=..., components=c, regularity=1, restrictions=[...])` | Declara variaciones admisibles antes de seleccionar la base. |
| Base | `V.basis(family, size=N, ...)` | N es la dimensión total; las restricciones preceden al espectro. |
| Forma | `weak(u,v,dx,ds)` con componentes explícitas | Una ecuación escalar usa `u[0]` y `v[0]`. |
| Campo | `GalerkinField(basis=basis, weak=weak)` | Las coordenadas pertenecen a esa base concreta. |

**Añadir `ZeroTrace` a un problema que no la tenía cambia el problema.** La base
laplaciana antigua, si no estaba restringida, usaba condiciones naturales en su
problema espectral auxiliar. No imponía extremos fijos por llamarse laplaciana.
La migración de sintaxis no debe añadir condiciones esenciales que no existían.

Los nombres públicos son `restrictions`, `size` y `component_sizes`. No hay entradas
genéricas `constraints`, `mass`, `zero_trace` o `modes` en `Space.basis`. Una matriz
pasada a `TransformedBasis` es una transformación de funciones; no se interpreta
como un sistema de ecuaciones de restricción.

## Componentes y tamaño total

| Cantidad | Base escalar anterior | Space con una componente | Space con c componentes |
|---|---|---|---|
| `basis.value_shape` | `()` | `(1,)` | `(c,)` |
| Valores de una función inicial | `[Q]` | `[Q,1]` | `[Q,c]` |
| `basis.evaluate(points)` | `[Q,N]` | `[Q,N,1]` | `[Q,N,c]` |
| Coordenadas y G | `[...,N]` | `[...,N]` | `[...,N]` |
| Reconstrucción | `[...,Q]` | `[...,Q,1]` | `[...,Q,c]` |

Por ejemplo, `torch.sin(x[:,0])` pasa a `torch.sin(x[:,:1])` al adoptar una componente
explícita. Un coeficiente usado como dato inicial también debe tener esa forma:
`Coefficient.cell(values)` recibe `[E,1]` en lugar de `[E]`. Esto no altera la forma
de un coeficiente físico escalar de la EDP: la conductividad puede seguir teniendo
`shape=()` y multiplicar `grad(u[0])`.

| Solicitud anterior | Solicitud equivalente de tamaño con Space |
|---|---|
| `ComponentBasis(scalar, components=2)`, con 8 modos escalares | `V.basis("laplacian", component_sizes=(8,8))`: 16 coordenadas. |
| `GalerkinBasis.build(..., modes=8)` para dos componentes | El entero antiguo era por componente: usar `(8,8)`, no `size=8`. |
| `GalerkinBasis.build(..., modes=(12,8))` | `component_sizes=(12,8)`, total 20. |
| Selección global de N modos sin reparto prescrito | `V.basis("laplacian", size=N)`. |

La equivalencia de tamaño no garantiza igualdad de funciones o coordenadas. En una
selección global, autovalores repetidos permiten mezclas entre componentes y no se
garantiza un reparto. Para fijarlo se usa `component_sizes`; su suma debe coincidir
con `size` si ambos se indican. `finite-element` y `custom` conservan su espacio
completo: `size` confirma la dimensión, no trunca la familia.

## Ejemplo completo de migración escalar

Este ejemplo conserva el mismo espacio aproximante y la misma forma de difusión
con reacción cúbica. Adapta una base escalar existente como fuente de una componente;
no añade restricciones de frontera. La comprobación compara funciones y velocidades
físicas, porque una preparación de base puede cambiar sus coordenadas.

```python
import numpy as np
import torch
from ngfield import (
    ComponentBasis,
    GalerkinField,
    GalerkinProblem,
    SimplicialDomain,
    Space,
    grad,
    inner,
)

vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(vertices=vertices, simplices=simplices)


def weak_old(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx - u**3 * v * dx


problem_old = GalerkinProblem(geometry=geometry, weak=weak_old)
basis_old = problem_old.basis("laplacian", size=4, degree=1)
G_old = problem_old.field(basis=basis_old, quadrature=6)

V = Space(geometry=geometry, components=1, regularity=1)
source = ComponentBasis(basis_old, components=1)
basis = V.basis("custom", source=source, quadrature_order=6)


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u[0]), grad(v[0])) * dx - u[0] ** 3 * v[0] * dx


G = GalerkinField(basis=basis, weak=weak, quadrature=6)
z_old = G_old.project(lambda x: torch.cos(torch.pi * x[:, 0]), quadrature=8)
z = G.project(lambda x: torch.cos(torch.pi * x[:, :1]), quadrature=8)
points = torch.linspace(0, 1, 101, dtype=G.dtype)[:, None]

torch.testing.assert_close(
    G.reconstruct(z, points)[..., 0], G_old.reconstruct(z_old, points), atol=1e-10, rtol=1e-10
)
torch.testing.assert_close(
    G.reconstruct(G(z), points)[..., 0],
    G_old.reconstruct(G_old(z_old), points),
    atol=1e-10,
    rtol=1e-10,
)
```

## Reutilizar o transferir coordenadas

Si la base ya proviene de Space y se conserva su instancia, cambiar únicamente
la construcción del campo conserva las coordenadas. Continuando el ejemplo:

```python
explicit = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis, quadrature=6)
torch.testing.assert_close(explicit(z), G(z), atol=1e-12, rtol=1e-12)
assert G.basis is explicit.basis is basis
```

Si se construye otra base, no se debe reutilizar z por coincidir su dimensión o sus
autovalores. Los modos pueden cambiar de signo, orden o rotar dentro de un autoespacio.
Para trasladar un estado anterior se reconstruye su función y se proyecta:

```python
z_transferred = G.project(lambda x: G_old.reconstruct(z_old, x).unsqueeze(-1), quadrature=8)
torch.testing.assert_close(z_transferred, z, atol=1e-10, rtol=1e-10)
```

El eje añadido adapta la forma escalar de este ejemplo. Si ambas bases ya tienen
la misma forma de componentes, no se añade. Cuando los espacios aproximantes son
distintos, la transferencia es una proyección y puede perder información. Si se
cambia la malla, la función fuente debe poder evaluarse en los nuevos puntos; en
superficies aproximadas distintas puede requerirse un mapa geométrico externo.
No hay una transferencia automática entre geometrías incompatibles.

## Persistencia sin perder las coordenadas

`FiniteElementBasis.save/load` conserva la representación funcional y el orden de
los modos, la geometría y los metadatos espectrales disponibles. No reconstruye
Space ni guarda la forma débil o un campo completo. Una base FEM cargada se usa
con un `GalerkinProblem` explícito. Este ejemplo continúa con la geometría y la
forma anteriores y utiliza un archivo temporal:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from ngfield import FiniteElementBasis

saved_basis = V.basis("laplacian", size=4, degree=1)
original = GalerkinField(basis=saved_basis, weak=weak, quadrature=6)

with TemporaryDirectory() as directory:
    path = Path(directory) / "basis.npz"
    saved_basis.save(path)
    restored = FiniteElementBasis.load(path)
    loaded = GalerkinProblem(geometry=restored.geometry, weak=weak).field(
        basis=restored, quadrature=6
    )
    state = torch.linspace(-0.2, 0.3, original.dimension, dtype=original.dtype)
    torch.testing.assert_close(loaded(state), original(state), atol=1e-12, rtol=1e-12)
    assert loaded.space is None
```

Adjuntar manualmente `.space` no impone ni verifica restricciones. Si se requiere
un vínculo a Space, se vuelve a preparar una fuente mediante `V.basis("custom",...)`
y se considera el posible cambio de coordenadas. Las bases `CallableBasis` y las
composiciones arbitrarias no adquieren serialización automática en esta etapa.

## Valores tensoriales y bases de la interfaz original

Space representa `(components,)`; declarar `components=4` no produce `(2,2)`.
Los programas que requieren una forma tensorial arbitraria conservan la ruta
general explícita. Por ejemplo, usando la base escalar anterior:

```python
tensor_basis = ComponentBasis(basis_old, value_shape=(2, 2))
tensor_problem = GalerkinProblem(geometry=geometry, weak=lambda u, v, dx, ds: inner(u, v) * dx)
tensor_field = tensor_problem.field(basis=tensor_basis)
assert tensor_field.value_shape == (2, 2)
```

La API original `Domain`/`FEMSpace`/`Problem`/`GalerkinBasis` también permanece. No
se añade un conversor automático desde su operador: adoptar Space exige expresar
la forma completa en `weak` y declarar las restricciones por componente. Sus bases
guardadas siguen usando `load_basis` y la construcción original; no se confunden
con los archivos de `FiniteElementBasis`.

## Restricciones, forma y controles numéricos

| Elemento | Responsabilidad |
|---|---|
| `ZeroTrace` | Traza esencial homogénea por componente sobre una frontera etiquetada. |
| `Periodic` | Biyección de vértices que preserve caras; extiende la igualdad a las trazas nodales. |
| `MeanZero` | Integral nula con la medida inducida; no es promedio de coeficientes ni de vértices. |
| Robin o Neumann | Términos de `weak` con datos espaciales fijos y signo de flujo explícito. |
| Dirichlet fija no homogénea | Levantamiento fijo `ell`, proyección de `T0-ell` y reconstrucción `ell+Phi(z)`. |
| `quadrature_order`, `validation_order` en `V.basis` | Preparación y validación de la base. |
| `quadrature` en G o `G.project` | Automática con `None`, orden entero o tolerancia real en (0,1). |
| `G.mass_matrix` | Gram numérico consultable; no sustituye el producto interno L2. |
| `regularity_verified` | Distingue construcciones conocidas de declaraciones de fuentes arbitrarias. |

Las combinaciones soportadas por familia figuran en la [guía de uso](usage.md#bases-del-espacio-admisible).
No se interpreta una base Fourier como una identificación periódica certificada.
`regularity>1` no dispone de construcción en `Space.basis`, aunque un evaluador
pueda devolver derivadas elementales de orden superior. Las restricciones no lineales,
los campos dependientes explícitamente del tiempo, las geometrías móviles y los
productos internos alternativos continúan fuera del recorrido implementado.

Al proporcionar un problema explícito junto a una base vinculada a Space, la malla
y los conjuntos de regiones y fronteras deben coincidir con los del espacio. Una
misma etiqueta no puede adquirir otra selección. Los coeficientes fijos se tabulan
al construir G; cambiar sus datos requiere construir otro campo.

Para comprobar una migración, comparar primero la reconstrucción y las velocidades
físicas, luego las restricciones pertinentes y finalmente las trayectorias. Los
[ejemplos de aceptación](acceptance-examples.md) muestran referencias adecuadas a
cada problema; coincidir en un número de modos no es una comprobación suficiente.
