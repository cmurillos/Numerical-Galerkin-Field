# Interfaz general de Numerical Galerkin Field

## Problema mínimo

La interfaz central sólo recibe la geometría simplicial y una forma débil completa:

```python
import numpy as np
from ngfield import GalerkinProblem, grad, inner, sin


vertices = np.array([[0.0], [0.5], [1.0]])
simplices = np.array([[0, 1], [1, 2]])


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx


problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
basis = problem.basis("laplacian", size=8, degree=1)
G = problem.field(basis=basis)
```

No se declara un tipo de condición de frontera. La base suministrada por el usuario
define el espacio admisible. Las medidas de frontera sólo añaden los términos escritos
en `weak`.

La misma geometría puede reutilizarse:

```python
from ngfield import SimplicialDomain

geometry = SimplicialDomain(
    vertices=vertices,
    simplices=simplices,
    boundaries=boundaries,
    regions=regions,
)
problem = GalerkinProblem(geometry=geometry, weak=weak)
```

No se combinan `geometry` y los argumentos geométricos directos en una misma llamada.

## Geometría simplicial

`vertices` tiene forma `[M,p]` y `simplices` forma `[E,k+1]`, con `1 <= k <= p`.
Se admiten dominios de dimensión intrínseca `k` en un ambiente euclídeo de dimensión
`p`, incluidos simplejos de dimensión arbitraria y complejos afines embebidos.

Los vértices de una cara exterior pueden agruparse por índices:

```python
boundaries = {"wall": np.array([[1, 2, 3, 4]])}
```

o mediante un predicado aplicado a los baricentros de todas las caras exteriores:

```python
boundaries = {"left": lambda x: x[:, 0] < 0}
```

La etiqueta reservada `"all"` existe siempre. El objeto valida rango local,
incidencias, caras y adyacencias de dominios de dimensión completa. El usuario debe
suministrar un complejo conforme; detectar todas las intersecciones globales entre
simplejos distantes queda fuera del contrato.

Las regiones interiores se etiquetan con índices, máscaras, conectividades o
predicados en los baricentros de los elementos:

```python
regions = {
    "material_a": [0, 1, 2],
    "material_b": lambda x: x[:, 0] >= 0.0,
}
```

Un toro triangulado en `R3` se representa con triángulos periódicamente conectados. Su
dimensión intrínseca es dos, la ambiente es tres y su frontera exterior es vacía. El
ejemplo completo está en `examples/embedded_torus.py`.

La cuadratura usa un mapa de Duffy y reglas de Gauss-Jacobi en el simplejo de referencia.
No contiene tablas específicas para 1D, 2D o 3D. En el uso normal no hay que elegirla:

```python
G = problem.field(basis=basis)  # automática
G = problem.field(basis=basis, quadrature=10)  # fija
G = problem.field(basis=basis, quadrature=1e-8)  # adaptativa
```

El modo automático infiere el orden cuando toda la forma es polinómica a trozos y se
adapta durante la construcción para `sin`, `exp`, coeficientes programables y otras
expresiones no polinómicas. Después, la regla queda fija: evaluar `G(z)` nunca cambia
los nodos. `max_quadrature_points` impide que un coste grande se acepte silenciosamente.

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

`dx` integra en todos los simplejos y `dx("name")` en una región etiquetada.
`ds("name")` integra en las caras etiquetadas y `ds` sin etiqueta equivale a
`ds("all")`. `dx.x` son las coordenadas espaciales y
`ds.normal` es la normal exterior; en un complejo embebido es la conormal unitaria
dentro del simplejo padre.

En un complejo embebido, `grad` es siempre el gradiente tangencial expresado en las
coordenadas ambientes. El paquete proyecta las derivadas de una base programable; el
usuario no debe hacerlo manualmente.

Se proporcionan `grad`, `inner`, `contract`, `dot`, `outer`, `transpose`, `trace`,
`div`, `sym_grad`, `stack`, `sin`, `cos`, `exp`, `log`, `sqrt` y `tanh`, además de
aritmética, `@`, potencias escalares e indexación. `grad` puede componerse para solicitar
derivadas superiores de la base. La conformidad Sobolev necesaria para una forma
concreta sigue siendo una propiedad de la base elegida.

## Coeficientes espaciales fijos

Los datos del operador se expresan con `Coefficient`. Se puede elegir la representación
que resulte natural para cada campo:

```python
from ngfield import Coefficient

# f(x), evaluada una sola vez al construir G
k_function = Coefficient(lambda x: 1.0 + x[:, 0] ** 2, shape=())

# una constante por simplex
k_cell = Coefficient.cell([1.0, 1.0, 4.0, 4.0])

# un valor por vértice, interpolado linealmente
k_vertex = Coefficient.vertex(vertex_values)
```

Las formas físicas también pueden ser vectoriales o tensoriales. Por ejemplo, una
difusión anisótropa en un ambiente de dimensión `p` se escribe:

```python
K = Coefficient(lambda x: diffusion_tensor(x), shape=(p, p))


def weak(u, v, dx, ds):
    return -inner(K @ grad(u), grad(v)) * dx
```

La función de un `Coefficient` recibe todos los puntos como tensor `[Q,p]` y debe
devolver `[Q,*shape]`. Los coeficientes quedan fijos al construir `G`: no dependen del
tiempo, no son parámetros entrenables y modificarlos después exige construir otro
campo. No existe una entrada de datos crudos por punto de cuadratura.

Para una no linealidad vectorizada que sí debe evaluarse con cada estado se usa
`pointwise`:

```python
from ngfield import pointwise


def weak(u, v, dx, ds):
    reaction = pointwise(lambda y: y - y**3, u, shape=u.shape)
    return inner(reaction, v) * dx
```

Los argumentos de `pointwise` no pueden contener `v`; así la forma continúa siendo
lineal en la función de prueba. La función debe usar operaciones PyTorch vectorizadas y
preserva autograd respecto de `z`. Si se necesita `grad` de un coeficiente externo o de
una caja negra `pointwise`, su derivada se expresa por separado: el núcleo no intenta
diferenciarla espacialmente.

Las caras interiores no forman parte del alcance actual. `ds.interior` falla de forma
explícita: D-010 aproxima datos discontinuos dentro de un espacio continuo. Una futura
extensión de Galerkin discontinuo deberá definir por separado trazas, saltos, promedios
y orientación.

## Evolución temporal

Una vez construido `G`, el dato inicial se proyecta y la EDO reducida se integra sin
cambiar la base ni el operador:

```python
import torch

z0 = G.project(u0)
times = torch.linspace(0, 1, 101, dtype=G.dtype, device=G.device)

Z = G.solve(z0, times)  # RK45 adaptativo
U = G.reconstruct(Z, points)
```

`Z` tiene forma `[T,*S,N]` cuando `z0` tiene forma `[*S,N]`. Los tiempos sólo indican
dónde se devuelve la solución; RK45 elige sus propios pasos internos. La tolerancia
puede fijarse explícitamente:

```python
Z = G.solve(z0, times, tolerance=1e-7)
```

Para RK4 fijo se entrega el máximo paso interno:

```python
Z = G.solve(z0, times, step=1e-3)
```

Cada intervalo de `times` se subdivide cuando es necesario, de modo que también se
admiten tiempos de salida no uniformes. `step` y `tolerance` no se combinan. Ambos
métodos son explícitos; una difusión rígida puede requerir un paso muy pequeño.

## Error y convergencia

Los tres diagnósticos básicos se mantienen separados:

```python
space = G.projection_error(u0)
time = G.time_error(z0, times, step=1e-3)
quadrature = G.quadrature_error(Z)
```

`space` estima `||u0-P_Nu0||_L2`. Admite la misma entrada y las mismas opciones de
cuadratura que `project`. Para verificar convergencia espacial se repite la llamada con
bases de mayor tamaño o con mallas sucesivamente refinadas.

`time` compara RK4 con pasos `h` y `h/2`. Si se entrega `tolerance` en lugar de `step`,
compara dos ejecuciones RK45 con tolerancias `tol` y `tol/2`. Su forma es `[T,*S]`.

`quadrature` compara `G(Z)` con el campo ensamblado temporalmente a orden `q+2`. Puede
indicarse otro orden superior:

```python
quadrature = G.quadrature_error(Z, order=14)
```

Los resultados son indicadores absolutos de sensibilidad al refinamiento, no cotas
certificadas. La base ortonormal hace que las diferencias de coordenadas usadas por
`time_error` y `quadrature_error` coincidan con diferencias `L2` entre las funciones
reconstruidas. El campo original nunca se modifica.

## Bases ortonormales

La ruta recomendada calcula directamente una base espectral adaptada a la geometría:

```python
basis = problem.basis("laplacian", size=N, degree=2)
G = problem.field(basis=basis)
```

Los modos satisfacen el problema propio discreto

```text
K c_j = lambda_j M c_j,
c_i^T M c_j = delta_ij,
```

y se ordenan por `lambda_j`. La construcción funciona también sobre complejos
embebidos: en una superficie triangulada usa el Laplace--Beltrami discreto. Cuando la
geometría tiene frontera y no se ha construido previamente un subespacio restringido,
el problema espectral auxiliar tiene el comportamiento de frontera natural. No se
elimina silenciosamente el modo constante.

Las familias disponibles son:

```python
laplacian = problem.basis("laplacian", size=N, degree=2)
polynomial = problem.basis("polynomial", size=N)
fourier = problem.basis("fourier", size=N, periods=periods)
full_fem = problem.basis("finite-element", degree=2)
custom = problem.basis("custom", source=raw_basis)
```

`"laplacian"` es la opción general. `"polynomial"` ordena monomios por grado total y
los ortonormaliza sobre la geometría. `"fourier"` usa senos y cosenos reales; es natural
en geometrías periódicas y, si se omite `periods`, infiere escalas de la caja envolvente.
`"finite-element"` ortonormaliza el espacio nodal completo, cuya dimensión queda fijada
por malla y grado. `"custom"` ortonormaliza una familia programada por el usuario. Una
base POD basada en snapshots queda aplazada hasta acordar su contrato de datos.

La base operacional siempre es real, fija y numéricamente ortonormal en
`L2(Omega_h)`. `problem.field` rechaza una familia cruda y no permite reemplazar el
producto interno posteriormente con una `mass_matrix` arbitraria.

## Bases personalizadas y subespacios

Una familia candidata personalizada implementa:

```python
class MyBasis:
    dimension = N
    value_shape = ()

    def evaluate(self, points, *, order=0, cells=None, barycentric=None):
        # [Q,N,*value_shape,*([ambient_dimension] * order)]
        ...
```

El paquete incluye estas construcciones de bajo nivel:

- `CallableBasis`: funciones Python diferenciables; deriva automáticamente con
  `torch.func` o acepta derivadas explícitas.
- `PolynomialBasis`: monomios de grado total o exponentes elegidos, en cualquier
  dimensión.
- `FiniteElementBasis`: Lagrange nodal continuo de grado arbitrario sobre simplejos.
- `ComponentBasis`: copias escalares para campos vectoriales o tensoriales.
- `ProductBasis`: bases escalares diferentes para las componentes de un producto.

`TransformedBasis` aplica una combinación lineal a cualquier base. Permite construir
primero el subespacio admisible y ortonormalizarlo después:

```python
raw = FiniteElementBasis(problem.geometry, degree=2)
admissible = TransformedBasis(raw, constraints)
basis = problem.orthonormalize(admissible)
G = problem.field(basis=basis)
```

Las bases son fijas. Sus tablas se congelan al construir `G`; modificar después un
parámetro capturado por `CallableBasis` no cambia el campo ya preparado. Autograd se
conserva con respecto a `z`.

Para una familia Lagrange se pueden entregar coeficientes
`[grados_globales,N,*value_shape]`. Sus columnas pueden codificar condiciones de
frontera, periodicidad, restricciones de divergencia, acoplamiento entre componentes
o cualquier otro subespacio lineal construido externamente. Las restricciones deben
aplicarse antes de la ortonormalización.

Para componentes con cantidades distintas de modos:

```python
from ngfield import ProductBasis

basis = ProductBasis(
    [
        problem.basis("laplacian", size=12),
        problem.basis("laplacian", size=8),
        problem.basis("polynomial", size=5),
    ]
)
```

La dimensión total es `12 + 8 + 5` y `value_shape == (3,)`. Los modos se ordenan por
componente. Si todas usan la misma base, `ComponentBasis(scalar, components=d)` es el
atajo correspondiente.

## Significado de G

Para `z` de forma `[N]` o `[B,N]`, la base define

```text
u_z = sum_j z_j phi_j.
```

Como la base operacional es ortonormal,

```text
G_i(z) = a(u_z; phi_i),
norm(u_z)_L2 = norm(z)_2.
```

Por ello `G(z)` contiene directamente la velocidad de los coeficientes de `u_z`, y las
distancias euclídeas en coordenadas son las distancias funcionales L2. El paquete
evalúa este campo y conserva autograd respecto de `z`; no integra una evolución
temporal.

## Proyección de una función

Para obtener los coeficientes de una condición inicial o de cualquier función física:

```python
u0 = lambda x: torch.sin(x[:, 0])
z0 = G.project(u0)
velocity = G(z0)
u0_projected = G.reconstruct(z0)
```

`G.project` calcula `z_i=<u0,phi_i>`. La ortonormalidad evita una corrección con la
matriz de masa. También se aceptan `Coefficient(function)`, `Coefficient.cell(values)`
y `Coefficient.vertex(values)`.

Una función con lotes devuelve `[*S,Q,*value_shape]` y la proyección conserva los ejes
libres como `[*S,N]`. No se aceptan valores crudos asociados implícitamente a los puntos
internos de cuadratura.

La selección de cuadratura mantiene la misma interfaz compacta:

```python
z0 = G.project(u0)  # automática
z0 = G.project(u0, quadrature=10)  # fija
z0 = G.project(u0, quadrature=1e-8)  # adaptativa
```

Una función PyTorch conserva autograd respecto de sus parámetros. Un `Coefficient` es
un dato fijo y se mantiene fuera del grafo. La proyección no modifica la base, el campo
ni sus tablas.

## Aproximación continua de una discontinuidad

Una discontinuidad alineada con las caras de la malla se describe naturalmente con un
valor por simplex y se proyecta sobre la base continua fija:

```python
u0 = Coefficient.cell([1.0, 1.0, 0.0, 0.0])
z0 = G.project(u0)

points = torch.tensor([[0.49], [0.50], [0.51]], dtype=G.dtype, device=G.device)
approximation = G.reconstruct(z0, points)
```

`G.project` produce la mejor aproximación en `L2` dentro de `V_N`. El resultado tiene
una sola traza continua en la interfaz; no se preserva un salto exacto. Al refinar una
base local, la transición se concentra en una región menor. Con bases globales pueden
aparecer oscilaciones de Gibbs.

Para un salto que no coincide con la malla se puede entregar una función y comprobar
la cuadratura explícitamente:

```python
z0 = G.project(
    lambda x: (x[:, 0] < 0.5).to(G.dtype),
    quadrature=20,
)
```

Esta aproximación se interpreta en `L2`. Si la formulación exige `grad(u)`, un salto
verdadero puede quedar fuera de `H1` y requerir cada vez gradientes mayores. Los
choques persistentes, fracturas o contactos imperfectos motivarán en el futuro un
contrato separado para Galerkin discontinuo y `ds.interior`.

## Evaluación en puntos físicos

Una vez conocidas las coordenadas `z`, el campo reconstruido y sus derivadas espaciales
se evalúan con tres métodos directos:

```python
points = torch.tensor([[0.2], [0.8]], dtype=G.dtype, device=G.device)

values = G.reconstruct(z, points)
gradients = G.grad(z, points)
Hessians = G.hessian(z, points)
```

Para `z:[*S,N]` y `points:[Q,p]`, las salidas tienen respectivamente las formas

```text
[*S,Q,*value_shape]
[*S,Q,*value_shape,p]
[*S,Q,*value_shape,p,p].
```

El paquete localiza automáticamente cada punto en la malla. Si un punto está sobre una
cara y la cantidad solicitada tiene trazas distintas, se selecciona el simplex padre
de forma explícita:

```python
cells = torch.tensor([0, 3], dtype=torch.int64, device=G.device)
gradients = G.grad(z, points, cells=cells)
```

`cells[q]` es el índice del simplex que contiene `points[q]`; la asignación se valida y
no permite extrapolación. En una malla embebida, `grad` y `hessian` son tangenciales al
simplex y conservan los `p` ejes ambientes. Son derivadas por elemento: para una base
`P1`, el Hessiano es cero dentro de cada simplex y el gradiente puede saltar en sus
caras.

Estos métodos derivan `u_z(x)` respecto de `x`. El Jacobiano o el Hessiano de
`G:R^N->R^N` respecto de `z` continúan calculándose con `torch.func`.

## Evaluación tensorial y diferenciación

El último eje siempre contiene los `N` coeficientes. Todos los ejes anteriores se
conservan como ejes de lote:

```python
velocity = G(z)  # [N] -> [N]
velocities = G(states)  # [B,N] -> [B,N]
grid_velocities = G(grid)  # [T,B,N] -> [T,B,N]
```

Esto equivale a evaluar el mismo campo de forma independiente en cada estado. También
se preservan lotes de rango superior y tamaños cero. `G.reconstruct(z)` reemplaza el
último eje de coeficientes por los puntos de cuadratura y la forma del valor:

```text
[*S,N] -> [*S,Q,*value_shape].
```

La entrada debe ser un tensor PyTorch con el mismo dispositivo y precisión que `G`;
no se realizan conversiones silenciosas. Para cambiar ambos de forma explícita:

```python
G.to(device="cuda", dtype=torch.float64)
z = z.to(device=G.device, dtype=G.dtype)
```

El campo funciona directamente con las transformaciones de PyTorch:

```python
J = torch.func.jacrev(G)(z)
_, Jw = torch.func.jvp(G, (z,), (w,))
value, pullback = torch.func.vjp(G, z)
```

Las funciones usadas en `pointwise` deben estar escritas con operaciones PyTorch
puras y compatibles con `torch.func`. La diferenciación se conserva respecto de `z`,
pero no respecto de la geometría, la base, los `Coefficient` ni la cuadratura ya
tabulada.

## Campos vectoriales y tensoriales

```python
scalar = problem.basis("polynomial", size=10)
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
    quadrature=8,
    device="cuda",
    dtype=torch.float64,
    max_quadrature_points=2_000_000,
)
```

La cuadratura comprueba que la matriz de Gram sea numéricamente la identidad. Una
familia no ortonormal produce un error con instrucciones para usar `problem.basis` o
`problem.orthonormalize`. La evaluación divide automáticamente los modos de prueba
para respetar `max_intermediate_entries`.

`G.to(device=..., dtype=...)` mueve las tablas, incluidos los `Coefficient` ya
tabulados. Se admiten `float32` y `float64`.

## Persistencia

`FiniteElementBasis.save(path)` conserva geometría, regiones, fronteras, numeración,
coeficientes, familia espectral, órdenes de cuadratura y valores propios en NPZ sin
pickle. Esto permite guardar exactamente la base laplaciana usada durante el
entrenamiento. `FiniteElementBasis.load(path)` restaura sus funciones y orden. Una
`CallableBasis` contiene código Python y no se serializa como datos ejecutables.

La API 0.1 basada en `Domain`, `FEMSpace`, `Problem` y `GalerkinBasis` se conserva por
compatibilidad. Los nuevos desarrollos deben usar `GalerkinProblem` y una base explícita.
