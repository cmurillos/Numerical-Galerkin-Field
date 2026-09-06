# Interfaz general de Numerical Galerkin Field

Esta guía conserva el recorrido de 0.9.0 y documenta `Space`, `ZeroTrace`, `Periodic`, `MeanZero` y
`V.basis(...)` añadidos en esta rama. El
[contrato D-013](design-contract.md#d-013--contrato-de-uso-con-espacio-admisible-explícito)
registra el recorrido completo, implementado hasta la construcción directa de
`GalerkinField` en la parte 6. Estas incorporaciones están en la rama de desarrollo;
la versión publicada 0.9.0 conserva las llamadas de compatibilidad descritas aquí.

## Problema mínimo — construcción desde Space

El usuario prepara la geometría, declara el espacio, elige la base y proporciona
la forma débil. Este ejemplo ejecutable construye el campo del calor con extremos
fijos en cero; no requiere todavía una condición inicial ni tiempos:

```python
import numpy as np
from ngfield import GalerkinField, SimplicialDomain, Space, ZeroTrace, grad, inner

vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(vertices, simplices)
V = Space(
    geometry=geometry,
    components=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
basis = V.basis("laplacian", size=8, degree=1)


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u[0]), grad(v[0])) * dx


G = GalerkinField(basis=basis, weak=weak)
```

`G.basis is basis`, `G.space is V` y `G.geometry is geometry`. Su dimensión es ocho
coordenadas, la forma de valor físico es `(1,)` y el campo cumple
`G: R^8 -> R^8`, con `G_i(z)=a(Phi(z);phi_i)`.

La base entrega la geometría y sus etiquetas; no se repiten esos argumentos en la
construcción del campo. La condición esencial ya fue aplicada al construir la
base, antes del problema espectral. La forma sigue usando componentes explícitas,
`dx`, `dx("region")` y `ds("boundary")`. Cambiar la geometría o las restricciones
requiere preparar un espacio y una base compatibles.

Los controles numéricos conservan sus nombres y significado:

```python
G = GalerkinField(
    basis=basis,
    weak=weak,
    quadrature=8,
    max_quadrature_points=1_000_000,
    max_intermediate_entries=10_000_000,
)
```

También admite `device` y `dtype`; por defecto usa CPU y `torch.float64`.
`quadrature=None` selecciona el procedimiento automático existente, un entero
selecciona el orden y un real en (0,1) solicita adaptación por tolerancia. La
construcción valida el Gram con su cuadratura; no normaliza nuevamente la base ni
cambia sus coordenadas para hacerla pasar. `G.mass_matrix` es un diagnóstico del
Gram calculado, no un argumento adicional que deba aportar el usuario.

| Consulta u operación | Significado |
|---|---|
| `G.space`, `G.geometry`, `G.basis` | Espacio asociado, geometría usada y base operacional. |
| `G.dimension`, `G.value_shape` | Dimensión reducida total N y componentes físicas `(c,)`. |
| `G(z)` | Estado de forma `[...,N]` a velocidad de la misma forma. |
| `G.project(u0)` | Función física a coordenadas; con Space, `u0(x)` debe devolver `[points,c]`. |
| `G.reconstruct(Z, points)` | Coordenadas a valores `[...,points,c]`. |
| `G.solve(z0, times)` | Integración temporal de la dinámica autónoma ya preparada. |

Las derivadas de PyTorch respecto de z, los lotes vacíos, el traslado de tablas y
los métodos de evaluación espacial conservan el comportamiento documentado. Los
coeficientes fijos se tabulan durante la preparación; evaluar G no vuelve a llamar
la forma débil ni las fuentes de coeficientes.

La entrada directa exige una base asociada a `Space`, normalmente obtenida con
`V.basis`. Comprueba la forma de valor y la malla. Si se usa además un problema
explícito con esa base, exige los mismos conjuntos etiquetados de fronteras y
regiones; acepta otra instancia geométrica con los mismos arreglos y conjuntos,
pero no admite cambiar el significado de una etiqueta. La geometría de `Space`
es la referencia para las etiquetas, incluso cuando la base candidata procedía
de un portador nodal con otras etiquetas sobre la misma malla.

La admisibilidad procede de la construcción de la base. G comprueba la consistencia
de la declaración, el contrato de evaluación, la forma y el Gram; no vuelve a
certificar la regularidad de callbacks ni aplica restricciones por añadir
manualmente un atributo `.space`. Las fuentes, la base y la geometría deben
permanecer fijas. `regularity_verified=False` conserva su significado de propiedad
declarada, aunque el Gram sea correcto.

`TransformedBasis` conserva el Space de su base al formar combinaciones lineales,
porque las restricciones implementadas son homogéneas. También lo conserva la
ortonormalización explícita de una familia obtenida con `V.restrict`. Una rotación
ortonormal puede usarse directamente; una transformación que altere el Gram se
rechaza hasta que el usuario normalice explícitamente su nueva base. Los autovalores
no se copian a una transformación arbitraria.

## Compatibilidad de construcción

| Llamada | Entradas |
|---|---|
| `GalerkinField(basis=basis, weak=weak)` | Base asociada a Space y forma débil; nuevo recorrido. |
| `problem.field(basis=basis)` | `GalerkinProblem` explícito; conserva la interfaz existente. |
| `GalerkinField(problem, basis)` | El mismo problema general como primer argumento. |
| `GalerkinField(problem=problem, basis=basis)` | La misma ruta general usando argumentos nombrados. |
| `GalerkinField(legacy_basis, legacy_problem)` | `GalerkinBasis` y `Problem` de la interfaz original; conserva su orden. |

La última ruta también admite ambos argumentos nombrados. `GeneralGalerkinField`
expone la clase general y acepta tanto el nuevo recorrido como el problema general
explícito. El nombre público `GalerkinField` sigue siendo una función que selecciona
la construcción correspondiente.

No se combinan `weak` y `problem` en una llamada, ni se proporciona una geometría
alternativa a la ruta directa. Una base sin Space usa el problema explícito; las
bases escalares existentes mantienen `value_shape=()` y su reconstrucción escalar.
En ese caso `G.space` es `None`. No hay promoción silenciosa de componentes.

La carga existente de una base FEM conserva sus funciones y geometría, pero no la
declaración de Space; sigue entrando mediante un `GalerkinProblem` explícito. La
serialización completa del espacio no se incorpora en esta etapa.

## Ejemplo compatible con GalerkinProblem

La construcción anterior sigue recibiendo la geometría simplicial y una forma débil completa:

```python
import numpy as np
from ngfield import GalerkinProblem, grad, inner, sin


vertices = np.array([[0.0], [0.5], [1.0]])
simplices = np.array([[0, 1], [1, 2]])


def weak(u, v, dx, ds):
    return -0.1 * inner(grad(u), grad(v)) * dx


problem = GalerkinProblem(vertices=vertices, simplices=simplices, weak=weak)
basis = problem.basis("laplacian", size=3, degree=1)
G = problem.field(basis=basis)
```

En este ejemplo de compatibilidad no se declaran restricciones esenciales; la base
suministrada define el espacio admisible. Las medidas de frontera sólo añaden los
términos escritos en `weak`.

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

## Descripción del espacio admisible — D-013, parte 2

Sobre un `SimplicialDomain` ya construido, se puede describir el espacio antes de
elegir una base:

```python
from ngfield import Space

V = Space(
    geometry=geometry,
    components=1,
    regularity=1,
    restrictions=[],
)
```

Este ejemplo declara H1 escalar sobre la geometría, con producto de estados L2.
`components` cuenta componentes físicas, independientemente de la dimensión del
dominio o del espacio ambiente. `V.value_shape` es `(1,)` para una componente y `(c,)`
para `c` componentes. Un toro superficial puede tener una componente y dimensiones
intrínseca dos y ambiente tres, consultables en `V.geometry`.

`regularity` declara un orden Sobolev entero no negativo, común a las componentes:
cero para L2, uno para H1. Su valor por defecto es uno. Declarar un orden superior no
construye ni certifica una base conforme de ese orden. El número de modos se decide
después, al elegir la base, y no es una propiedad de este descriptor.

`Space` conserva la geometría por referencia y no altera sus regiones, medidas ni
fronteras. Sus atributos no se reasignan y la lista de restricciones se copia
como tupla. La geometría compartida debe mantenerse fija, igual que al construir una
base o un campo.

`restrictions` admite una lista o tupla de objetos `ZeroTrace`, `Periodic` y
`MeanZero`; omitir el argumento
equivale a la tupla vacía. Los tipos no implementados se rechazan explícitamente.
Una etiqueta de frontera nunca impone por sí sola una restricción.

La base se construye con `V.basis(...)`, como se describe en la parte 4 a continuación.
El campo puede construirse directamente con `GalerkinField(basis=basis, weak=weak)`
o mediante `GalerkinProblem.field`. Las bases escalares
existentes con `value_shape=()` conservan su comportamiento.

## Traza cero por componente — D-013, parte 3

Sobre una geometría que incluya la etiqueta de frontera `"fixed"`:

```python
from ngfield import ComponentBasis, FiniteElementBasis, Space, ZeroTrace

V = Space(
    geometry=geometry,
    components=1,
    regularity=1,
    restrictions=[ZeroTrace(component=0, boundary="fixed")],
)
raw = ComponentBasis(FiniteElementBasis(geometry, degree=2), components=V.components)
admissible = V.restrict(raw)
```

`admissible` representa todas las combinaciones de la familia candidata cuya
componente cero se anula en esa frontera, hasta la tolerancia algebraica de la
construcción. Se imponen todos los grados de libertad de las caras, incluidos los
nodos interiores en grado alto. La traza polinómica completa queda determinada por
esos valores; no se comprueba solamente en los vértices.

Se admiten bases nodales `FiniteElementBasis` y sus composiciones con `ComponentBasis`,
`ProductBasis` y `TransformedBasis`, incluidos modos acoplados. La familia candidata
debe tener forma de valor `(V.components,)` y la misma malla que `V.geometry`.
Varias restricciones pueden dirigirse a componentes diferentes; las declaraciones
idénticas y las caras superpuestas no duplican las condiciones independientes.

La familia devuelta todavía debe ortonormalizarse para construir el campo:

```python
problem = GalerkinProblem(geometry=geometry, weak=weak)
basis = problem.orthonormalize(admissible)
G = problem.field(basis=basis)
```

Las restricciones no alteran los términos de `weak`: Robin y Neumann se siguen
escribiendo mediante las medidas de frontera. `V.restrict` tampoco selecciona los
modos laplacianos; `V.basis("laplacian", ...)` realiza esa selección en el espacio
restringido, como se describe a continuación.

El cálculo del núcleo usa SVD y permite `tolerance=1e-12` y
`max_matrix_entries=10_000_000` como valores predeterminados. El segundo limita la
preparación algebraica densa. El resultado registra `restriction_rank` y un residual
normalizado `restriction_error`, y conserva `space=V`. No modifica la base de entrada;
con restricciones vacías devuelve la familia admitida sin cambios.

Se rechazan etiquetas inexistentes o vacías, componentes fuera de rango, traza sobre
un espacio declarado sólo L2, familias dependientes, un subespacio resultante nulo y
bases programables sin representación nodal verificable. Esta operación admite
conformidad H1 y rechaza órdenes mayores; declarar H2 no convierte elementos Lagrange
continuos en elementos conformes H2. Para un toro cerrado se dejan vacías las
restricciones de frontera.

## Bases del espacio admisible — D-013, parte 4

La base se prepara antes de proporcionar la forma débil. Este ejemplo completo
construye cuatro modos de H1 con extremos fijos:

```python
import numpy as np
from ngfield import SimplicialDomain, Space, ZeroTrace

vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(vertices, simplices)
V = Space(
    geometry=geometry,
    components=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
basis = V.basis("laplacian", size=4, degree=1)
```

`basis.dimension == 4`, `basis.value_shape == (1,)` y `basis.space is V`.
La biblioteca elimina los grados de libertad de las caras restringidas del problema
FEM completo **antes de calcular los autovectores**. Conserva nodos de caras de grado
alto y geometrías embebidas bajo el mismo contrato. Una superficie cerrada, como el
toro triangulado, usa `restrictions=[]` y conserva el modo constante si se incluye
el menor autovalor. Se trabaja sobre la superficie afín triangulada.

`size=N` siempre es la dimensión total. Para varias componentes hay dos elecciones:

```python
W = Space(geometry=geometry, components=2)
global_basis = W.basis("laplacian", size=16)
allocated_basis = W.basis("laplacian", size=16, component_sizes=(8, 8))
```

La primera selecciona los 16 menores autovalores de la suma directa de los espacios
restringidos. Los autovalores repetidos pueden dar modos que mezclan componentes;
no se garantiza un reparto ni que cada componente esté representada cuando N es
pequeño. `global_basis.component_sizes` es `None`.

La segunda reserva ocho modos para cada componente y los ordena por componente,
con autovalores crecientes dentro de cada bloque. `component_sizes` acepta una lista
o tupla de enteros no negativos; su suma debe ser positiva y coincidir con `size`
si ambos se proporcionan. También se puede omitir `size` y usar sólo esa tupla.
Un cero excluye esa componente de la aproximación sin cambiar la forma del estado.
No se puede exceder la dimensión admisible de ninguna componente.

| Familia en `V.basis` | Tamaño | Restricciones admitidas en esta etapa |
|---|---|---|
| `"laplacian"` | `size` total o `component_sizes`; selección espectral restringida. | `ZeroTrace`, `Periodic`, `MeanZero` y sus combinaciones nodales. |
| `"finite-element"` | Todos los grados de libertad libres; `size` opcional comprueba la dimensión. | `ZeroTrace`, `Periodic`, `MeanZero` y sus combinaciones nodales. |
| `"polynomial"` | `size` total o `degree` para todos los monomios por componente. | Sin restricciones adicionales. |
| `"fourier"` | `size` total o `component_sizes`. | Sin restricciones adicionales; no certifica identificaciones periódicas de la geometría. |
| `"custom"` | Todo el espacio de la fuente después de restringirlo; `size` opcional comprueba la dimensión. | Las mismas combinaciones en representaciones nodales de la parte 3. |

Para polinomios y Fourier, el reparto predeterminado es equilibrado: con
`size = q*c + r` se asignan `q+1` modos a las primeras `r` componentes y `q` a las
restantes. `component_sizes` permite cambiarlo; los modos quedan agrupados por
componente. Las opciones escalares de las familias existentes (`degree`, `periods`,
`origin`, `center`, `scale`) conservan su significado. Los polinomios son funciones
de coordenadas ambientes: una familia dependiente al restringirse a la geometría
puede fallar la ortonormalización y no se elimina automáticamente.

En `finite-element`, el grado y la malla determinan la dimensión por componente,
menos los nodos restringidos. No se trunca la base nodal; un `component_sizes`
explícito debe coincidir con esas dimensiones. En `custom`, se conserva toda la
familia admisible y no se acepta `component_sizes`, porque sus modos pueden estar
acoplados. Por ejemplo:

```python
from ngfield import ComponentBasis, FiniteElementBasis

source = ComponentBasis(FiniteElementBasis(geometry, degree=2), components=1)
custom_basis = V.basis("custom", source=source, quadrature_order=6)
```

La fuente debe tener forma de valor `(V.components,)`. Cuando hay restricciones, primero
se calcula su núcleo nodal y después se ortonormaliza. Un callback arbitrario sin
restricciones también puede entrar como fuente, pero su pertenencia al espacio
Sobolev queda bajo responsabilidad del usuario: `basis.regularity_verified` será
`False`. En las familias construidas y las composiciones nodales/polinómicas
conocidas será `True` para la regularidad soportada. Las evaluaciones numéricas no
certifican regularidad de un callback ni su traza completa. `regularity>1` se rechaza
por ahora en todas las familias de `V.basis`.

La preparación conserva `quadrature_order`, `validation_order` y
`orthonormality_error`. En las familias nodales el orden predeterminado es
`2*degree`, la validación usa dos órdenes más, y ambos órdenes deben ser al menos
`2*degree`. Los modos laplacianos incluyen `eigenvalues`; las dos familias nodales
incluyen `admissible_dofs` por componente y el rango de las restricciones.
`restriction_error` registra un residual normalizado previo a la normalización: cero
para las identificaciones y fijaciones nodales, o el residual del núcleo integral
cuando hay medias. En `custom` sigue el significado de la SVD de la parte 3.
No es una cota de error físico.

Se conservan los controles de cuadratura y tolerancia de las familias anteriores.
`max_matrix_entries` limita las matrices principales de preparación (20 millones
por defecto; 10 millones en `custom`), sin ser una cota exacta de memoria total.
`max_dofs` limita la construcción nodal y `max_quadrature_points` limita la cuadratura.
La ortonormalización de la familia FEM completa utiliza matrices densas: para una
aproximación reducida se puede elegir `"laplacian"`. La preparación usa CPU y float64;
el campo posterior puede tabular la base en otro dispositivo y precisión.

El vínculo `.space` y los diagnósticos nuevos se conservan en el objeto en memoria.
La serialización existente de `FiniteElementBasis` conserva los coeficientes y su
geometría, pero todavía no reconstruye la declaración de `Space` ni estos metadatos.
No hay cambios silenciosos de coordenadas al construir el campo. La conexión
explícita siguiente permanece disponible junto a la llamada directa de la parte 6:

```python
from ngfield import GalerkinProblem, grad, inner


def weak(u, v, dx, ds):
    return -inner(grad(u[0]), grad(v[0])) * dx


G = GalerkinProblem(geometry=geometry, weak=weak).field(basis=basis)
```

Las condiciones naturales del problema espectral sólo definen la familia de
aproximación. Los coeficientes y términos Robin/Neumann de la evolución siguen en
`weak`, usando `dx`, `dx("region")` y `ds("boundary")`.

## Periodicidad, media cero y fronteras fijas — D-013, parte 5

La [guía de periodicidad y fronteras fijas](periodic-and-fixed-boundaries.md) contiene
la API, ejemplos ejecutables, el contrato algebraico y los límites de esta etapa.
`Periodic` empareja vértices de fronteras con conectividad compatible y extiende la
igualdad a las trazas completas de grado alto. `MeanZero` integra una componente
sobre el dominio o una región etiquetada. Ambas se combinan con `ZeroTrace` antes
de seleccionar modos en `laplacian`, `finite-element` y fuentes nodales `custom`.

La guía también desarrolla Robin y Neumann con datos espaciales fijos y Dirichlet
no homogénea mediante un levantamiento explícito `T=ell+w`. Explica la proyección
de `T0-ell`, la reconstrucción de T y cuándo media cero es compatible con la EDP.

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
