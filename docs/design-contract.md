# Contrato de diseño

Este documento registra decisiones estables de Numerical Galerkin Field. Una decisión
aceptada sólo cambia mediante una revisión explícita del contrato.

## D-001 — Bases fijas

**Estado:** aceptada.

El espacio reducido se define mediante una base fija

```text
V_N = span{phi_1, ..., phi_N}.
```

Al construir `G = problem.field(basis=basis)`, la base se evalúa, se valida y sus tablas
se separan del grafo de diferenciación. Cambiar posteriormente un parámetro capturado
por una base programable no modifica `G`. Autograd se conserva respecto de los
coeficientes `z`, no respecto de los parámetros usados para definir la base.

## D-002 — Interfaz canónica

**Estado:** aceptada.

La construcción principal es

```python
problem = GalerkinProblem(
    vertices=P,
    simplices=T,
    boundaries=boundaries,
    regions=regions,
    weak=weak,
)

G = problem.field(basis=basis)
```

Una geometría reutilizable ofrece una entrada equivalente:

```python
geometry = SimplicialDomain(
    vertices=P,
    simplices=T,
    boundaries=boundaries,
    regions=regions,
)

problem = GalerkinProblem(geometry=geometry, weak=weak)
```

`geometry` y los argumentos geométricos directos son mutuamente excluyentes. El objeto
geométrico permite reutilizar el dominio y deja un punto de extensión para futuras
geometrías mapeadas sin cambiar `GalerkinProblem`.

## D-003 — Geometría simplicial afín ND

**Estado:** aceptada.

### Representación

La geometría discreta es

```text
G_h = (P, T, R, B),
```

con

```text
P in R^(n_v x p),
T in {0, ..., n_v-1}^(n_K x (k+1)),
1 <= k <= p.
```

Cada fila de `T` define

```text
K_e = conv{P[T_e,0], ..., P[T_e,k]},
Omega_h = union_e K_e.
```

La dimensión ambiente `p` y la dimensión intrínseca `k` se infieren de los arreglos.
No se reciben como parámetros independientes.

El contrato admite dominios no convexos, desconectados, con agujeros y complejos
inmersos en una dimensión mayor. En particular, un toro triangulado es un complejo con
`k=2`, `p=3` y frontera vacía. La superficie suave se representa mediante su
aproximación simplicial afín.

### Hipótesis geométricas

El complejo debe ser:

- puro: todos los elementos máximos tienen dimensión `k`;
- no degenerado: cada simplex tiene rango afín `k`;
- conforme: dos elementos se intersectan sólo en una cara común o no se intersectan;
- manifold en codimensión uno: cada faceta pertenece a uno o dos elementos.

Las facetas con un elemento incidente forman la frontera exterior. Las facetas con dos
elementos incidentes son interiores. Una faceta con más de dos elementos se rechaza
como incidencia no-manifold.

La biblioteca valida índices, duplicados, rango local e incidencias. El contrato exige
conformidad global, pero no promete detectar todas las intersecciones entre elementos
distantes en dimensión arbitraria.

### Fronteras

La frontera completa se deriva de `T`; `boundaries` sólo asigna nombres a subconjuntos
de sus facetas. No codifica Dirichlet, Neumann, Robin ni otro tipo de condición.

Una etiqueta puede darse mediante conectividades de facetas o mediante un predicado
booleano evaluado en sus baricentros:

```python
boundaries = {
    "inlet": inlet_facets,
    "outlet": lambda x: x[:, 0] > 0.99,
}
```

La etiqueta reservada `"all"` representa toda la frontera y puede estar vacía. Una
faceta puede pertenecer a varias etiquetas. En las formas débiles,

```text
ds             integra en toda la frontera,
ds("name")     integra en la frontera etiquetada.
```

### Regiones

`regions` asigna nombres a subconjuntos de los símplices máximos. Una región admite
índices de elementos, máscara booleana, conectividades completas o un predicado en los
baricentros:

```python
regions = {
    "material_a": cells_a,
    "material_b": lambda x: x[:, 0] >= 0.0,
}
```

La etiqueta reservada `"all"` contiene todos los elementos. Las regiones pueden
superponerse; cada integral cuenta exactamente los elementos de su etiqueta:

```text
dx                   integra en Omega_h,
dx("material_a")     integra en la región etiquetada.
```

### Medida y gradiente tangencial

Para un elemento con matriz de aristas

```text
B_e = [P[T_e,1]-P[T_e,0], ..., P[T_e,k]-P[T_e,0]],
```

la medida inducida usa

```text
J_e = sqrt(det(B_e^T B_e)).
```

Si `k=p`, coincide con `abs(det(B_e))`. Si `k<p`, la medida es la medida
`k`-dimensional inducida por la inmersión.

`grad(u)` siempre representa el gradiente tangencial. Si `Q_e` tiene columnas
ortonormales que generan el espacio tangente de `K_e`, se usa

```text
Pi_e = Q_e Q_e^T,
grad_Omega u = Pi_e grad(tilde u).
```

El resultado se expresa en las `p` coordenadas ambientes, no en coordenadas locales.
La biblioteca proyecta cada eje derivativo suministrado por la base. En particular, el
resultado no depende de cómo una función definida sobre `Omega_h` se extienda fuera de
un complejo inmerso.

### Conormal exterior

`ds.normal` representa la conormal exterior `nu_(K,F)`. Es un vector unitario en
`R^p` que satisface

```text
nu_(K,F) in T_K,
nu_(K,F) perpendicular a T_F,
norm(nu_(K,F)) = 1,
```

y apunta fuera del elemento padre. Para `k=p` es la normal exterior usual. Para
`k<p` es tangente al elemento y normal a su faceta. No se define todavía una normal
ambiental orientada de una superficie.

### Orientación y geometrías curvas

El núcleo es no orientado. `dx`, `ds`, el gradiente tangencial y la conormal exterior
no requieren que el usuario suministre una orientación global.

Los elementos son afines. Una geometría curva se aproxima mediante la malla simplicial.
Una representación curva exacta requerirá un contrato posterior para geometrías
mapeadas o isoparamétricas.

## Criterios de aceptación de D-003

La implementación debe verificar al menos:

1. símplices de dimensión arbitraria con `1 <= k <= p`;
2. medida inducida y conormales en complejos inmersos;
3. un toro triangulado cerrado en `R3` con frontera vacía;
4. regiones nombradas mediante `dx("name")`;
5. fronteras nombradas mediante `ds("name")`;
6. invariancia del gradiente tangencial frente a extensiones ambientales;
7. equivalencia entre la entrada directa y `geometry=...`;
8. persistencia de regiones y fronteras en bases de elementos finitos.

## D-004 — Lenguaje de formas débiles y coeficientes fijos

**Estado:** aceptada.

### Una sola forma débil

El usuario describe la aplicación completa

```text
a(u;v) = suma de integrales de volumen y de frontera
```

mediante una sola función:

```python
def weak(u, v, dx, ds):
    return ...
```

`u` es el estado reconstruido y puede aparecer de forma no lineal. Cada integrando
no nulo debe depender exactamente de forma lineal de `v`. La biblioteca rechaza
integrandos independientes de `v` y dependencias cuadráticas, no lineales o en
denominadores que contengan `v`.

`dx`, `dx("region")`, `ds` y `ds("boundary")` sólo seleccionan medidas. No introducen
tipos de condición de frontera. `dx.x` proporciona el punto físico y `ds.normal` la
conormal exterior definida en D-003.

### Coeficientes espaciales

`Coefficient` representa un campo espacial fijo que forma parte del operador. En la
primera versión se admiten exactamente tres fuentes:

```python
Coefficient(function, shape=value_shape)
Coefficient.cell(values)  # [numero_de_simplices, *value_shape]
Coefficient.vertex(values)  # [numero_de_vertices, *value_shape]
```

La función recibe puntos físicos `[Q,p]` y devuelve `[Q,*value_shape]`. Los valores por
simplex se interpretan como constantes a trozos. Los valores por vértice se interpolan
linealmente mediante coordenadas baricéntricas, también sobre caras de frontera.

Todos estos coeficientes son autónomos, no entrenables y no variables en el tiempo.
Se copian o evalúan al construir `G`, se separan del grafo de autograd y después se
mueven con `G.to(...)`. Cambiar la función, sus parámetros capturados o los datos de
origen no modifica un campo ya construido.

No se aceptan inicialmente valores crudos por punto de cuadratura. Esos datos estarían
ligados a una regla, orden y enumeración internos y no definirían por sí solos un campo
geométrico reutilizable.

### Operadores ND

El lenguaje incluye aritmética e indexación, `grad`, `inner`, `contract`, `dot`,
`outer`, `transpose`, `trace`, `div`, `sym_grad` y `stack`. Todos operan sobre los ejes
físicos sin contraer los ejes internos de lote, función de prueba o cuadratura.

`inner(a,b)` contrae todos los ejes físicos de igual forma. `a @ b` equivale a
`contract(a,b,axes=1)`. `contract` admite un número de ejes finales/iniciales o dos
listas explícitas de ejes. `transpose` intercambia los dos últimos ejes, `trace` los
contrae, `div` requiere que el último eje tenga tamaño `p`, y `sym_grad` requiere un
campo vectorial de forma `(p,)`.

`pointwise(function, *values, shape=...)` permite una operación PyTorch vectorizada no
lineal. Sus argumentos pueden depender de `u`, de coordenadas y de coeficientes, pero
no de `v`. El usuario declara la forma física de la salida. La función se evalúa al
evaluar `G`, por lo que autograd respecto de los coeficientes de estado `z` se conserva.

Las derivadas espaciales de `u`, `v`, coordenadas y expresiones algebraicas se obtienen
componiendo `grad`. D-004 no intenta diferenciar espacialmente una función externa de
`Coefficient` ni una caja negra `pointwise`; esas derivadas deben suministrarse como
expresiones o coeficientes separados.

### Caras interiores

Las caras interiores no se activan en D-004. La expresión reservada `ds.interior`
produce un error explícito hasta definir un contrato independiente para trazas de ambos
lados, saltos, promedios y orientación. Las integrales de frontera exterior sí forman
parte de D-004.

## Criterios de aceptación de D-004

La implementación debe verificar al menos:

1. coeficientes escalares y tensoriales definidos por función;
2. coeficientes constantes por simplex e interpolados por vértice;
3. tabulación fija sin gradientes hacia los datos externos;
4. uso de coeficientes tanto en `dx` como en `ds`;
5. operadores tensoriales ND y contracciones que preserven lotes;
6. no linealidades `pointwise` diferenciables respecto de `z`;
7. rechazo de toda forma que no sea exactamente lineal en `v`;
8. rechazo explícito de caras interiores hasta acordar su contrato.

## D-005 — Contrato de bases ortonormales

**Estado:** aceptada.

### Base matemática y coordenadas

Una familia candidata es una lista real, finita y ordenada

```text
B = (phi_1, ..., phi_N),
phi_j: Omega_h -> R^(s_1 x ... x s_r).
```

`dimension=N` y `value_shape=(s_1,...,s_r)` forman parte de su identidad. El índice no
se reordena al construir un campo: `z_j` es siempre el coeficiente de `phi_j`. Para
campos escalares `value_shape=()`.

La misma familia se usa para reconstruir el estado y para probar la forma débil. Una
elección distinta de espacios de prueba y estado sería Petrov--Galerkin y requiere un
contrato posterior.

La base operacional que recibe `problem.field` debe satisfacer

```text
<phi_i, phi_j>_L2 = delta_ij.
```

El producto interno de valores es el producto euclídeo o de Frobenius estándar. Por
ello la síntesis `Phi_N z=sum_j z_j phi_j` es una isometría, `norm(Phi_N z)_L2=norm(z)_2`
y `Phi_N(B_R)=V_N` intersección `B_L2(0,R)`. No se admite una `mass_matrix` posterior
que cambie esta métrica.

Las bases son reales para conservar `G:R^N->R^N`. Fourier se representa mediante
senos y cosenos reales. El soporte complejo modificaría el producto interno y se deja
para otro contrato.

### Protocolo mínimo

Una familia candidata no necesita heredar de una clase concreta. Debe proporcionar:

```python
class MyBasis:
    dimension = N
    value_shape = ()

    def evaluate(self, points, *, order=0, cells=None, barycentric=None): ...
```

Para `points` de forma `[Q,p]`, el resultado tiene forma

```text
[Q,N,*value_shape,*([p]*order)].
```

`cells[q]` identifica el simplex padre y `barycentric[q]` sus coordenadas
baricéntricas. Una base global puede ignorarlos; una base local o discontinua puede
usarlos. Sólo se solicitan los órdenes derivativos presentes en `weak`. Las derivadas
se expresan en coordenadas ambientes y se proyectan tangencialmente según D-003.

Al construir `G`, todas las tablas se separan del grafo de autograd. La base no depende
del estado ni del tiempo y no es entrenable.

### Construcción desde la geometría

La ruta canónica es

```python
basis = problem.basis("laplacian", size=N, degree=q)
G = problem.field(basis=basis)
```

Para una familia Lagrange `psi_a`, se ensamblan

```text
M_ab = integral_Omega_h psi_a psi_b,
K_ab = integral_Omega_h <grad psi_a, grad psi_b>,
```

y se resuelve

```text
K c_j = lambda_j M c_j,
c_i^T M c_j = delta_ij.
```

Los modos `phi_j=sum_a (c_j)_a psi_a` se ordenan por valor propio creciente. En un
complejo embebido, `K` usa el gradiente tangencial y representa el Laplace--Beltrami
discreto. Si existe frontera y no se construyó previamente un subespacio restringido,
el problema auxiliar conserva su condición natural. Los modos nulos, incluida la
constante cuando corresponde, no se eliminan silenciosamente.

La orientación de signo se fija haciendo positiva la entrada de mayor magnitud de
cada vector propio. Un espacio propio múltiple aún admite rotaciones ortogonales; por
eso la base numérica concreta se guarda y reutiliza, en lugar de regenerarse para el
entrenamiento.

### Familias públicas

`problem.basis` ofrece:

```text
laplacian       modos geométricos ordenados; opción general predeterminada;
polynomial      monomios por grado total, restringidos y ortonormalizados;
fourier         senos y cosenos reales, restringidos y ortonormalizados;
finite-element  espacio Lagrange completo ortonormalizado;
custom          una familia suministrada por el usuario y ortonormalizada.
```

Una futura base POD basada en snapshots queda fuera de D-005 hasta acordar el contrato
de esos datos.

`ComponentBasis` repite una base escalar en todas las componentes. `ProductBasis`
combina bases escalares distintas, posiblemente con cantidades de modos diferentes,
en el producto directo. Los modos se ordenan primero por componente y luego dentro de
cada base. `TransformedBasis` aplica `phi_new[j]=sum_i phi_old[i] C[i,j]` y permite
codificar cualquier subespacio lineal; las restricciones se aplican antes de la
ortonormalización.

### Ortonormalización y validación

Una familia cruda se prepara explícitamente:

```python
admissible = TransformedBasis(raw_basis, constraints)
basis = problem.orthonormalize(admissible)
```

Si `M=L L^T`, se usa la transformación `L^(-T)`. La operación devuelve un objeto nuevo
y nunca cambia silenciosamente las coordenadas de una base existente.

La construcción y la validación pueden usar órdenes de cuadratura distintos. Se
registra

```text
orthonormality_error = max_ij |M_ij-delta_ij|,
```

y se rechaza la base si supera la tolerancia. `problem.field` repite la comprobación
con su propia cuadratura. Una base preparada recomienda el orden con el que debe
evaluarse cuando el usuario no lo especifica.

La representación densa de tablas es un detalle de la implementación actual, no una
parte permanente del contrato; se podrá reemplazar por almacenamiento local o disperso
sin cambiar la interfaz matemática.

### Persistencia

Una base espectral de elementos finitos guarda sin `pickle` la geometría, numeración,
coeficientes, orden, familia, órdenes de cuadratura y valores propios. Cargarla restaura
las mismas funciones coordenadas. Una familia basada en código Python no se serializa
automáticamente como código ejecutable.

## Criterios de aceptación de D-005

La implementación debe verificar al menos:

1. base laplaciana ortonormal en una geometría ND;
2. base laplaciana sobre una superficie cerrada embebida en `R3`;
3. orden creciente de valores propios y conservación del modo nulo;
4. familias polinómica, Fourier, de elementos finitos y personalizada;
5. productos con cantidades distintas de modos por componente;
6. rechazo de una familia operacional no ortonormal;
7. detección de una falsa ortonormalidad producida por subintegración;
8. persistencia exacta de los modos laplacianos y sus metadatos;
9. igualdad `G_i(z)=a(Phi_N z;phi_i)` sin una corrección de masa posterior.

## D-006 — Campo, lotes y diferenciación

**Estado:** aceptada.

### Objeto matemático

Una vez fijadas la geometría, la forma débil y la base operacional, la construcción

```python
G = problem.field(basis=basis)
```

produce un campo autónomo y reutilizable

```text
G: R^N -> R^N,
G_i(z) = a(Phi_N z; phi_i).
```

La geometría, la base, el orden de sus modos, los coeficientes espaciales, la forma
débil y la cuadratura quedan fijados al construir `G`. Evaluar el campo no cambia esos
datos ni integra una trayectoria temporal.

### Contrato tensorial

El último eje de la entrada contiene siempre las coordenadas de Galerkin. Todos los
ejes anteriores son ejes de lote libres:

```text
z:    [*S,N],
G(z): [*S,N],
```

donde `S` es cualquier tupla de tamaños, incluida la tupla vacía y tuplas con tamaños
cero. Por tanto se admiten `[N]`, `[B,N]`, `[T,B,N]` y tensores de rango superior con
una única regla. La evaluación equivale a aplicar el mismo campo a cada estado de
forma independiente; la implementación puede aplanar temporalmente `S`, pero debe
restaurarlo exactamente.

No existen conversiones implícitas de listas, arreglos NumPy, dispositivo o precisión.
La entrada es un tensor PyTorch con el mismo `dtype` y `device` que `G`. El campo admite
`float32` y `float64`, y `G.to(device=..., dtype=...)` mueve en conjunto todas sus
tablas fijas.

### Reconstrucción

Para los puntos de cuadratura ya tabulados por el campo,

```python
u = G.reconstruct(z)
```

preserva los mismos ejes de lote y devuelve

```text
[*S,Q,*value_shape].
```

En particular, una entrada `[N]` produce `[Q,*value_shape]`, sin introducir un eje de
lote artificial. La localización y evaluación en puntos físicos arbitrarios requieren
un contrato geométrico posterior y no forman parte de D-006.

### Diferenciación

La evaluación se expresa mediante operaciones PyTorch y conserva autograd respecto de
`z`. Debe ser compatible con las transformaciones funcionales estándar:

```python
J = torch.func.jacrev(G)(z)
_, Jw = torch.func.jvp(G, (z,), (w,))
_, pullback = torch.func.vjp(G, z)
```

También se admiten derivadas de orden superior cuando las operaciones de la forma
débil las poseen. Las funciones entregadas mediante `pointwise` deben ser puras, sin
efectos laterales y estar escritas con operaciones PyTorch compatibles con esas
transformaciones.

No se conservan gradientes hacia la geometría, la base, los datos de cuadratura ni los
`Coefficient`, pues todos ellos definen el operador fijo. D-006 no añade parámetros
variables en tiempo de evaluación.

### Separación de responsabilidades

Construir `G` puede ensamblar, integrar, validar y tabular. Llamar `G(z)` sólo evalúa el
campo ya preparado. El contrato de construcción y evaluación de D-006 no resuelve

```text
z'(t)=G(z(t)),
```

no recibe el tiempo y no altera `G` con un integrador. D-011 añade después una operación
separada `G.solve(...)`: el campo continúa siendo autónomo, fijo y reutilizable.

## Criterios de aceptación de D-006

La implementación debe verificar al menos:

1. conservación exacta de entradas y salidas `[*S,N]` para varios rangos de `S`;
2. equivalencia entre la evaluación tensorial y la evaluación estado por estado;
3. soporte de lotes con tamaños cero;
4. reconstrucción con forma `[*S,Q,*value_shape]`;
5. equivalencia entre `G(batch)` y `torch.vmap(G)(batch)`;
6. jacobianos, JVP, VJP y derivadas de segundo orden mediante `torch.func`;
7. ejecución en `float32`, `float64`, CPU y, cuando esté disponible, CUDA;
8. rechazo de entradas con dimensión, tipo, precisión o dispositivo incompatibles.

## D-007 — Contrato de cuadratura

**Estado:** aceptada.

### Interfaz única

La cuadratura del campo se controla con un solo argumento opcional:

```python
G = problem.field(basis=basis)  # automática
G = problem.field(basis=basis, quadrature=10)  # orden fijo
G = problem.field(basis=basis, quadrature=1e-8)  # tolerancia adaptativa
```

Un entero no negativo representa el grado polinómico que la regla debe integrar
exactamente en cada simplejo de referencia. Un número real estrictamente entre cero y
uno representa una tolerancia. Los booleanos, los reales mayores o iguales que uno y
las cadenas se rechazan para que ninguna entrada tenga dos interpretaciones posibles.

`quadrature_order` y `quadrature_rule` no forman parte de la interfaz general de
`problem.field`. El método geométrico de bajo nivel puede seguir recibiendo una regla
de referencia personalizada, pero esta posibilidad no complica el uso normal del
campo.

### Selección automática

Si `quadrature` se omite, la biblioteca inspecciona la base y el árbol de la forma
débil. Cuando ambos son polinómicos a trozos, calcula el grado del integrando completo,
incluidos estado, función de prueba, derivadas, coordenadas y coeficientes nodales, y
elige un orden exacto no menor que el recomendado para validar la base.

Una base programable, un `Coefficient(function)`, `pointwise`, una división por una
expresión espacial o funciones como `sin` y `exp` pueden impedir esa inferencia. En ese
caso se usa automáticamente la tolerancia `1e-8` en `float64` y `5e-5` en `float32`.

### Adaptación durante la construcción

La adaptación prueba los órdenes `q, q+2, ...`, comenzando en el orden de validación de
la base. Para cada candidato evalúa el campo en un conjunto determinista y acotado de
estados de calibración dentro de la bola unitaria de coeficientes. Se detiene cuando

```text
max |G_q2(z)-G_q(z)| / (1+|G_q2(z)|) <= tolerance
```

en todos esos estados y componentes. Se consideran conjuntamente todas las integrales
de volumen y de frontera. El orden máximo interno es 64 y se conserva el límite
`max_quadrature_points`; si se alcanza alguno antes de converger, la construcción falla
explícitamente en vez de aceptar una integral no verificada.

La adaptación sólo refina el orden de integración sobre cada simplejo afín. No modifica
la malla, no subdivide elementos y no cambia ni reortonormaliza la base.

Esta comparación es una estimación numérica sobre los estados de calibración, no una
cota uniforme para una forma no lineal arbitraria en todo `R^N`. Tal cota no puede
deducirse de una tolerancia sin hipótesis adicionales sobre el operador y el conjunto
de estados. Cuando se requiera control fuera de esa escala, el usuario debe fijar y
validar un orden entero apropiado para el problema.

### Inmutabilidad operacional

Toda adaptación ocurre al construir `G`. Después quedan fijos los puntos, pesos,
coeficientes tabulados y valores de la base. En particular, llamar `G(z)` nunca cambia
la cuadratura. Esto conserva el campo autónomo acordado en D-006 y su compatibilidad
con lotes, `torch.func` y autograd.

El resultado registra:

```text
quadrature_mode             fixed | automatic-exact | adaptive | automatic-adaptive
quadrature_order            orden finalmente usado
quadrature_tolerance        tolerancia solicitada, o None
quadrature_error_estimate   última diferencia escalada, 0 o None
```

### Criterios de aceptación de D-007

La implementación debe verificar al menos:

1. inferencia de un orden suficiente para formas polinómicas a trozos;
2. adaptación de formas que contienen funciones no polinómicas;
3. selección inequívoca mediante `None`, un entero o una tolerancia real;
4. coincidencia del resultado adaptativo con una regla fija refinada;
5. conservación del orden y de las tablas después de construir `G`;
6. registro del modo, orden, tolerancia y error estimado;
7. fallo explícito al agotar el orden o el presupuesto de puntos;
8. aplicación conjunta a integrales de volumen y de frontera.

## D-008 — Proyección a coordenadas de Galerkin

**Estado:** aceptada.

### Objeto matemático

Para una función real `u` con la misma forma de valor que la base operacional, se
define

```text
P_N u = sum_i z_i phi_i,
z_i = <u,phi_i>_L2.
```

Como la base es ortonormal, `z` se obtiene directamente mediante esos productos
internos. No se ensambla ni se invierte una matriz de masa. Además,

```text
norm(u-P_N u)_L2 = inf_{w in V_N} norm(u-w)_L2,
```

y proyectar una función que ya tiene coordenadas `z` en `V_N` recupera las mismas
coordenadas, salvo el error de cuadratura.

### Interfaz compacta

El campo preparado ofrece un único método nuevo:

```python
z0 = G.project(u0)
dz0 = G(z0)
u0N = G.reconstruct(z0)
```

`project` acepta una función evaluable o cualquiera de las representaciones espaciales
fijas acordadas en D-004:

```python
z0 = G.project(lambda x: torch.sin(x[:, 0]))
z0 = G.project(Coefficient(function, shape=value_shape))
z0 = G.project(Coefficient.cell(values))
z0 = G.project(Coefficient.vertex(values))
```

No acepta un arreglo crudo de valores en puntos de cuadratura, pues ese arreglo no
identifica por sí mismo los puntos, la regla ni la geometría que representa.

### Contrato tensorial

La función recibe puntos `x:[Q,p]`. Para un lote arbitrario `S`, devuelve

```text
u(x): [*S,Q,*value_shape]
```

y la proyección produce

```text
G.project(u): [*S,N].
```

Una función individual usa `S=()`. Un `Coefficient` describe una sola función y por
tanto produce `[N]`. Los valores deben ser reales, finitos y tener exactamente la forma
física de la base. Los lotes vacíos se admiten con cuadratura fija; una adaptación no
puede estimar un error a partir de un lote vacío y lo rechaza explícitamente.

### Cuadratura

La proyección reutiliza sin variaciones el lenguaje de D-007:

```python
z0 = G.project(u0)  # automática
z0 = G.project(u0, quadrature=10)  # fija
z0 = G.project(u0, quadrature=1e-8)  # adaptativa
```

Para `Coefficient.cell` y `Coefficient.vertex`, una base polinómica reconocida permite
inferir un orden exacto. Una función Python o `Coefficient(function)` se considera en
general no polinómica y activa la adaptación. En este caso se comparan directamente
las coordenadas obtenidas con `q` y `q+2`; no hacen falta estados de calibración porque
la función que se está proyectando ya está determinada.

La cuadratura usada por `project` es temporal e independiente de la empleada para
construir el campo. La operación no altera `quadrature_order`, las tablas, la base ni
ningún otro dato de `G`.

### Diferenciación y alcance

Si una función Python devuelve operaciones PyTorch dependientes de parámetros con
gradiente, la proyección final conserva autograd respecto de esos parámetros. Los
`Coefficient` mantienen el contrato fijo de D-004 y se proyectan separados del grafo.
La selección adaptativa del orden es una decisión numérica discreta; la diferenciación
corresponde a la integral evaluada con el orden finalmente seleccionado.

`G.reconstruct(z)` continúa evaluando en los puntos ya preparados por el campo. La
evaluación adicional en puntos físicos se especifica en D-009. D-008 tampoco integra
la ecuación temporal ni modifica el espacio `V_N`.

### Criterios de aceptación de D-008

La implementación debe verificar al menos:

1. identidad entre proyección y síntesis para funciones pertenecientes a `V_N`;
2. proyección de funciones escalares, vectoriales y tensoriales;
3. conservación de ejes de lote arbitrarios;
4. soporte de `Coefficient` por función, simplex y vértice;
5. cuadratura automática, fija y adaptativa con el contrato de D-007;
6. conservación de autograd para funciones PyTorch;
7. inmutabilidad de la base y de las tablas del campo;
8. rechazo de valores crudos, complejos, no finitos o con forma incompatible.

## D-009 — Evaluación espacial del campo reconstruido

**Estado:** aceptada.

### Objeto matemático

Para unas coordenadas de Galerkin `z`, la síntesis espacial es

```text
u_z(x) = sum_j z_j phi_j(x).
```

D-009 permite evaluar `u_z` y sus dos primeras derivadas espaciales en puntos físicos
arbitrarios de `Omega_h`. Estas operaciones no evalúan `G` en los puntos físicos ni
representan derivadas de `G:R^N->R^N` respecto de `z`.

### Interfaz compacta

```python
values = G.reconstruct(z, points)
gradients = G.grad(z, points)
Hessians = G.hessian(z, points)
```

El comportamiento anterior permanece intacto:

```python
values_at_quadrature = G.reconstruct(z)
values_at_boundary_quadrature = G.reconstruct(z, boundary="wall")
```

No se introduce un argumento `order`: el nombre de cada método determina sin
ambigüedad la cantidad solicitada. `points` debe ser un tensor PyTorch `[Q,p]`, real y
finito, con el mismo `dtype` y `device` que `G`. No se convierten silenciosamente
listas, arreglos NumPy, dispositivos ni precisiones.

Si `z:[*S,N]`, las formas de salida son

```text
G.reconstruct(z, points): [*S,Q,*value_shape]
G.grad(z, points):        [*S,Q,*value_shape,p]
G.hessian(z, points):     [*S,Q,*value_shape,p,p].
```

Se conservan todos los ejes de lote, incluidos tamaños cero. Los ejes derivativos se
añaden al final y siempre se expresan en las coordenadas ambientes.

### Localización y selección del simplex

Por defecto, el campo localiza automáticamente cada punto en los simplejos afines de
`Omega_h` y calcula sus coordenadas baricéntricas. Un punto que no pertenece al
complejo —incluido un punto fuera del soporte afín de una variedad embebida— se rechaza
explícitamente.

El argumento avanzado `cells:[Q]`, entero `torch.int64`, permite indicar el simplex
padre de cada punto:

```python
values = G.reconstruct(z, points, cells=cells)
gradients = G.grad(z, points, cells=cells)
Hessians = G.hessian(z, points, cells=cells)
```

Cada asignación se valida geométricamente. `cells` no cambia el punto ni extrapola la
base fuera del simplex seleccionado.

Un punto interior tiene un único simplex padre. Sobre una cara compartida puede haber
varias trazas. Si todas las evaluaciones de la base para la cantidad solicitada
coinciden dentro de tolerancia numérica, el resultado es único y no se exige `cells`.
Si difieren, la llamada falla y requiere que el usuario seleccione la traza. Así se
evita elegir silenciosamente un lado según la numeración de la malla. En particular,
una base continua puede tener valores únicos pero gradientes distintos en una cara.

### Derivadas elementales y geometrías embebidas

Las operaciones calculan

```text
grad_Omega_h u_z = sum_j z_j grad_Omega_h phi_j,
Hess_Omega_h u_z = sum_j z_j Hess_Omega_h phi_j.
```

En un simplex `K_e` embebido, cada eje derivativo se proyecta con el proyector
tangencial `Pi_e`. Por tanto el gradiente tiene `p` componentes y es tangente a `K_e`;
el Hessiano tiene forma `[p,p]` y es tangencial en ambos índices. Para la geometría
afín a trozos de D-003, el Hessiano es el Hessiano tangencial clásico dentro de cada
simplex.

Las derivadas son elementales. No se añaden contribuciones distribucionales sobre
caras y no se afirma conformidad Sobolev global. Por ejemplo, una base Lagrange `P1`
tiene gradiente constante y Hessiano cero en el interior de cada simplex, aunque el
gradiente pueda saltar entre elementos. Bases `P2` o superiores y bases globales
suaves pueden tener Hessianos elementales no nulos.

### Diferenciación y base fija

Los tres métodos son lineales en `z` y conservan autograd respecto de sus coordenadas.
Las derivadas de `G` como aplicación de coeficientes continúan obteniéndose con

```python
J = torch.func.jacrev(G)(z)
H = torch.func.jacrev(torch.func.jacrev(G))(z)
```

Los puntos, la geometría, la localización y la base quedan fuera del grafo. Una base
programable forma parte del dato fijo: su evaluador debe representar la misma familia
que se validó al construir `G` y no debe mutarse después.

### Criterios de aceptación de D-009

La implementación debe verificar al menos:

1. reconstrucción, gradiente y Hessiano en puntos físicos interiores;
2. conservación de ejes de lote y formas de valor escalares, vectoriales y tensoriales;
3. derivadas tangenciales con `p` componentes sobre geometrías embebidas;
4. localización automática en complejos de simplejos de dimensión arbitraria;
5. rechazo explícito de puntos exteriores y asignaciones `cells` incompatibles;
6. detección de valores o derivadas ambiguos sobre caras compartidas;
7. selección explícita de cada traza mediante `cells`;
8. interpretación elemental de las derivadas, incluido el Hessiano nulo de `P1`;
9. conservación de autograd respecto de `z` sin gradientes hacia datos fijos;
10. conservación del comportamiento previo de `G.reconstruct(z)`.

## D-010 — Aproximación continua de datos discontinuos

**Estado:** aceptada para el alcance actual.

### Decisión

El núcleo no introduce todavía trazas dobles ni integrales sobre caras interiores. Una
función física discontinua se lleva al espacio operacional continuo mediante la
proyección ortogonal ya definida en D-008:

```text
P_N u = sum_i <u,phi_i>_L2 phi_i.
```

Aunque `u` sea discontinua, cada aproximación `P_N u` es continua cuando la base lo es.
La propiedad relevante es

```text
norm(u-P_N u)_L2 = inf_{w in V_N} norm(u-w)_L2.
```

Por tanto no se suavizan manualmente los datos ni se modifica la base durante la
evolución. La discontinuidad queda representada mediante una transición continua cuya
resolución depende del espacio fijo `V_N`.

### Interfaz

No se añade una nueva llamada pública. Para una discontinuidad alineada con la malla,
la representación preferida es exacta por simplex:

```python
u0 = Coefficient.cell(cell_values)
z0 = G.project(u0)
u0N = G.reconstruct(z0, points)
```

Una discontinuidad general también puede darse mediante una función PyTorch:

```python
z0 = G.project(lambda x: (x[:, 0] < 0.5).to(G.dtype), quadrature=20)
```

Cuando el salto corta el interior de los simplejos, la cuadratura deja de ser
polinómica a trozos respecto de esa malla. El usuario debe refinar la malla, emplear
una tolerancia adaptativa que converja o fijar un orden comprobado. La adaptación de
D-007 estima la integral numérica, pero no convierte esa estimación en una cota
uniforme del error de aproximación funcional.

### Alcance matemático

La convergencia de aproximaciones continuas a una función con saltos se entiende aquí
en `L2`. No se promete convergencia puntual sobre el salto. Bases globales pueden
mostrar oscilaciones de Gibbs y una base local concentra el error en una vecindad de
la interfaz que disminuye al refinar el espacio.

Una función con un salto verdadero no pertenece en general a `H1`. Si la forma débil
requiere `grad(u)`, la norma de los gradientes de aproximaciones continuas puede crecer
al intentar resolver el salto. D-010 no afirma convergencia en `H1` ni reemplaza las
hipótesis funcionales exigidas por la ecuación concreta. Para problemas difusivos, el
suavizado de la evolución puede hacer suficiente esta representación; para choques
persistentes o interfaces físicas, puede no serlo.

`ds.interior` continúa produciendo un error explícito. No existen por ahora
`u.minus`, `u.plus`, saltos, promedios, flujos numéricos ni penalizaciones automáticas.
Esto evita asignar una semántica de Galerkin discontinuo a una formulación que todavía
usa un único valor reconstruido.

### Mejora futura

Si los experimentos requieren discontinuidades persistentes, el contrato deberá
extenderse sin cambiar la API ya aceptada. La extensión futura incluirá:

1. enumeración y etiquetado de caras interiores;
2. dos simplejos padres y dos trazas para estado, prueba y coeficientes;
3. dos conormales exteriores, necesarias también en complejos embebidos;
4. una medida `ds.interior` compatible con regiones nombradas;
5. primitivas explícitas para construir saltos, promedios y flujos conservativos;
6. bases discontinuas ortonormales y reglas de cuadratura por cara;
7. pruebas de conservación, orientación e invariancia frente a la numeración.

Esta extensión correspondería a un contrato posterior de Galerkin discontinuo; no se
implementará silenciosamente dentro de D-010.

### Criterios de aceptación de D-010

La implementación debe verificar al menos:

1. proyección de valores constantes por simplex sobre una base continua;
2. unicidad de la reconstrucción en una cara compartida;
3. reducción del error `L2` al refinar un espacio continuo local;
4. conservación de autograd respecto de parámetros de una función PyTorch;
5. rechazo explícito ya existente de `ds.interior`;
6. ausencia de nuevas clases de condición de frontera o integración temporal.

## D-011 — Evolución temporal por Runge--Kutta

**Estado:** aceptada.

### Objeto matemático

Fijados el campo y el dato inicial, la trayectoria reducida satisface

```text
z'(t) = G(z(t)),       z(t_0) = z_0.
```

La integración es posterior a la construcción de `G`: no cambia la geometría, la forma
débil, la base, los coeficientes ni la cuadratura. Si el dato se entrega como función
física, la cadena completa permanece explícita:

```python
z0 = G.project(u0)
Z = G.solve(z0, times)
U = G.reconstruct(Z, points)
```

### Interfaz compacta

`times:[T]` contiene los tiempos en los que se devuelve la trayectoria. Debe ser un
tensor PyTorch real, finito, estrictamente monótono, no vacío y con el mismo `dtype` y
`device` que `G`. Se admiten tiempos crecientes y decrecientes.

```python
Z = G.solve(z0, times)  # RK45 adaptativo
Z = G.solve(z0, times, tolerance=1e-7)  # RK45 adaptativo
Z = G.solve(z0, times, step=1e-3)  # RK4 fijo
```

No se introduce una clase de trayectoria. Si `z0:[*S,N]`, entonces

```text
G.solve(z0,times): [T,*S,N].
```

Así, la trayectoria completa puede pasarse directamente a `reconstruct`, `grad`,
`hessian` o de nuevo a `G`. Para `T=1` se devuelve solamente el estado inicial con el
eje temporal añadido. Los lotes vacíos conservan exactamente su forma.

### RK4 fijo

Cuando se especifica `step=h>0`, cada intervalo entre dos tiempos pedidos se divide en
`ceil(abs(Delta t)/h)` subintervalos iguales. Por tanto `step` es el máximo paso interno,
los estados se entregan exactamente en `times` y se admiten mallas temporales no
uniformes. En cada subintervalo se usa el esquema clásico

```text
k1 = G(z_n),
k2 = G(z_n + h k1/2),
k3 = G(z_n + h k2/2),
k4 = G(z_n + h k3),
z_(n+1) = z_n + h(k1 + 2k2 + 2k3 + k4)/6.
```

### RK45 adaptativo

Si se omite `step`, se usa el par embebido Dormand--Prince 5(4). La estimación local se
normaliza componente a componente mediante

```text
tolerance * (1 + max(abs(z_n), abs(z_(n+1)))).
```

y el máximo se toma sobre todos los modos y todos los lotes. El algoritmo elige y puede
rechazar pasos internos sin modificar los tiempos de salida. La tolerancia por defecto
es `5e-5` en `float32` y `1e-8` en `float64`. `step` y `tolerance` representan modos
distintos y no pueden combinarse.

### Diferenciación y límites

Las operaciones aceptadas conservan autograd respecto de `z0` y de las operaciones
dinámicas diferenciables evaluadas por `G`. La aceptación o rechazo de un paso
adaptativo es una decisión numérica discreta; no se promete diferenciabilidad respecto
de esa decisión ni respecto de `times`.

Toda velocidad y todo estado producido deben ser finitos. Existe un presupuesto interno
para impedir bucles ilimitados. RK4 y RK45 son métodos explícitos: D-011 no garantiza
estabilidad eficiente para difusión severa u otros campos rígidos. Métodos implícitos o
IMEX podrán añadirse sin cambiar el significado de `G.solve`.

### Criterios de aceptación de D-011

La implementación debe verificar al menos:

1. exactitud esperada sobre un campo lineal con solución conocida;
2. RK4 fijo y RK45 adaptativo en tiempos de salida no uniformes;
3. integración hacia adelante y hacia atrás;
4. conservación de ejes de lote, lotes vacíos y tiempo único;
5. compatibilidad directa con la reconstrucción de toda la trayectoria;
6. conservación de autograd respecto del estado inicial;
7. ejecución en `float32`, `float64`, CPU y, cuando esté disponible, CUDA;
8. rechazo de estados, tiempos, pasos y tolerancias inválidos o no finitos.

## D-012 — Indicadores de error y convergencia

**Estado:** aceptada.

### Decisión

El paquete separa tres fuentes numéricas sin presentarlas como cotas rigurosas a
posteriori:

```text
aproximación espacial, integración temporal y cuadratura.
```

Cada indicador es absoluto. Esta elección evita ocultar una división inestable cuando
la solución o el campo de referencia es cercano a cero. El usuario puede normalizarlo
con la escala física apropiada para su problema.

### Error espacial de proyección

```python
space_error = G.projection_error(u0)
space_error = G.projection_error(u0, quadrature=12)
```

El método estima mediante cuadratura

```text
||u - P_N u||_L2(Omega_h).
```

Acepta exactamente las mismas funciones y `Coefficient` que `G.project`, conserva sus
ejes de lote y reutiliza los modos automático, fijo y adaptativo de D-007. Para estudiar
convergencia espacial se construyen bases fijas de tamaños o mallas sucesivas y se
comparan sus errores sobre la misma función. El método mide el mejor error de
representación del dato; no demuestra por sí solo convergencia de toda una trayectoria.

### Indicador temporal

```python
time_error = G.time_error(z0, times, step=1e-3)
time_error = G.time_error(z0, times, tolerance=1e-7)
```

Con paso fijo devuelve

```text
||Z_h(t_j) - Z_(h/2)(t_j)||_2,
```

y con RK45 compara las tolerancias `tol` y `tol/2`. La salida tiene forma `[T,*S]`.
Como la síntesis es una isometría, esta norma euclídea coincide exactamente con la
distancia `L2` entre ambas reconstrucciones. Es un indicador de refinamiento, no una
cota certificada del error respecto de la solución exacta.

### Indicador de cuadratura

```python
quadrature_error = G.quadrature_error(z)
quadrature_error = G.quadrature_error(z, order=12)
```

El campo se vuelve a ensamblar temporalmente con un orden fijo mayor y se calcula

```text
||G_q(z) - G_qref(z)||_2.
```

Por defecto `qref=q+2`; un orden explícito debe ser estrictamente mayor que el usado por
`G`. La salida conserva todos los ejes de lote anteriores a `N`. Por ortonormalidad,
también es la distancia `L2` entre las velocidades funcionales reconstruidas. La
operación puede ser costosa porque prepara nuevas tablas, pero no muta el campo original.

### Interpretación y cantidades físicas

Los indicadores sólo aíslan cambios al refinar una decisión numérica. No garantizan que
la forma débil modele correctamente la ecuación ni sustituyen una estimación analítica.
La convergencia práctica requiere que los indicadores decrezcan en una sucesión de
refinamientos.

No existe una energía universal inferible de una forma débil arbitraria. Para una
trayectoria `Z`, la norma `L2` se obtiene exactamente con

```python
torch.linalg.vector_norm(Z, dim=-1)
```

y cualquier masa, energía o invariante adicional debe programarse como el observable
matemático correspondiente. El núcleo no declara conservación donde el usuario no la
ha especificado.

### Criterios de aceptación de D-012

La implementación debe verificar al menos:

1. error de proyección nulo, salvo precisión numérica, sobre el espacio fijo;
2. reducción del error de proyección al enriquecer una familia convergente;
3. indicadores temporales fijo y adaptativo con lotes arbitrarios;
4. reducción del indicador temporal al refinar el paso en un problema regular;
5. indicador de cuadratura con forma `[*S]` y orden de referencia superior;
6. ausencia de mutaciones sobre `G`, su base o sus tablas;
7. documentación explícita de que los tres resultados son indicadores, no cotas;
8. compatibilidad con CPU, CUDA, `float32` y `float64` bajo los contratos previos.
