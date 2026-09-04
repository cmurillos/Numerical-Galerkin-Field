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
