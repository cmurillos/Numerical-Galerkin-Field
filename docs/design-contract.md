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
