# Contrato matemático

## Geometría

Sean `P` una matriz de `M` puntos de R^p y `T` una matriz de índices con `k+1`
columnas. Cada fila determina el k-simplejo

```text
K_e = conv{P[T_e,0], ..., P[T_e,k]},       1 <= k <= p.
```

Se supone que estos simplejos forman un complejo conforme, no degenerado, con
interiores disjuntos. Su unión define el dominio discreto Omega_h. Para `k=p` se usa
la medida de Lebesgue; para `k<p`, la medida k-dimensional inducida por la inmersión
afín.

El mapa del simplejo de referencia es

```text
F_e(xi) = P[T_e,0] + B_e xi,
B_e = [P[T_e,1]-P[T_e,0], ..., P[T_e,k]-P[T_e,0]].
```

El factor geométrico común a ambos casos es

```text
J_e = sqrt(det(B_e^T B_e)).
```

Para `k=p`, coincide con `abs(det(B_e))`. En consecuencia,

```text
integral_K_e f dmu_k = J_e integral_Khat f(F_e(xi)) dxi.
```

Las caras pertenecientes a un único elemento constituyen la frontera. Una cara tiene
dimensión `k-1`, y su medida satisface

```text
J_face = J_e norm(grad lambda_opposite).
```

El vector exterior usado por `ds.normal` es el opuesto del gradiente normalizado de
esa coordenada baricéntrica. Si `k<p`, es una conormal contenida en el espacio tangente
del elemento.

Las etiquetas de `boundaries` seleccionan facetas exteriores y las etiquetas de
`regions` seleccionan símplices máximos. No imponen interpretaciones físicas. Por ello
`ds("Gamma")` y `dx("Omega_1")` sólo restringen la medida a los elementos nombrados.

Si `Q_e` contiene una base ortonormal de la imagen de `B_e`, el proyector tangencial es

```text
Pi_e = Q_e Q_e^T.
```

Toda derivada espacial se representa en las `p` coordenadas ambientes y cada eje
derivativo se proyecta con `Pi_e`. Así, para `k<p`, `grad(u)` es el gradiente tangencial
y no depende de la extensión ambiental usada para programar una función de base.

Para `x` dentro de un simplex padre `K_e`, la síntesis y sus primeras derivadas son

```text
u_z(x) = sum_j z_j phi_j(x),
grad_Omega_h u_z(x) = sum_j z_j grad_Omega_h phi_j(x),
Hess_Omega_h u_z(x) = sum_j z_j Hess_Omega_h phi_j(x).
```

Cada eje del Hessiano se proyecta con `Pi_e`. Estas son derivadas clásicas por
simplex; no incluyen términos distribucionales asociados a saltos entre elementos. Si
`x` pertenece a varios simplejos y las trazas no coinciden, el simplex padre forma
parte necesaria de la evaluación.

## Espacio y base

Sea H = L2(Omega_h; R^s), donde `s` puede reemplazarse por una forma tensorial
arbitraria. El usuario suministra funciones linealmente independientes

```text
phi_1, ..., phi_N in H,
V_N = span{phi_1, ..., phi_N}.
```

La familia contiene toda restricción que defina el espacio admisible. La interfaz central
no interpreta tipos de condición de frontera. Las condiciones naturales se expresan
en la forma débil; las restricciones esenciales, periódicas, de simetría o de otra
clase pueden incorporarse al construir las funciones `phi_i`.

La base operacional es real, fija y ortonormal. Al preparar el campo, sus valores y
derivadas se tabulan y se separan del grafo de diferenciación. Autograd actúa sobre los
coeficientes de estado, no sobre parámetros capturados por la implementación de la base.

La síntesis es

```text
Phi: R^N -> V_N,
Phi z = sum_j z_j phi_j = u_z.
```

La ortonormalidad respecto del producto interno L2 estándar significa

```text
(phi_j, phi_i)_H = delta_ij.
```

En consecuencia, `Phi` es una isometría, `norm(Phi z)_H=norm(z)_2`, y la bola funcional
`V_N` intersección `B_H(0,R)` se identifica exactamente con la bola euclídea
`B_R` en coordenadas. El programa verifica numéricamente la matriz de Gram y rechaza
una familia no ortonormal. El producto interno no puede sustituirse después mediante
una matriz arbitraria.

## Forma débil y campo reducido

La función Python describe una única aplicación

```text
a: D_a x V_N -> R,
```

posiblemente no lineal en el estado `u` y lineal en la función de prueba `v`. Una
expresión puede contener integrales de volumen y de frontera:

```text
a(u;v)
 = sum_r integral_Omega_h I_r(x,u,grad u,...,v,grad v,...) dmu_k
 + sum_Gamma integral_Gamma J_Gamma(x,n,u,...,v,...) dmu_(k-1).
```

Las integrales pueden restringirse a regiones y fronteras nombradas sin introducir
tipos especiales de condición de frontera.

Los coeficientes espaciales fijos que aparecen en los integrandos pueden ser campos
escalares, vectoriales o tensoriales. Se definen mediante una función de `x`, valores
constantes por simplex o valores nodales con interpolación baricéntrica P1. Su
tabulación se realiza una sola vez al construir el campo, de modo que forman parte del
operador fijo y no del estado dinámico. En particular, ni estos valores ni parámetros
capturados por su función reciben gradientes.

Las operaciones no lineales de punto se aplican a expresiones independientes de `v`.
Se evalúan de nuevo para cada `z` y conservan la diferenciación automática respecto de
ese estado. Cada integrando completo debe ser exactamente de grado uno en `v`; no se
admiten términos afines independientes de la función de prueba.

El vector de acciones es

```text
g_i(z) = a(Phi z; phi_i).
```

El campo de Galerkin es directamente la velocidad coordenada

```text
G(z) = g(z).
```

Equivalentemente, la función

```text
w_z = Phi G(z)
```

es el representante en `V_N` definido por

```text
(w_z, v_N)_H = a(Phi z; v_N)       para todo v_N en V_N.
```

Si existe una realización fuerte `A(Phi z)` en `H`, entonces `w_z` es la proyección de
ese campo sobre `V_N`.

Una trayectoria de Galerkin satisface

```text
z'(t) = G(z(t)),
u_N(t) = Phi z(t).
```

Para una tupla arbitraria de tamaños de lote `S=(S_1,...,S_r)` y un multiíndice
`alpha`, la extensión tensorial no define un campo diferente: actúa componente a
componente,

```text
(G_S(Z))_alpha = G(Z_alpha),
Z in R^(S_1 x ... x S_r x N).
```

En consecuencia, `G_S` conserva todos los ejes anteriores al último y su diferencial
es diagonal respecto de los índices de lote. La síntesis se extiende de la misma forma:

```text
(Phi_S Z)_alpha = Phi(Z_alpha).
```

La realización numérica es diferenciable respecto de cada `Z_alpha`. Las tablas de la
geometría, la base, los coeficientes espaciales y la cuadratura representan parámetros
fijos del operador y se mantienen fuera de ese grafo de diferenciación.

La integración de esta EDO es posterior e independiente de la construcción de `G`.
El método `G.solve` implementa RK4 fijo o Dormand--Prince 5(4) adaptativo sin modificar
el campo. Para tiempos pedidos `t_0,...,t_J` y estados iniciales `z_0` con ejes de lote
arbitrarios, devuelve el tensor de trayectorias

```text
Z = (z(t_0),...,z(t_J)) in R^((J+1) x ... x N).
```

La base ortonormal convierte diferencias euclídeas de coordenadas en diferencias
funcionales exactas dentro de `V_N`:

```text
norm(Phi z - Phi y)_L2 = norm(z-y)_2.
```

Esta identidad fundamenta los indicadores temporales y de cuadratura de D-012. El
indicador temporal compara dos integraciones refinadas y el de cuadratura compara dos
campos ensamblados con órdenes distintos. Ambos miden sensibilidad al refinamiento;
no son cotas del error frente a una solución exacta.

Para una función física `u`, el indicador espacial se evalúa directamente como

```text
norm(u-P_N u)_L2,
```

usando cuadratura sobre el residuo reconstruido. Separar estas tres cantidades evita
confundir truncamiento espacial, integración temporal y error de ensamblaje. Ninguna
de ellas sustituye las hipótesis de estabilidad y convergencia propias del problema.

Los métodos RK de D-011 son explícitos. En problemas rígidos, en particular al refinar
discretizaciones difusivas, la estabilidad puede exigir pasos mucho menores que los
dictados por la precisión. Una futura extensión implícita o IMEX deberá conservar el
mismo campo reducido y cambiar únicamente el integrador temporal.

Una función discontinua de `L2` puede proyectarse sobre una sucesión de espacios
continuos. Cada proyección es continua y puede converger a la función original en
`L2`, sin converger puntualmente sobre el salto. Esta es la convención actual de
D-010. No implica convergencia en `H1`: un salto verdadero no posee en general un
gradiente débil en `L2`, y las normas de los gradientes de sus aproximaciones continuas
pueden crecer.

Por ello el soporte actual es adecuado cuando la métrica relevante es `L2` o la
evolución regulariza los datos. Si el modelo conserva choques o prescribe leyes de
interfaz, será necesario introducir en el futuro espacios rotos, trazas laterales e
integrales sobre caras interiores.

Este contrato sólo integra en los elementos máximos y en la frontera exterior. Las
caras interiores requieren distinguir las dos trazas incidentes y fijar convenciones
de salto, promedio y orientación; se mantienen reservadas hasta definir ese contrato.

## Cuadratura en dimensión arbitraria

Se usa la transformación de Duffy del cubo unitario al simplejo de referencia. Su
jacobiano se absorbe mediante reglas de Gauss-Jacobi. Con suficientes nodos por eje,
la regla integra exactamente polinomios de un grado total solicitado, en aritmética
exacta. El número de puntos crece exponencialmente con `k`, por lo que la API expone un
presupuesto máximo y permite reemplazar la regla completa.

Una regla externa devuelve:

```text
barycentric: [Q,k+1],    sum_i barycentric[q,i] = 1,
weights:     [Q].
```

Los pesos corresponden a la medida del simplejo de referencia. El dominio aplica los
factores geométricos físicos.

## Alcance numérico

No hay un límite lógico de dimensión codificado en la geometría, las funciones
polinómicas o la cuadratura. Permanecen límites computacionales: crecimiento del número
de simplejos, grados de libertad, modos y puntos de integración.

Los simplejos son afines. Las fronteras curvas se aproximan con el complejo o requieren
una futura extensión isoparamétrica. `FiniteElementBasis` produce Lagrange continuo de
grado arbitrario; las derivadas de orden superior son elementales y no implican por sí
mismas conformidad global en H2 u otros espacios. La validez analítica de una forma
respecto del espacio elegido continúa siendo una condición matemática del problema.
