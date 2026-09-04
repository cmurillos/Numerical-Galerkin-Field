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

## Espacio y base

Sea H = L2(Omega_h; R^s), donde `s` puede reemplazarse por una forma tensorial
arbitraria. El usuario suministra funciones linealmente independientes

```text
phi_1, ..., phi_N in H,
V_N = span{phi_1, ..., phi_N}.
```

La base contiene toda restricción que defina el espacio admisible. La interfaz central
no interpreta tipos de condición de frontera. Las condiciones naturales se expresan
en la forma débil; las restricciones esenciales, periódicas, de simetría o de otra
clase pueden incorporarse al construir las funciones `phi_i`.

La síntesis es

```text
Phi: R^N -> V_N,
Phi z = sum_j z_j phi_j = u_z.
```

La matriz de Gram es

```text
M_ij = (phi_j, phi_i)_H.
```

El programa calcula esta matriz mediante la misma cuadratura del volumen. También
acepta una matriz simétrica positiva definida suministrada por el usuario para
representar otro producto interno fijo.

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

El vector de acciones es

```text
g_i(z) = a(Phi z; phi_i).
```

El campo de Galerkin es la velocidad coordenada única que satisface

```text
M G(z) = g(z).
```

Equivalentemente, la función

```text
w_z = Phi G(z)
```

es el representante en `V_N` definido por

```text
(w_z, v_N)_H = a(Phi z; v_N)       para todo v_N en V_N.
```

Si la base es ortonormal, `M=I` y `G_i(z)=a(Phi z;phi_i)`. Si existe una realización
fuerte `A(Phi z)` en `H`, entonces `w_z` es la proyección de ese campo sobre `V_N`.

Una trayectoria de Galerkin satisface

```text
z'(t) = G(z(t)),
u_N(t) = Phi z(t).
```

La biblioteca construye y evalúa `G`; la integración de esta EDO es una operación
posterior e independiente.

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
