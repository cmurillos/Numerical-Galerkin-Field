# Periodicidad, media cero y fronteras fijas — D-013, parte 5

El recorrido sigue siendo geometría, espacio, base y campo autónomo. Las
restricciones nuevas se combinan con `ZeroTrace` antes de seleccionar modos.

## Periodicidad y media cero

Este ejemplo completo prepara seis modos periódicos con integral cero en [0,1]:

```python
import numpy as np
from ngfield import GalerkinField, MeanZero, Periodic, SimplicialDomain, Space, grad, inner

vertices = np.linspace(0, 1, 17)[:, None]
simplices = np.column_stack((np.arange(16), np.arange(1, 17)))
geometry = SimplicialDomain(
    vertices,
    simplices,
    boundaries={"left": [[0]], "right": [[16]]},
)
V = Space(
    geometry=geometry,
    components=1,
    restrictions=[
        Periodic(component=0, boundaries=("left", "right"), vertex_pairs=[(0, 16)]),
        MeanZero(component=0),
    ],
)
basis = V.basis("laplacian", size=6, degree=2)


def weak(u, v, dx, ds):
    return -inner(grad(u[0]), grad(v[0])) * dx


G = GalerkinField(basis=basis, weak=weak)
```

El espacio es `V={u in H1(0,1): u(0)=u(1), integral(u)=0}`. `size` sigue contando
la dimensión total; `component_sizes` conserva el significado de la parte 4.

`Periodic(component=r, boundaries=(source,target), vertex_pairs=pairs)` identifica
la traza de una componente escalar entre dos fronteras etiquetadas. Cada fila
contiene índices de vértices del dominio, no coordenadas físicas ni modos. El mapa
se extiende afínmente sobre cada cara. Debe cubrir exactamente los vértices de
ambas fronteras, ser biyectivo y preservar toda la conectividad de caras. Se
rechazan triangulaciones incompatibles aunque tengan el mismo número de vértices.

Todos los nodos Lagrange de las caras se emparejan mediante pesos baricéntricos
enteros, incluidos nodos interiores de aristas y caras de grado alto. Esto impone
igualdad de trazas polinómicas completas. No se certifica una traza muestreando
puntos arbitrarios. El usuario elige la correspondencia física: para una
traslación periódica debe proporcionar los pares de esa traslación.

Se admiten identificaciones múltiples que se encuentran en esquinas y mapas
específicos por componente. Las equivalencias se cierran transitivamente: una
traza cero que toca una clase fija toda la clase. No se implementan rotaciones
entre componentes, desfases de valor ni interpolación entre fronteras incompatibles.
La periodicidad exige `regularity>=1` y etiquetas distintas, existentes y no vacías.

La geometría conserva sus caras y etiquetas. Si sólo se pretende periodicidad,
no se añaden términos de intercambio con un medio exterior sobre las caras
emparejadas. El balance periódico de flujos procede de la formulación variacional
con pruebas periódicas; no se exige igualdad puntual de derivadas normales FEM.
Un toro cerrado por su conectividad no necesita pares de fronteras.

`MeanZero(component=r, region=None)` impone `integral_region u_r dmu_h=0`.
Omitir `region`, o usar `"all"`, selecciona todo el dominio. Una etiqueta, por
ejemplo `MeanZero(component=0, region="heated")`, usa los mismos elementos y
medida inducida que `dx("heated")`. No se promedian vértices ni coeficientes.
En una superficie se integra con área. La condición tiene sentido en L2 y admite
`regularity=0`; la región debe existir y contener elementos.

Las medias se combinan entre sí y con traza cero y periodicidad. Las ecuaciones
redundantes no eliminan dimensión dos veces. En un dominio desconectado, media
global cero elimina una combinación constante, sin imponer una media por
componente conexa. Para eso se etiquetan las regiones respectivas.

**Compatibilidad con la EDP:** una base de media cero siempre reconstruye estados
de media cero, pero eso no demuestra que la EDP original preserve ese conjunto.
El calor periódico o aislado sin fuentes conserva la media global. Una fuente
con integral no nula, un flujo neto o intercambio Robin pueden cambiarla. Imponer
media cero en esos casos define una evolución restringida que puede modificar el
problema físico. Una media regional también requiere justificación.

## Preparación y límites

Las combinaciones se implementan en `laplacian`, `finite-element` y `custom` con
las representaciones nodales certificadas de la parte 3. Las familias polinómica y
Fourier y los callbacks arbitrarios todavía no incorporan estos constructores de
restricciones. Una familia Fourier por sí sola no certifica la periodicidad elegida.

Para cada componente se construye una prolongación dispersa S que identifica y
fija nodos. Se integran las funciones nodales con orden al menos igual al grado
polinómico sobre simplejos afines. Cada funcional integral b se escala por
`sum(abs(b))`, antes de evaluarlo en los candidatos, para evitar que cambios de
escala geométrica borren una restricción o que el redondeo de una media ya nula
se convierta en una restricción espuria. Las filas `b^T S` determinan un núcleo Q:

```text
P = S Q,
M_adm = P^T M P,
K_adm = P^T K P.
```

Sin medias, P=S. El espectro y la normalización L2 se calculan después. Para una
fuente nodal acoplada se apilan filas de traza, diferencias periódicas y medias,
con el escalado de columnas de `V.restrict`.

El núcleo de las medias usa álgebra densa y puede densificar las matrices
espectrales. `max_matrix_entries` limita esa preparación incluso si se piden pocos
modos. `max_quadrature_points` limita la integración. El rango se controla con
`restriction_tolerance=1e-12` en `V.basis`, o `tolerance` en `V.restrict`. Un
espacio total nulo se rechaza. `restriction_rank` cuenta las condiciones
independientes; `restriction_error` es un residual normalizado previo a la
normalización de la base, no una cota física. Las identificaciones y fijaciones
nodales solas producen residual cero. Las declaraciones se copian y son inmutables.

## Fronteras independientes del tiempo

Robin y Neumann admiten datos espaciales fijos en la forma débil. Con la convención
`q_out=-kappa*grad(T) dot n`, el término Neumann es `-q_out*v[0]*ds("wall")` y el
Robin es `-alpha*(T-g)*v[0]*ds("exchange")`. Los datos `alpha`, `g` y `q_out`
pueden variar espacialmente y permanecen independientes del tiempo; la temperatura
y el flujo resultante sí evolucionan.

Para Dirichlet fija no homogénea `T|Gamma_D=g_D`, el usuario proporciona un
levantamiento fijo ell con esa traza, de regularidad adecuada, y escribe:

```text
T(t) = ell + w(t),      w(t) in V,
G_i(z) = a(ell + Phi(z); phi_i).
```

Los tests de V tienen traza cero en Gamma_D. No aparece `-d_t ell` porque ell es
fijo. El campo puede ser afín o no lineal en z y sigue siendo autónomo. El estado
físico completo debe sustituirse también en las reacciones y términos Robin.
`Space` describe w; `ZeroTrace` conserva su significado homogéneo.

En [0,1], para `T(0)=1` y `T(1)=2`, basta `ell(x)=1+x`:

```python
from ngfield import ZeroTrace

W = Space(
    geometry=geometry,
    components=1,
    restrictions=[ZeroTrace(component=0, boundary="all")],
)
lifted_basis = W.basis("laplacian", size=6)


def weak(w, v, dx, ds):
    temperature = w[0] + 1 + dx.x[0]
    return -inner(grad(temperature), grad(v[0])) * dx


G = GalerkinField(basis=lifted_basis, weak=weak)
```

Con datos iniciales y tiempos proporcionados por el usuario, la conexión física es:

```python
z0 = G.project(lambda x: initial_temperature(x) - (1 + x[:, :1]))
Z = G.solve(z0, times)
temperature = 1 + points[:, :1] + G.reconstruct(Z, points)
```

`initial_temperature` devuelve `[points,1]`. La reconstrucción del campo es w:
se suma ell para recuperar T. Una `MeanZero` declarada recae sobre w, no sobre T
salvo que ell también tenga media cero. Lo mismo se debe comprobar para periodicidad.

`dx.x` y las operaciones del lenguaje permiten derivar ell mediante `grad`.
Si se aporta un levantamiento externo, sus valores y gradientes necesarios deben
proporcionarse explícitamente: `Coefficient` no admite derivación espacial
automática en la API actual. Generar ell desde datos de frontera y representar
espacios afines con un objeto propio siguen pendientes.

## Lámina con condiciones mixtas

En el cuadrado [0,1]^2, se impone `ZeroTrace(component=0, boundary="left")` sobre w.
La temperatura `ell=1+x+y^2` es un equilibrio exacto para esta forma:

```python
def weak(w, v, dx, ds):
    x, y = dx.x[0], dx.x[1]
    temperature = w[0] + 1 + x + y**2
    return (
        -inner(grad(temperature), grad(v[0])) * dx
        - 2 * v[0] * dx
        + v[0] * ds("right")
        - (temperature - (4 + x)) * v[0] * ds("top")
    )
```

Los datos físicos son `T(0,y)=1+y^2`, fuente `f=-2`, flujo saliente `q_out=-1`
a la derecha, aislamiento abajo y Robin `alpha=1`, `g=4+x` arriba. Todas las
funciones son fijas en el tiempo. El caso está verificado con elementos de grado
dos; el campo se anula en w=0 y su Jacobiano es simétrico y negativo definido.
