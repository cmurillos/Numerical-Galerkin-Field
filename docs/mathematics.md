# Formulación matemática y realización numérica

## 1. Objeto del paquete

Sean un dominio acotado Ω ⊂ ℝᵐ y el espacio real H = L²(Ω; ℝᵈ), con el
producto interno que suma las integrales de las componentes. Los símbolos son:

| Símbolo | Significado |
|---|---|
| m | Dimensión espacial: 1, 2 o 3 en la implementación actual. |
| d | Número de componentes del estado. |
| nₕ | Número de grados de libertad FEM **escalares**. |
| Nᵣ | Modos retenidos en la componente r. |
| N = Σᵣ Nᵣ | Dimensión total de entrada y salida del campo reducido. |
| Q | Número total de puntos de cuadratura volumétrica, sumando los elementos. |
| B | Número de estados independientes de un lote. |

El operador de evolución se suministra mediante su acción débil a(u;v), lineal
en v. No se exige linealidad, simetría ni una estructura de gradiente en u.
Para una base ortonormal admisible {φᵢ} de un subespacio V_N, la síntesis es

$$
\Phi:\mathbb R^N\to V_N,\qquad \Phi z=\sum_{i=1}^N z_i\varphi_i,
\qquad [G(z)]_i=a(\Phi z;\varphi_i).
$$

El paquete aproxima esta evaluación numérica. Obtener G(z) no requiere integrar
la EDO ż = G(z), construir trayectorias ni disponer de soluciones de referencia.

## 2. Dominio, fronteras y espacio FEM

La entrada efectiva es una malla afín de segmentos, triángulos o tetraedros.
Un dominio curvo está aproximado por su malla Ωₕ; este error geométrico se debe
distinguir del error en los modos y en las integrales. Se utilizan elementos
continuos P1 o P2 sobre esa geometría, con una base escalar {ψₖ}.

Las condiciones esenciales homogéneas se incorporan eliminando exactamente sus
grados de libertad en el cálculo de modos. Cada componente puede tener un
conjunto distinto de etiquetas esenciales. Todas comparten malla y grado FEM.
Los términos naturales de frontera pertenecen a la forma débil del usuario.

El dominio y la fórmula diferencial no bastan para especificar una EDP: el
espacio variacional y sus condiciones de frontera también son parte del problema.
La conformidad H¹ de P1/P2 no basta por sí sola para formas que exijan H²,
restricciones de divergencia o trazas especiales.

## 3. Base de referencia

Se usa el Laplaciano positivo como operador auxiliar. Se ensamblan

$$
(M_h)_{k\ell}=\int_{\Omega_h}\psi_\ell\psi_k\,dx,
\qquad (K_h)_{k\ell}=\int_{\Omega_h}\nabla\psi_\ell\cdot\nabla\psi_k\,dx.
$$

Ambas matrices tienen tamaño nₕ × nₕ. Después de eliminar los grados esenciales
de cada componente, se resuelve Kₕc = λMₕc y se seleccionan los valores propios
más pequeños. En el resto de la frontera, el **problema auxiliar** tiene la
condición natural de Neumann. Esto no obliga al operador real a ser un Laplaciano
ni impide incorporar términos Robin en su acción débil.

El caso Neumann incluye sus modos nulos; no se elimina automáticamente la media.
En dominios desconectados puede haber más de un modo nulo. El solver disperso
usa un desplazamiento negativo para evitar invertir una rigidez singular.

Para cada componente r se guarda Cᵣ ∈ ℝⁿʰˣᴺʳ, con ceros en los grados esenciales,
y se normaliza de modo que CᵣᵀMₕCᵣ = I. La matriz vectorial conceptual es

$$
C=\operatorname{diag}(C_1,\ldots,C_d),\qquad
\mathcal M_h=\operatorname{diag}(M_h,\ldots,M_h),\qquad
C^\top\mathcal M_h C=I_N.
$$

No se almacena esa gran matriz de bloques: coefficients[r] contiene Cᵣ.
El orden de los coeficientes es por componente, y dentro de cada componente
por valor propio creciente. Se fija el signo de cada modo usando su coeficiente
de mayor módulo. Esto no elimina la libertad de rotación de autoespacios múltiples.
Para reproducibilidad se guarda la base calculada, en lugar de regenerarla.

La masa se calcula con una regla de grado al menos 2p, donde p es el grado FEM.
En geometría afín integra exactamente los productos polinómicos de las funciones
de forma, salvo redondeo. La identidad de masa corresponde al dominio mallado
y al producto discreto elegido; no afirma exactitud de los modos del dominio Ω.

## 4. Campo discreto y proyección directa

Si rₕ,Q es la acción débil ensamblada contra la base FEM vectorial, entonces

$$
G_{h,Q}(z)=C^\top r_{h,Q}(Cz).
$$

En efecto, por linealidad respecto de la función de prueba,

$$
\big[C^\top r_{h,Q}(Cz)\big]_i
=\sum_k C_{ki}\,a_{h,Q}(\Psi_h Cz;\psi_k)
=a_{h,Q}(\Phi_h z;\varphi_{i,h}).
$$

No se necesita resolver un sistema con la masa FEM para cada estado. Si se
representara primero una acción fuerte por y mediante ℳₕy = rₕ,Q, su proyección
sería Cᵀℳₕy = Cᵀrₕ,Q. La cancelación no identifica ℳₕ con la identidad.

La implementación evita incluso formar el vector rₕ,Q cuando evalúa el campo:
precalcula los modos reducidos y sus gradientes en los puntos de integración,
reconstruye allí el estado y contrae directamente con las funciones de prueba.
Ambas realizaciones coinciden algebraicamente si utilizan la misma cuadratura.

## 5. Contrato de forma débil

La primera implementación admite acciones locales de la forma

$$
a(u;v)=\int_{\Omega_h}
\left[f_0(x,u,\nabla u)\cdot v+f_1(x,u,\nabla u):\nabla v\right]dx
+\sum_{\Gamma}\int_{\Gamma}b_\Gamma(x,u,\nabla u,n)\cdot v\,ds.
$$

Aquí f₀ tiene d componentes, f₁ es una matriz d × m y n es la normal exterior.
Las funciones pueden acoplar todas las componentes del estado. Los callbacks
reciben el lote completo, pero deben actuar independientemente sobre cada estado.
La proyección no impone una convención extra de signo: para difusión
uₜ = div(κ∇u) + g(u), el término volumétrico es f₀ = g(u), f₁ = −κ∇u.
El flujo de frontera aparece con el signo de la integración por partes.

Los tensores calculados se organizan como

$$
Z\in\mathbb R^{B\times N},\quad
U\in\mathbb R^{B\times d\times Q},\quad
\nabla U\in\mathbb R^{B\times d\times Q\times m},\quad
G^{[B]}(Z)\in\mathbb R^{B\times N}.
$$

Las contracciones se vectorizan sobre los estados y los modos. Sólo hay bucles
de evaluación sobre componentes físicas y etiquetas de frontera. Los primeros
ejes nunca representan una mezcla entre estados independientes.

## 6. Precisión y alcance de las comprobaciones

Se distinguen: truncamiento del espacio reducido, aproximación de geometría y
modos por FEM, cuadratura de la acción y redondeo. La ortonormalidad no elimina
los errores de integración de una no linealidad. Para estudiar cuadratura se
mantiene **la misma base guardada** y se cambia únicamente quadrature_order.

Al comparar mallas, las coordenadas de sus modos pueden tener signos o rotaciones
distintos, especialmente ante valores propios múltiples. No se deben restar
campos coordenados entre mallas sin alinear las bases o comparar cantidades
invariantes/funcionales. El test de refinamiento usa el primer modo de calor en
un intervalo, cuyo límite y significado están fijados analíticamente.

La suite compara el campo con el espectro discreto conocido de P1, con ensamblaje
FEM independiente para una forma acoplada no lineal, con identidades de frontera
y con refinamientos de malla/cuadratura. También verifica derivación automática,
persistencia y lotes. La prueba CUDA se ejecuta sólo si hay GPU; la CI alojada
en runners estándar verifica CPU y no certifica rendimiento ni precisión CUDA.

Estos tests verifican implementaciones y casos de referencia; no son un teorema
de convergencia para cualquier operador suministrado por el usuario.

## Referencias

- [scikit-fem: API de mallas, bases, cuadratura y ensamblaje](https://scikit-fem.readthedocs.io/en/latest/api.html).
- [FreeFEM: problema variacional de valores propios del Laplaciano](https://doc.freefem.org/models/eigen-value-problems.html).
- [SciPy: eigsh y problemas generalizados con matriz de masa](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
- [SciPy: eigh para problemas simétricos generalizados](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
- [PyTorch: contracciones tensoriales con einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html).
