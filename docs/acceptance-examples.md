# Ejemplos de aceptación — D-013, parte 7

Los cuatro programas recorren la API implementada: vértices, simplejos y subconjuntos;
`Space`; `V.basis(...)`; forma débil; `GalerkinField(basis=basis, weak=weak)`.
Cada archivo contiene `build_field()` para obtener G sin integrar y `run()` para
proyectar datos iniciales, integrar, reconstruir y comprobar el resultado.
La geometría y todos los datos del operador son fijos: G es autónomo.
La [guía de uso](usage.md) describe cada operación y la [migración](migration.md)
explica cómo adaptar programas existentes a este mismo recorrido.

| Programa | Espacio de variaciones | Aproximación | Referencia principal |
|---|---|---|---|
| [Toro](../examples/acceptance_torus.py) | H1 de la superficie cerrada triangulada | 12 modos laplacianos P1, incluida la constante | Evolución espectral de la misma discretización, conservación de media y disipación. |
| [Lámina](../examples/acceptance_mixed_plate.py) | H1 del cuadrado, traza cero a la izquierda | Espacio FEM P2 completo: 20 coordenadas | Equilibrio físico conocido con fuente localizada y fronteras mixtas. |
| [Dos componentes](../examples/acceptance_components.py) | H1 x H1, con trazas cero en extremos distintos | Espacio FEM P2 completo: 8 + 8 coordenadas | Equilibrio no lineal conocido, acoplamiento y disipación respecto de él. |
| [Periodicidad](../examples/acceptance_periodic.py) | H1 periódico de media cero | 4 modos laplacianos P1 en dos mallas | Solución exacta de la EDP y reducción del error espacial. |

Los tamaños de los espacios FEM completos se deducen de la malla, el grado y las
restricciones; no se truncan. En las familias laplacianas `size` cuenta la dimensión
reducida total. Incluso las ecuaciones escalares utilizan `u[0]` y valores `[points,1]`.

## Ejecución

Desde un checkout que contenga esta entrega, instalar el paquete y las herramientas
de desarrollo permite ejecutar los cuatro ejemplos con sus comprobaciones:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_acceptance_examples.py
```

También se pueden ejecutar por separado, con el paquete instalado:

```bash
python examples/acceptance_torus.py
python examples/acceptance_mixed_plate.py
python examples/acceptance_components.py
python examples/acceptance_periodic.py
```

Cada programa imprime un informe JSON cuando sus comprobaciones terminan y falla
si se incumple un umbral. Ejecutar Python sin `-O`, para conservar los `assert`.
Los ejemplos usan CPU y float64; no requieren gráficos, archivos externos ni datos
de entrenamiento. Pertenecen al desarrollo posterior a 0.9.0, no al código de esa
versión publicada. El test parametrizado ejecuta los mismos archivos mediante
`runpy`; el workflow existente de CI los incluye al descubrir todos los tests.

## Calor en el toro

La parametrización con radios R=2 y r=0.5 genera 96 vértices y 192 triángulos. Las
identificaciones se hacen en la conectividad: no quedan caras exteriores ni hacen
falta restricciones `Periodic`. El problema sobre la superficie poliédrica Gamma_h es

```text
T_t = 0.1 Delta_Gamma_h T,
a(T;v) = -0.1 integral_Gamma_h grad_Gamma_h(T) dot grad_Gamma_h(v) dS_h.
```

El dato inicial es la proyección de `2 + 0.2*x_1 + 0.3*x_3`. El espacio reducido
incluye la constante, por lo que preserva la media 2. Se verifican en los tiempos
muestreados la media, la disminución de la energía respecto de la media y la identidad

```text
d/dt [1/2 integral_Gamma_h (T-mean(T))^2 dS_h]
    = -0.1 integral_Gamma_h |grad_Gamma_h(T)|^2 dS_h.
```

Las integrales de diagnóstico usan una cuadratura espacial independiente de orden 6.
La trayectoria de referencia usa los autovalores FEM de la base:
`z_j(t)=exp(-0.1*lambda_j*t)*z_j(0)`. Comprueba el ensamblaje y el integrador sobre
la misma malla. **No es una solución exacta sobre el toro curvo**, ni estima el error
geométrico o la convergencia de esa superficie al refinar la triangulación.

## Lámina con fuente localizada y fronteras mixtas

En Omega=(0,1)^2 se resuelve

```text
T_t = Delta T + 2*1_{x>1/2},
T(0,y) = 1+y,
q_out = +1 abajo,   q_out = -1 arriba,
q_out = (1+y)*(T-(1.75+y)) a la derecha,
q_out = -grad(T) dot n.
```

El subconjunto `heated` contiene exactamente los triángulos con x>1/2; las cuatro
fronteras tienen sus propias etiquetas. La fuente total es 1. El flujo Neumann
entra con signo `-q_out*v*ds` y el intercambio Robin con
`-(1+y)*(T-(1.75+y))*v*ds("right")`.

El levantamiento fijo es `ell=1+y`: se representa `T=ell+w`, con w de traza cero a
la izquierda. El programa proyecta `T0-ell` y suma ell al reconstruir. El equilibrio

```text
T_*(x,y) = 1+y+x-max(x-1/2,0)^2
```

cumple todos los datos. Es C1 y cuadrático por partes, y la malla está alineada con
x=1/2; por ello `T_*-ell` pertenece exactamente al espacio FEM P2 elegido. Se comprueban
la reconstrucción de ese equilibrio, `G(z_*)=0`, la traza física fija y

```text
e = T-T_*,
d/dt [1/2 integral_Omega e^2 dx]
    = -integral_Omega |grad(e)|^2 dx
      -integral_right (1+y)*e^2 ds.
```

La trayectoria empieza en `T0=ell`. Una segunda referencia utiliza
`z(t)=z_*+exp(t*A)*(z0-z_*)`, con `A=DG(z_*)`. SciPy calcula esta exponencial;
**esta comparación sólo verifica la integración temporal del campo ensamblado**.
La comprobación espacial proviene del equilibrio prescrito y del balance integral.

## Dos componentes con fronteras diferentes

En (0,1), con kappa=(0.2,0.1) y beta=0.3:

```text
u_t = kappa_0*u_xx + beta*(v-u) - u^3 + f_0(x),
v_t = kappa_1*v_xx + beta*(u-v) - v^3 + f_1(x),
u(0)=0,   u_x(1)=0,   v_x(0)=0,   v(1)=0.
```

Las restricciones se declaran por componente: `ZeroTrace(component=0, boundary="left")`
y `ZeroTrace(component=1, boundary="right")`. Los flujos
nulos en los otros extremos son naturales. Las fuentes fijas se eligen a partir de
`a=x*(2-x)` y `b=1-x^2`:

```text
f_0 = 2*kappa_0 - beta*(b-a) + a^3,
f_1 = 2*kappa_1 - beta*(a-b) + b^3.
```

Entonces `(a,b)` es un equilibrio exacto representable en P2. El estado inicial
es `(0.7*a,1.2*b)`. La integración y la reconstrucción conservan el orden de las
dos componentes. Se comprueban sus trazas esenciales, que las otras trazas no se
anulen por accidente, y que el Jacobiano tenga acoplamiento entre ambas componentes.

Para `e_0=u-a`, `e_1=v-b`, la identidad comprobada es

```text
d/dt [1/2 integral (e_0^2+e_1^2) dx]
  = -integral [kappa_0*|e_0'|^2 + kappa_1*|e_1'|^2
               + beta*(e_0-e_1)^2
               + e_0*(u^3-a^3) + e_1*(v^3-b^3)] dx.
```

Cada término del integrando es no negativo; para los cúbicos se usa
`(s-t)*(s^3-t^3) >= 0`. El programa compara ambos lados y comprueba el descenso
de energía. También verifica que `DG(z_*)` sea simétrico y negativo definido.
La cuadratura de orden 8 integra los productos polinómicos de esta formulación.
**No se prescribe una solución transitoria exacta para este sistema no lineal.**

## Difusión periódica de media cero

El problema `u_t=0.05*u_xx` en (0,1), con periodicidad y media cero, tiene la solución

```text
u(t,x) = exp(-0.05*(2*pi)^2*t)*sin(2*pi*x)
         + 0.25*exp(-0.05*(4*pi)^2*t)*cos(4*pi*x).
```

Se identifican los vértices de los extremos con `Periodic`, se añade `MeanZero` y
se seleccionan cuatro modos sobre mallas de 16 y 32 intervalos. Los modos P1 son
aproximaciones de las dos parejas seno/coseno; la solución continua no pertenece
exactamente a estos espacios. Se mide el error L2 mediante cuadratura de orden 8
en siete tiempos de [0,0.3], incluida la proyección inicial.

La aceptación exige que el máximo de esos errores en la malla fina sea menor que
0.003 y que el cociente fino/grueso sea menor que 0.35, compatible con la reducción
por un factor cercano a cuatro esperada para este caso P1 suave. No constituye un
teorema de convergencia ni un estudio general de refinamiento en N.

La referencia exponencial con los autovalores FEM separa el error temporal del
error frente a la EDP. También se comprueban la igualdad de trazas y la integral
cero en ambas mallas. La fuente y el flujo neto son nulos, de modo que la media cero
es compatible con la dinámica original.

## Interpretación de los informes

Una ejecución de referencia en CPU/float64 produjo estos valores redondeados. Los
últimos dígitos pueden cambiar entre plataformas; la aceptación usa los umbrales
del programa, no igualdad con esta tabla.

| Ejemplo | Diagnóstico observado |
|---|---|
| Toro | Error temporal discreto 6.60e-12; error de media 1.11e-15. |
| Lámina | Residual del equilibrio 6.70e-14; error frente a la exponencial matricial 5.48e-11. |
| Dos componentes | Residual del equilibrio 8.41e-14; fracción de energía final 0.0281. |
| Periodicidad | Error L2: 6.016e-3 con 16 celdas y 1.453e-3 con 32; cociente 0.2416. |

`time_error_*` es el máximo error absoluto en coordenadas, sobre los tiempos
muestreados. `max_L2_error_vs_PDE` corresponde a las dos mallas periódicas; incluye
aproximación espacial, proyección, cuadratura y el error temporal restante.
`equilibrium_residual` mide coordenadas de G, mientras que `equilibrium_value_error`
mide valores físicos. `energy_balance_error` es el residual absoluto de las
identidades anteriores. `final_energy_fraction` compara energía final e inicial
respecto de la media o del equilibrio, según el caso.

Los umbrales algebraicos están entre 1e-12 y 1e-10; las comparaciones temporales
permiten tolerancias absoluta y relativa de hasta 3e-8. El código contiene los
umbrales exactos. Son criterios de aceptación de estos casos en float64, no cotas
certificadas de error para una EDP arbitraria. Los diagnósticos no cambian G, su
espacio, su cuadratura ni las coordenadas de su base.
