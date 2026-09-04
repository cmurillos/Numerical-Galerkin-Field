# Uso y contratos de Numerical Galerkin Field

## Instalación

Python 3.11 o superior. Desde el repositorio:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

En Windows se activa con .venv\Scripts\activate. Para CUDA, instalar primero la
distribución apropiada de PyTorch siguiendo su [selector oficial](https://pytorch.org/get-started/locally/).
La preparación FEM y espectral utiliza CPU; el campo acepta CPU o CUDA.
La licencia de distribución está pendiente de elección por el propietario.

## Ejemplo mínimo

```python
import torch
from skfem import MeshTri
from ngfield import Domain, FEMSpace, Problem, GalerkinBasis, GalerkinField


def volume(x, u, grad_u):
    return u - u**3, -0.1 * grad_u


domain = Domain(MeshTri.init_lshaped().refined(3))
problem = Problem(components=1, volume=volume, dirichlet=(("all",),))
space = FEMSpace(domain, degree=1)
basis = GalerkinBasis.build(space, problem, modes=12)
G = GalerkinField(basis, problem, quadrature_order=6)

Z = torch.randn(32, basis.dimension, dtype=G.dtype, device=G.device)
Y = G(Z)  # [32,12]
```

El objeto G está listo para evaluar estados; no resuelve una evolución temporal.
Las nuevas EDP se definen en los scripts del usuario mediante callbacks.

## Dominio y fronteras

Domain acepta mallas afines de segmentos, triángulos o tetraedros de scikit-fem.
También puede importar archivos mediante meshio, por ejemplo .msh o .vtu:

```python
domain = Domain.from_file("geometria.msh")
```

La geometría y las etiquetas físicas admitidas por el importador se conservan.
Se debe inspeccionar domain.mesh.boundaries para verificar los nombres importados.
La etiqueta reservada "all" siempre denota toda la frontera exterior.
Para etiquetar por predicados evaluados en puntos medios de facetas:

```python
domain = domain.with_boundaries(left=lambda x: x[0] < -0.999)
```

Los predicados reciben coordenadas con convenio scikit-fem [m,facetas]; esto es
distinto del convenio [Q,m] de los callbacks del campo. Un predicado debe seleccionar
la frontera geométrica pretendida, teniendo en cuenta la tolerancia de la malla.
Las etiquetas desconocidas o vacías producen un error al usarlas.
La malla no se debe modificar después de crear el espacio FEM.

dirichlet contiene una tupla de nombres **por componente**. Por ejemplo:

```python
problem = Problem(2, volume_coupled, dirichlet=(("all",), ("left",)))
```

Las condiciones esenciales de esta versión son homogéneas. Si se omite
dirichlet, todas las componentes usan el espacio sin restricciones esenciales.
La ausencia de un callback de frontera significa contribución natural nula.
Los datos Dirichlet no homogéneos necesitan un levantamiento; no se implementan
mediante asignaciones posteriores a los modos.

## Componentes y truncamiento

modes=8 solicita ocho modos para **cada** componente. Para d=3 devuelve N=24.
modes=(8,6,4) fija N=18 con distinto truncamiento por componente. No confundir N
con la cantidad de nodos, grados FEM escalares, coordenadas espaciales o componentes.

```python
basis = GalerkinBasis.build(space, problem, modes=(8, 6))
print(basis.dimension)  # 14
print(basis.slices)  # bloques de coeficientes por componente
print(basis.diagnostics())  # ortonormalidad, residuos y frontera
```

Cada modes[r] debe ser positivo y no superar los grados de libertad libres de
esa componente. Para resolver más modos puede ser necesario refinar la malla.
Los modos nulos de Neumann se incluyen y cuentan dentro del truncamiento.
Si un corte divide un autoespacio múltiple, el subespacio seleccionado no tiene
una orientación canónica: guardar la base es esencial para reproducir coordenadas.

## Forma débil y formas de los tensores

| Objeto | Forma |
|---|---|
| Z | [B,N], o [N] para una evaluación individual. |
| x | [Q,m]. |
| u | [B,d,Q]. |
| grad_u | [B,d,Q,m]. |
| f0 | [B,d,Q], o escalar/tensor compatible por broadcasting. |
| f1 | [B,d,Q,m], o escalar/tensor compatible por broadcasting. |
| normal, sólo en frontera | [Q,m]. |
| Resultado G(Z) | La misma forma de coeficientes que Z. |

El callback volume(x,u,grad_u) devuelve (f0,f1). Para difusión con coeficientes
por componente, construir el vector de coeficientes con forma [1,d,1,1].
Los callbacks deben preservar la independencia del lote: no promediar ni reducir
sobre B. El paquete no puede verificar esta propiedad para una función arbitraria.

Las contribuciones naturales se suministran como boundary={nombre: callback}:

```python
def robin(x, u, grad_u, normal):
    return -2.0 * u


problem = Problem(1, volume, boundary={"all": robin})
```

Cada callback devuelve la densidad multiplicada por el valor de la función de
prueba; el paquete realiza la integral. Si varias etiquetas se solapan, sus
contribuciones **se suman**, de manera intencional. No se deduplican términos
físicos. Los valores del callback de una componente esencial no cambian sus
grados de libertad: las funciones de prueba ya tienen traza cero allí.

Las funciones pueden depender de x, u y grad_u y de parámetros capturados por
una función o un objeto callable. Deben devolver tensores con el mismo dispositivo
y dtype que los estados. Para constantes nuevas usar u.new_tensor(...), o
escalares Python. G no mueve automáticamente los parámetros externos del usuario.

## Dispositivos y derivadas

```python
G = GalerkinField(basis, problem, device="cuda", dtype=torch.float64)
Z = torch.zeros(16, basis.dimension, device=G.device, dtype=G.dtype, requires_grad=True)
Y = G(Z)
(gradient,) = torch.autograd.grad(Y.square().sum(), Z)
```

G.to(device=...,dtype=...) mueve sus tablas **en el mismo objeto** y devuelve G.
Se admiten float64 y float32; float64 es el valor inicial para verificaciones.
La base FEM se prepara y guarda en CPU/float64. Se conservan las derivadas de
PyTorch con respecto a Z y a los parámetros de callbacks que mantengan su grafo.
No se diferencia a través de la malla ni del cálculo de autovectores de SciPy.

G.reconstruct(Z) devuelve valores y gradientes en cuadratura, conservando siempre
el eje B. G.quadrature_points() devuelve una copia de los puntos físicos.
G.table_bytes informa la memoria de las tablas, sin incluir estados y temporales.
La memoria depende de Q, N y m; para lotes grandes se puede evaluar por bloques
de estados desde el código llamador.

## Guardar y reutilizar la base

```python
from ngfield import save_basis, load_basis

save_basis("base.npz", basis)
restored = load_basis("base.npz")
G = GalerkinField(restored, problem, quadrature_order=8)
```

El archivo conserva malla, etiquetas, numeración de grados de libertad, modos,
valores propios y metadatos de versión. La carga exige la misma versión de
scikit-fem y verifica numeración, condiciones esenciales y matriz de Gram.
No recalcula los modos, no usa pickle y no ejecuta código del problema.
Para reproducir un experimento completo, conservar también el código y parámetros
del callback, el orden de cuadratura, las versiones instaladas y los estados usados.
Los archivos existentes se protegen salvo overwrite=True.

La base puede reutilizarse con otra acción débil que comparta componentes y
condiciones esenciales. Los nombres esenciales normalizados deben coincidir.
La cuadratura del campo puede refinarse sin modificar la masa ni los coeficientes.

## Verificación y rendimiento

```bash
python -m pytest
python examples/diffusion.py
python examples/reaction_diffusion.py
python benchmarks/field_evaluation.py --device cpu --output outputs/benchmark.json
```

Los ejemplos usan dominios no rectangulares. La suite incluye pruebas de malla,
espectro, forma débil, frontera, lotes, autograd, persistencia y refinamiento.
En una máquina CUDA, la prueba marcada cuda se ejecuta; en CI CPU se omite.
El benchmark separa preparación y evaluación, sincroniza CUDA para medir tiempo
y registra versiones. El muestreo, la preparación y las pruebas no integran tiempo.

## Alcance inicial

Implementado: H = L²(Ω;ℝᵈ), geometrías afines malladas en 1D–3D, FEM continuo
P1/P2, fronteras esenciales homogéneas por componente, modos del Laplaciano,
acciones locales volumétricas con valores/primeras derivadas y cargas naturales
de frontera, evaluación CPU/CUDA y persistencia de la base.

Requieren extensiones específicas: espacios con otra métrica, condiciones
periódicas por identificación de grados, geometría curvilínea de orden alto,
levantamientos no homogéneos, formas no locales, derivadas de orden superior y
espacios restringidos como campos de divergencia nula. Estas variantes no se
interpretan silenciosamente como el problema implementado.
