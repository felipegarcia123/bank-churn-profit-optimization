# Retention Copilot

Predicción de abandono bancario y priorización de campañas por **beneficio incremental esperado**. Una app Streamlit convierte una base local en una lista de contactos, recalculada al cambiar costo y tasa de éxito.

**Estado:** prototipo reproducible para portafolio. Las cifras económicas son simulaciones bajo supuestos, no ingresos observados ni efectos causales demostrados.

## Inicio rápido

Desde la raíz del proyecto, con Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Coloca `Churn_Modelling.csv` en `data/raw/`. La fuente utilizada es [Bank Customer Churn en Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling). Los datos no se distribuyen con este repositorio.

```bash
python -m src.train_pipeline
streamlit run app.py
```

La app lee el CSV directamente de la carpeta; no requiere cargarlo desde el navegador. Admite el esquema original de Kaggle o los nombres normalizados definidos en `src/data_processing.py`. Para scoring, `Exited` es opcional; las diez variables predictoras originales son obligatorias. `CustomerId` es opcional y, si existe, debe ser único y no vacío.

Alternativa con Conda, incluyendo herramientas de notebook:

```bash
conda env create -f environment.yml
conda activate churn
```

Las versiones directas están fijadas y se verificaron con Python 3.11. No se incluye un lockfile de todas las dependencias transitivas. En algunos sistemas XGBoost requiere un runtime OpenMP compatible.

## Decisión económica

Se estima un **ingreso anual por cliente**, no un CLV de vida completa:

`valor = 0,025 × saldo + 80 × productos + 150 × indicador de tarjeta`

Los coeficientes son supuestos ilustrativos. El saldo utilizado se limita al percentil 99 aprendido en entrenamiento.

Para cada cliente:

`beneficio esperado de contactar = P(abandono) × valor × tasa de éxito − costo`

Solo se recomiendan contactos con beneficio esperado positivo. Se ordenan por ese beneficio; con costo y éxito constantes, equivale a ordenar por probabilidad × valor. El riesgo mínimo manual es un filtro adicional opcional, no el criterio principal.

El beneficio retrospectivo incremental se calcula como `éxito × valor de churners contactados − costo total`. No contactar tiene beneficio incremental cero. Las pérdidas de ingreso que siguen en riesgo se muestran aparte y no se descuentan otra vez. El ROI divide el beneficio incremental por el costo; **maximizar beneficio no equivale a maximizar ROI**.

## Metodología

1. División estratificada reproducible: **60% entrenamiento / 20% validación / 20% test**.
2. Tres candidatos: regresión logística, Random Forest y XGBoost.
3. Calibración sigmoide de tres folds dentro del entrenamiento. Cada fold ajusta también imputación, límites p99, escalado y codificación.
4. Selección por beneficio de la política económica en validación. No se ajusta un umbral global sobre test.
5. Evaluación del ganador en test reservado; comparación con no actuar, llamar a todos y azar de la misma cobertura.

El baseline aleatorio usa su esperanza analítica al seleccionar el mismo número de clientes sin reemplazo, evitando variación por una sola semilla. Las transformaciones se persisten: un cliente recibe el mismo procesamiento independientemente del tamaño del lote de scoring.

## Resultados de la ejecución incluida

Dataset: 10.000 filas; test: **2.000 clientes**, 407 con abandono. Costo: **US$15**; éxito supuesto: **30%**. Todas las filas de esta tabla usan el mismo test.

| Política | Contactos | Costo | Beneficio incremental simulado |
|---|---:|---:|---:|
| No actuar | 0 | US$0 | US$0 |
| Llamar a todos | 2.000 | US$30.000 | US$271.627,87 |
| Azar, misma cobertura (esperanza) | 1.370 | US$20.550 | US$186.065,09 |
| XGBoost + política económica | 1.370 | US$20.550 | **US$278.032,87** |

Mejora frente a llamar a todos: **US$6.405**, con **31,5% menos contactos**. ROC-AUC: **0,855**; average precision: **0,698**; Brier score: **0,104**. ROI incremental simulado: **1.353%**.

XGBoost superó a Random Forest por solo US$54 en validación. Esa diferencia pequeña no demuestra superioridad robusta; estas cifras corresponden a una partición fija y no incluyen intervalos de confianza.

Los valores y el hash del dataset están en [el reporte JSON](reports/executive_report.json). No deben compararse directamente con versiones anteriores que seleccionaban el modelo sobre test. La app puntúa la base completa —que puede contener filas de entrenamiento— y sus cifras no sustituyen esta evaluación reservada.

## Pruebas y notebook

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
jupyter notebook notebooks/01_churn_financial_analysis.ipynb
```

Las pruebas usan datos sintéticos y artefactos temporales: no requieren descargar el dataset ni entrenar el modelo real. CI ejecuta las mismas pruebas en Python 3.11. El notebook lee el reporte agregado; no vuelve a seleccionar modelos sobre test ni incluye identificadores o salidas antiguas.

## Estructura

```text
app.py                         Dashboard y carga local
src/data_processing.py         Validación, split y transformaciones persistidas
src/modeling.py                Clasificadores y calibración
src/financial_evaluator.py     Política, beneficio, baselines y ranking
src/train_pipeline.py          Entrenamiento y exportación reproducibles
tests/                         Pruebas unitarias y de Streamlit
notebooks/                     Lectura explicativa de resultados
reports/executive_report.*     Resultados agregados versionados
data/raw/                      Dataset local excluido de Git
models/                        Artefacto local excluido de Git
docs/PORTFOLIO.md               Caso de estudio y texto para el portafolio
docs/PUBLISHING.md              Guardado en Git y publicación posterior
.github/workflows/ci.yml        Pruebas en cada push y pull request
```

`reports/priority_calls.csv` contiene solo recomendaciones del test y se excluye de Git. El CSV descargado desde la app corresponde a la base local que esté puntuando.

## Configuración y mantenimiento

```bash
python -m src.train_pipeline --retention-cost 25 --retention-success 0.40
```

El CLI acepta rutas mediante `--data`, `--out-model`, `--out-report` y `--out-calls`. La app acepta `CHURN_DATA_PATH` y `CHURN_MODEL_PATH` como variables de entorno; por defecto resuelve las rutas desde la ubicación de `app.py`. `.env.example` documenta estas opciones; el archivo `.env` no se carga automáticamente.

Al modificar módulos Python, reinicia el servidor con Ctrl+C y `streamlit run app.py`; recargar solo el navegador puede conservar módulos anteriores en memoria. Al regenerar el modelo, la app invalida su caché según la fecha de modificación. Los artefactos v1 requieren reentrenamiento.

## Límites

- No hay datos de intervención: el éxito de retención es un supuesto homogéneo, no una estimación causal.
- El valor anual omite horizonte de vida, descuento, costos completos y heterogeneidad de ofertas.
- La validación es aleatoria; no demuestra estabilidad temporal ni generalización a otro banco.
- La calibración se ajusta con CV, pero requiere evaluación adicional por segmentos y fuera de muestra.
- El modelo incluye geografía y género. No se ha completado una evaluación de equidad para uso operativo.
- La app es local, sin autenticación ni controles para una base real en producción.


