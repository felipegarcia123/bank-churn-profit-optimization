# Retention Copilot — caso de estudio

**Objetivo:** decidir a qué clientes contactar en una campaña de retención bancaria considerando riesgo de abandono, ingreso anual estimado y costo de intervención.

## Problema

Un ranking por probabilidad de abandono no tiene en cuenta cuánto valor podría preservar una llamada. Además, un umbral fijo de 0,5 no responde a cambios en el costo o la eficacia de la campaña.

## Solución implementada

Aplicación Streamlit que carga una base local, permite modificar los supuestos y genera una lista priorizada de contactos. La política recomienda una llamada cuando su beneficio incremental esperado es positivo:

`P(abandono) × ingreso anual estimado × éxito de retención − costo > 0`

Los predictores se comparan en validación por beneficio simulado. La preparación y calibración se ajustan dentro de entrenamiento; el conjunto de test queda reservado para evaluar al ganador. Los datos originales, modelos binarios y listas con identificadores se mantienen fuera de Git.

## Resultados reproducidos

En una partición de 6.000 clientes de entrenamiento, 2.000 de validación y 2.000 de test, con costo de US$15 y éxito supuesto del 30%:

- XGBoost fue seleccionado en validación. Superó a Random Forest por solo US$54 en beneficio simulado; no es evidencia de una superioridad robusta.
- En test, recomendó contactar a 1.370 de 2.000 clientes.
- Beneficio incremental simulado: **US$278.032,87**.
- Mejora frente a contactar a todos: **US$6.405**, con 630 contactos menos.
- Mejora frente a una selección aleatoria de igual tamaño, en esperanza: **US$91.967,78**.
- ROC-AUC de test: **0,855**; average precision: **0,698**; Brier score: **0,104**.

Fuente reproducible: [reporte agregado](../reports/executive_report.json). Son resultados sobre una partición fija, sin intervalos de confianza.

## Qué demuestra

- Integración de clasificación, calibración de probabilidades y decisiones económicas.
- Prevención de contaminación entre entrenamiento, validación y test.
- Consistencia entre las métricas de la app y las filas que se exportan.
- Manejo de lotes pequeños, categorías desconocidas, entradas inválidas y campañas vacías.
- Pruebas con datos sintéticos y CI sin necesidad de distribuir el dataset.

## Límites y siguiente experimento

El valor es una aproximación de ingreso anual, no CLV completo ni beneficio contable. La tasa de éxito no se estima con datos de intervención. El proyecto predice abandono y simula decisiones; no demuestra que una llamada cause retención.

Antes de un uso real harían falta validación temporal y externa, evaluación por segmentos, estimación de costos completos y un experimento controlado que mida el efecto incremental de contactar. La calibración utilizada no garantiza probabilidades perfectas fuera de la muestra.

## Texto breve para el portafolio

> Desarrollé Retention Copilot, una aplicación de machine learning para priorizar campañas de retención bancaria por beneficio esperado. Integra tres clasificadores con calibración, separación 60/20/20 y un dashboard que recalcula recomendaciones según costo y eficacia supuesta. En 2.000 clientes de test, la política seleccionada simuló US$6.405 adicionales frente a contactar a todos, con un 31,5% menos de contactos. El proyecto incluye pruebas automatizadas, CI y documentación de los supuestos y límites del análisis.

## Guion de demostración

1. Mostrar la carga automática de la base y el modelo activo.
2. Cambiar costo y tasa de éxito; observar cómo cambia la cantidad recomendada.
3. Llevar la tasa de éxito a cero para demostrar que se recomienda no realizar campaña.
4. Volver a los supuestos iniciales y descargar el ranking.
5. Explicar que la app sobre la base completa es una simulación; las métricas reservadas están en el reporte de test.
