# Cambios

## 2.0.0 — 2026-09-18

- Carga automática del CSV local y corrección del estado global al cambiar controles.
- Separación 60/20/20, selección en validación y evaluación reservada del ganador.
- Calibración de tres folds con transformaciones ajustadas dentro de cada fold.
- Preprocesamiento persistido e independiente del lote de scoring.
- Política basada en beneficio esperado positivo, actualizada con los parámetros de campaña.
- Definición única de beneficio incremental y baseline aleatorio de cobertura exacta en esperanza.
- Validación de esquema y manejo de clientes individuales, empates y campañas vacías.
- Artefacto v2 con metadatos, hash del dataset y versiones de entrenamiento.
- Reportes, README y notebook regenerados; caso de estudio para portafolio.
- 17 pruebas con datos sintéticos y workflow de GitHub Actions.

### Migración

Los modelos v1 no son compatibles. Ejecuta `python -m src.train_pipeline` con las dependencias fijadas y reinicia Streamlit. No compares directamente las métricas antiguas con las nuevas: cambian el protocolo de evaluación, la calibración y la política.
