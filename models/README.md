El entrenamiento genera aquí `best_model.pkl`. El binario se excluye de Git.

```bash
python -m src.train_pipeline
```

El artefacto v2 contiene el predictor calibrado, el preprocesador de valor anual, los supuestos y los metadatos de evaluación. Carga únicamente artefactos generados por ti o de una fuente de confianza: joblib utiliza serialización pickle.
