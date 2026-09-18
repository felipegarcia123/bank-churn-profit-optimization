# Guardar en Git y presentar el proyecto

## Contenido del repositorio

Versionar código, pruebas, configuración, notebook sin salidas y reportes agregados. `.gitignore` excluye el dataset, modelos serializados, credenciales y listas de clientes.

```bash
git status --short
git diff --check
git add .
git diff --cached --stat
git commit -m "Prepare retention copilot for portfolio"
```

Si la carpeta todavía no tiene repositorio, ejecutar primero `git init -b main`. Configura tu nombre y correo de Git si no están disponibles.

## Conectar un remoto existente

Crea un repositorio vacío en tu proveedor y utiliza su URL real:

```bash
git remote add origin <URL_DE_TU_REPOSITORIO>
git push -u origin main
```

Si `origin` ya existe, revisa `git remote -v` antes de cambiarlo. Esta preparación no crea ni publica automáticamente un repositorio remoto.

## Portafolio

Utiliza [el caso de estudio](PORTFOLIO.md), enlaza el repositorio y añade una captura de la app ya ejecutada. Presenta los resultados como simulaciones bajo supuestos; el reporte de test es la fuente de las métricas.

## Ejecutar o desplegar la demo más adelante

La instalación necesita Python 3.11, `requirements.txt`, el CSV local y un modelo generado con esas versiones. Entrena antes de lanzar `streamlit run app.py`. El dataset y el artefacto no se incluyen en Git, por lo que un despliegue desde un clon necesita provisionarlos explícitamente.

No uses una base de clientes reales en una demo pública: la descarga expone las filas recomendadas. Para una demostración pública, utiliza una base que puedas redistribuir o datos sintéticos y describe su origen. Añadir autenticación, alojamiento y aprovisionamiento queda fuera de esta versión local.
