# 🫘 Dry Bean MLOps

Clasificación de **7 variedades de frijol seco** (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) a partir de **16 medidas de forma** extraídas de imágenes. Proyecto final de Data Science G8: datos → entrenamiento → API → interfaz → Docker → CI/CD.

- **Problema:** un clasificador multiclase que identifica la variedad de un grano; útil para control de calidad y clasificación automática en acopio.
- **Dataset:** [UCI Dry Bean (id 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) — 13 611 registros, 16 variables, sin nulos. CC BY 4.0. DOI 10.24432/C50S4B. Ref.: Koklu & Özkan (2020), *Computers and Electronics in Agriculture*.
- **Modelo:** Regresión logística multinomial + `StandardScaler`.
- **Métrica:** **F1 macro** (las clases están desbalanceadas; Bombay tiene ~522 vs. Dermason ~3 546). Puerta de calidad: F1 macro ≥ 0.88, de lo contrario no se guarda el modelo.

## Estructura
```
src/drybean_mlops/
  config.py   rutas, semilla, parámetros, rangos
  data.py     descarga y validación (nulos, tipos, rangos, clases)
  train.py    entrenamiento + puerta de calidad + models/metrics.json
  predict.py  carga del modelo y predicción
  api.py      FastAPI: /health, /model-info, /predict
app/gradio_app.py   interfaz que consume la API
tests/              pytest de datos, modelo y API
Dockerfile · compose.yaml · .github/workflows/{ci,cd}.yml
```

## Ejecutar desde cero (uv)
```bash
uv sync                 # instala dependencias fijadas en uv.lock
uv run datos            # descarga y valida el dataset
uv run entrenar         # entrena y genera models/metrics.json
uv run pytest -v        # pruebas
uv run api              # API en http://localhost:8000/docs
API_URL=http://localhost:8000 uv run python app/gradio_app.py   # UI en :7860
```

## Docker
```bash
docker compose up --build   # API :8000 + interfaz :7860
```

## CI/CD
`ci.yml` ejecuta `uv sync --locked`, datos, entrenamiento y pytest. `cd.yml` dispara el deploy en Render solo si CI pasa; el hook va en el secreto `RENDER_DEPLOY_HOOK`.

## Limitaciones y mejoras
Los modelos lineales confunden Dermason/Sira/Seker (formas similares); un MLP pequeño o gradient boosting podría mejorar. Pendientes: MLflow, monitoreo y detección de data drift.
