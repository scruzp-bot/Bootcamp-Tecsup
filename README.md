# Clasificación de frijol seco con Dry Bean (UCI)

Proyecto final del curso Ciencia de Datos G8. Lleva un modelo de Machine Learning de punta a punta:

```
datos → entrenamiento → API → interfaz web → contenedor → CI/CD y nube
```

La estructura sigue el proyecto de referencia del curso (`iris_mlops`), adaptado a un problema distinto: **identificar la variedad de un grano de frijol seco a partir de 16 medidas de forma**.

## Demo en línea

| Servicio | URL |
|---|---|
| Interfaz web (Gradio) | https://drybean-ui.onrender.com |
| API (FastAPI) | https://drybean-api.onrender.com |
| Documentación de la API | https://drybean-api.onrender.com/docs |

Ambos servicios corren en el plan gratuito de Render y se duermen tras 15 minutos sin uso. Si llevan un rato inactivos, la primera petición puede tardar alrededor de un minuto.

## El problema

- **Qué se predice:** la variedad del grano: Barbunya, Bombay, Cali, Dermason, Horoz, Seker o Sira.
- **Para quién:** control de calidad y clasificación automática de granos en acopio y selección.
- **Dataset:** [Dry Bean Dataset (UCI, id 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). 13 611 granos, 16 variables (12 dimensiones y 4 factores de forma) extraídas de imágenes, sin valores nulos. Licencia CC BY 4.0, DOI 10.24432/C50S4B.
- **Referencia:** Koklu, M. y Özkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques*. Computers and Electronics in Agriculture.

## Modelo y evaluación

- **Modelo:** regresión logística multinomial con `StandardScaler`.
- **Métrica principal:** F1 macro, porque las clases están desbalanceadas (Bombay tiene unos 520 granos y Dermason cerca de 3 500).
- **Puerta de calidad:** si el F1 macro es menor a 0.88, el modelo no se guarda y el entrenamiento falla.
- **Resultado (CI, semilla 42, 20 % de prueba):** F1 macro **0.9335** y accuracy **0.9214**.
- La ficha del modelo queda en `models/metrics.json` (versión, fecha, parámetros y métricas).

## Estructura

```
src/drybean_mlops/
  config.py    rutas, semilla, parámetros y rangos válidos
  data.py      descarga y validación (nulos, tipos, rangos, clases)
  train.py     entrenamiento, puerta de calidad y ficha del modelo
  predict.py   carga del modelo y predicción
  api.py       FastAPI: /health, /model-info y /predict
app/gradio_app.py    interfaz que consume la API por HTTP
tests/               pruebas de datos, modelo y API
Dockerfile · compose.yaml · render.yaml
.github/workflows/   ci.yml (y cd.yml)
```

## Ejecutar desde cero (uv)

```bash
uv sync                # instala dependencias
uv run datos           # descarga y valida el dataset
uv run entrenar        # entrena y genera models/metrics.json
uv run pytest -v       # pruebas
uv run api             # API en http://localhost:8000/docs
uv run python app/gradio_app.py   # interfaz en http://localhost:7860
```

## Docker

```bash
docker compose up --build    # API en :8000 e interfaz en :7860
```

## CI/CD y despliegue

- `ci.yml`: instala con `uv`, descarga los datos, entrena con puerta de calidad, ejecuta pytest y guarda el modelo como artefacto.
- **Render:** `render.yaml` define dos servicios (API e interfaz) que se despliegan solos con cada push a `main`.

## Limitaciones y mejoras

Los modelos lineales confunden variedades de forma parecida (Dermason, Sira y Seker). Un MLP pequeño o gradient boosting podría mejorar el F1. Pendientes: MLflow, monitoreo y detección de data drift.

