# Clasificación de variedades de frijol seco con Dry Bean (UCI)

Proyecto final del curso Data Science G8 (TECSUP). Lleva un modelo de Machine Learning de punta a punta:

```
datos → entrenamiento → API → interfaz web → contenedor → CI/CD y nube
```

La estructura sigue el proyecto de referencia del curso ([iris_mlops](https://github.com/marcell-llerena/iris_mlops)), adaptado a un problema distinto: identificar la variedad de un grano de frijol seco a partir de medidas de forma.

## Demo en línea

| Servicio | URL |
|---|---|
| Interfaz web (Gradio) | https://drybean-ui.onrender.com |
| API (FastAPI) | https://drybean-api.onrender.com |
| Documentación de la API | https://drybean-api.onrender.com/docs |

Ambos servicios corren en el plan gratuito de Render y se duermen tras 15 minutos sin uso. Si llevan un rato inactivos, la primera petición puede tardar alrededor de un minuto.

## El problema

Un acopiador o una planta de selección recibe lotes con varias variedades de frijol mezcladas. Clasificarlas a mano es lento y depende del criterio de cada persona. El **responsable de control de calidad** necesita saber de qué variedad es cada grano para separar el lote y verificar su pureza.

El modelo recibe 16 medidas de forma del grano, obtenidas por visión computacional, y predice su variedad (`Class`) con la probabilidad de cada una.

## El dataset

[Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) (UCI, id 602). Contiene 13 611 granos de 7 variedades registradas, fotografiados con una cámara de alta resolución. Un sistema de visión segmentó las imágenes y extrajo 16 variables: 12 dimensiones y 4 factores de forma. No tiene valores nulos. Licencia CC BY 4.0, DOI 10.24432/C50S4B.

El dataset se descarga desde UCI con `uv run datos` y se guarda en `data/dry_bean.csv`. Esa carpeta no se sube a git.

| Variedad | Granos | Proporción |
|---|---|---|
| Dermason | 3 546 | 26.1 % |
| Sira | 2 636 | 19.4 % |
| Seker | 2 027 | 14.9 % |
| Horoz | 1 928 | 14.2 % |
| Cali | 1 630 | 12.0 % |
| Barbunya | 1 322 | 9.7 % |
| Bombay | 522 | 3.8 % |

Variables de entrada:

| Variable | Descripción |
|---|---|
| `Area` | Píxeles dentro del borde del grano |
| `Perimeter` | Longitud del borde |
| `MajorAxisLength` | Largo del eje mayor |
| `MinorAxisLength` | Largo del eje menor, perpendicular al mayor |
| `AspectRation` | Relación entre eje mayor y eje menor |
| `Eccentricity` | Excentricidad de la elipse con los mismos momentos que el grano |
| `ConvexArea` | Píxeles del menor polígono convexo que contiene al grano |
| `EquivDiameter` | Diámetro de un círculo con la misma área |
| `Extent` | Razón entre el área del grano y la de su caja envolvente |
| `Solidity` | Razón entre el área del grano y la de su envolvente convexa |
| `roundness` | 4πA / P², con A el área y P el perímetro |
| `Compactness` | Diámetro equivalente dividido por el eje mayor |
| `ShapeFactor1` a `ShapeFactor4` | Factores de forma |

El archivo original escribe `AspectRation` y `roundness` con esos nombres; el código los conserva tal cual y unifica mayúsculas y minúsculas al cargar los datos.

## Por qué F1 macro y no solo accuracy

Las clases están desbalanceadas: Dermason tiene casi siete veces más granos que Bombay. La accuracy puede verse bien aunque el modelo falle en la variedad minoritaria. Por eso la métrica principal es el **F1 macro**: calcula el F1 de cada variedad y las promedia con el mismo peso, así que Bombay cuenta tanto como Dermason. La accuracy aparece solo como dato informativo.

## Resultados del modelo

Regresión logística multinomial con `StandardScaler` (`C=1.0`, `max_iter=2000`), con partición estratificada 80/20 y `random_state=42`. Las métricas son sobre el conjunto de prueba (2 723 granos):

| Métrica | Valor | Mínimo exigido |
|---|---|---|
| F1 macro | **0.9335** | 0.88 |
| Accuracy (informativa) | 0.9214 | – |

Resultado obtenido en la ejecución del CI, con el dataset descargado de UCI.

### Puerta de calidad

`train.py` compara el F1 macro con `MIN_F1_MACRO = 0.88`, definido en `config.py`, unos 0.05 por debajo del resultado obtenido. Si el modelo no lo alcanza, el entrenamiento lanza `QualityGateError` y **no guarda el modelo**. En Docker y en Render el modelo se entrena durante la construcción, así que un modelo malo tampoco llega a desplegarse.

Cada entrenamiento exitoso guarda `models/model.joblib` y `models/metrics.json` (versión, fecha, semilla, parámetros, variables, clases, tamaños de train y test, y métricas).

## Arquitectura

```
                 UCI Dry Bean (zip → xlsx)
                          │
                          ▼
   data.py ──── descarga, normaliza columnas y valida
                          │
                          ▼
   train.py ─── StandardScaler + LogisticRegression
                │  puerta de calidad (F1 macro)
                ▼
        models/model.joblib + models/metrics.json
                          │
                          ▼
   predict.py ─ carga el modelo una sola vez (caché)
                          │
                          ▼
   api.py (FastAPI :8000)   GET /health   GET /model-info   POST /predict
                          ▲
                          │ HTTP (API_URL)
   app/gradio_app.py (Gradio :7860) ── responsable de control de calidad

   GitHub push ─► CI (datos → entrenar → pytest) ─► CD (deploy hooks) ─► Render
```

```
Bootcamp-Tecsup/
├── .github/workflows/      ci.yml y cd.yml
├── app/gradio_app.py       interfaz web (consume la API)
├── models/                 se genera al entrenar
├── src/drybean_mlops/
│   ├── config.py           rutas, semilla, variables, rangos, parámetros y umbral
│   ├── data.py             descarga, validación y normalización
│   ├── train.py            entrenamiento, métricas y puerta de calidad
│   ├── predict.py          predicción con el modelo cacheado
│   └── api.py              API FastAPI
├── tests/                  pruebas de datos, modelo y API
├── Dockerfile, compose.yaml, .dockerignore, render.yaml
└── pyproject.toml          (uv.lock se genera con uv sync)
```

## Cómo ejecutarlo desde cero

Requisitos: git y conexión a internet. uv se encarga de instalar Python si hace falta.

**1. Instalar uv**

Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clonar el repositorio e instalar las dependencias**

```bash
git clone https://github.com/scruzp-bot/Bootcamp-Tecsup.git
```

```bash
cd Bootcamp-Tecsup
```

```bash
uv sync
```

**3. Descargar y validar los datos**

```bash
uv run datos
```

Imprime el número de registros y el conteo por variedad. Si el archivo tiene nulos, valores fuera de rango, columnas faltantes o una variedad desconocida, se detiene con un error.

**4. Entrenar el modelo**

```bash
uv run entrenar
```

Imprime el F1 macro y la accuracy, y guarda el modelo y la ficha en `models/`. Si no pasa la puerta de calidad, termina con error y no guarda nada.

**5. Levantar la API** (deja esta terminal abierta)

```bash
uv run api
```

La documentación interactiva queda en http://localhost:8000/docs, con un ejemplo ya cargado en `/predict`.

**6. Levantar la interfaz Gradio** (en otra terminal)

```bash
uv run python app/gradio_app.py
```

Se abre en http://localhost:7860. Por defecto busca la API en `http://localhost:8000`; se puede cambiar con la variable de entorno `API_URL`.

**7. Correr las pruebas**

```bash
uv run pytest -v
```

**8. Todo con Docker** (requiere Docker Desktop)

```bash
docker compose up --build
```

Levanta la API en el puerto 8000 y Gradio en el 7860. El modelo se entrena durante la construcción de la imagen.

## Endpoints de la API

| Método | Ruta | Respuesta |
|---|---|---|
| GET | `/health` | Estado del servicio: `{"status": "ok"}` |
| GET | `/model-info` | Contenido de `metrics.json`. 503 si no hay modelo entrenado |
| POST | `/predict` | Variedad predicha, confianza y probabilidad de cada variedad. 422 si la entrada no es válida, 503 si no hay modelo |

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Area\":28395,\"Perimeter\":610.291,\"MajorAxisLength\":208.178,\"MinorAxisLength\":173.889,\"AspectRation\":1.197,\"Eccentricity\":0.55,\"ConvexArea\":28715,\"EquivDiameter\":190.141,\"Extent\":0.764,\"Solidity\":0.989,\"roundness\":0.958,\"Compactness\":0.913,\"ShapeFactor1\":0.00733,\"ShapeFactor2\":0.00314,\"ShapeFactor3\":0.834,\"ShapeFactor4\":0.999}"
```

Formato de la respuesta (los valores dependen del modelo entrenado):

```json
{
  "variety": "<variedad>",
  "confidence": 0.0,
  "probabilities": {"BARBUNYA": 0.0, "BOMBAY": 0.0, "CALI": 0.0, "DERMASON": 0.0, "HOROZ": 0.0, "SEKER": 0.0, "SIRA": 0.0}
}
```

- `variety`: la variedad con mayor probabilidad.
- `confidence`: la probabilidad de esa variedad.

La API valida cada campo con Pydantic: los 16 campos son obligatorios, no se aceptan campos extra y cada variable debe estar dentro de los rangos de `config.py`. Si falta un campo o un valor está fuera de rango, la respuesta es **422**. Si no hay modelo entrenado, es **503**.

## Pruebas

15 pruebas en tres capas:

- **Datos** (`test_data.py`): datos válidos, y rechazo de columnas faltantes, nulos, valores fuera de rango y variedades desconocidas; unificación de nombres de columnas.
- **Modelo** (`test_model.py`): la ficha `metrics.json` trae sus campos y supera la puerta de calidad, `predict()` devuelve variedad, confianza y probabilidades que suman 1, y la puerta de calidad bloquea un modelo que no alcanza el mínimo.
- **API** (`test_api.py`): `/health` y `/model-info` responden 200, `/predict` válido da 200, un valor fuera de rango o un campo faltante da 422 y la ausencia del modelo da 503.

Las pruebas usan datos sintéticos, así que no dependen de la descarga desde UCI.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): en cada push a `main` instala dependencias con `uv sync`, descarga los datos, entrena (con puerta de calidad), corre `pytest -v` y sube la carpeta `models/` como artefacto.
- **CD** (`.github/workflows/cd.yml`): se ejecuta cuando termina el CI y solo si terminó en verde. Llama a los deploy hooks de Render, guardados como secretos de GitHub (`RENDER_DEPLOY_API` y `RENDER_DEPLOY_UI`). Ninguna clave está escrita en el código.

## Despliegue en Render

Estos pasos se hacen a mano una sola vez.

1. **Verificar el CI.** En la pestaña *Actions* del repositorio, el CI debe terminar en verde.
2. **Crear los servicios con el Blueprint.** En Render: *New → Blueprint*, conectar el repositorio y aplicar. `render.yaml` crea `drybean-api` y `drybean-ui` con Auto-Deploy desactivado, para que solo despliegue el CD después de un CI en verde.
3. **Probar la API.** Abrir `https://drybean-api.onrender.com/docs`. Si Render asignó otra dirección, actualizar `API_URL` en el servicio de la interfaz y las URLs de este README.
4. **Copiar los deploy hooks.** En cada servicio: *Settings → Deploy Hook*.
5. **Guardarlos como secretos en GitHub.** *Settings → Secrets and variables → Actions → New repository secret*:
   - `RENDER_DEPLOY_API` = deploy hook de la API
   - `RENDER_DEPLOY_UI` = deploy hook de la interfaz
6. **Probar la cadena completa.** Hacer un push a `main`: primero corre el CI y, si pasa, el CD dispara los dos despliegues.

En el plan gratuito de Render los servicios se duermen tras un rato sin uso. La primera petición puede tardar cerca de un minuto.

## Uso de herramientas de IA

Este proyecto se desarrolló con Claude (Anthropic) como asistente de programación: generó el código base, las pruebas, los archivos de Docker, el CI/CD y la documentación a partir del problema y el dataset elegidos. El autor armó el repositorio en GitHub, ajustó la estructura de carpetas, ejecutó el CI y verificó los resultados del entrenamiento.

## Limitaciones

- **Medidas obtenidas de imágenes en condiciones controladas.** Las variables salen de fotografías tomadas con una cámara y una configuración fijas. El modelo puede degradarse si las imágenes vienen de otra cámara, otra iluminación o granos de otra cosecha.
- **Variedades de forma parecida.** Los modelos lineales tienden a confundir variedades similares, como Dermason, Sira y Seker. El F1 global no muestra qué pares se confunden; para eso hace falta la matriz de confusión.
- **Solo siete variedades.** El modelo siempre responde con una de las siete, incluso si recibe un grano de una variedad que no conoce.
- **Dependencia de UCI.** El CI y el despliegue descargan el dataset en cada ejecución; si el sitio de UCI no responde, fallan.
- **Rangos de validación amplios.** Los rangos de `config.py` son holgados a propósito para no rechazar granos válidos, así que no detectan medidas poco realistas dentro de ellos.

## Mejoras futuras

- Probar un MLP pequeño o gradient boosting y comparar su F1 macro con el de la regresión logística.
- Guardar la matriz de confusión y el F1 por variedad en `metrics.json`.
- **MLflow** para registrar cada experimento (parámetros, métricas y modelo) y comparar versiones.
- **Monitoreo de drift:** comparar la distribución de las peticiones a la API con la de entrenamiento y avisar cuando se alejen.
- Incluir una copia del dataset en el repositorio para que el CI y Docker no dependan de UCI.

