# ECG Arrhythmia Classification — MLOps Project

**Autores:** Carlos Ruiz, Alejandro Gomez

Clasificación de latidos ECG con el dataset MIT-BIH aplicando metodologías MLOps:
entrenamiento reproducible, seguimiento de experimentos con W&B, API REST con FastAPI
y contenedorización con Docker.

## Estructura

```text
Practica MLOPS/
├── config/
│   └── config.yaml          # Hiperparámetros del experimento
├── data/                    # mitbih_train.csv y mitbih_test.csv (no incluidos en repo)
├── img/                     # Reportes y visualizaciones
├── models/                  # Artefactos entrenados (generados por train.py)
├── notebooks/               # Exploración y resumen de resultados
├── src/
│   ├── config.py            # Constantes globales
│   ├── data.py              # Carga y particionado de datos
│   ├── features.py          # Extracción de features temporales
│   ├── evaluation.py        # Métricas y tabla comparativa
│   ├── models.py            # Definición y entrenamiento de modelos
│   ├── train.py             # Orquestación, comparación y exportación
│   └── inference_api.py     # API FastAPI para servir predicciones
├── tests/
├── .env.example             # Plantilla de variables de entorno (copiar a .env)
├── .gitignore               # .env no se incluye en el repo por seguridad
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── README.md
```

## Clases predichas (MIT-BIH)

| Clase | Descripción |
|-------|-------------|
| 0 | N — latido normal |
| 1 | S — supraventricular |
| 2 | V — ventricular |
| 3 | F — fusionado |
| 4 | Q — desconocido |

## Configuración del entorno local

El proyecto usa [uv](https://docs.astral.sh/uv/) para gestionar el entorno virtual y las dependencias definidas en `pyproject.toml`.

Las dependencias están divididas en dos grupos:
- **Producción** (`dependencies`): lo que necesita la API y el entrenamiento.
- **Dev** (`[project.optional-dependencies] dev`): herramientas de desarrollo (`pytest`, `jupyter`) que no se instalan en Docker para mantener la imagen ligera.

```bash
# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear .venv e instalar todo (producción + dev)
uv sync --extra dev

# Solo producción (equivalente a lo que hace Docker)
uv sync --no-dev
```

Colocar los ficheros de datos en `data/`:
- `data/mitbih_train.csv`
- `data/mitbih_test.csv`

### Configuración de W&B (Weights & Biases)

**IMPORTANTE:** El archivo `.env` ha sido eliminado del repositorio por razones de seguridad. No debe contener credenciales en control de versiones.

El proyecto carga automáticamente las credenciales de W&B desde un archivo `.env` local. Para configurar el acceso:

**Opción 1: Crear un archivo `.env`** (Recomendado):
```bash
# Crear archivo .env en la raíz del proyecto (local, no se sube a GitHub)
echo "WANDB_API_KEY=tu_api_key_aqui" > .env
```

**Opción 2: Usar `wandb login`** (Alternativo):
```bash
wandb login
```

El `.env` está incluido en `.gitignore` para evitar exponer credenciales accidentalmente.

## Entrenamiento

Con la configuración por defecto (`config/config.yaml`):

```bash
python src/train.py --config config/config.yaml --wandb-project ecg-classification
```

Entrenamiento rápido para pruebas (sin W&B):

```bash
python src/train.py --fast-run --no-wandb
```

Artefactos generados en `models/`:
- `random_forest_ecg.joblib`
- `mlp.keras`
- `cnn_1d.keras`
- `scaler.joblib`
- `metrics_summary.csv`

### Seguimiento de Experimentos con Weights & Biases

Durante el entrenamiento, el proyecto registra automáticamente en W&B:

**Configuración:**
- Hiperparámetros (learning rate, epochs, batch size)
- Arquitectura del modelo
- Versión del dataset
- Semilla (seed)
- Entorno de ejecución

**Métricas:**
- Pérdida de entrenamiento y validación por época
- Accuracy de validación
- Métricas finales: precisión, recall, F1-score, accuracy balanceado

**Artefactos:**
- Modelos entrenados (.joblib, .keras)
- Scaler para normalización
- Resumen de métricas en CSV

**Comparación de Experimentos:**
Accede a https://wandb.ai para:
- Comparar múltiples entrenamientos lado a lado
- Visualizar curvas de aprendizaje
- Analizar qué configuraciones dieron mejores resultados
- Reproducir cualquier experimento con sus parámetros exactos

## API de inferencia

Requiere tener los modelos entrenados en `models/` antes de lanzar la API.

```bash
uvicorn src.inference_api:app --reload
```

Documentación interactiva disponible en: `http://localhost:8000/docs`

Ejemplo de petición:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.0, 0.1, ...], "model_name": "random_forest"}'
```

Modelos disponibles: `random_forest`, `mlp`, `cnn_1d`.

### Ejemplo de petición y respuesta (modelo MLP)

Petición:
<div align="center">
  <img src="img/inference_request.jpg" alt="Ejemplo petición al modelo MLP" width="800">
</div>

Respuesta:
<div align="center">
  <img src="img/inference_response.jpg" alt="Ejemplo respuesta del modelo MLP" width="800">
</div>

## Resultados de Experimentos

<div align="center">
  <img src="img/reporte_mlops.jpg" alt="Reporte MLOps" width="800">
</div>

<div align="center">
  <img src="img/runs.jpg" alt="Runs MLOps" width="800">
</div>

Para explorar todos los experimentos en detalle, consulta el [Proyecto W&B](https://api.wandb.ai/links/practica-mlops-upm/83i5bibb).

## Docker

Construir y lanzar el contenedor (sirve la API de inferencia):

```bash
docker build -t ecg-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models ecg-api
```

La API quedará disponible en `http://localhost:8000`.

## Tests

```bash
PYTHONPATH=. uv run pytest -v
```

## CI/CD

El proyecto cuenta con un workflow de GitHub Actions (`.github/workflows/tests.yml`) que:
- Ejecuta automáticamente los tests en cada push y pull request
- Prueba en Python 3.10, 3.11 y 3.12
- Valida la calidad del código

## Enlaces

- **GitHub:** https://github.com/CarlosRuiz4102/MLOPS
- **W&B Report:** https://api.wandb.ai/links/practica-mlops-upm/83i5bibb
- **Endpoint en producción:** N/A (se ha ejecutado en local)

