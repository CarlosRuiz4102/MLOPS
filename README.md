# PruebaClase

Estructura base de un proyecto MLOps para clasificacion de latidos ECG con el dataset MIT-BIH.

## Estructura

```text
PruebaClase/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference_api.py
│   ├── models.py
│   └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Que hay en cada carpeta

- `data/`: contiene `mitbih_train.csv` y `mitbih_test.csv`.
- `models/`: aqui se guardan los artefactos entrenados y las metricas exportadas.
- `notebooks/`: notebooks de exploracion y resumen de resultados.
- `src/`: codigo principal del proyecto, descompuesto por responsabilidades.
- `tests/`: pruebas basicas del codigo.

## Archivos principales

- `src/config.py`: constantes globales del proyecto.
- `src/data.py`: carga de datos y preparacion de particiones.
- `src/features.py`: extraccion de features temporales.
- `src/evaluation.py`: metricas y tabla comparativa.
- `src/models.py`: definicion y entrenamiento de modelos.
- `src/train.py`: orquestacion completa de la comparacion de modelos y exportacion de artefactos.
- `src/inference_api.py`: API con FastAPI para servir predicciones con varios modelos entrenados.

## Ejecucion

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Entrenar modelos y exportar artefactos:

```bash
python src/train.py
```

Esto generara archivos en `models/` como:

```text
random_forest_ecg.joblib
mlp.keras
cnn_1d.keras
scaler.joblib
metrics_summary.csv
```

Lanzar API:

```bash
uvicorn src.inference_api:app --reload
```

Ejecutar tests:

```bash
pytest
```
