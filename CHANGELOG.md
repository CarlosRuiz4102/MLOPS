# Changelog

## 1.0.0 (07/05/2026)

- `feat`: integración con W&B — tracking de métricas por época (MLP/CNN), tablas comparativas y artefactos del modelo
- `feat`: selección de modelos a entrenar desde `config/config.yaml` (`models: [logreg, mlp, ...]`)
- `feat`: `Dockerfile` para contenedorizar la API FastAPI en el puerto 8000
- `feat`: `config/config.yaml` con hiperparámetros centralizados y lista de modelos
- `refactor`: `extra_callbacks` en `train_mlp` y `train_cnn1d` para inyectar callbacks externos
- `chore`: añadido `wandb` y `pyyaml` a `requirements.txt`
- `chore`: `.dockerignore`, `.gitignore` y `pytest.ini` actualizados
- `docs`: `README.md` reescrito con instrucciones de entrenamiento, API y Docker
- `docs`: `CHANGELOG.md` añadido
