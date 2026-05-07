# Texto para la entrega

Autores: Carlos Ruiz Oyarzun y Alejandro Gomez

Enlace al proyecto en GitHub:

https://github.com/CarlosRuiz4102/MLOPS

Enlace al proyecto en Weights & Biases:

https://api.wandb.ai/links/practica-mlops-upm/83i5bibb

Endpoint accesible con el servicio/aplicacion:

http://localhost:8000/docs

Nota: el servicio esta preparado para ejecutarse como API dockerizada. Para levantarlo:

```bash
docker build -t ecg-api .
docker run -p 8000:8000 -v "$(pwd)/models:/app/models" ecg-api
```
