# LSM · Demo web

Página de pruebas para `huespedes.onnx`. Carga el modelo en el navegador con
**ONNX Runtime Web** y los landmarks de manos/pose con **MediaPipe Tasks Web**.
Replica exactamente el preprocesamiento y los criterios de rechazo de
`sign_classifier.py` para que las predicciones sean coherentes con la versión
de Python.

## Arrancar

```bash
cd web
python serve.py          # http://localhost:8000
```

Luego abre `http://localhost:8000` en Chrome/Edge. La primera vez tarda unos
segundos en bajar los assets de MediaPipe + ONNX Runtime desde CDN.

> La cámara solo se concede en orígenes seguros. `localhost` cuenta como
> seguro, no hace falta HTTPS.

## Estructura

```
web/
  index.html        # UI
  style.css         # estilos
  app.js            # pipeline: webcam → MediaPipe → ONNX → texto
  serve.py          # server estático con MIME types correctos
  models/
    huespedes.onnx
    label_map.json
```

## Cuando reentrenes el modelo

```bash
cd model
python train_model.py
python export_onnx.py
cp huespedes.onnx label_map.json ../web/models/
```

## Filtros aplicados (espejo del Python)

1. **Filtro de manos por dueño** — descarta manos cuya muñeca esté a más de
   `0.18` (en coords normalizadas 0-1) de cualquiera de las muñecas del pose
   principal. Las manos descartadas se dibujan tenues en rojo.
2. **Rechazo `sin manos`** — si menos del 40% del buffer de 90 frames tiene
   alguna mano válida, no se ejecuta la inferencia.
3. **Rechazo `baja confianza`** — descarta si `max(prob) < 0.70`.
4. **Rechazo `red insegura`** — descarta si la diferencia entre top-1 y top-2
   es menor a `0.20` (la red duda entre dos clases).

Cualquier ajuste a estos umbrales debe hacerse a la vez en `model/sign_classifier.py`
y en `web/app.js` (constantes al inicio de cada archivo).
