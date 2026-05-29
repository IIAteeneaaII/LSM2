# Integración del front + modelo LSM (handoff)

Doc para conectar este front (`app.js`) con `models/huespedes.onnx` en otro proyecto.
El punto crítico: **el front replica byte-a-byte el preprocesamiento del Python**
(`model/sign_classifier.py`). Si una sola normalización difiere, MediaPipe entrega
features distintos a los del entrenamiento y el modelo predice basura aunque el ONNX
esté bien. Esta es la lista de contratos que deben mantenerse idénticos.

## Cinco modelos en el mismo front

El front carga **cinco modelos** y se cambia entre ellos con el selector (`#model-select`).
Comparten TODO el pipeline de captura (MediaPipe, espejado, crop 4:3, filtros de
rechazo) pero **difieren en el vector de features** — esto es lo más importante:

| modo      | onnx                | label_map                  | features | clases | qué usa                          |
|-----------|---------------------|----------------------------|----------|--------|----------------------------------|
| huespedes | `huespedes.onnx`    | `label_map.json`           | **168**  | 5  | pose(42) + mano-izq(63) + mano-der(63), con filtro de dueño |
| numeros   | `numeros.onnx`      | `numeros_label_map.json`   | **63**   | 10 | SOLO mano izquierda; sin pose ni filtro de dueño |
| meses     | `meses.onnx`        | `meses_label_map.json`     | **63**   | 12 | SOLO mano izquierda; sin pose ni filtro de dueño |
| dias      | `dias.onnx`         | `dias_label_map.json`      | **63**   | 7  | SOLO mano izquierda; sin pose ni filtro de dueño |
| genero    | `genero.onnx`       | `genero_label_map.json`    | **63**   | 2  | SOLO mano izquierda; sin pose ni filtro de dueño |

`numeros`, `meses`, `dias` y `genero` comparten exactamente la misma extracción de
features (`extractLeftHand`) y el mismo modo de dibujo/pose-off; solo cambian el `.onnx`
y su `label_map`. La generalización en `app.js` es: **todo modo que no sea `huespedes`
usa 63 features (solo mano izquierda) y NO corre pose**. `genero` (femenino/masculino)
solo marca el detectado y emite `{ genero }`; no entra en la reservación.

### Panel de reservación (combina numeros + meses + dias)

El panel `#card-reserva` (visible fuera de huespedes) acumula en el objeto `reserva`:
- **día** del mes → se firma en modo `numeros` enfocando el campo "Día de llegada".
- **mes** → modo `meses` (vía `setMonth`, mapea a número 1-12 con `MONTH_NUM`).
- **día de la semana** → modo `dias` (vía `setDiaSemana`).

`recomputeReserva()` arma la fecha resolviendo el año a la próxima ocurrencia ≥ hoy,
valida que caiga en la ventana **[hoy, hoy+30]** (`RESERVA_MAX_DIAS`) y cruza el día de
la semana firmado contra el real (`JS_WEEKDAY`). Todo es **solo aviso** (array `avisos`),
nunca bloquea. Usa `new Date()` del navegador, así que la ventana es relativa al día real.

`buildKeypoints()` en `app.js` arma el vector según el modo activo; `loadModel()`
cachea cada sesión ONNX y, al cambiar, **vacía el buffer** porque el feature dim
cambia. El front deriva el feature dim del propio snapshot al inferir, así que la
forma del tensor `[1, 90, featureDim]` siempre cuadra con la sesión.

## Contrato de entrada/salida del ONNX

- **Input**: nombre dinámico (`session.inputNames[0]`), shape `[1, 90, featureDim]`, `float32`.
  - 90 = frames (`TOTAL_FRAMES`); featureDim = 168 (huespedes) o 63 (numeros).
- **Output**: softmax `[1, N]` (`session.outputNames[0]`). huespedes N=5 (`1,2,3,adulto,nino`),
  numeros N=10 (`0..9`). El índice del argmax se mapea con el `label_map` del modo.

> El modelo de números se entrena con `model/train_model.py` (FEATURE_DIM=63, solo
> mano izquierda) y se exporta con `python export_onnx.py numeros.onnx`. El de
> huéspedes usa 168 features (pose + 2 manos). Son arquitecturas/entradas distintas,
> NO intercambiables: cada uno requiere su propia extracción de features en el front.

## Layout de los 168 features (orden EXACTO)

Se concatenan en este orden — ver `app.js` (`keypoints.set`) y Python (`np.concatenate`):

| offset | tamaño | qué                         |
|--------|--------|-----------------------------|
| 0      | 42     | pose superior (14 pts × 3)  |
| 42     | 63     | mano izquierda (21 pts × 3) |
| 105    | 63     | mano derecha (21 pts × 3)   |

Si una mano no está presente → sus 63 valores quedan en 0.

## Normalizaciones que DEBEN coincidir

**Pose** (`extract_pose_upper` en Python / `extractPoseUpper` en JS):
1. Tomar landmarks de pose con índices 11..24 (14 puntos), cada uno `[x,y,z]`.
2. Restar `hip_mid = (LM23 + LM24) / 2` a todos (centra en la cadera).
3. Dividir todo por `shoulder_dist = dist(LM11, LM12)` (escala por ancho de hombros).
   - Ojo: en el array local, LM23/LM24 son los índices 12/13 y LM11/LM12 son 0/1.

**Manos** (`extract_hands` / `extractHands`):
1. Para cada mano, 21 puntos `[x,y,z]`.
2. Restar la muñeca `pts[0]` a todos (centra en la muñeca).
3. Dividir por `dist(muñeca → pts[9])` (escala por nudillo medio).
4. Asignar a izquierda/derecha según `handedness` de MediaPipe (`"Left"`/`"Right"`).

**Filtro de manos por dueño** (descarta manos de gente detrás):
- Una mano es válida solo si su muñeca está a menos de `HAND_OWNER_MAX_DIST`
  (coords normalizadas 0-1) de la muñeca de pose correspondiente (LM15 izq, LM16 der).
- ⚠️ **Divergencia conocida**: Python usa `0.18`, `app.js` usa `0.30` (se subió porque
  manos válidas con el brazo extendido quedaban fuera). Es intencional, pero tenlo
  presente si afinas este umbral — cámbialo en ambos lados si quieres paridad estricta.

## Detalles del front que son fáciles de romper

Estos NO están en el Python como código pero son la traducción de lo que OpenCV hacía
implícito. Si los omites, las predicciones fallan:

1. **Espejado (mirror)**: Python hace `cv2.flip(frame, 1)` antes de detectar. El front
   lo replica pintando el video con `detectCtx.scale(-1, 1)` en un canvas oculto.
   Sin esto, la asignación Left/Right de las manos se invierte vs el entrenamiento y
   el modelo no detecta nada.
2. **Crop 4:3 antes de reducir a 320×240**: si la cámara da 16:9 y estiras directo a
   320×240, la persona se comprime horizontalmente y los features normalizados ya no
   coinciden. Hay que recortar al centro a 4:3 primero.
3. **El overlay NO se espeja**: los landmarks ya vienen en coords del frame espejado,
   así que el canvas de dibujo se deja sin `scaleX(-1)` (el CSS sí espeja el `<video>`).
4. **Timestamps estrictamente crecientes** para `detectForVideo` — `performance.now()`
   puede repetirse entre frames, por eso se usa `Math.max(round(now), frameCount+1)`.

## Cadencia de inferencia y filtros de rechazo

Idénticos en ambos (constantes al inicio de `app.js` / `sign_classifier.py`):

- Buffer circular de `TOTAL_FRAMES = 90`. Se infiere cada `INFERENCE_EVERY = 90` frames
  (≈ una predicción cada 3 s a 30 fps).
- **`sin manos`**: si <40% (`MIN_HAND_RATIO`) del buffer tuvo alguna mano válida → no se infiere.
- **`baja confianza`**: descarta si `max(prob) < 0.70` (`MIN_CONFIDENCE`).
- **`red insegura`**: descarta si `top1 - top2 < 0.20` (`MIN_MARGIN`).

Diferencia de implementación (no de resultado): Python infiere síncrono; el front corre
ONNX en background (`inferenceBusy`) para no bloquear la captura de frames.

## Cómo está conectado el formulario (gramática número → tipo)

Cuando se acepta una predicción y **no** hay un `<input>` enfocado, la seña alimenta el
constructor de huéspedes en vez de escribirse como texto:
- `1/2/3` → deja un número *pendiente*.
- `adulto/nino` → confirma un grupo `(cantidad, tipo)`; sin número pendiente da error.
- Al enviar se valida ≥1 grupo y se emite `{ nombre, grupos[], adultos, ninos, total }`.

Ver `handleGuestSign`, `renderGuests`, `submitGuests` en `app.js`.

## Dependencias y arranque

No hay npm: MediaPipe Tasks y ONNX Runtime Web se cargan desde CDN dentro de `app.js`.
Sírvelo con `python serve.py` (da los MIME correctos para `.onnx`/`.wasm` y los headers
COOP/COEP). Abrir `index.html` con `file://` NO funciona (cámara + ONNX requieren origen
seguro; `localhost` cuenta como seguro).

## Si reentrenas el modelo

Reemplaza **a la vez** `models/huespedes.onnx` y `models/label_map.json` (el server los
sirve con `Cache-Control: no-store`). Si dejas un `label_map` viejo con un ONNX nuevo,
el código revienta con labels `undefined`. El front no asume el número de clases:
construye las filas de probabilidad desde `label_map`, así que añadir/quitar señas
funciona sin tocar `app.js` — salvo la gramática del formulario, que sí asume
`1/2/3` + `adulto/nino`.
