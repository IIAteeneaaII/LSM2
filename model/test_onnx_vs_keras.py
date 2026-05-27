"""
Compara las predicciones del modelo Keras vs el ONNX exportado con las MISMAS
muestras del dataset. Si dan resultados muy distintos, la conversión a ONNX
rompió algo; si dan resultados casi iguales, el bug está fuera del modelo.
"""
import os
import json
import numpy as np
import tensorflow as tf
import onnxruntime as ort

DATA_DIR  = "data"
KERAS     = "model.keras"
ONNX_PATH = "huespedes.onnx"
LABEL_MAP = "label_map.json"

with open(LABEL_MAP, "r", encoding="utf-8") as f:
    label_map = {int(k): v for k, v in json.load(f).items()}
print(f"Clases: {label_map}\n")

keras_model = tf.keras.models.load_model(KERAS)
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_in = sess.get_inputs()[0].name

max_diff = 0.0
mismatches = 0
total = 0

for class_name in sorted(os.listdir(DATA_DIR)):
    cls_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(cls_dir):
        continue
    files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".npy"))
    # Probar 3 muestras por clase
    for fname in files[:3]:
        seq = np.load(os.path.join(cls_dir, fname)).astype(np.float32)
        x = seq[np.newaxis]  # (1, 90, 168)

        keras_probs = keras_model.predict(x, verbose=0)[0]
        onnx_probs  = sess.run(None, {onnx_in: x})[0][0]

        diff = float(np.max(np.abs(keras_probs - onnx_probs)))
        max_diff = max(max_diff, diff)

        k_idx = int(np.argmax(keras_probs))
        o_idx = int(np.argmax(onnx_probs))

        match = "OK " if k_idx == o_idx else "!! "
        if k_idx != o_idx:
            mismatches += 1
        total += 1

        print(f"  {match} clase={class_name}  {fname}  "
              f"Keras->{label_map[k_idx]}({keras_probs[k_idx]*100:.0f}%)  "
              f"ONNX->{label_map[o_idx]}({onnx_probs[o_idx]*100:.0f}%)  "
              f"max|diff|={diff:.5f}")

print(f"\nMáxima diferencia absoluta entre probs: {max_diff:.6f}")
print(f"Predicciones distintas: {mismatches}/{total}")
if max_diff < 1e-4:
    print("==> ONNX y Keras son prácticamente idénticos. El bug NO está en el ONNX.")
elif mismatches == 0:
    print("==> Predicciones top-1 coinciden, pero las probs difieren un poco.")
else:
    print("==> El ONNX da predicciones distintas a Keras. La conversión está rota.")
