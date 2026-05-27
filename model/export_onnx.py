"""
========================================================
  Exportacion del modelo Keras LSTM -> ONNX
  -------------------------------------------------------
  Dependencias:
      pip install tensorflow tf2onnx onnx onnxruntime

  Uso:
      python export_onnx.py

  Entrada:
      model.keras       -- modelo Keras entrenado
  Salida:
      huespedes.onnx    -- modelo ONNX listo para onnxruntime web / TouchDesigner

  Por qué este enfoque:
    `tf2onnx.convert.from_keras` y la ruta via SavedModel CLI fallan o
    producen ONNX roto cuando el modelo es Sequential + LSTM en Keras 3
    (los nombres de tensor cambian a "keras_tensor_N" y tf2onnx no los
    mapea). La solución robusta: reconstruir como Functional model con
    nombre explícito, envolver en tf.function con signature, y convertir
    con tf2onnx.convert.from_function que sí preserva LSTM correctamente.
========================================================
"""

import os
import tensorflow as tf
import tf2onnx
import onnx

KERAS_PATH   = "model.keras"
ONNX_PATH    = "huespedes.onnx"
TOTAL_FRAMES = 90
FEATURE_DIM  = 168
OPSET        = 17


def main():
    if not os.path.exists(KERAS_PATH):
        raise FileNotFoundError(
            f"No se encontro '{KERAS_PATH}'. Entrena primero con train_model.py"
        )

    print(f"Cargando modelo Keras desde '{KERAS_PATH}'...")
    src = tf.keras.models.load_model(KERAS_PATH)
    src.summary()

    # Reconstruir como Functional con input nombrado para que tf2onnx
    # pueda mapear los tensores de salida sin chocar con "keras_tensor_N".
    inputs = tf.keras.Input(shape=(TOTAL_FRAMES, FEATURE_DIM),
                            dtype=tf.float32, name="input")
    x = inputs
    for layer in src.layers:
        x = layer(x)
    model = tf.keras.Model(inputs=inputs, outputs=x, name="signs")

    # Wrappear en tf.function con signature explícita.
    input_signature = [
        tf.TensorSpec([None, TOTAL_FRAMES, FEATURE_DIM], tf.float32, name="input"),
    ]

    @tf.function(input_signature=input_signature)
    def serving_fn(x):
        return model(x)

    print(f"\nConvirtiendo a ONNX (opset {OPSET})...")
    onnx_model, _ = tf2onnx.convert.from_function(
        serving_fn,
        input_signature=input_signature,
        opset=OPSET,
    )
    onnx.save(onnx_model, ONNX_PATH)

    onnx.checker.check_model(ONNX_PATH)
    print(f"\nOK Modelo ONNX guardado -> {ONNX_PATH}")

    m = onnx.load(ONNX_PATH)
    print("\nEntradas:")
    for i in m.graph.input:
        dims = [d.dim_value or d.dim_param or "?" for d in i.type.tensor_type.shape.dim]
        print(f"  - {i.name}  shape={dims}  dtype=float32")
    print("Salidas:")
    for o in m.graph.output:
        dims = [d.dim_value or d.dim_param or "?" for d in o.type.tensor_type.shape.dim]
        print(f"  - {o.name}  shape={dims}  dtype=float32")

    # ── Sanity check Keras vs ONNX en muestras reales ───────────────────────
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name

        sample_paths = []
        if os.path.isdir("data"):
            for cls in sorted(os.listdir("data")):
                cls_dir = os.path.join("data", cls)
                if os.path.isdir(cls_dir):
                    for f in sorted(os.listdir(cls_dir))[:2]:
                        if f.endswith(".npy"):
                            sample_paths.append((cls, os.path.join(cls_dir, f)))

        if sample_paths:
            print("\nComparando Keras vs ONNX en muestras reales:")
            max_diff = 0.0
            mismatches = 0
            for cls, path in sample_paths:
                seq = np.load(path).astype(np.float32)[np.newaxis]
                k = src.predict(seq, verbose=0)[0]
                o = sess.run(None, {in_name: seq})[0][0]
                diff = float(np.max(np.abs(k - o)))
                max_diff = max(max_diff, diff)
                k_idx, o_idx = int(np.argmax(k)), int(np.argmax(o))
                if k_idx != o_idx:
                    mismatches += 1
                tag = "OK " if k_idx == o_idx and diff < 1e-3 else "!! "
                print(f"  {tag} clase={cls}  Keras={k_idx}({k[k_idx]*100:.0f}%)  "
                      f"ONNX={o_idx}({o[o_idx]*100:.0f}%)  max|diff|={diff:.5f}")
            verdict = ("EXCELENTE — ONNX listo para web"
                       if max_diff < 1e-3 and mismatches == 0
                       else "REVISAR — ONNX todavía no coincide con Keras")
            print(f"\n  Máxima diferencia: {max_diff:.6f}  ({verdict})")
        else:
            dummy = np.zeros((1, TOTAL_FRAMES, FEATURE_DIM), dtype=np.float32)
            out = sess.run(None, {in_name: dummy})[0]
            print(f"\nForma de salida con zeros: {out.shape}, suma probs: {out.sum():.4f}")
    except Exception as e:
        print(f"\n(Aviso) sanity check fallo: {e}")


if __name__ == "__main__":
    main()
