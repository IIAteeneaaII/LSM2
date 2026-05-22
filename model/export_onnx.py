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
      model.onnx        -- modelo ONNX listo para TouchDesigner / runtimes ONNX
========================================================
"""

import os
import sys
import shutil
import tempfile
import subprocess
import tensorflow as tf
import onnx

KERAS_PATH   = "model.keras"
ONNX_PATH    = "huespedes.onnx"
TOTAL_FRAMES = 90
FEATURE_DIM  = 168
OPSET        = 17  # compatible con onnxruntime moderno y TouchDesigner


def main():
    if not os.path.exists(KERAS_PATH):
        raise FileNotFoundError(
            f"No se encontro '{KERAS_PATH}'. Entrena primero con train_model.py"
        )

    print(f"Cargando modelo Keras desde '{KERAS_PATH}'...")
    model = tf.keras.models.load_model(KERAS_PATH)
    model.summary()

    # 1) Exportar a SavedModel (API oficial de Keras 3)
    tmpdir = tempfile.mkdtemp(prefix="lsm_savedmodel_")
    try:
        print(f"\nExportando SavedModel intermedio a '{tmpdir}'...")
        # Keras 3: model.export() produce un SavedModel listo para servir
        model.export(tmpdir)

        # 2) Convertir SavedModel -> ONNX usando el CLI de tf2onnx
        print(f"\nConvirtiendo SavedModel -> ONNX (opset {OPSET})...")
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", tmpdir,
            "--output", ONNX_PATH,
            "--opset", str(OPSET),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError("tf2onnx fallo al convertir el SavedModel")
        print(result.stdout.strip().splitlines()[-1] if result.stdout else "")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # 3) Verificacion + info del grafo
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
        print(f"  - {o.name}  shape={dims}  dtype=float32  (softmax)")

    # 4) Sanity check con onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        dummy = np.zeros((1, TOTAL_FRAMES, FEATURE_DIM), dtype=np.float32)
        out = sess.run(None, {in_name: dummy})[0]
        print(f"\nInferencia de prueba OK. Shape salida: {out.shape}, suma probs: {out.sum():.4f}")
    except Exception as e:
        print(f"\n(Aviso) sanity check con onnxruntime fallo: {e}")


if __name__ == "__main__":
    main()
