"""
========================================================
  export_to_onnx.py — convierte model.keras → model.onnx
  -------------------------------------------------------
  Dependencias:
      pip install tf2onnx onnxruntime

  Uso:
      python export_to_onnx.py

  Salida:
      model.onnx               -- modelo en formato ONNX (opset 17)
      onnx_verification.txt    -- log de verificación (paridad con Keras)

  El modelo ONNX es el que consume el frontend Next.js vía
  onnxruntime-web. El input/output esperados son:
      input  : float32 [batch, 90, 168]
      output : float32 [batch, 18]   (probabilidades softmax)
========================================================
"""

import os
import sys
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import MODEL_PATH, ONNX_PATH, TOTAL_FRAMES, FEATURE_DIM

OPSET = 17
TOL_RTOL = 1e-4
TOL_ATOL = 1e-4
LOG_PATH = "onnx_verification.txt"


def main():
    import shutil
    import tempfile
    import subprocess
    import keras
    import tensorflow as tf
    import onnxruntime as ort

    print(f"Cargando {MODEL_PATH}...")
    keras_model = keras.models.load_model(MODEL_PATH)

    # Keras 3 + tf2onnx 1.16 no son compatibles directamente
    # (tf2onnx espera model.output_names que Keras 3 no expone). Vía indirecta:
    # exportar a SavedModel y convertir la SavedModel con tf2onnx CLI.
    tmp_saved = tempfile.mkdtemp(prefix="keras3_savedmodel_")
    try:
        print(f"Exportando SavedModel intermedio → {tmp_saved}")
        keras_model.export(tmp_saved)

        print(f"Convirtiendo a ONNX opset {OPSET}...")
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", tmp_saved,
            "--output", ONNX_PATH,
            "--opset", str(OPSET),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("tf2onnx conversion failed")
    finally:
        shutil.rmtree(tmp_saved, ignore_errors=True)

    print(f"✓ Modelo ONNX guardado → {ONNX_PATH}")
    print(f"  Tamaño: {os.path.getsize(ONNX_PATH)/1024:.1f} KB")

    # ── Verificación de paridad ──────────────────────────────────────────────
    print("\nVerificando paridad Keras ↔ ONNX...")
    np.random.seed(42)
    n_samples = 8
    dummy = np.random.randn(n_samples, TOTAL_FRAMES, FEATURE_DIM).astype(np.float32)

    pred_keras = keras_model.predict(dummy, verbose=0)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    pred_onnx = sess.run(None, {in_name: dummy})[0]

    max_diff = float(np.max(np.abs(pred_keras - pred_onnx)))
    is_close = np.allclose(pred_keras, pred_onnx, rtol=TOL_RTOL, atol=TOL_ATOL)
    argmax_match = float(np.mean(pred_keras.argmax(1) == pred_onnx.argmax(1)))

    lines = [
        f"Modelo Keras  : {MODEL_PATH}",
        f"Modelo ONNX   : {ONNX_PATH}",
        f"Opset         : {OPSET}",
        f"Muestras dummy: {n_samples}",
        f"Max |Δ|       : {max_diff:.2e}",
        f"Tolerancia    : rtol={TOL_RTOL}, atol={TOL_ATOL}",
        f"Paridad numérica  : {'OK' if is_close else 'FALLA'}",
        f"Paridad argmax    : {argmax_match*100:.1f}% ({int(argmax_match*n_samples)}/{n_samples})",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n✓ Log de verificación → {LOG_PATH}")

    if not is_close or argmax_match < 1.0:
        print("\n[!] La paridad no es perfecta. Revisar tolerancias o conversión.")
        sys.exit(1)


if __name__ == "__main__":
    main()
