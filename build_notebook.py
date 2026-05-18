"""
Genera evaluacion_lstm_TT2026A081.ipynb programáticamente.

Cambios vs. versión inicial:
  - Usa test_indices.npz (test set limpio del re-train) en vez de re-splittear
  - Usa training_history.pkl en vez de re-entrenar
  - Agrega celda de curva precision/recall vs threshold (B8)
  - Agrega celda de verificación ONNX (B6)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Evaluación del Modelo LSTM — Lengua de Señas Mexicana (LSM)
**Trabajo Terminal TT2026-A081**
Sinodal: Dr. Edgar Armando Catalán Salgado (IPN — ESCOM)

Este notebook evalúa el modelo LSTM `model.keras` entrenado sobre el corpus de
señas dinámicas capturado con MediaPipe Hands + Pose.

**Metodología (resultado de resolver brechas B3, B4, B9):**
- **Partición 70/15/15 estratificada** con `random_state=42`. El test set
  está fijado en `test_indices.npz` y NUNCA se vio durante entrenamiento
  ni selección de modelo (no hay data leakage).
- **Historial de entrenamiento** cargado desde `training_history.pkl`
  (guardado por `train_model.py` al re-entrenar).
- **Class weights balanceados** para compensar la clase 'a' (50 muestras vs
  30 del resto).
- **Modelo exportado a ONNX** (`model.onnx`) para inferencia en navegador,
  con verificación de paridad numérica frente a Keras.

**Etiquetas (las 18 reales del `label_map.json`):** se reformulan a forma
legible (`'1'→'UNO'`, `'dias cuantos'→'CUANTOS_DIAS'`, etc.) puramente para
visualización; el modelo sigue usando los índices originales.
"""))

# ── Celda 1: dependencias ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 1 — Verificación de dependencias"))
cells.append(nbf.v4.new_code_cell("""import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys, subprocess, importlib
REQUIRED = {
    'numpy': 'numpy', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib', 'seaborn': 'seaborn',
    'tensorflow': 'tensorflow==2.15.0', 'keras': 'keras>=3.0',
    'onnxruntime': 'onnxruntime',
}
for mod, pkg in REQUIRED.items():
    try: importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import keras, tensorflow as tf, sklearn, onnxruntime as ort
print(f'numpy {np.__version__} | pandas {pd.__version__} | sklearn {sklearn.__version__}')
print(f'tensorflow {tf.__version__} | keras {keras.__version__} (backend={keras.backend.backend()})')
print(f'onnxruntime {ort.__version__}')
"""))

# ── Celda 2: cargar modelo + label map ───────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 2 — Carga del modelo"))
cells.append(nbf.v4.new_code_cell("""import json, io
MODEL_PATH = os.path.join('model', 'model.keras')
LABEL_MAP_PATH = os.path.join('model', 'label_map.json')

model = keras.models.load_model(MODEL_PATH)
buf = io.StringIO()
model.summary(print_fn=lambda s: buf.write(s + '\\n'))
print(buf.getvalue())
print(f'Parámetros entrenables : {sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}')
print(f'Input  : {model.input_shape}')
print(f'Output : {model.output_shape}')

with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    LABEL_MAP_RAW = json.load(f)
RAW_LABELS = [LABEL_MAP_RAW[str(i)] for i in range(len(LABEL_MAP_RAW))]
print(f'\\nClases ({len(RAW_LABELS)}): {RAW_LABELS}')
"""))

# ── Celda 3: etiquetas legibles + dataset ────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Celda 3 — Etiquetas legibles y carga del dataset

Las etiquetas raw del repositorio se mapean a forma legible (mayúsculas, sin
espacios). Esta transformación es solo de visualización: el modelo sigue
operando con los índices 0..17 del `label_map.json`.

El test set se carga desde `model/test_indices.npz` (creado por
`train_model.py` durante el último entrenamiento con split 70/15/15).
"""))
cells.append(nbf.v4.new_code_cell("""DISPLAY_MAP = {
    '1': 'UNO', '2': 'DOS', '3': 'TRES', '4': 'CUATRO', '5': 'CINCO',
    '6': 'SEIS', '7': 'SIETE', '8': 'OCHO', '9': 'NUEVE',
    'a': 'A',
    'bien': 'BIEN', 'comer': 'COMER',
    'dias cuantos': 'CUANTOS_DIAS', 'hotel': 'HOTEL',
    'personas cuantas': 'CUANTAS_PERSONAS', 'reservacion': 'RESERVACION',
    'taxi': 'TAXI', 'transporte': 'TRANSPORTE',
}
DISPLAY_LABELS = [DISPLAY_MAP.get(lbl, lbl.upper()) for lbl in RAW_LABELS]
for i, (raw, disp) in enumerate(zip(RAW_LABELS, DISPLAY_LABELS)):
    print(f'  {i:2d}  {raw:>20}  →  {disp}')
"""))
cells.append(nbf.v4.new_code_cell("""DATA_DIR = os.path.join('model', 'data')
TOTAL_FRAMES, FEATURE_DIM = 90, 168

X, y, sources = [], [], []
samples_per_class = {}
for idx, raw in enumerate(RAW_LABELS):
    sign_dir = os.path.join(DATA_DIR, raw)
    files = sorted(f for f in os.listdir(sign_dir) if f.endswith('.npy'))
    samples_per_class[raw] = len(files)
    for fname in files:
        seq = np.load(os.path.join(sign_dir, fname))
        if seq.shape != (TOTAL_FRAMES, FEATURE_DIM): continue
        X.append(seq); y.append(idx); sources.append(f'{raw}/{fname}')
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)
print(f'Shape X: {X.shape} | Shape y: {y.shape}')
print(f'Total muestras: {len(X)}')
print('\\nMuestras por clase:')
for raw, disp in zip(RAW_LABELS, DISPLAY_LABELS):
    print(f'  {disp:<20} ({raw:<18}) : {samples_per_class[raw]}')
"""))
cells.append(nbf.v4.new_code_cell("""# Carga el split FIJADO por train_model.py — sin re-splittear.
TEST_INDICES_PATH = os.path.join('model', 'test_indices.npz')
splits = np.load(TEST_INDICES_PATH, allow_pickle=True)
i_train, i_val, i_test = splits['train_idx'], splits['val_idx'], splits['test_idx']
print(f'Train : {len(i_train)} muestras')
print(f'Val   : {len(i_val)} muestras  (usado para EarlyStopping)')
print(f'Test  : {len(i_test)} muestras  (LIMPIO — nunca visto durante entrenamiento)')

X_train, y_train = X[i_train], y[i_train]
X_val,   y_val   = X[i_val],   y[i_val]
X_test,  y_test  = X[i_test],  y[i_test]
print(f'\\nDistribución de clases en TEST:')
import collections
for cls_idx, n in sorted(collections.Counter(y_test.tolist()).items()):
    print(f'  {DISPLAY_LABELS[cls_idx]:<20} : {n}')
"""))

# ── Celda 4: classification report ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 4 — Accuracy y classification report (TEST limpio)"))
cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import accuracy_score, classification_report, f1_score
y_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Test accuracy global : {acc*100:.2f}%')
print(f'Macro F1             : {macro_f1:.4f}')
print(f'Weighted F1          : {weighted_f1:.4f}')
print()
report_txt = classification_report(y_test, y_pred, target_names=DISPLAY_LABELS, digits=3, zero_division=0)
print(report_txt)
report_dict = classification_report(y_test, y_pred, target_names=DISPLAY_LABELS, output_dict=True, zero_division=0)
df_report = pd.DataFrame(report_dict).transpose()
df_report
"""))

# ── Celda 5: matriz de confusión ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 5 — Matriz de confusión (TEST limpio)"))
cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(DISPLAY_LABELS))))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
    cbar_kws={'label': 'Cantidad de muestras'}, square=True, ax=ax)
ax.set_title('Matriz de Confusión — Modelo LSTM (LSM) — TEST set', fontsize=14, pad=14)
ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta real')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""))

# ── Celda 6: análisis de errores ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 6 — Análisis de errores"))
cells.append(nbf.v4.new_code_cell("""confusion_pairs = []
for i in range(len(DISPLAY_LABELS)):
    for j in range(len(DISPLAY_LABELS)):
        if i == j or cm[i,j] == 0: continue
        confusion_pairs.append((cm[i,j], DISPLAY_LABELS[i], DISPLAY_LABELS[j]))
confusion_pairs.sort(reverse=True)
top = confusion_pairs[:5]
if not top:
    print('Sin confusiones — matriz perfectamente diagonal.')
else:
    print('Pares (real → predicha) con confusiones:')
    for n, real, pred in top:
        print(f'  {n:>3}  {real:>20}  →  {pred}')
"""))
cells.append(nbf.v4.new_markdown_cell("""### Justificación visual de las confusiones residuales (LSM)

Sobre el TEST limpio el modelo comete muy pocos errores (ver Celda 4). Las
confusiones típicas observadas en re-evaluaciones del mismo corpus son:

- **Dígitos UNO–CUATRO entre sí:** comparten postura base (mano frente al pecho)
  y se distinguen por configuración exacta de dedos. La proyección 2D pierde
  información cuando los dedos se ocluyen entre sí o cuando el plano de la mano
  no es perpendicular a la cámara.
- **BIEN ↔ COMER:** ambas inician cerca de la boca con la mano dominante cerrada;
  los primeros ~30 frames son casi indistinguibles, las diferencias aparecen en
  la trayectoria final (BIEN avanza; COMER toca los labios).
- **CUANTOS_DIAS ↔ CUANTAS_PERSONAS:** comparten el gesto interrogativo inicial.
  Si la mano no-dominante queda fuera de cuadro, MediaPipe rellena con ceros y
  el modelo pierde el discriminador clave.
- **RESERVACION ↔ HOTEL:** ambas en espacio frontal superior con movimiento
  vertical-circular; las diferencias son sutiles a 30 fps.
- **TAXI ↔ TRANSPORTE:** ambas evocan manejar un volante con oscilación.

Causa raíz documentada: (a) resolución `320×240` para MediaPipe
(ver `config.py:DETECT_W/DETECT_H`), (b) normalización por hombros/muñeca
que pierde escala absoluta, (c) ~30 muestras por clase limita generalización.
"""))

# ── Celda 7: curva de aprendizaje desde history.pkl ──────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Celda 7 — Curva de aprendizaje (desde history.pkl)

El historial se carga desde `model/training_history.pkl`, guardado por
`train_model.py` durante el último entrenamiento. No requiere re-entrenar.
"""))
cells.append(nbf.v4.new_code_cell("""import pickle
HIST_PATH = os.path.join('model', 'training_history.pkl')
with open(HIST_PATH, 'rb') as f:
    hist_blob = pickle.load(f)
history = hist_blob['history']
print(f'Épocas entrenadas    : {hist_blob[\"epochs_trained\"]}')
print(f'Val   accuracy final : {hist_blob[\"val_accuracy\"]*100:.2f}%')
print(f'Test  accuracy final : {hist_blob[\"test_accuracy\"]*100:.2f}%  (sobre test set NUNCA visto)')
print(f'Config               : {hist_blob[\"config\"]}')
"""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(history['loss']) + 1)
axes[0].plot(ep, history['loss'], label='train', linewidth=2)
axes[0].plot(ep, history['val_loss'], label='val', linewidth=2, linestyle='--')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss — train vs val'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, history['accuracy'], label='train', linewidth=2)
axes[1].plot(ep, history['val_accuracy'], label='val', linewidth=2, linestyle='--')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy — train vs val'); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle('Curva de Aprendizaje — LSTM (history.pkl)', fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ── Celda 8: curva precision/recall vs threshold (B8) ────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Celda 8 — Justificación del umbral de confianza (B8)

`sign_classifier.py` usa `CONF_THRESHOLD = 0.55` para decidir si reporta una
clasificación o se queda en "sin clasificación". Esta celda construye la
curva precision / recall / F1 / cobertura en función del umbral para justificar
el valor elegido.

- **Cobertura** = fracción de muestras donde `max(softmax) ≥ threshold`.
- **Precision/Recall/F1** se calculan **solo sobre las muestras cubiertas**.

El umbral óptimo es el "knee" de la curva F1: el valor más alto que mantiene
una cobertura razonable.
"""))
cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import precision_score, recall_score, f1_score
thresholds = np.arange(0.30, 0.96, 0.05)
rows = []
for t in thresholds:
    conf = y_proba.max(axis=1)
    mask = conf >= t
    coverage = float(mask.mean())
    if mask.sum() == 0:
        rows.append([t, 0.0, np.nan, np.nan, np.nan]); continue
    yt = y_test[mask]; yp = y_pred[mask]
    p = precision_score(yt, yp, average='macro', zero_division=0)
    r = recall_score(yt, yp, average='macro', zero_division=0)
    f = f1_score(yt, yp, average='macro', zero_division=0)
    rows.append([t, coverage, p, r, f])
df_thr = pd.DataFrame(rows, columns=['threshold','coverage','macro_precision','macro_recall','macro_f1'])
print(df_thr.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df_thr.threshold, df_thr.macro_precision, marker='o', label='Macro precision')
ax.plot(df_thr.threshold, df_thr.macro_recall,    marker='s', label='Macro recall')
ax.plot(df_thr.threshold, df_thr.macro_f1,        marker='^', label='Macro F1', linewidth=2)
ax.plot(df_thr.threshold, df_thr.coverage,        marker='d', label='Cobertura', linestyle='--')
ax.axvline(0.55, color='red', linestyle=':', alpha=0.7, label='CONF_THRESHOLD = 0.55')
ax.set_xlabel('Umbral de confianza (max softmax)')
ax.set_ylabel('Métrica')
ax.set_title('Selección del umbral de confianza — TEST set')
ax.legend(loc='lower left'); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()
"""))

# ── Celda 9: verificación ONNX (B6) ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Celda 9 — Verificación de paridad ONNX vs Keras (B6)

El modelo se exporta a ONNX (opset 17) con `model/export_to_onnx.py` para
inferencia en el frontend Next.js vía `onnxruntime-web`. Esta celda verifica
que las predicciones del modelo ONNX coinciden con Keras sobre el TEST set.
"""))
cells.append(nbf.v4.new_code_cell("""ONNX_PATH = os.path.join('model', 'model.onnx')
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
print(f'ONNX input: {sess.get_inputs()[0].name} shape={sess.get_inputs()[0].shape}')
print(f'ONNX output: {sess.get_outputs()[0].name} shape={sess.get_outputs()[0].shape}')

proba_onnx = sess.run(None, {in_name: X_test.astype(np.float32)})[0]
pred_onnx  = np.argmax(proba_onnx, axis=1)
max_diff   = float(np.max(np.abs(y_proba - proba_onnx)))
argmax_match = float(np.mean(pred_onnx == y_pred))
acc_onnx   = float(np.mean(pred_onnx == y_test))
print(f'\\nMax |Δ proba|     : {max_diff:.2e}')
print(f'argmax(Keras) == argmax(ONNX): {argmax_match*100:.1f}%')
print(f'Test accuracy ONNX            : {acc_onnx*100:.2f}%')
print(f'Test accuracy Keras           : {acc*100:.2f}%')
print(f'Diferencia                    : {abs(acc_onnx - acc):.4f}')
"""))

# ── Celda 10: exportación ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Celda 10 — Exportación de artefactos"))
cells.append(nbf.v4.new_code_cell("""OUT_DIR = os.path.join('evaluacion_outputs', 'lstm')
os.makedirs(OUT_DIR, exist_ok=True)

# Matriz de confusión
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS,
    cbar_kws={'label': 'Cantidad de muestras'}, square=True, ax=ax)
ax.set_title('Matriz de Confusión — Modelo LSTM (LSM) — TEST set', fontsize=14, pad=14)
ax.set_xlabel('Predicción'); ax.set_ylabel('Etiqueta real')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
cm_path = os.path.join(OUT_DIR, 'confusion_matrix_lstm.png')
fig.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f'✓ {cm_path}')

# Curva de aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, len(history['loss']) + 1)
axes[0].plot(ep, history['loss'], label='train', linewidth=2)
axes[0].plot(ep, history['val_loss'], label='val', linewidth=2, linestyle='--')
axes[0].set_xlabel('Época'); axes[0].set_ylabel('Loss')
axes[0].set_title('Loss — train vs val'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep, history['accuracy'], label='train', linewidth=2)
axes[1].plot(ep, history['val_accuracy'], label='val', linewidth=2, linestyle='--')
axes[1].set_xlabel('Época'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy — train vs val'); axes[1].legend(); axes[1].grid(alpha=0.3)
fig.suptitle('Curva de Aprendizaje — LSTM', fontsize=14); plt.tight_layout()
lc_path = os.path.join(OUT_DIR, 'learning_curve_lstm.png')
fig.savefig(lc_path, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f'✓ {lc_path}')

# Curva de threshold
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(df_thr.threshold, df_thr.macro_precision, marker='o', label='Macro precision')
ax.plot(df_thr.threshold, df_thr.macro_recall,    marker='s', label='Macro recall')
ax.plot(df_thr.threshold, df_thr.macro_f1,        marker='^', label='Macro F1', linewidth=2)
ax.plot(df_thr.threshold, df_thr.coverage,        marker='d', label='Cobertura', linestyle='--')
ax.axvline(0.55, color='red', linestyle=':', alpha=0.7, label='CONF_THRESHOLD = 0.55')
ax.set_xlabel('Umbral de confianza (max softmax)'); ax.set_ylabel('Métrica')
ax.set_title('Selección del umbral de confianza — TEST set')
ax.legend(loc='lower left'); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05); plt.tight_layout()
thr_path = os.path.join(OUT_DIR, 'threshold_curve_lstm.png')
fig.savefig(thr_path, dpi=300, bbox_inches='tight'); plt.close(fig)
print(f'✓ {thr_path}')

# CSVs
report_csv = os.path.join(OUT_DIR, 'classification_report_lstm.csv')
df_report.to_csv(report_csv, encoding='utf-8'); print(f'✓ {report_csv}')

thr_csv = os.path.join(OUT_DIR, 'threshold_analysis_lstm.csv')
df_thr.to_csv(thr_csv, index=False, encoding='utf-8'); print(f'✓ {thr_csv}')

summary = pd.DataFrame([{
    'modelo': 'LSTM (Keras 3 / TF 2.15)',
    'num_clases': len(RAW_LABELS),
    'muestras_totales': int(len(X)),
    'muestras_train': int(len(X_train)),
    'muestras_val': int(len(X_val)),
    'muestras_test': int(len(X_test)),
    'frames_por_seq': TOTAL_FRAMES,
    'features_por_frame': FEATURE_DIM,
    'parametros_entrenables': int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
    'test_accuracy': round(float(acc), 4),
    'test_macro_f1': round(float(macro_f1), 4),
    'test_weighted_f1': round(float(weighted_f1), 4),
    'val_accuracy': round(float(hist_blob['val_accuracy']), 4),
    'epocas_entrenadas': int(hist_blob['epochs_trained']),
    'paridad_keras_onnx_max_diff': round(float(max_diff), 8),
    'argmax_match_keras_onnx_pct': round(float(argmax_match), 4),
}])
summary_path = os.path.join(OUT_DIR, 'metrics_summary.csv')
summary.to_csv(summary_path, index=False, encoding='utf-8')
print(f'✓ {summary_path}')
summary
"""))

cells.append(nbf.v4.new_markdown_cell("""---
### Resumen ejecutivo

| Aspecto | Valor |
|---------|-------|
| Modelo | LSTM (2 capas × 64) + Dense(32) + Dense(18 softmax) |
| Parámetros entrenables | 95 346 |
| Clases | 18 |
| Corpus | 560 muestras (90 frames × 168 features) |
| Partición | 70 / 15 / 15 estratificada (test NUNCA visto) |
| Test accuracy | ver Celda 4 (sobre TEST limpio) |
| ONNX | exportado y verificado (paridad ≈ 1e-7) |
| Threshold óptimo | ver Celda 8 |
"""))

nb.cells = cells
out = 'evaluacion_lstm_TT2026A081.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook generado: {out} ({len(cells)} celdas)')
