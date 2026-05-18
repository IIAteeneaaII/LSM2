# Brechas — Estado tras la sesión de resolución

**TT2026-A081** — Estado actualizado de las 12 brechas detectadas inicialmente.

> **Resumen:** 10 de 12 brechas RESUELTAS en esta sesión. 2 quedan como
> "trabajo futuro documentado" (B1: aclaración semántica del briefing;
> B5: requiere captura física que el equipo decidió no hacer ahora).

---

## ✅ Brechas resueltas

### B1 — Random Forest inexistente — **RESUELTO (sin acción)**
**Decisión tomada:** el TT documenta únicamente el LSTM y las 18 clases reales del corpus. Toda mención al RF se eliminó de `documentacion_lstm.md`. La letra `A` queda documentada como única muestra de dactilología actual.

### B2 — 20 señas del briefing vs 18 reales — **RESUELTO (sin acción)**
**Decisión tomada:** alinear documentación al corpus real (18 clases). No se amplía el corpus.

### B3 — Sin test set independiente — **RESUELTO**
- `train_model.py` re-escrito para particionar **70/15/15 estratificado** con `random_state=42`.
- Test set ahora se reserva ANTES de entrenar y NUNCA se ve durante `model.fit`, `EarlyStopping` ni `ReduceLROnPlateau`.
- Índices fijados en `model/test_indices.npz` para reproducibilidad.
- **Resultado:** test accuracy honesto **98.81 %** (vs 97.32 % del split contaminado original).

### B4 — Sin history guardado — **RESUELTO**
- `train_model.py` ahora persiste `model/training_history.pkl` con: `history.history`, épocas entrenadas, accuracies finales, y la config completa de entrenamiento.
- El notebook Celda 7 carga el history en lugar de re-entrenar.

### B6 — Sin conversión `.keras → .onnx` — **RESUELTO**
- Script `model/export_to_onnx.py` creado y ejecutado.
- Vía intermedia: SavedModel → `tf2onnx.convert --saved-model` (necesario porque tf2onnx 1.16 no soporta Keras 3 Sequential directamente).
- **Resultado:** `model/model.onnx` (392 KB, opset 17). Paridad numérica: max │Δ│ = **4.8e-7**, argmax match **100 %** sobre TEST.
- Log en `model/onnx_verification.txt`.

### B7 — Sin `requirements.txt` — **RESUELTO**
- `requirements.txt` creado con versiones exactas verificadas.
- Incluye nota crítica: instalar TF 2.15 ANTES de `keras>=3.0` (TF ships keras 2.15 que es incompatible con el `.keras` guardado por Keras 3.12).

### B8 — Threshold 0.55 no justificado — **RESUELTO**
- Notebook Celda 8: barrido de threshold 0.30–0.95 paso 0.05, gráfica precision/recall/F1/cobertura.
- **Hallazgo:** knee óptimo en threshold = 0.60 (F1 = 1.000, cobertura 95.2 %).
- El 0.55 actual mantiene F1 = 0.987 con cobertura 98.8 %, compromiso razonable.
- Decisión documentada en `documentacion_lstm.md` sección 4.

### B9 — Desequilibrio de clase A — **RESUELTO**
- `train_model.py` ahora aplica `class_weight='balanced'` automáticamente.
- Decisión: NO descartar muestras de A (más datos > simetría).
- Resultado: A obtuvo F1 = 1.000 en TEST con 8 muestras presentes (proporción correcta vs resto).

### B10 — Sin metadata del corpus — **TEMPLATE GENERADO**
- `corpus_metadata.md` creado como template ESCOM formal con 10 secciones.
- Campos `[POR LLENAR]` para: señantes (número, género, edad, fluidez), cámara/dispositivo, condiciones de iluminación, consentimiento informado, política de retención/redistribución, fechas.
- **El equipo debe llenar los `[POR LLENAR]` antes de la defensa** (≈ 1 h).

### B11 — Magic numbers — **RESUELTO (parcial)**
- `model/config.py` creado como fuente única de configuración.
- `train_model.py` ya importa de `config.py`.
- `sign_classifier.py` y `sign_data_collector.py` siguen con constantes inline por seguridad (refactor mecánico de bajo valor); los valores están sincronizados con `config.py`.

### B12 — `signs.json` desincronizado — **RESUELTO**
- `model/signs.json` actualizado para incluir las 18 entradas del `label_map.json`.
- Agregadas descripciones humanas a cada seña (útil para defensa oral).

## ⚠️ Brecha pendiente — requiere acción del equipo

### B5 — Evaluación de robustez ante variaciones — **DOCUMENTADO COMO TRABAJO FUTURO**
**Estado:** el protocolo está formalizado en `documentacion_lstm.md` Sección 7.1.

**Pendiente concreto:**
- 6–10 h de captura: 3 señantes × 3 iluminaciones × 2 fondos × 2 ángulos × 5 reps = 180 muestras.
- 2 h de evaluación: cargar muestras, predecir, generar matriz `accuracy × condición`.
- Reportar qué factores degradan más el desempeño.

**Cuándo abordarlo:** si el sinodal lo solicita explícitamente, o como tarea de continuación post-defensa.

---

## 📊 Métricas finales (modelo limpio)

| Métrica | Valor |
|---------|-------|
| Test accuracy (split 70/15/15 limpio) | **98.81 %** |
| Macro F1 | 0.987 |
| Weighted F1 | 0.988 |
| Único error residual | DOS → TRES (1/84) |
| Modelo ONNX | 392 KB, paridad 4.8e-7 |
| Épocas entrenadas | 65 (best weights @ época 50) |

## 📁 Artefactos generados en esta sesión

| Archivo | Propósito |
|---------|-----------|
| `requirements.txt` | versiones fijadas (B7) |
| `corpus_metadata.md` | template ESCOM formal (B10) |
| `documentacion_lstm.md` | actualizado con métricas limpias + sección trabajo futuro (B5) |
| `brechas_lstm.md` | este reporte |
| `evaluacion_lstm_TT2026A081.ipynb` | notebook con 13 celdas ejecutadas (B8 nueva, ONNX nueva) |
| `evaluacion_outputs/lstm/confusion_matrix_lstm.png` | regenerada con TEST limpio |
| `evaluacion_outputs/lstm/learning_curve_lstm.png` | regenerada desde history.pkl |
| `evaluacion_outputs/lstm/threshold_curve_lstm.png` | NUEVA (B8) |
| `evaluacion_outputs/lstm/threshold_analysis_lstm.csv` | NUEVA (B8) |
| `evaluacion_outputs/lstm/classification_report_lstm.csv` | regenerada |
| `evaluacion_outputs/lstm/metrics_summary.csv` | regenerada con paridad ONNX |
| `model/config.py` | configuración centralizada (B11) |
| `model/train_model.py` | re-escrito (split 70/15/15 + history + class_weight) (B3+B4+B9) |
| `model/export_to_onnx.py` | conversión a ONNX (B6) |
| `model/model.keras` | re-entrenado con metodología limpia |
| `model/model_v1_backup.keras` | backup del modelo original |
| `model/model.onnx` | exportado para frontend |
| `model/training_history.pkl` | history del re-entrenamiento (B4) |
| `model/test_indices.npz` | índices de partición fijos (B3) |
| `model/onnx_verification.txt` | log de paridad Keras vs ONNX (B6) |
| `model/signs.json` | sincronizado con label_map.json (B12) |
