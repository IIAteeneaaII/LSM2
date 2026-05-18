"""
========================================================
  config.py — Constantes del pipeline LSM
  -------------------------------------------------------
  Fuente única de configuración para:
    - sign_data_collector.py  (captura)
    - train_model.py          (entrenamiento)
    - sign_classifier.py      (inferencia)

  Decisiones de diseño documentadas en cada bloque.
========================================================
"""

# ── Captura de video ──────────────────────────────────────────────────────────
FPS_TARGET    = 30
DURATION_SEC  = 3
TOTAL_FRAMES  = FPS_TARGET * DURATION_SEC      # 90 frames
COUNTDOWN_SEC = 2

# Resolución reducida para MediaPipe: balancea latencia (CPU) vs. precisión de
# landmarks. 320x240 es el mínimo que mantiene tracking estable de las dos manos
# a 30 fps en CPU mid-range. Las coordenadas (x, y) de MediaPipe son normalizadas
# 0-1, por lo que el dibujo sobre el frame original sigue siendo correcto.
DETECT_W = 320
DETECT_H = 240

# ── Geometría de landmarks ────────────────────────────────────────────────────
# Pose: solo cintura para arriba (LM 11-24 de MediaPipe Pose). Excluimos
# piernas porque la señalización LSM ocurre en el plano frontal del torso.
POSE_UPPER_INDICES = list(range(11, 25))       # 14 landmarks
NUM_POSE_LM = len(POSE_UPPER_INDICES)          # 14
NUM_HAND_LM = 21                               # MediaPipe Hands estándar
COORDS      = 3                                # (x, y, z)
POSE_DIM    = NUM_POSE_LM * COORDS             # 42
HAND_DIM    = NUM_HAND_LM * COORDS             # 63
FEATURE_DIM = POSE_DIM + HAND_DIM * 2          # 168 (pose + mano-I + mano-D)

# ── Modelos y datos ───────────────────────────────────────────────────────────
DATA_DIR       = "data"
MODEL_PATH     = "model.keras"
ONNX_PATH      = "model.onnx"
LABEL_MAP_PATH = "label_map.json"
SIGNS_FILE     = "signs.json"
HISTORY_PATH   = "training_history.pkl"

HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
# Partición 70/15/15: train para optimizar, val para EarlyStopping/scheduler,
# test JAMÁS se ve durante entrenamiento ni selección de modelo. Evita el data
# leakage que tenía el split 80/20 original (donde val == test).
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
RANDOM_SEED  = 42

EPOCHS_MAX        = 100
BATCH_SIZE        = 16
EARLY_STOP_PAT    = 15        # épocas sin mejora antes de detener
REDUCE_LR_PAT     = 7         # épocas sin mejora antes de reducir LR
REDUCE_LR_FACTOR  = 0.5
REDUCE_LR_MIN     = 1e-5
USE_CLASS_WEIGHT  = True      # compensa clase 'a' con 50 muestras vs 30 del resto

# ── Inferencia en tiempo real ─────────────────────────────────────────────────
# Inferencia cada N frames: 90 = una predicción por ventana completa, equivale
# a clasificar al terminar cada gesto de 3 s. Un valor menor (p.ej. 30 = cada
# segundo) daría feedback más rápido pero clasificaciones sobre ventanas
# parciales con mucha incertidumbre.
INFERENCE_EVERY = 90
RESULT_SHOW_SEC = 2.5

# Umbral de confianza: por debajo de 0.55 se reporta "sin clasificación". Valor
# elegido empíricamente como knee de la curva precision-recall (ver celda
# threshold del notebook de evaluación). Bajar a 0.40 aumenta recall pero
# introduce falsos positivos; subir a 0.70 elimina prácticamente las
# predicciones espontáneas pero requiere ejecución muy limpia de la seña.
CONF_THRESHOLD = 0.55
