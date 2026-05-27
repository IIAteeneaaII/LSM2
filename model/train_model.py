"""
========================================================
  Entrenamiento del modelo LSTM para Lengua de Señas
  -------------------------------------------------------
  Arquitectura: LSTM(64) → LSTM(64) → Dense(32) → softmax
  con Dropout 0.5 + augmentación leve específica para LSM
  (simula la variación natural entre la sesión de captura
  y la sesión de uso en vivo, sin tocar la semántica
  de la seña: nada de mirror, frame-dropout ni time-shift).

  Dependencias:
      pip install tensorflow scikit-learn

  Uso:
      python train_model.py

  Salida:
      model.keras       -- modelo entrenado
      label_map.json    -- mapeo índice → nombre de seña
========================================================
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_DIR       = "data"
MODEL_PATH     = "model.keras"
LABEL_MAP_PATH = "label_map.json"
TOTAL_FRAMES   = 90
FEATURE_DIM    = 168     # 42 pose + 63 mano-izq + 63 mano-der
TEST_SIZE      = 0.2
RANDOM_SEED    = 42
BATCH_SIZE     = 16
EPOCHS         = 100

# Dropout más fuerte: fuerza al modelo a no depender de trayectorias exactas
# memorizadas durante la captura.
DROPOUT_RATE   = 0.5

# Augmentación LEVE para LSM. Sólo simula variaciones entre la sesión de
# grabado y la sesión real (jitter de MediaPipe, distancia a cámara,
# posición del cuerpo, velocidad del signante). NO altera la seña.
AUG_NOISE_STD       = 0.005   # jitter típico de MediaPipe
AUG_SCALE_RANGE     = 0.03    # ±3 % de distancia a cámara
AUG_TRANSLATE_RANGE = 0.02    # ±2 % de posición residual del cuerpo
AUG_TIME_STRETCH    = 0.10    # ±10 % de velocidad del signante

tf.keras.utils.set_random_seed(RANDOM_SEED)


# ── Carga del dataset ─────────────────────────────────────────────────────────
# Lee TODAS las carpetas dentro de data/ — al grabar nuevas señas con
# sign_data_collector.py, automáticamente se incluyen en el próximo entreno.

def load_dataset():
    signs = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    if not signs:
        raise RuntimeError(f"No se encontraron señas en '{DATA_DIR}/'")

    label_map = {i: name for i, name in enumerate(signs)}
    name_to_idx = {name: i for i, name in label_map.items()}

    X, y = [], []
    for name in signs:
        sign_dir = os.path.join(DATA_DIR, name)
        files = sorted([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        if not files:
            print(f"  [!] Sin muestras en '{name}', se omite.")
            continue
        for fname in files:
            seq = np.load(os.path.join(sign_dir, fname))  # (90, 168)
            if seq.shape != (TOTAL_FRAMES, FEATURE_DIM):
                print(f"  [!] Shape inesperado en {fname}: {seq.shape}, se omite.")
                continue
            X.append(seq)
            y.append(name_to_idx[name])
        print(f"  ✓ {name}: {len(files)} muestras")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


# ── Augmentación específica para LSM ─────────────────────────────────────────
# Cada transformación está justificada como simulación de una variación REAL
# que ocurre entre la captura y la inferencia en vivo.

@tf.function
def augment(seq, label):
    # 1) Ruido gaussiano leve: MediaPipe nunca devuelve coords idénticas
    #    entre dos sesiones aunque la pose sea la misma.
    seq = seq + tf.random.normal(tf.shape(seq), stddev=AUG_NOISE_STD,
                                 dtype=seq.dtype)

    # 2) Escala global: el usuario puede estar más cerca/lejos de la cámara
    #    que cuando grabó. Las features ya están normalizadas, pero la
    #    normalización no es perfecta entre sesiones.
    scale = 1.0 + tf.random.uniform((), -AUG_SCALE_RANGE, AUG_SCALE_RANGE,
                                    dtype=seq.dtype)
    seq = seq * scale

    # 3) Traslación residual: posición del cuerpo en el frame puede variar
    #    ligeramente. Mismo offset para toda la secuencia → no rompe la
    #    dinámica del movimiento.
    offset = tf.random.uniform((1, FEATURE_DIM),
                               -AUG_TRANSLATE_RANGE, AUG_TRANSLATE_RANGE,
                               dtype=seq.dtype)
    seq = seq + offset

    # 4) Time stretch: el usuario puede signar un poco más rápido/lento que
    #    cuando grabó. Re-muestreamos por interpolación lineal — la seña
    #    sigue siendo la misma seña, sólo a otra velocidad.
    factor = tf.random.uniform((), 1.0 - AUG_TIME_STRETCH,
                                  1.0 + AUG_TIME_STRETCH, dtype=tf.float32)
    src_idx = tf.cast(tf.range(TOTAL_FRAMES), tf.float32) / factor
    src_idx = tf.clip_by_value(src_idx, 0.0, float(TOTAL_FRAMES - 1))
    i0 = tf.cast(tf.floor(src_idx), tf.int32)
    i1 = tf.minimum(i0 + 1, TOTAL_FRAMES - 1)
    w  = tf.cast(src_idx - tf.cast(i0, tf.float32), seq.dtype)
    w  = tf.reshape(w, (-1, 1))
    seq = (1.0 - w) * tf.gather(seq, i0) + w * tf.gather(seq, i1)

    return seq, label


def make_dataset(X, y_onehot, training):
    ds = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    if training:
        ds = ds.shuffle(len(X), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── Arquitectura del modelo ───────────────────────────────────────────────────
# Misma arquitectura LSTM apilada del original, pero con Dropout 0.5
# en cada capa para que el modelo NO memorice trayectorias exactas
# y sea forzado a aprender patrones robustos.

def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(TOTAL_FRAMES, FEATURE_DIM)),
        LSTM(64, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(64),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 50)
    print("  ENTRENAMIENTO — Lengua de Señas")
    print("═" * 50)

    print("\nCargando dataset...")
    X, y, label_map = load_dataset()
    num_classes = len(label_map)
    print(f"\nTotal muestras : {len(X)}")
    print(f"Señas          : {list(label_map.values())}")
    print(f"Shape entrada  : {X.shape}")

    # Split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    y_train_cat = to_categorical(y_train, num_classes).astype(np.float32)
    y_val_cat   = to_categorical(y_val,   num_classes).astype(np.float32)
    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}")

    # Pipelines tf.data — augment SÓLO en train, val queda intacto
    train_ds = make_dataset(X_train, y_train_cat, training=True)
    val_ds   = make_dataset(X_val,   y_val_cat,   training=False)

    # Modelo
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5, verbose=1),
    ]

    print("\nEntrenando...\n")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluación
    print("\n" + "─" * 50)
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"Val accuracy : {acc * 100:.1f}%")
    print(f"Val loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    target_names = [label_map[i] for i in range(num_classes)]
    print("\nReporte de clasificación:")
    print(classification_report(y_val, y_pred, target_names=target_names))

    cm = confusion_matrix(y_val, y_pred)
    print("Matriz de confusión:")
    header = "        " + "  ".join(f"{n:>8}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>6}  " + "  ".join(f"{v:>8}" for v in row))

    # Guardar
    model.save(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Modelo guardado  → {MODEL_PATH}")
    print(f"✓ Mapa de etiquetas → {LABEL_MAP_PATH}")
    print("\n¡Listo! Ejecuta 'python sign_classifier.py' para clasificar en tiempo real.")


if __name__ == "__main__":
    main()
