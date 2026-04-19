"""
========================================================
  Entrenamiento del modelo LSTM para Lengua de Señas
  -------------------------------------------------------
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
FEATURE_DIM    = 168   # 42 pose + 63 mano-izq + 63 mano-der
TEST_SIZE      = 0.2
RANDOM_SEED    = 42


# ── Carga del dataset ─────────────────────────────────────────────────────────

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
            seq = np.load(os.path.join(sign_dir, fname))  # (90, 63)
            if seq.shape != (TOTAL_FRAMES, FEATURE_DIM):
                print(f"  [!] Shape inesperado en {fname}: {seq.shape}, se omite.")
                continue
            X.append(seq)
            y.append(name_to_idx[name])
        print(f"  ✓ {name}: {len(files)} muestras")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


# ── Arquitectura del modelo ───────────────────────────────────────────────────

def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(TOTAL_FRAMES, FEATURE_DIM)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
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

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)

    print(f"\nTrain: {len(X_train)}  |  Val: {len(X_val)}")

    # Modelo
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5, verbose=1),
    ]

    print("\nEntrenando...\n")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluación
    print("\n" + "─" * 50)
    loss, acc = model.evaluate(X_val, y_val_cat, verbose=0)
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
