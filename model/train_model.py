"""
========================================================
  Entrenamiento del modelo LSTM para Lengua de Señas
  -------------------------------------------------------
  Dependencias:
      pip install -r requirements.txt

  Uso:
      python train_model.py

  Salida:
      model.keras            -- modelo entrenado (mejor val_loss)
      label_map.json         -- mapeo índice → nombre de seña
      training_history.pkl   -- history.history para curvas de aprendizaje
      test_indices.npz       -- índices de muestras del test set
                                (para evaluación reproducible posterior)

  Cambios vs. versión original (resolución de brechas B3/B4/B9/B11):
    - Split 70/15/15 (antes 80/20 sin test). El test set NUNCA se ve
      durante el entrenamiento ni durante la selección de modelo.
    - Guarda training_history.pkl para reproducir curvas de aprendizaje
      sin necesidad de re-entrenar.
    - Aplica class_weight='balanced' para compensar la clase 'a' (50
      muestras vs 30 del resto).
    - Configuración centralizada en config.py.
========================================================
"""

import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Backend Keras 3 antes de importar keras
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    DATA_DIR, MODEL_PATH, LABEL_MAP_PATH, HISTORY_PATH,
    TOTAL_FRAMES, FEATURE_DIM,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED,
    EPOCHS_MAX, BATCH_SIZE,
    EARLY_STOP_PAT, REDUCE_LR_PAT, REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    USE_CLASS_WEIGHT,
)

TEST_INDICES_PATH = "test_indices.npz"


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

    X, y, src = [], [], []
    for name in signs:
        sign_dir = os.path.join(DATA_DIR, name)
        files = sorted([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
        if not files:
            print(f"  [!] Sin muestras en '{name}', se omite.")
            continue
        for fname in files:
            seq = np.load(os.path.join(sign_dir, fname))
            if seq.shape != (TOTAL_FRAMES, FEATURE_DIM):
                print(f"  [!] Shape inesperado en {fname}: {seq.shape}, se omite.")
                continue
            X.append(seq)
            y.append(name_to_idx[name])
            src.append(f"{name}/{fname}")
        print(f"  ✓ {name}: {len(files)} muestras")

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32),
        label_map,
        src,
    )


# ── Arquitectura del modelo ───────────────────────────────────────────────────

def build_model(num_classes: int) -> keras.Model:
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


# ── Particiones 70 / 15 / 15 estratificadas ──────────────────────────────────

def stratified_split_70_15_15(X, y, sources):
    """
    Devuelve seis arrays + tres listas de fuentes: train / val / test.
    Usa dos pasadas de train_test_split: primero separa test, luego val
    de lo restante. Mantiene la estratificación por clase en ambas etapas.
    """
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9
    idx = np.arange(len(X))

    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=RANDOM_SEED, stratify=y,
    )
    val_frac_rel = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac_rel,
        random_state=RANDOM_SEED, stratify=y[idx_trainval],
    )

    def take(i):
        return X[i], y[i], [sources[k] for k in i]

    return take(idx_train), take(idx_val), take(idx_test), (idx_train, idx_val, idx_test)


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 50)
    print("  ENTRENAMIENTO — Lengua de Señas (split 70/15/15)")
    print("═" * 50)

    keras.utils.set_random_seed(RANDOM_SEED)

    print("\nCargando dataset...")
    X, y, label_map, sources = load_dataset()
    num_classes = len(label_map)
    print(f"\nTotal muestras : {len(X)}")
    print(f"Clases ({num_classes}) : {list(label_map.values())}")
    print(f"Shape entrada  : {X.shape}")

    (Xtr, ytr, src_tr), (Xv, yv, src_v), (Xts, yts, src_ts), (i_tr, i_v, i_ts) = (
        stratified_split_70_15_15(X, y, sources)
    )
    print(f"\nTrain: {len(Xtr)}  |  Val: {len(Xv)}  |  Test: {len(Xts)}")
    print("(El test set NO se usa durante entrenamiento ni selección de modelo.)")

    np.savez(
        TEST_INDICES_PATH,
        train_idx=i_tr, val_idx=i_v, test_idx=i_ts,
        sources_train=np.array(src_tr), sources_val=np.array(src_v),
        sources_test=np.array(src_ts),
    )
    print(f"✓ Índices de partición → {TEST_INDICES_PATH}")

    ytr_cat = keras.utils.to_categorical(ytr, num_classes)
    yv_cat  = keras.utils.to_categorical(yv,  num_classes)
    yts_cat = keras.utils.to_categorical(yts, num_classes)

    # Class weights (compensa el desbalance de la clase 'a' con 50 muestras
    # vs 30 del resto).
    class_weight = None
    if USE_CLASS_WEIGHT:
        cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr)
        class_weight = {i: float(w) for i, w in enumerate(cw)}
        print(f"\nClass weights: {class_weight}")

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=EARLY_STOP_PAT, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(
            factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PAT,
            min_lr=REDUCE_LR_MIN, verbose=1,
        ),
    ]

    print("\nEntrenando...\n")
    history = model.fit(
        Xtr, ytr_cat,
        validation_data=(Xv, yv_cat),
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluación FINAL sobre test set (nunca visto) ───────────────────────
    print("\n" + "─" * 50)
    val_loss, val_acc = model.evaluate(Xv,  yv_cat,  verbose=0)
    test_loss, test_acc = model.evaluate(Xts, yts_cat, verbose=0)
    print(f"Val  accuracy : {val_acc*100:.2f}%   (usado por EarlyStopping)")
    print(f"Test accuracy : {test_acc*100:.2f}%  (LIMPIO, nunca visto durante entrenamiento)")

    y_pred = np.argmax(model.predict(Xts, verbose=0), axis=1)
    target_names = [label_map[i] for i in range(num_classes)]
    print("\nReporte de clasificación (TEST):")
    print(classification_report(yts, y_pred, target_names=target_names, zero_division=0))

    # ── Persistencia ─────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # history.pkl para reproducir curvas de aprendizaje sin re-entrenar
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump({
            "history": history.history,
            "epochs_trained": len(history.history["loss"]),
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "val_accuracy": float(val_acc),
            "config": {
                "split": [TRAIN_FRAC, VAL_FRAC, TEST_FRAC],
                "random_seed": RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "epochs_max": EPOCHS_MAX,
                "early_stop_patience": EARLY_STOP_PAT,
                "use_class_weight": USE_CLASS_WEIGHT,
                "class_weight": class_weight,
            },
        }, f)

    print(f"\n✓ Modelo guardado    → {MODEL_PATH}")
    print(f"✓ Mapa de etiquetas  → {LABEL_MAP_PATH}")
    print(f"✓ Historial          → {HISTORY_PATH}")
    print("\n¡Listo! Ejecuta 'python sign_classifier.py' para clasificar en tiempo real.")


if __name__ == "__main__":
    main()
