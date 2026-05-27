"""
========================================================
  Recolector de datos para Lengua de Señas Dinámica
  -------------------------------------------------------
  Dependencias:
      pip install opencv-python mediapipe numpy

  Uso:
      python sign_data_collector.py

  Estructura del dataset generado:
      data/
        ├── hola/
        │   ├── rep_000.npy   # shape (90, 63)
        │   └── ...
        └── gracias/
            └── ...

  Cada frame contiene 63 valores:
    -  63 valores: mano izquierda normalizada (21 landmarks x,y,z)
========================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import json
import time
import sys
import urllib.request

# ── Modelos MediaPipe (Tasks API) ─────────────────────────────────────────────
HAND_MODEL     = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
])

# ── Configuración global ──────────────────────────────────────────────────────
FPS_TARGET    = 30
DURATION_SEC  = 3
TOTAL_FRAMES  = FPS_TARGET * DURATION_SEC   # 90
COUNTDOWN_SEC = 2                            # reducido de 3 a 2 s

# Resolución de detección: MediaPipe recibe una imagen más pequeña para ir más
# rápido. Las coordenadas x,y son normalizadas (0-1), así que el dibujo sobre
# el frame original no se ve afectado.
DETECT_W = 320
DETECT_H = 240
REPS_DEFAULT  = 50
DATA_DIR      = "data"
SIGNS_FILE    = "signs.json"

NUM_HAND_LM  = 21
COORDS       = 3
HAND_DIM     = NUM_HAND_LM * COORDS         # 63
FEATURE_DIM  = HAND_DIM                     # 63 — solo mano izquierda

# ── Colores (BGR) ──────────────────────────────────────────────────────────────
C_GREEN   = (72, 199, 142)
C_BLUE    = (237, 139, 55)
C_RED     = (60, 60, 220)
C_YELLOW  = (0, 220, 220)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0, 0, 0)
C_GRAY    = (150, 150, 150)
C_PURPLE  = (200, 100, 220)
C_CYAN    = (220, 180, 0)

ZONE_COLOR = {
    0: C_WHITE,
    1: C_PURPLE, 2: C_PURPLE, 3: C_PURPLE, 4: C_PURPLE,
    5: C_GREEN,  6: C_GREEN,  7: C_GREEN,  8: C_GREEN,
    9: C_BLUE,  10: C_BLUE,  11: C_BLUE,  12: C_BLUE,
   13: C_YELLOW,14: C_YELLOW,15: C_YELLOW,16: C_YELLOW,
   17: C_RED,   18: C_RED,   19: C_RED,   20: C_RED,
}

# ── Estado ────────────────────────────────────────────────────────────────────
class AppState:
    IDLE      = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    SAVED     = "saved"


# ── Descarga de modelos ────────────────────────────────────────────────────────

def ensure_models():
    for path, url in [(HAND_MODEL, HAND_MODEL_URL)]:
        if not os.path.exists(path):
            print(f"Descargando {path}...")
            urllib.request.urlretrieve(url, path)
            print(f"  OK: {path}")


def create_detector():
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(hand_opts)


# ── Extracción de features ────────────────────────────────────────────────────

def extract_left_hand(hand_result):
    """
    Normaliza la mano izquierda:
      centro = muñeca, escala = dist muñeca→MCP-medio.
    Devuelve (left(63,), left_valid).
    Si hay varias manos detectadas, se queda con la primera etiquetada "Left".
    """
    left = np.zeros(HAND_DIM, dtype=np.float32)
    left_valid = False

    for lm_list, handedness_list in zip(hand_result.hand_landmarks,
                                        hand_result.handedness):
        label = handedness_list[0].category_name   # "Left" o "Right"
        if label != "Left":
            continue

        pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
        pts -= pts[0].copy()
        scale = np.linalg.norm(pts[9])
        if scale > 1e-6:
            pts /= scale
        left = pts.flatten()
        left_valid = True
        break
    return left, left_valid


# ── Gestión del catálogo de señas ──────────────────────────────────────────────

def load_signs():
    if os.path.exists(SIGNS_FILE):
        with open(SIGNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_signs(signs: dict):
    with open(SIGNS_FILE, "w", encoding="utf-8") as f:
        json.dump(signs, f, ensure_ascii=False, indent=2)


def get_rep_count(sign_name: str) -> int:
    sign_dir = os.path.join(DATA_DIR, sign_name)
    if not os.path.isdir(sign_dir):
        return 0
    return len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])


# ── Dibujo ────────────────────────────────────────────────────────────────────

def put_text(frame, text, pos, size=0.7, color=C_WHITE, thickness=1, shadow=True):
    if shadow:
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, C_BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def draw_left_hand(frame, hand_result, w, h):
    for lm_list, handedness_list in zip(hand_result.hand_landmarks,
                                        hand_result.handedness):
        if handedness_list[0].category_name != "Left":
            continue
        pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lm_list)}
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
        for idx, (x, y) in pts.items():
            cv2.circle(frame, (x, y), 5, ZONE_COLOR.get(idx, C_WHITE), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, C_WHITE, 1, cv2.LINE_AA)
        break


def draw_detection_status(frame, left_ok):
    w = frame.shape[1]
    color = C_GREEN if left_ok else C_RED
    x0 = w - 115
    y = 65
    cv2.circle(frame, (x0, y), 6, color, -1)
    put_text(frame, "mano-I", (x0 + 12, y + 5), size=0.45, color=color, shadow=False)


def draw_hud(frame, state, sign_name, rep_done, reps_target,
             frame_idx, countdown_val, fps_actual):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), 1)

    label = sign_name if sign_name else "—"
    put_text(frame, f"Senal: {label}", (10, 32), size=0.8, color=C_GREEN, thickness=2)
    put_text(frame, f"Rep: {rep_done}/{reps_target}", (w // 2 - 60, 32), size=0.75, color=C_WHITE)
    put_text(frame, f"{fps_actual:.0f} fps", (w - 90, 32), size=0.6, color=C_GRAY)

    if state == AppState.IDLE:
        put_text(frame, "LISTO  |  [ESPACIO] grabar  [N] nueva sena  [Q] salir",
                 (10, h - 15), size=0.5, color=C_GRAY)

    elif state == AppState.COUNTDOWN:
        num_txt = str(int(countdown_val) + 1)
        scale = 4.0
        (tw, th), _ = cv2.getTextSize(num_txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 6)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        put_text(frame, num_txt, (cx, cy), size=scale, color=C_YELLOW, thickness=6)
        put_text(frame, "Preparate!", (w // 2 - 70, cy + 60), size=0.8, color=C_YELLOW)

    elif state == AppState.RECORDING:
        progress = frame_idx / TOTAL_FRAMES
        bar_x, bar_y, bar_w, bar_h = 20, h - 40, w - 40, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h), C_RED, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), C_WHITE, 1)
        remaining = (TOTAL_FRAMES - frame_idx) / FPS_TARGET
        put_text(frame, f"GRABANDO  {remaining:.1f}s",
                 (bar_x, bar_y - 10), size=0.6, color=C_RED, thickness=2)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (w - 25, h - 32), 7, C_RED, -1)

    elif state == AppState.SAVED:
        put_text(frame, f"Guardada  (rep {rep_done}/{reps_target})",
                 (w // 2 - 130, h // 2), size=1.0, color=C_GREEN, thickness=2)


# ── Menú de consola ────────────────────────────────────────────────────────────

def console_menu(signs: dict):
    print("\n" + "═" * 50)
    print("  GESTOR DE SENAS")
    print("═" * 50)
    if signs:
        print("\nSenas registradas:")
        for i, (name, meta) in enumerate(signs.items(), 1):
            reps = get_rep_count(name)
            target = meta.get("reps_target", REPS_DEFAULT)
            print(f"  {i:2}. {name:<20}  {reps}/{target} reps")
    else:
        print("  (Sin senas aun)")
    print("\n  [1] Seleccionar sena existente")
    print("  [2] Crear nueva sena")
    print("  [3] Eliminar sena")
    print("  [0] Salir del programa")
    print()
    return input("Elige opcion: ").strip()


def select_sign(signs: dict):
    if not signs:
        print("No hay senas registradas. Crea una primero.")
        return None, None
    names = list(signs.keys())
    for i, n in enumerate(names, 1):
        reps = get_rep_count(n)
        target = signs[n].get("reps_target", REPS_DEFAULT)
        print(f"  {i}. {n}  ({reps}/{target} reps)")
    idx = input("Numero de sena: ").strip()
    try:
        chosen = names[int(idx) - 1]
        return chosen, signs[chosen]
    except (ValueError, IndexError):
        print("Opcion invalida.")
        return None, None


def create_sign(signs: dict):
    name = input("Nombre de la sena (ej. 'hola', 'gracias'): ").strip().lower()
    if not name:
        return None, None
    if name in signs:
        print(f"'{name}' ya existe.")
        return name, signs[name]
    try:
        reps = int(input(f"Cuantas repeticiones? (Enter = {REPS_DEFAULT}): ").strip() or REPS_DEFAULT)
    except ValueError:
        reps = REPS_DEFAULT
    signs[name] = {"reps_target": reps, "description": ""}
    os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)
    save_signs(signs)
    print(f"Sena '{name}' creada con {reps} repeticiones objetivo.")
    return name, signs[name]


def delete_sign(signs: dict):
    if not signs:
        return
    names = list(signs.keys())
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    idx = input("Numero a eliminar: ").strip()
    try:
        chosen = names[int(idx) - 1]
    except (ValueError, IndexError):
        print("Opcion invalida.")
        return
    confirm = input(f"Eliminar '{chosen}' y todos sus datos? (s/N): ").strip().lower()
    if confirm == "s":
        import shutil
        sign_dir = os.path.join(DATA_DIR, chosen)
        if os.path.isdir(sign_dir):
            shutil.rmtree(sign_dir)
        del signs[chosen]
        save_signs(signs)
        print(f"'{chosen}' eliminada.")


# ── Loop principal de captura ──────────────────────────────────────────────────

def capture_loop(sign_name: str, meta: dict):
    reps_target = meta.get("reps_target", REPS_DEFAULT)
    reps_done   = get_rep_count(sign_name)
    sign_dir    = os.path.join(DATA_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    hand_det = create_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se puede abrir la camara.")
        hand_det.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    state           = AppState.IDLE
    frame_buffer    = []
    countdown_start = 0.0
    save_flash_end  = 0.0
    prev_time       = time.time()
    fps_actual      = 30.0
    start_ns        = time.monotonic_ns()

    print(f"\n── Grabando: '{sign_name}'  ({reps_done}/{reps_target} reps) ──")
    print(f"   Features/frame: {FEATURE_DIM}  (mano-I={HAND_DIM})")
    print("   [ESPACIO] iniciar repeticion   [Q] volver al menu\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        fps_actual = 0.9 * fps_actual + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time  = now

        # ── Deteccion con Tasks API (imagen reducida para mayor velocidad) ──
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small        = cv2.resize(rgb, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000

        hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)

        # ── Extraer features (63,) solo mano izquierda ──
        left_feat, left_ok = extract_left_hand(hand_result)
        keypoints          = left_feat

        # ── Dibujar ──
        draw_left_hand(frame, hand_result, w, h)
        draw_detection_status(frame, left_ok)

        # ── Maquina de estados ──
        if state == AppState.COUNTDOWN:
            elapsed   = now - countdown_start
            remaining = COUNTDOWN_SEC - elapsed
            if remaining <= 0:
                state        = AppState.RECORDING
                frame_buffer = []
            else:
                draw_hud(frame, state, sign_name, reps_done, reps_target,
                         0, remaining, fps_actual)

        elif state == AppState.RECORDING:
            frame_buffer.append(keypoints)

            if len(frame_buffer) >= TOTAL_FRAMES:
                seq      = np.array(frame_buffer[:TOTAL_FRAMES], dtype=np.float32)
                rep_path = os.path.join(sign_dir, f"rep_{reps_done:03d}.npy")
                np.save(rep_path, seq)
                reps_done += 1
                print(f"  Rep {reps_done}/{reps_target} guardada -> {rep_path}  shape={seq.shape}")

                state          = AppState.SAVED
                save_flash_end = now + 1.0

                if reps_done >= reps_target:
                    print(f"\nCompletadas {reps_target} reps para '{sign_name}'!")

        elif state == AppState.SAVED:
            if now >= save_flash_end:
                state = AppState.IDLE

        if state != AppState.COUNTDOWN:
            draw_hud(frame, state, sign_name, reps_done, reps_target,
                     len(frame_buffer), 0, fps_actual)

        cv2.imshow("Lengua de Senas - Recolector de Datos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and state == AppState.IDLE:
            if reps_done >= reps_target:
                print("  Ya se completaron todas las repeticiones para esta sena.")
            else:
                state           = AppState.COUNTDOWN
                countdown_start = now
        elif key == ord('n'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()


# ── Punto de entrada ───────────────────────────────────────────────────────────

def main():
    ensure_models()
    os.makedirs(DATA_DIR, exist_ok=True)
    signs = load_signs()

    while True:
        option = console_menu(signs)

        if option == "1":
            sign, meta = select_sign(signs)
            if sign:
                capture_loop(sign, meta)

        elif option == "2":
            sign, meta = create_sign(signs)
            if sign:
                start = input(f"Empezar a grabar '{sign}' ahora? (S/n): ").strip().lower()
                if start != "n":
                    capture_loop(sign, meta)

        elif option == "3":
            delete_sign(signs)
            signs = load_signs()

        elif option == "0":
            print("Hasta luego.")
            sys.exit(0)

        else:
            print("Opcion no valida.")


if __name__ == "__main__":
    main()
