"""
========================================================
  Clasificador en Tiempo Real -- Lengua de Senas
  -------------------------------------------------------
  Dependencias:
      pip install tensorflow opencv-python mediapipe numpy

  Uso:
      python sign_classifier.py

  Requiere haber ejecutado primero:
      python train_model.py
========================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import json
import time
import math
import collections
import sys
import os
import urllib.request
import tensorflow as tf

# ── Modelos MediaPipe (Tasks API) ─────────────────────────────────────────────
HAND_MODEL     = "hand_landmarker.task"
POSE_MODEL     = "pose_landmarker_lite.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# ── Indices de pose cintura-para-arriba ───────────────────────────────────────
POSE_UPPER_INDICES = list(range(11, 25))   # 14 landmarks

POSE_UPPER_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
]

HAND_CONNECTIONS = frozenset([
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
])

# ── Configuracion ─────────────────────────────────────────────────────────────
MODEL_PATH      = "model.keras"
LABEL_MAP_PATH  = "label_map.json"
TOTAL_FRAMES    = 90
FEATURE_DIM     = 168   # 42 pose + 63 mano-izq + 63 mano-der
INFERENCE_EVERY = 90
RESULT_SHOW_SEC = 2.5
CONF_THRESHOLD  = 0.55
DETECT_W        = 320
DETECT_H        = 240

# ── Paleta de colores (BGR) ───────────────────────────────────────────────────
BG       = (28, 20, 38)
TEAL     = (200, 230,  0)
PURPLE   = (255,  80, 180)
GREEN    = ( 80, 220, 100)
ORANGE   = (  0, 155, 255)
YELLOW   = (  0, 220, 220)
RED_SOFT = ( 60,  60, 220)
CYAN     = (220, 180,  0)
WHITE    = (248, 248, 252)
GRAY     = (120, 115, 140)
DIM      = ( 55,  50,  70)

ZONE_COLOR = {
    0:  WHITE,
    1:  PURPLE,  2: PURPLE,  3: PURPLE,  4: PURPLE,
    5:  GREEN,   6: GREEN,   7: GREEN,   8: GREEN,
    9:  TEAL,   10: TEAL,   11: TEAL,   12: TEAL,
   13:  YELLOW, 14: YELLOW, 15: YELLOW, 16: YELLOW,
   17:  ORANGE, 18: ORANGE, 19: ORANGE, 20: ORANGE,
}


def conf_color(c: float):
    if c >= 0.82:
        return GREEN
    elif c >= 0.65:
        return YELLOW
    else:
        return ORANGE


# ── Descarga de modelos ────────────────────────────────────────────────────────

def ensure_models():
    for path, url in [(HAND_MODEL, HAND_MODEL_URL), (POSE_MODEL, POSE_MODEL_URL)]:
        if not os.path.exists(path):
            print(f"Descargando {path}...", flush=True)
            urllib.request.urlretrieve(url, path)
            print(f"  OK: {path}")


def create_detectors():
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (mp_vision.HandLandmarker.create_from_options(hand_opts),
            mp_vision.PoseLandmarker.create_from_options(pose_opts))


# ── Extraccion de features (identica al collector) ────────────────────────────

def extract_pose_upper(pose_result):
    if not pose_result.pose_landmarks:
        return np.zeros(len(POSE_UPPER_INDICES) * 3, dtype=np.float32)
    lm = pose_result.pose_landmarks[0]
    pts = np.array(
        [[lm[i].x, lm[i].y, lm[i].z] for i in POSE_UPPER_INDICES],
        dtype=np.float32,
    )
    hip_mid = (pts[12] + pts[13]) / 2.0
    pts -= hip_mid
    shoulder_dist = np.linalg.norm(pts[0] - pts[1])
    if shoulder_dist > 1e-6:
        pts /= shoulder_dist
    return pts.flatten()


def extract_hands(hand_result):
    left  = np.zeros(21 * 3, dtype=np.float32)
    right = np.zeros(21 * 3, dtype=np.float32)
    for lm_list, handedness_list in zip(hand_result.hand_landmarks,
                                        hand_result.handedness):
        label = handedness_list[0].category_name
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
        pts -= pts[0].copy()
        scale = np.linalg.norm(pts[9])
        if scale > 1e-6:
            pts /= scale
        if label == "Left":
            left = pts.flatten()
        else:
            right = pts.flatten()
    return left, right


# ── Helpers de dibujo ─────────────────────────────────────────────────────────

def fill_rect(img, x1, y1, x2, y2, color, alpha):
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    solid = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(solid, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def fill_rounded(img, x1, y1, x2, y2, color, alpha, r=14):
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def stroke_rounded(img, x1, y1, x2, y2, color, r=14, thickness=1):
    r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.line(img, (x1+r, y1),  (x2-r, y1),  color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1+r, y2),  (x2-r, y2),  color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1+r),  (x1, y2-r),  color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1+r),  (x2, y2-r),  color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, thickness, cv2.LINE_AA)


def text(img, msg, x, y, size=0.65, color=WHITE, bold=False, shadow=True):
    thick = 2 if bold else 1
    if shadow:
        cv2.putText(img, msg, (x+1, y+1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def text_centered(img, msg, cy, size=0.8, color=WHITE, bold=False, shadow=True):
    thick = 2 if bold else 1
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, size, thick)
    w = img.shape[1]
    x = (w - tw) // 2
    y = cy + th // 2
    if shadow:
        cv2.putText(img, msg, (x+1, y+1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def draw_pose_upper(frame, pose_result, w, h):
    if not pose_result.pose_landmarks:
        return
    lm = pose_result.pose_landmarks[0]
    pts = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in POSE_UPPER_INDICES}
    for a, b in POSE_UPPER_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (160, 160, 160), 2, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        cv2.circle(frame, (x, y), 6, CYAN, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, WHITE, 1, cv2.LINE_AA)


def draw_all_hands(frame, hand_result, w, h):
    for lm_list in hand_result.hand_landmarks:
        pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lm_list)}
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], DIM, 4, cv2.LINE_AA)
            cv2.line(frame, pts[a], pts[b], (160, 155, 175), 1, cv2.LINE_AA)
        for idx, (x, y) in pts.items():
            color = ZONE_COLOR.get(idx, WHITE)
            cv2.circle(frame, (x, y), 9, (*color[:2], max(0, color[2]-40)), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 6, WHITE,  1, cv2.LINE_AA)


# ── Componentes de UI ─────────────────────────────────────────────────────────

def draw_top_bar(frame, fps, pose_ok, left_ok, right_ok, now):
    h, w = frame.shape[:2]
    fill_rect(frame, 0, 0, w, 54, BG, 0.88)

    pulse = 0.55 + 0.45 * math.sin(now * 3.5)
    dot_c = tuple(int(c * pulse) for c in GREEN)
    cv2.circle(frame, (18, 27), 7, dot_c, -1, cv2.LINE_AA)
    cv2.circle(frame, (18, 27), 7, GREEN, 1,  cv2.LINE_AA)
    text(frame, "LIVE", 30, 33, size=0.42, color=GREEN, shadow=False)

    text_centered(frame, "LENGUA DE SENAS", 28, size=0.72, color=WHITE, bold=True)

    # Indicadores: P=pose, I=mano-izq, D=mano-der
    indicators = [("P", pose_ok), ("I", left_ok), ("D", right_ok)]
    x0 = w - 18
    for label, ok in reversed(indicators):
        color = GREEN if ok else RED_SOFT
        cv2.circle(frame, (x0, 20), 6, color, -1, cv2.LINE_AA)
        text(frame, label, x0 - 5, 38, size=0.32, color=color, shadow=False)
        x0 -= 22

    text(frame, f"{fps:>3.0f} fps", x0 - 55, 33, size=0.48, color=GRAY, shadow=False)


def draw_bottom_bar(frame, buf_len, frame_count, last_inf_frame, now):
    h, w = frame.shape[:2]
    BAR_H = 52
    fill_rect(frame, 0, h - BAR_H, w, h, BG, 0.88)

    margin = 20
    bar_w  = w - 2 * margin
    bar_y  = h - 22
    bar_h  = 10

    if buf_len < TOTAL_FRAMES:
        progress   = buf_len / TOTAL_FRAMES
        label      = f"Cargando buffer...  {buf_len}/{TOTAL_FRAMES}"
        fill_color = TEAL
    else:
        frames_since = frame_count - last_inf_frame
        frames_left  = max(0, INFERENCE_EVERY - frames_since)
        progress     = 1.0 - frames_left / INFERENCE_EVERY
        secs_left    = frames_left / 30.0
        label        = f"Siguiente prediccion en  {secs_left:.1f} s"
        fill_color   = PURPLE

    cv2.rectangle(frame, (margin, bar_y), (margin + bar_w, bar_y + bar_h), (45, 40, 58), -1)
    filled = int(bar_w * progress)
    if filled > 0:
        cv2.rectangle(frame, (margin, bar_y),
                      (margin + filled, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(frame, (margin, bar_y), (margin + bar_w, bar_y + bar_h), GRAY, 1)
    text(frame, label, margin, bar_y - 6, size=0.44, color=GRAY, shadow=False)


def draw_prediction_card(frame, label, confidence, shown_at, now):
    h, w = frame.shape[:2]
    age  = now - shown_at
    fade = min(age / 0.25, 1.0)
    if fade <= 0:
        return

    c_col = conf_color(confidence)
    cw = min(380, w - 60)
    ch = 155
    cx = (w - cw) // 2
    cy = h // 2 - ch // 2 - 15

    fill_rounded(frame, cx+5, cy+5, cx+cw+5, cy+ch+5, (0,0,0), 0.45 * fade, r=16)
    fill_rounded(frame, cx, cy, cx+cw, cy+ch, BG, 0.92 * fade, r=16)
    if fade > 0.5:
        stroke_rounded(frame, cx, cy, cx+cw, cy+ch, c_col, r=16, thickness=2)

    text_centered(frame, label.upper(), cy + 68, size=1.55, color=c_col,
                  bold=True, shadow=True)

    bx = cx + 24
    by = cy + ch - 34
    bw2 = cw - 48
    bh2 = 9
    cv2.rectangle(frame, (bx, by), (bx+bw2, by+bh2), (50,45,62), -1)
    conf_px = int(bw2 * confidence)
    if conf_px > 0:
        cv2.rectangle(frame, (bx, by), (bx+conf_px, by+bh2), c_col, -1)
    cv2.rectangle(frame, (bx, by), (bx+bw2, by+bh2), GRAY, 1)

    pct_txt = f"{confidence*100:.0f}%"
    (tw, _), _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text(frame, pct_txt, bx + bw2 + 6, by + bh2, size=0.5, color=c_col, shadow=False)


def draw_no_detection(frame, now):
    h, w = frame.shape[:2]
    alpha = 0.5 + 0.45 * math.sin(now * 2.2)
    c = tuple(int(v * alpha) for v in ORANGE)
    text_centered(frame, "Muestre cintura para arriba", h // 2 + 50,
                  size=0.65, color=c, shadow=False)


def draw_debug_probs(frame, probs, label_map):
    h, w = frame.shape[:2]
    n = len(label_map)
    panel_w = 200
    row_h   = 26
    pad     = 10
    px      = pad
    py      = 60

    fill_rounded(frame, px - 4, py - 6,
                 px + panel_w + 4, py + n * row_h + 8, BG, 0.80, r=8)
    text(frame, "PROBABILIDADES", px, py + 8, size=0.38, color=GRAY, shadow=False)

    for i in range(n):
        lbl  = label_map[i]
        p    = float(probs[i])
        y    = py + 20 + i * row_h
        bw2  = panel_w - 60
        col  = conf_color(p)
        cv2.rectangle(frame, (px + 50, y), (px + 50 + bw2, y + 12), (45,40,58), -1)
        fw = int(bw2 * p)
        if fw > 0:
            cv2.rectangle(frame, (px + 50, y), (px + 50 + fw, y + 12), col, -1)
        cv2.rectangle(frame, (px + 50, y), (px + 50 + bw2, y + 12), GRAY, 1)
        text(frame, lbl,          px,              y + 10, size=0.42, color=WHITE, shadow=False)
        text(frame, f"{p*100:.0f}%", px + 50 + bw2 + 4, y + 10, size=0.40, color=col, shadow=False)


def draw_history(frame, history, now):
    if not history:
        return
    h, w = frame.shape[:2]
    recent = list(history)[-4:]
    px, py = w - 160, 60
    panel_h = 20 + len(recent) * 22
    fill_rounded(frame, px - 8, py - 10, w - 8, py + panel_h, BG, 0.72, r=10)
    text(frame, "HISTORIAL", px, py + 6, size=0.38, color=GRAY, shadow=False)
    for i, (lbl, conf) in enumerate(recent):
        y    = py + 26 + i * 21
        col  = conf_color(conf) if i == len(recent) - 1 else GRAY
        bold = (i == len(recent) - 1)
        text(frame, f"{lbl.upper()}  {conf*100:.0f}%", px, y,
             size=0.46, color=col, bold=bold, shadow=False)


# ── Loop principal ────────────────────────────────────────────────────────────

def run():
    ensure_models()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encontro '{MODEL_PATH}'.")
        print("Ejecuta primero: python train_model.py")
        sys.exit(1)

    print("Cargando modelo...", end=" ", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    print("listo.")
    print(f"Senas: {list(label_map.values())}")
    print("[Q] / [ESC] para salir  |  [D] debug probabilidades\n")

    # Warmup
    _ = model.predict(np.zeros((1, TOTAL_FRAMES, FEATURE_DIM), dtype=np.float32), verbose=0)

    hand_det, pose_det = create_detectors()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se puede abrir la camara.")
        hand_det.close(); pose_det.close()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_buffer   = collections.deque(maxlen=TOTAL_FRAMES)
    pred_history   = collections.deque(maxlen=8)
    current_label  = None
    current_conf   = 0.0
    last_probs     = np.ones(len(label_map), dtype=np.float32) / len(label_map)
    shown_at       = -999.0
    frame_count    = 0
    last_inf_frame = 0
    prev_time      = time.time()
    fps_smooth     = 30.0
    start_ns       = time.monotonic_ns()
    show_debug     = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time  = now

        # ── Deteccion con Tasks API (imagen reducida para mayor velocidad) ──
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small        = cv2.resize(rgb, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
        timestamp_ms = (time.monotonic_ns() - start_ns) // 1_000_000

        hand_result = hand_det.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_det.detect_for_video(mp_image, timestamp_ms)

        pose_ok  = bool(pose_result.pose_landmarks)
        left_ok  = any(h[0].category_name == "Left"  for h in hand_result.handedness)
        right_ok = any(h[0].category_name == "Right" for h in hand_result.handedness)

        # ── Dibujar ──
        draw_pose_upper(frame, pose_result, w, h)
        draw_all_hands(frame, hand_result, w, h)

        # ── Extraer features (168,) ──
        pose_feat        = extract_pose_upper(pose_result)
        left_feat, right_feat = extract_hands(hand_result)
        keypoints        = np.concatenate([pose_feat, left_feat, right_feat])

        frame_buffer.append(keypoints)
        frame_count += 1

        # ── Inferencia cada INFERENCE_EVERY frames ──
        if (len(frame_buffer) == TOTAL_FRAMES
                and frame_count - last_inf_frame >= INFERENCE_EVERY):
            seq   = np.array(frame_buffer, dtype=np.float32)[np.newaxis]
            probs = model.predict(seq, verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            last_inf_frame = frame_count
            last_probs     = probs

            prob_str = "  ".join(
                f"{label_map[i]}={probs[i]*100:.0f}%" for i in range(len(label_map))
            )
            print(f"[pred] {prob_str}  -> {label_map[idx]} ({conf*100:.0f}%)")

            if conf >= CONF_THRESHOLD:
                current_label = label_map[idx]
                current_conf  = conf
                shown_at      = now
                pred_history.append((current_label, current_conf))

        # ── UI ──
        if not pose_ok:
            draw_no_detection(frame, now)

        if current_label and (now - shown_at) < RESULT_SHOW_SEC:
            draw_prediction_card(frame, current_label, current_conf, shown_at, now)

        if show_debug:
            draw_debug_probs(frame, last_probs, label_map)
        draw_history(frame, pred_history, now)
        draw_top_bar(frame, fps_smooth, pose_ok, left_ok, right_ok, now)
        draw_bottom_bar(frame, len(frame_buffer), frame_count, last_inf_frame, now)

        cv2.imshow("Sign Language Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    hand_det.close()
    pose_det.close()


if __name__ == "__main__":
    run()
