// ============================================================================
//  LSM · Clasificador en vivo (browser)
//  Pipeline:
//    MediaPipe Tasks (Hand + Pose, WASM) → features (168) → ONNX Runtime Web
//  Las constantes y normalizaciones DEBEN coincidir con sign_classifier.py
// ============================================================================

import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// onnxruntime-web (UMD vía CDN, expone window.ort)
await new Promise((resolve, reject) => {
  const s = document.createElement("script");
  s.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
  s.onload = resolve;
  s.onerror = reject;
  document.head.appendChild(s);
});
const ort = window.ort;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

// ── Config (espejo de sign_classifier.py) ──────────────────────────────────
const TOTAL_FRAMES   = 90;
const INFERENCE_EVERY = 90;
const MIN_HAND_RATIO  = 0.40;
const MIN_CONFIDENCE  = 0.70;
const MIN_MARGIN      = 0.20;
// 0.30 es más permisivo que el 0.18 original — manos válidas con brazo
// extendido quedaban fuera. Aún descarta manos de personas detrás.
const HAND_OWNER_MAX_DIST = 0.30;
const POSE_WRIST_LEFT  = 15;
const POSE_WRIST_RIGHT = 16;
const DETECT_W = 320;
const DETECT_H = 240;
const POSE_UPPER_INDICES = [11,12,13,14,15,16,17,18,19,20,21,22,23,24];
const POSE_UPPER_CONNECTIONS = [
  [11,12], [11,13], [13,15], [12,14], [14,16],
  [11,23], [12,24], [23,24],
];
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

// Dos modelos en el mismo front. OJO: tienen FEATURE_DIM distinto —
//   huespedes = 168 (pose 42 + mano-izq 63 + mano-der 63)
//   numeros   = 63  (SOLO mano izquierda; sin pose ni filtro de dueño)
// El pipeline de captura/normalización es el mismo; lo que cambia es qué
// landmarks se arman en el vector de features (ver buildKeypoints).
const MODELS = {
  huespedes: { onnx: "./models/huespedes.onnx", labels: "./models/label_map.json",         featureDim: 168 },
  numeros:   { onnx: "./models/numeros.onnx",   labels: "./models/numeros_label_map.json", featureDim: 63  },
  meses:     { onnx: "./models/meses.onnx",     labels: "./models/meses_label_map.json",   featureDim: 63  },
  dias:      { onnx: "./models/dias.onnx",      labels: "./models/dias_label_map.json",    featureDim: 63  },
  genero:    { onnx: "./models/genero.onnx",    labels: "./models/genero_label_map.json",  featureDim: 63  },
};
const GENERO_ORDER = ["femenino", "masculino"];
// Solo huespedes usa 168 features (pose + 2 manos); el resto usa 63 (solo mano izq).
const MONTH_NUM = {
  enero: 1, febrero: 2, marzo: 3, abril: 4, mayo: 5, junio: 6,
  julio: 7, agosto: 8, septiembre: 9, octubre: 10, noviembre: 11, diciembre: 12,
};
const MONTHS_ORDER = Object.keys(MONTH_NUM);
// Orden lunes-primero para la grilla; el cruce con la fecha real usa JS_WEEKDAY.
const WEEKDAYS_ORDER = ["lunes","martes","miercoles","jueves","viernes","sabado","domingo"];
// Mapeo de Date.getDay() (0=domingo) → etiqueta del modelo (sin acentos).
const JS_WEEKDAY = ["domingo","lunes","martes","miercoles","jueves","viernes","sabado"];
const RESERVA_MAX_DIAS = 30;
const DEFAULT_MODE = "huespedes";
const EMPTY_POSE   = { landmarks: [] };   // placeholder cuando no corremos pose

// ── DOM ────────────────────────────────────────────────────────────────────
const video      = document.getElementById("webcam");
const canvas     = document.getElementById("overlay");
const ctx        = canvas.getContext("2d");
const splash     = document.getElementById("splash");
const splashMsg  = document.getElementById("splash-msg");
const fpsEl      = document.getElementById("status-fps");
const chipPose   = document.getElementById("status-pose");
const chipLeft   = document.getElementById("status-left");
const chipRight  = document.getElementById("status-right");
const predLabel  = document.getElementById("pred-label");
const predDetail = document.getElementById("pred-detail");
const confFill   = document.getElementById("conf-fill");
const textArea   = document.getElementById("text-output");
const btnSpace   = document.getElementById("btn-space");
const btnBack    = document.getElementById("btn-backspace");
const btnClear   = document.getElementById("btn-clear");
const autoToggle = document.getElementById("auto-append");
const probsEl    = document.getElementById("probs");
const historyEl  = document.getElementById("history");
const formEl     = document.getElementById("demo-form");
const formOut    = document.getElementById("form-output");
const btnSubmit  = document.getElementById("btn-submit");
const guestsPending = document.getElementById("guests-pending");
const guestsList    = document.getElementById("guests-list");
const guestsError   = document.getElementById("guests-error");
const btnGuestUndo  = document.getElementById("btn-guest-undo");
const btnGuestClear = document.getElementById("btn-guest-clear");
const modelSelect   = document.getElementById("model-select");
const cardHuespedes = document.getElementById("card-huespedes");
const cardNumeros   = document.getElementById("card-numeros");
const numForm       = document.getElementById("num-form");
const numOut        = document.getElementById("num-output");
const btnNumBack    = document.getElementById("btn-num-back");
const btnNumClear   = document.getElementById("btn-num-clear");
const btnNumSubmit  = document.getElementById("btn-num-submit");
const hintTelefono  = document.getElementById("hint-telefono");
const hintDia       = document.getElementById("hint-dia");
const hintAnio      = document.getElementById("hint-anio");
const cardMeses     = document.getElementById("card-meses");
const mesInput      = document.getElementById("mes-input");
const mesOut        = document.getElementById("mes-output");
const monthsGrid    = document.getElementById("months-grid");
const btnMesClear   = document.getElementById("btn-mes-clear");
const cardDias      = document.getElementById("card-dias");
const diasGrid      = document.getElementById("dias-grid");
const btnDiaClear   = document.getElementById("btn-dia-clear");
const cardReserva   = document.getElementById("card-reserva");
const resDiaInput   = document.getElementById("res-dia");
const resMesEl      = document.getElementById("res-mes");
const resSemanaEl   = document.getElementById("res-semana");
const reservaOut    = document.getElementById("reserva-output");
const btnReservaClear = document.getElementById("btn-reserva-clear");
const cardGenero    = document.getElementById("card-genero");
const generoGrid    = document.getElementById("genero-grid");
const generoOut     = document.getElementById("genero-output");
const btnGeneroClear = document.getElementById("btn-genero-clear");

// ── Estado ─────────────────────────────────────────────────────────────────
let session, labelMap, handLM, poseLM;
let activeMode  = DEFAULT_MODE;          // "huespedes" | "numeros"
let featureDim  = MODELS[DEFAULT_MODE].featureDim;
const sessionCache = {};                 // mode → InferenceSession (cargado on-demand)
const labelCache   = {};                 // mode → label_map
// Canvas oculto donde pintamos el video ESPEJADO a baja resolución antes de
// mandarlo a MediaPipe. Esto replica exactamente cv2.flip(frame, 1) + resize
// que hace sign_classifier.py — sin esto la asignación Left/Right de las
// manos queda invertida vs los datos de entrenamiento y el modelo no detecta.
const detectCanvas = document.createElement("canvas");
detectCanvas.width  = DETECT_W;
detectCanvas.height = DETECT_H;
const detectCtx = detectCanvas.getContext("2d", { willReadFrequently: true });
const frameBuffer  = [];      // últimos TOTAL_FRAMES Float32Array(168)
const handPresence = [];      // 1 si alguna mano válida ese frame
let frameCount     = 0;
let lastInfFrame   = 0;
let currentLabel   = null;
let currentConf    = 0;
let lastProbs      = null;
let lastReject     = "";
let predHistory    = [];
let lastT          = performance.now();
let fpsSmooth      = 30;
let activeField    = null;    // input del formulario actualmente seleccionado
let pendingCount   = null;    // número firmado a la espera de un tipo (adulto/nino)
let guests         = [];      // [{ cantidad, tipo }] grupos confirmados
// Reservación: se llena combinando los modos números (día), meses y días.
let reserva        = { dia: null, mes: null, mesNum: null, diaSemana: null };
let inferenceBusy  = false;   // true mientras se ejecuta runInference en background
let mpMsSmooth     = 0;       // ms promedio que tarda MediaPipe (hand+pose)
let ortMsLast      = 0;       // ms que tardó la última inferencia ONNX

// ============================================================================
//  Init
// ============================================================================

async function init() {
  splashMsg.textContent = "Cargando ONNX…";
  await loadModel(DEFAULT_MODE);
  buildProbRows();

  splashMsg.textContent = "Cargando MediaPipe…";
  await loadMediapipe();

  splashMsg.textContent = "Solicitando cámara…";
  await startWebcam();

  splash.classList.add("hidden");
  setTimeout(() => splash.remove(), 400);

  hookUI();
  requestAnimationFrame(loop);
}

// Carga (y cachea) el modelo del modo dado, lo deja como activo y reinicia el
// buffer porque el FEATURE_DIM puede cambiar entre modelos.
async function loadModel(mode) {
  const cfg = MODELS[mode];
  if (!sessionCache[mode]) {
    const [sess, labels] = await Promise.all([
      ort.InferenceSession.create(cfg.onnx, { executionProviders: ["wasm"] }),
      fetch(cfg.labels).then(r => r.json()),
    ]);
    const dummy = new ort.Tensor("float32",
      new Float32Array(TOTAL_FRAMES * cfg.featureDim),
      [1, TOTAL_FRAMES, cfg.featureDim]);
    await sess.run({ [sess.inputNames[0]]: dummy });
    sessionCache[mode] = sess;
    labelCache[mode]   = labels;
  }
  session    = sessionCache[mode];
  labelMap   = labelCache[mode];
  featureDim = cfg.featureDim;
  activeMode = mode;
  // El vector de features cambia de tamaño → el buffer viejo es inválido.
  frameBuffer.length  = 0;
  handPresence.length = 0;
  lastInfFrame        = frameCount;
  lastProbs           = null;
}

async function switchModel(mode) {
  if (mode === activeMode && sessionCache[mode]) return;
  predDetail.textContent = `cargando modelo «${mode}»…`;
  await loadModel(mode);
  buildProbRows();
  cardHuespedes.style.display = mode === "huespedes" ? "" : "none";
  cardNumeros.style.display   = mode === "numeros"   ? "" : "none";
  cardMeses.style.display     = mode === "meses"     ? "" : "none";
  cardDias.style.display      = mode === "dias"      ? "" : "none";
  cardGenero.style.display    = mode === "genero"    ? "" : "none";
  // El panel de reservación solo se muestra en los modos que lo alimentan.
  cardReserva.style.display   = ["numeros","meses","dias"].includes(mode) ? "" : "none";
  if (activeField) { activeField.removeAttribute("data-active"); activeField = null; }
  showReject(`modelo «${mode}» listo`);
}

async function loadMediapipe() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLM = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  poseLM = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  await video.play();
  // El overlay usa las dims del detectCanvas (espacio donde MediaPipe da
  // las coords normalizadas). El CSS lo estira al tamaño visual.
  canvas.width  = DETECT_W;
  canvas.height = DETECT_H;
  console.log(`[webcam] resolución real: ${video.videoWidth}×${video.videoHeight}`);
}

// ============================================================================
//  Loop principal
// ============================================================================

function loop() {
  const now = performance.now();
  const dt  = Math.max((now - lastT) / 1000, 1e-3);
  lastT = now;
  fpsSmooth = 0.9 * fpsSmooth + 0.1 * (1 / dt);

  if (video.readyState >= 2) {
    // Timestamps deben ser estrictamente crecientes; performance.now() puede
    // repetirse entre frames consecutivos.
    const ts = Math.max(Math.round(now), frameCount + 1);

    // Espejar + recortar a 4:3 + reducir antes de mandarlo a MediaPipe.
    // El crop 4:3 es crítico: si la cámara da 16:9 (1280×720) y estiramos
    // a 320×240, la persona se comprime horizontalmente y los features
    // normalizados ya no coinciden con los del entrenamiento.
    const vw = video.videoWidth, vh = video.videoHeight;
    const targetAspect = DETECT_W / DETECT_H;       // 4/3
    const videoAspect  = vw / vh;
    let sx, sy, sw, sh;
    if (videoAspect > targetAspect) {
      sh = vh; sw = vh * targetAspect;
      sx = (vw - sw) / 2; sy = 0;
    } else {
      sw = vw; sh = vw / targetAspect;
      sx = 0; sy = (vh - sh) / 2;
    }
    detectCtx.save();
    detectCtx.scale(-1, 1);
    detectCtx.drawImage(video, sx, sy, sw, sh, -DETECT_W, 0, DETECT_W, DETECT_H);
    detectCtx.restore();

    const t0 = performance.now();
    const handRes = handLM.detectForVideo(detectCanvas, ts);
    // En modo números NO corremos pose: el modelo solo usa la mano izquierda y
    // el detector de pose añade ~10-30ms/frame que bajan los FPS. Como el LSTM
    // toma 90 frames consecutivos sin normalizar el tiempo, ese costo estira el
    // gesto vs los datos de números (capturados solo-mano). Esto replica
    // exactamente el pipeline de sign_classifier.py en su versión de números.
    const poseRes = activeMode === "huespedes"
      ? poseLM.detectForVideo(detectCanvas, ts)
      : EMPTY_POSE;
    const mpMs = performance.now() - t0;
    mpMsSmooth = 0.9 * mpMsSmooth + 0.1 * mpMs;

    const poseOk = poseRes.landmarks && poseRes.landmarks.length > 0;
    const { keypoints, leftValid, rightValid } = buildKeypoints(handRes, poseRes);

    frameBuffer.push(keypoints);
    if (frameBuffer.length > TOTAL_FRAMES) frameBuffer.shift();
    handPresence.push(leftValid || rightValid ? 1 : 0);
    if (handPresence.length > TOTAL_FRAMES) handPresence.shift();
    frameCount++;

    drawOverlay(handRes, poseRes);
    // Fuera de huéspedes solo importa la mano izquierda (no hay pose).
    setChip(chipPose,  activeMode === "huespedes" ? poseOk : false);
    setChip(chipLeft,  leftValid);
    setChip(chipRight, activeMode === "huespedes" ? rightValid : false);

    // Inferencia ONNX en background — NO se hace await, así que el loop
    // sigue capturando frames mientras corre. inferenceBusy evita lanzar
    // dos a la vez.
    if (!inferenceBusy &&
        frameBuffer.length === TOTAL_FRAMES &&
        frameCount - lastInfFrame >= INFERENCE_EVERY) {
      lastInfFrame = frameCount;
      inferenceBusy = true;
      // Snapshot del buffer ahora (antes de que siga creciendo)
      const fd = featureDim;
      const snapshot = new Float32Array(TOTAL_FRAMES * fd);
      for (let i = 0; i < TOTAL_FRAMES; i++) snapshot.set(frameBuffer[i], i * fd);
      const handRatio = handPresence.reduce((a,b)=>a+b,0) / TOTAL_FRAMES;
      runInference(snapshot, handRatio).finally(() => { inferenceBusy = false; });
    }
  }

  fpsEl.textContent = `${fpsSmooth.toFixed(0)} fps · mp ${mpMsSmooth.toFixed(0)}ms · ort ${ortMsLast.toFixed(0)}ms`;
  requestAnimationFrame(loop);
}

// ============================================================================
//  Features (idénticas a Python)
// ============================================================================

function extractPoseUpper(poseRes) {
  const out = new Float32Array(POSE_UPPER_INDICES.length * 3);
  if (!poseRes.landmarks || poseRes.landmarks.length === 0) return out;
  const lm = poseRes.landmarks[0];
  const pts = POSE_UPPER_INDICES.map(i => [lm[i].x, lm[i].y, lm[i].z]);
  // hip_mid = (pts[12] + pts[13]) / 2  →  índices locales de 23,24 son 12,13
  const hipMid = [
    (pts[12][0] + pts[13][0]) / 2,
    (pts[12][1] + pts[13][1]) / 2,
    (pts[12][2] + pts[13][2]) / 2,
  ];
  for (const p of pts) { p[0]-=hipMid[0]; p[1]-=hipMid[1]; p[2]-=hipMid[2]; }
  // shoulder_dist = dist(pts[0]=LM11, pts[1]=LM12)
  const sx = pts[0][0]-pts[1][0], sy = pts[0][1]-pts[1][1], sz = pts[0][2]-pts[1][2];
  const sd = Math.sqrt(sx*sx + sy*sy + sz*sz);
  if (sd > 1e-6) for (const p of pts) { p[0]/=sd; p[1]/=sd; p[2]/=sd; }
  for (let i = 0; i < pts.length; i++) {
    out[i*3]   = pts[i][0];
    out[i*3+1] = pts[i][1];
    out[i*3+2] = pts[i][2];
  }
  return out;
}

function poseWrist(poseRes, idx) {
  if (!poseRes.landmarks || poseRes.landmarks.length === 0) return null;
  const lm = poseRes.landmarks[0][idx];
  return [lm.x, lm.y];
}

function extractHands(handRes, poseRes) {
  const left  = new Float32Array(63);
  const right = new Float32Array(63);
  let leftValid = false, rightValid = false;

  if (!handRes.landmarks) return { left, right, leftValid, rightValid };

  const wL = poseWrist(poseRes, POSE_WRIST_LEFT);
  const wR = poseWrist(poseRes, POSE_WRIST_RIGHT);

  for (let h = 0; h < handRes.landmarks.length; h++) {
    const pts = handRes.landmarks[h];
    const handed = handRes.handedness[h][0].categoryName;  // "Left" o "Right"
    const wrist = [pts[0].x, pts[0].y];

    if (wL || wR) {
      const dL = wL ? Math.hypot(wrist[0]-wL[0], wrist[1]-wL[1]) : 1e9;
      const dR = wR ? Math.hypot(wrist[0]-wR[0], wrist[1]-wR[1]) : 1e9;
      if (Math.min(dL, dR) > HAND_OWNER_MAX_DIST) continue;  // mano huérfana
    }

    // Normalizar: centro=wrist, escala=dist(wrist→pts[9])
    const norm = pts.map(p => [p.x - pts[0].x, p.y - pts[0].y, p.z - pts[0].z]);
    const s = Math.hypot(norm[9][0], norm[9][1], norm[9][2]);
    if (s > 1e-6) for (const p of norm) { p[0]/=s; p[1]/=s; p[2]/=s; }

    const flat = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      flat[i*3]   = norm[i][0];
      flat[i*3+1] = norm[i][1];
      flat[i*3+2] = norm[i][2];
    }
    if (handed === "Left") { left.set(flat);  leftValid  = true; }
    else                   { right.set(flat); rightValid = true; }
  }
  return { left, right, leftValid, rightValid };
}

// Modo números: SOLO la mano izquierda, 63 features, sin pose ni filtro de
// dueño. Espejo de extract_left_hand() en sign_classifier.py.
function extractLeftHand(handRes) {
  const left = new Float32Array(63);
  let leftValid = false;
  if (!handRes.landmarks) return { left, leftValid };

  for (let h = 0; h < handRes.landmarks.length; h++) {
    if (handRes.handedness[h][0].categoryName !== "Left") continue;
    const pts = handRes.landmarks[h];
    const norm = pts.map(p => [p.x - pts[0].x, p.y - pts[0].y, p.z - pts[0].z]);
    const s = Math.hypot(norm[9][0], norm[9][1], norm[9][2]);
    if (s > 1e-6) for (const p of norm) { p[0]/=s; p[1]/=s; p[2]/=s; }
    for (let i = 0; i < 21; i++) {
      left[i*3]   = norm[i][0];
      left[i*3+1] = norm[i][1];
      left[i*3+2] = norm[i][2];
    }
    leftValid = true;
    break;
  }
  return { left, leftValid };
}

// Arma el vector de features según el modelo activo.
function buildKeypoints(handRes, poseRes) {
  // numeros y meses: 63 features, solo mano izquierda, sin pose.
  if (activeMode !== "huespedes") {
    const { left, leftValid } = extractLeftHand(handRes);
    return { keypoints: left, leftValid, rightValid: false };
  }
  // huespedes: 168 = pose(42) + mano-izq(63) + mano-der(63)
  const { left, right, leftValid, rightValid } = extractHands(handRes, poseRes);
  const poseFeat = extractPoseUpper(poseRes);
  const keypoints = new Float32Array(168);
  keypoints.set(poseFeat, 0);
  keypoints.set(left, 42);
  keypoints.set(right, 105);
  return { keypoints, leftValid, rightValid };
}

// ============================================================================
//  Inferencia + rechazo
// ============================================================================

async function runInference(snapshot, handRatio) {
  if (handRatio < MIN_HAND_RATIO) {
    lastProbs = null;
    showReject(`sin manos (${(handRatio*100|0)}% del buffer)`);
    return;
  }

  // El feature dim se deriva del snapshot (no del global) por si el modelo
  // se cambió mientras esta inferencia estaba en vuelo.
  const fd = snapshot.length / TOTAL_FRAMES;
  const tensor = new ort.Tensor("float32", snapshot, [1, TOTAL_FRAMES, fd]);
  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: tensor });
  ortMsLast = performance.now() - t0;
  const probs = out[session.outputNames[0]].data;
  lastProbs = probs;
  updateProbRows(probs);

  // top-1 y top-2
  let i1 = 0, p1 = probs[0], p2 = -1;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > p1) { p2 = p1; p1 = probs[i]; i1 = i; }
    else if (probs[i] > p2) { p2 = probs[i]; }
  }
  if (p2 < 0) p2 = 0;
  const margin = p1 - p2;
  const labelStr = labelMap[i1];

  if (labelStr === undefined) {
    console.warn(`[pred] modelo dio idx=${i1} pero label_map solo tiene ${Object.keys(labelMap).length} entradas. ` +
                 `Probablemente label_map.json está cacheado y desactualizado — Ctrl+Shift+R para forzar recarga.`);
    showReject(`label_map desactualizado (idx=${i1} no existe)`);
    return;
  }

  console.log(`[pred] ${labelStr} conf=${(p1*100).toFixed(0)}% margin=${(margin*100).toFixed(0)}%`);

  if (p1 < MIN_CONFIDENCE) {
    showReject(`baja confianza (${(p1*100|0)}%)`);
  } else if (margin < MIN_MARGIN) {
    showReject(`red insegura (margen ${(margin*100|0)}%)`);
  } else {
    acceptPrediction(labelStr, p1);
  }
}

function acceptPrediction(label, conf) {
  currentLabel = label;
  currentConf  = conf;
  lastReject   = "";
  predLabel.textContent = label.toUpperCase();
  predLabel.classList.remove("weak");
  predDetail.textContent = `confianza ${(conf*100).toFixed(0)}%`;
  confFill.style.width = `${(conf*100).toFixed(0)}%`;

  predHistory.unshift({ label, conf });
  if (predHistory.length > 8) predHistory.pop();
  renderHistory();

  if (!autoToggle.checked) return;

  if (activeMode === "numeros") {
    // Los dígitos van al campo seleccionado (teléfono / día / año).
    if (activeField) appendText(label);
    else predDetail.textContent = "Selecciona un campo (teléfono, día o año)";
    return;
  }
  if (activeMode === "meses") {
    setMonth(label);   // el mes detectado llena el campo automáticamente
    return;
  }
  if (activeMode === "dias") {
    setDiaSemana(label);
    return;
  }
  if (activeMode === "genero") {
    setGenero(label);
    return;
  }
  // Huéspedes: si hay campo enfocado escribe ahí; si no, gramática número→tipo.
  if (activeField) appendText(label);
  else handleGuestSign(label);
}

function showReject(reason) {
  lastReject   = reason;
  currentLabel = null;
  predLabel.textContent = "—";
  predLabel.classList.add("weak");
  predDetail.textContent = reason;
  confFill.style.width = "0%";
}

// ============================================================================
//  Texto / formulario
// ============================================================================

function appendText(s) {
  if (activeField) {
    // Respeta maxlength (los campos numéricos lo usan; nombre no lo define).
    const max = activeField.maxLength;
    if (max > 0 && activeField.value.length >= max) return;
    activeField.value += s;
    activeField.dispatchEvent(new Event("input"));
  } else {
    textArea.value += s;
  }
}

function backspace() {
  const target = activeField ?? textArea;
  target.value = target.value.slice(0, -1);
  if (activeField) activeField.dispatchEvent(new Event("input"));
}

function hookUI() {
  btnSpace.onclick = () => appendText(" ");
  btnBack.onclick  = () => backspace();
  btnClear.onclick = () => {
    if (activeField) activeField.value = "";
    else textArea.value = "";
  };
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.code === "Space")     { appendText(" "); e.preventDefault(); }
    if (e.code === "Backspace") { backspace();     e.preventDefault(); }
    if (e.key.toLowerCase() === "d") {
      probsEl.parentElement.style.display =
        probsEl.parentElement.style.display === "none" ? "" : "none";
    }
  });

  for (const inp of document.querySelectorAll("input[data-fillable]")) {
    inp.addEventListener("focus", () => {
      if (activeField) activeField.removeAttribute("data-active");
      activeField = inp;
      inp.setAttribute("data-active", "true");
    });
    inp.addEventListener("blur", () => {
      // pequeño delay para no perder el focus al hacer click en un botón
      setTimeout(() => {
        if (document.activeElement !== inp) {
          inp.removeAttribute("data-active");
          if (activeField === inp) activeField = null;
        }
      }, 100);
    });
  }
  btnGuestUndo.onclick = () => {
    // Primero deshace el número pendiente; si no hay, quita el último grupo.
    if (pendingCount != null) pendingCount = null;
    else guests.pop();
    guestsError.textContent = "";
    renderGuests();
  };
  btnGuestClear.onclick = () => {
    guests = [];
    pendingCount = null;
    guestsError.textContent = "";
    renderGuests();
  };
  btnSubmit.onclick = submitGuests;
  renderGuests();

  // ── Selector de modelo + formulario de números ──
  modelSelect.onchange = () => switchModel(modelSelect.value);

  const numFields = [
    [numForm.telefono, hintTelefono, validateTelefono],
    [numForm.dia,      hintDia,      validateDia],
    [numForm.anio,     hintAnio,     validateAnio],
  ];
  for (const [inp, hintEl, fn] of numFields) {
    inp.addEventListener("input", () => applyHint(inp, hintEl, fn(inp.value)));
  }
  btnNumBack.onclick  = () => backspace();
  btnNumClear.onclick = () => {
    if (activeField) { activeField.value = ""; activeField.dispatchEvent(new Event("input")); }
  };
  btnNumSubmit.onclick = submitNumeros;

  // ── Formulario de meses ──
  buildMonthsGrid();
  btnMesClear.onclick = () => {
    mesInput.value = "";
    mesOut.textContent = "";
    highlightMonth(null);
  };

  // ── Días + panel de reservación ──
  buildWeekdayGrid();
  btnDiaClear.onclick = () => {
    reserva.diaSemana = null;
    highlightWeekday(null);
    updateReservaSummary();
    recomputeReserva();
  };
  resDiaInput.addEventListener("input", () => {
    const n = parseInt(resDiaInput.value, 10);
    reserva.dia = Number.isInteger(n) ? n : null;
    recomputeReserva();
  });
  btnReservaClear.onclick = () => {
    reserva = { dia: null, mes: null, mesNum: null, diaSemana: null };
    resDiaInput.value = "";
    highlightWeekday(null);
    highlightMonth(null);
    updateReservaSummary();
    reservaOut.textContent = "";
  };

  // ── Género ──
  buildGeneroGrid();
  btnGeneroClear.onclick = () => {
    highlightGenero(null);
    generoOut.textContent = "";
  };
}

// ── Género (femenino / masculino) ────────────────────────────────────────────

function buildGeneroGrid() {
  generoGrid.innerHTML = "";
  GENERO_ORDER.forEach(g => {
    const chip = document.createElement("div");
    chip.className = "month-chip";
    chip.dataset.genero = g;
    chip.textContent = g;
    generoGrid.appendChild(chip);
  });
}

function highlightGenero(label) {
  for (const chip of generoGrid.children) {
    chip.classList.toggle("active", chip.dataset.genero === label);
  }
}

function setGenero(label) {
  if (!GENERO_ORDER.includes(label)) return;
  highlightGenero(label);
  generoOut.textContent = JSON.stringify({ genero: label }, null, 2);
}

// ── Formulario de meses (mes de nacimiento) ──────────────────────────────────

function buildMonthsGrid() {
  monthsGrid.innerHTML = "";
  MONTHS_ORDER.forEach((m, i) => {
    const chip = document.createElement("div");
    chip.className = "month-chip";
    chip.dataset.month = m;
    chip.textContent = `${i + 1}. ${m}`;
    monthsGrid.appendChild(chip);
  });
}

function highlightMonth(label) {
  for (const chip of monthsGrid.children) {
    chip.classList.toggle("active", chip.dataset.month === label);
  }
}

function setMonth(label) {
  const numero = MONTH_NUM[label];
  if (!numero) return;   // ignora etiquetas inesperadas
  mesInput.value = label;
  highlightMonth(label);
  mesOut.textContent = JSON.stringify({ mes: label, numero }, null, 2);
  // El mes también alimenta la fecha de reservación.
  reserva.mes = label;
  reserva.mesNum = numero;
  updateReservaSummary();
  recomputeReserva();
}

// ── Días de la semana + reservación ──────────────────────────────────────────

function buildWeekdayGrid() {
  diasGrid.innerHTML = "";
  WEEKDAYS_ORDER.forEach(d => {
    const chip = document.createElement("div");
    chip.className = "month-chip";
    chip.dataset.day = d;
    chip.textContent = d;
    diasGrid.appendChild(chip);
  });
}

function highlightWeekday(label) {
  for (const chip of diasGrid.children) {
    chip.classList.toggle("active", chip.dataset.day === label);
  }
}

function setDiaSemana(label) {
  if (!JS_WEEKDAY.includes(label)) return;
  reserva.diaSemana = label;
  highlightWeekday(label);
  updateReservaSummary();
  recomputeReserva();
}

function updateReservaSummary() {
  resMesEl.textContent    = reserva.mes ?? "—";
  resSemanaEl.textContent = reserva.diaSemana ?? "—";
}

function pad2(n) { return String(n).padStart(2, "0"); }

// Construye la fecha de llegada con día+mes (resolviendo el año a la próxima
// ocurrencia >= hoy), valida la ventana de 30 días y cruza el día de la semana.
function recomputeReserva() {
  const { dia, mesNum, diaSemana } = reserva;
  if (!dia || !mesNum) { reservaOut.textContent = ""; return; }

  const today = new Date(); today.setHours(0, 0, 0, 0);
  let year = today.getFullYear();
  let d = new Date(year, mesNum - 1, dia); d.setHours(0, 0, 0, 0);
  if (d < today) { d = new Date(year + 1, mesNum - 1, dia); d.setHours(0, 0, 0, 0); }

  // JS hace overflow (31 de feb → marzo); detectamos día inexistente.
  const diaValido = d.getDate() === dia && d.getMonth() === mesNum - 1;
  const diff = Math.round((d - today) / 86400000);
  const semanaReal = JS_WEEKDAY[d.getDay()];

  const avisos = [];
  if (!diaValido) avisos.push(`el día ${dia} no existe en ese mes`);
  if (diff < 0 || diff > RESERVA_MAX_DIAS)
    avisos.push(`fuera de rango: la reservación debe ser de hoy a ${RESERVA_MAX_DIAS} días (esta cae en ${diff} días)`);
  if (diaSemana && diaSemana !== semanaReal)
    avisos.push(`firmaste «${diaSemana}» pero esa fecha cae en ${semanaReal}`);

  reservaOut.textContent = JSON.stringify({
    fecha: diaValido ? `${d.getFullYear()}-${pad2(mesNum)}-${pad2(dia)}` : null,
    diaSemanaReal: diaValido ? semanaReal : null,
    diaSemanaFirmado: diaSemana,
    dentroDe30Dias: diaValido && diff >= 0 && diff <= RESERVA_MAX_DIAS,
    avisos,
  }, null, 2);
}

// ── Formulario de números (teléfono / día / año) ─────────────────────────────
// Validación SOLO informativa: marca en rojo y avisa, pero nunca bloquea.

function validateTelefono(v) {
  if (v === "")            return { ok: null,  msg: "" };
  if (!/^\d+$/.test(v))    return { ok: false, msg: "solo dígitos" };
  if (v.length < 10)       return { ok: false, msg: `faltan ${10 - v.length} dígitos` };
  if (v.length > 10)       return { ok: false, msg: "máximo 10 dígitos" };
  if (!/^(55|56)/.test(v)) return { ok: false, msg: "normalmente inicia con 55 o 56" };
  return { ok: true, msg: "✓ válido" };
}

function validateDia(v) {
  if (v === "")         return { ok: null,  msg: "" };
  if (!/^\d+$/.test(v)) return { ok: false, msg: "solo dígitos" };
  const n = +v;
  if (n < 1 || n > 31)  return { ok: false, msg: "debe estar entre 1 y 31" };
  return { ok: true, msg: "✓ válido" };
}

function validateAnio(v) {
  if (v === "")         return { ok: null,  msg: "" };
  if (!/^\d+$/.test(v)) return { ok: false, msg: "solo dígitos" };
  if (v.length < 4)     return { ok: false, msg: "deben ser 4 dígitos" };
  const year = new Date().getFullYear();
  const max  = year - 18;    // mayor de 18
  const min  = year - 100;   // raro pasar de 100 años
  const n = +v;
  if (n > max) return { ok: false, msg: `debe ser mayor de 18 (≤ ${max})` };
  if (n < min) return { ok: false, msg: `¿más de 100 años? (≥ ${min})` };
  return { ok: true, msg: "✓ válido" };
}

function applyHint(input, hintEl, result) {
  hintEl.textContent = result.msg;
  hintEl.classList.toggle("ok",  result.ok === true);
  hintEl.classList.toggle("bad", result.ok === false);
  input.classList.toggle("invalid", result.ok === false);
}

function submitNumeros() {
  const tel  = numForm.telefono.value.trim();
  const dia  = numForm.dia.value.trim();
  const anio = numForm.anio.value.trim();
  const vt = validateTelefono(tel), vd = validateDia(dia), va = validateAnio(anio);
  applyHint(numForm.telefono, hintTelefono, vt);
  applyHint(numForm.dia,      hintDia,      vd);
  applyHint(numForm.anio,     hintAnio,     va);

  const avisos = [];
  if (vt.ok !== true) avisos.push(`teléfono: ${vt.msg || "vacío"}`);
  if (vd.ok !== true) avisos.push(`día: ${vd.msg || "vacío"}`);
  if (va.ok !== true) avisos.push(`año: ${va.msg || "vacío"}`);

  numOut.textContent = JSON.stringify({
    telefono: tel,
    dia:  dia  ? +dia  : null,
    anio: anio ? +anio : null,
    avisos,   // informativo — el envío no se bloquea
  }, null, 2);
}

// ── Constructor de huéspedes (gramática número → adulto/niño) ────────────────

function handleGuestSign(label) {
  if (label === "1" || label === "2" || label === "3") {
    pendingCount = +label;
    guestsError.textContent = "";
    renderGuests();
    return;
  }
  if (label === "adulto" || label === "nino") {
    if (pendingCount == null) {
      guestsError.textContent = "Firma primero un número (1-3), luego adulto o niño.";
      return;
    }
    guests.push({ cantidad: pendingCount, tipo: label });
    pendingCount = null;
    guestsError.textContent = "";
    renderGuests();
  }
}

function tipoLabel(tipo, cantidad) {
  if (tipo === "adulto") return cantidad === 1 ? "adulto" : "adultos";
  return cantidad === 1 ? "niño" : "niños";
}

function renderGuests() {
  guestsPending.textContent = pendingCount == null
    ? "Pendiente: firma un número (1-3)…"
    : `Pendiente: ${pendingCount} → ahora firma adulto o niño`;
  guestsPending.classList.toggle("active", pendingCount != null);

  guestsList.innerHTML = "";
  guests.forEach((g, i) => {
    const li = document.createElement("li");
    const span = document.createElement("span");
    span.textContent = `${g.cantidad} ${tipoLabel(g.tipo, g.cantidad)}`;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "guest-del";
    del.textContent = "✕";
    del.title = "Quitar grupo";
    del.onclick = () => { guests.splice(i, 1); renderGuests(); };
    li.append(span, del);
    guestsList.appendChild(li);
  });
}

function submitGuests() {
  if (guests.length === 0) {
    guestsError.textContent = "Agrega al menos un grupo: un número (1-3) + adulto o niño.";
    formOut.textContent = "";
    return;
  }
  const adultos = guests.filter(g => g.tipo === "adulto").reduce((a, g) => a + g.cantidad, 0);
  const ninos   = guests.filter(g => g.tipo === "nino").reduce((a, g) => a + g.cantidad, 0);
  guestsError.textContent = "";
  formOut.textContent = JSON.stringify({
    nombre: formEl.nombre.value.trim(),
    grupos: guests.map(g => ({ cantidad: g.cantidad, tipo: g.tipo })),
    adultos,
    ninos,
    total: adultos + ninos,
  }, null, 2);
}

// ============================================================================
//  Dibujo en canvas
// ============================================================================

function drawOverlay(handRes, poseRes) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Modos números/meses: sin pose. Mano izquierda en color, derecha tenue.
  if (activeMode !== "huespedes") {
    if (handRes.landmarks) {
      for (let i = 0; i < handRes.landmarks.length; i++) {
        const isLeft = handRes.handedness[i][0].categoryName === "Left";
        drawHand(handRes.landmarks[i], w, h, isLeft);
      }
    }
    return;
  }

  if (poseRes.landmarks && poseRes.landmarks.length > 0) {
    const lm = poseRes.landmarks[0];
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(180,180,200,0.75)";
    for (const [a, b] of POSE_UPPER_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * w, lm[a].y * h);
      ctx.lineTo(lm[b].x * w, lm[b].y * h);
      ctx.stroke();
    }
    for (const i of POSE_UPPER_INDICES) {
      drawDot(lm[i].x * w, lm[i].y * h, "#00e6c4", 5);
    }
  }

  if (handRes.landmarks) {
    const wL = poseWrist(poseRes, POSE_WRIST_LEFT);
    const wR = poseWrist(poseRes, POSE_WRIST_RIGHT);
    for (let h2 = 0; h2 < handRes.landmarks.length; h2++) {
      const pts = handRes.landmarks[h2];
      const wrist = [pts[0].x, pts[0].y];
      let owned = true;
      if (wL || wR) {
        const dL = wL ? Math.hypot(wrist[0]-wL[0], wrist[1]-wL[1]) : 1e9;
        const dR = wR ? Math.hypot(wrist[0]-wR[0], wrist[1]-wR[1]) : 1e9;
        owned = Math.min(dL, dR) <= HAND_OWNER_MAX_DIST;
      }
      drawHand(pts, w, h, owned);
    }
  }
}

function drawHand(pts, w, h, owned) {
  if (!owned) {
    ctx.strokeStyle = "rgba(255,108,112,0.45)";
    ctx.lineWidth = 1;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
      ctx.stroke();
    }
    for (const p of pts) drawDot(p.x * w, p.y * h, "rgba(255,108,112,0.5)", 2.5);
    return;
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(220,220,235,0.7)";
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(pts[a].x * w, pts[a].y * h);
    ctx.lineTo(pts[b].x * w, pts[b].y * h);
    ctx.stroke();
  }
  const palette = ["#fff","#b466ff","#b466ff","#b466ff","#b466ff",
                   "#5ad776","#5ad776","#5ad776","#5ad776",
                   "#00e6c4","#00e6c4","#00e6c4","#00e6c4",
                   "#ffe05a","#ffe05a","#ffe05a","#ffe05a",
                   "#ff9f4a","#ff9f4a","#ff9f4a","#ff9f4a"];
  for (let i = 0; i < pts.length; i++) {
    drawDot(pts[i].x * w, pts[i].y * h, palette[i] || "#fff", 5);
  }
}

function drawDot(x, y, color, r) {
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
}

// ============================================================================
//  UI helpers
// ============================================================================

function setChip(el, ok) {
  el.classList.toggle("on", ok);
  el.classList.toggle("off", !ok);
}

function buildProbRows() {
  probsEl.innerHTML = "";
  for (const k of Object.keys(labelMap).sort((a,b)=>+a-+b)) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.dataset.idx = k;
    row.innerHTML = `
      <span>${labelMap[k]}</span>
      <div class="prob-bar"><div style="width:0%;background:#aaa1c4"></div></div>
      <span class="prob-pct" style="text-align:right;color:#aaa1c4">0%</span>`;
    probsEl.appendChild(row);
  }
}

function updateProbRows(probs) {
  for (const row of probsEl.children) {
    const idx = +row.dataset.idx;
    const p = probs[idx];
    const bar = row.querySelector(".prob-bar > div");
    const pct = row.querySelector(".prob-pct");
    bar.style.width = `${(p*100).toFixed(0)}%`;
    bar.style.background = colorForConf(p);
    pct.textContent = `${(p*100).toFixed(0)}%`;
    pct.style.color = colorForConf(p);
  }
}

function colorForConf(c) {
  if (c >= 0.82) return "#5ad776";
  if (c >= 0.65) return "#ffe05a";
  return "#ff9f4a";
}

function renderHistory() {
  historyEl.innerHTML = "";
  for (const { label, conf } of predHistory) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${label.toUpperCase()}</span><span class="h-conf">${(conf*100).toFixed(0)}%</span>`;
    historyEl.appendChild(li);
  }
}

// ── Arrancar ───────────────────────────────────────────────────────────────
init().catch(err => {
  console.error(err);
  splashMsg.textContent = `Error: ${err.message || err}`;
});
