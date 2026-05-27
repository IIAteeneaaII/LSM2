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
const FEATURE_DIM    = 168;
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

const MODEL_URL     = "./models/huespedes.onnx";
const LABEL_MAP_URL = "./models/label_map.json";

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

// ── Estado ─────────────────────────────────────────────────────────────────
let session, labelMap, handLM, poseLM;
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
let inferenceBusy  = false;   // true mientras se ejecuta runInference en background
let mpMsSmooth     = 0;       // ms promedio que tarda MediaPipe (hand+pose)
let ortMsLast      = 0;       // ms que tardó la última inferencia ONNX

// ============================================================================
//  Init
// ============================================================================

async function init() {
  splashMsg.textContent = "Cargando ONNX…";
  const [labelRes] = await Promise.all([
    fetch(LABEL_MAP_URL).then(r => r.json()),
    loadOnnx(),
  ]);
  labelMap = labelRes;
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

async function loadOnnx() {
  session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
  });
  // warmup
  const dummy = new ort.Tensor("float32",
    new Float32Array(TOTAL_FRAMES * FEATURE_DIM),
    [1, TOTAL_FRAMES, FEATURE_DIM]);
  await session.run({ [session.inputNames[0]]: dummy });
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
    const poseRes = poseLM.detectForVideo(detectCanvas, ts);
    const mpMs = performance.now() - t0;
    mpMsSmooth = 0.9 * mpMsSmooth + 0.1 * mpMs;

    const poseOk = poseRes.landmarks && poseRes.landmarks.length > 0;
    const { left, right, leftValid, rightValid } = extractHands(handRes, poseRes);
    const poseFeat = extractPoseUpper(poseRes);
    const keypoints = new Float32Array(FEATURE_DIM);
    keypoints.set(poseFeat, 0);
    keypoints.set(left, 42);
    keypoints.set(right, 105);

    frameBuffer.push(keypoints);
    if (frameBuffer.length > TOTAL_FRAMES) frameBuffer.shift();
    handPresence.push(leftValid || rightValid ? 1 : 0);
    if (handPresence.length > TOTAL_FRAMES) handPresence.shift();
    frameCount++;

    drawOverlay(handRes, poseRes);
    setChip(chipPose, poseOk);
    setChip(chipLeft, leftValid);
    setChip(chipRight, rightValid);

    // Inferencia ONNX en background — NO se hace await, así que el loop
    // sigue capturando frames mientras corre. inferenceBusy evita lanzar
    // dos a la vez.
    if (!inferenceBusy &&
        frameBuffer.length === TOTAL_FRAMES &&
        frameCount - lastInfFrame >= INFERENCE_EVERY) {
      lastInfFrame = frameCount;
      inferenceBusy = true;
      // Snapshot del buffer ahora (antes de que siga creciendo)
      const snapshot = new Float32Array(TOTAL_FRAMES * FEATURE_DIM);
      for (let i = 0; i < TOTAL_FRAMES; i++) snapshot.set(frameBuffer[i], i * FEATURE_DIM);
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

// ============================================================================
//  Inferencia + rechazo
// ============================================================================

async function runInference(snapshot, handRatio) {
  if (handRatio < MIN_HAND_RATIO) {
    lastProbs = null;
    showReject(`sin manos (${(handRatio*100|0)}% del buffer)`);
    return;
  }

  const tensor = new ort.Tensor("float32", snapshot, [1, TOTAL_FRAMES, FEATURE_DIM]);
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

  if (autoToggle.checked) appendText(label);
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
    activeField.value += s;
    activeField.dispatchEvent(new Event("input"));
  } else {
    textArea.value += s;
  }
}

function backspace() {
  const target = activeField ?? textArea;
  target.value = target.value.slice(0, -1);
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

  for (const inp of formEl.querySelectorAll("input[data-fillable]")) {
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
  btnSubmit.onclick = () => {
    const data = Object.fromEntries(new FormData(formEl).entries());
    formOut.textContent = JSON.stringify(data, null, 2);
  };
}

// ============================================================================
//  Dibujo en canvas
// ============================================================================

function drawOverlay(handRes, poseRes) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

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
