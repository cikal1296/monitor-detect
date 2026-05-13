import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/esm/ort.min.js';

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const MODEL_PATH = '/model/best.onnx';
const INPUT_SIZE  = 640;
const CONF_THRESH = 0.25;
const IOU_THRESH  = 0.45;
const CLASS_NAMES = ['mati', 'menyala', 'objects'];
const CLASS_COLORS = {
  menyala: '#00ff88',
  mati:    '#ff3b3b',
  objects: '#ffaa00',
};
const CLASS_EMOJI = {
  menyala: '🟢',
  mati:    '🔴',
  objects: '🟡',
};

// ─── STATE ────────────────────────────────────────────────────────────────────
let session = null;
let stream  = null;
let rafId   = null;
let backend = 'wasm';

// ─── ELEMENTS ─────────────────────────────────────────────────────────────────
const modelStatus    = document.getElementById('modelStatus');
const btnCamera      = document.getElementById('btnCamera');
const btnUpload      = document.getElementById('btnUpload');
const fileInput      = document.getElementById('fileInput');
const cameraControls = document.getElementById('cameraControls');
const btnStopCamera  = document.getElementById('btnStopCamera');
const btnSnapshot    = document.getElementById('btnSnapshot');
const videoEl        = document.getElementById('videoEl');
const outputCanvas   = document.getElementById('outputCanvas');
const overlayCanvas  = document.getElementById('overlayCanvas');
const canvasWrapper  = document.getElementById('canvasWrapper');
const placeholder    = document.getElementById('placeholder');
const resultsPanel   = document.getElementById('resultsPanel');
const resultsGrid    = document.getElementById('resultsGrid');
const resultsSummary = document.getElementById('resultsSummary');
const statsBar       = document.getElementById('statsBar');
const statInference  = document.getElementById('statInference');
const statObjects    = document.getElementById('statObjects');
const statBackend    = document.getElementById('statBackend');

const ctx     = outputCanvas.getContext('2d');
const oCtx    = overlayCanvas.getContext('2d');

// ─── INIT ─────────────────────────────────────────────────────────────────────
async function initModel() {
  try {
    setStatus('Memuat Model...', '');

    // Try WebGL first, fallback to WASM
    const execProviders = ['webgl', 'wasm'];
    for (const ep of execProviders) {
      try {
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
        session = await ort.InferenceSession.create(MODEL_PATH, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        backend = ep.toUpperCase();
        break;
      } catch (e) {
        console.warn(`${ep} failed, trying next...`);
      }
    }

    if (!session) throw new Error('Semua backend gagal dimuat');

    setStatus('Model Siap ✓', 'ready');
    statBackend.textContent = backend;
    showToast('Model berhasil dimuat!', 'success');
    btnCamera.disabled = false;
    btnUpload.disabled = false;

  } catch (err) {
    console.error(err);
    setStatus('Gagal Memuat Model', 'error');
    showToast('Gagal memuat model: ' + err.message, 'error');
  }
}

// ─── PREPROCESSING ────────────────────────────────────────────────────────────
function preprocessCanvas(sourceCanvas) {
  const offscreen = document.createElement('canvas');
  offscreen.width  = INPUT_SIZE;
  offscreen.height = INPUT_SIZE;
  const octx = offscreen.getContext('2d');
  octx.drawImage(sourceCanvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const imageData = octx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  // HWC → CHW, normalize [0,1]
  const float32 = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    float32[i]                           = imageData[i * 4]     / 255;
    float32[INPUT_SIZE * INPUT_SIZE + i] = imageData[i * 4 + 1] / 255;
    float32[2 * INPUT_SIZE * INPUT_SIZE + i] = imageData[i * 4 + 2] / 255;
  }
  return float32;
}

// ─── INFERENCE ────────────────────────────────────────────────────────────────
async function runInference(imgCanvas) {
  if (!session) return null;

  const t0      = performance.now();
  const tensor  = new ort.Tensor('float32', preprocessCanvas(imgCanvas), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds   = { images: tensor };
  const results = await session.run(feeds);
  const elapsed = performance.now() - t0;

  // output0: [1, 7, 8400]  → [cx, cy, w, h, cls0, cls1, cls2]
  const output = results['output0'].data;
  const numDet = 8400;
  const numCls = CLASS_NAMES.length;

  const detections = [];
  for (let i = 0; i < numDet; i++) {
    const cx = output[i];
    const cy = output[numDet + i];
    const w  = output[2 * numDet + i];
    const h  = output[3 * numDet + i];

    // Get max class score
    let maxScore = -Infinity, maxCls = 0;
    for (let c = 0; c < numCls; c++) {
      const score = output[(4 + c) * numDet + i];
      if (score > maxScore) { maxScore = score; maxCls = c; }
    }

    if (maxScore < CONF_THRESH) continue;

    detections.push({
      x1: (cx - w / 2) / INPUT_SIZE,
      y1: (cy - h / 2) / INPUT_SIZE,
      x2: (cx + w / 2) / INPUT_SIZE,
      y2: (cy + h / 2) / INPUT_SIZE,
      conf: maxScore,
      cls: maxCls,
      label: CLASS_NAMES[maxCls],
    });
  }

  const filtered = nms(detections, IOU_THRESH);
  return { detections: filtered, elapsed };
}

// ─── NMS ──────────────────────────────────────────────────────────────────────
function nms(dets, iouThresh) {
  dets.sort((a, b) => b.conf - a.conf);
  const keep = [];
  const used = new Set();
  for (let i = 0; i < dets.length; i++) {
    if (used.has(i)) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (used.has(j)) continue;
      if (iou(dets[i], dets[j]) > iouThresh) used.add(j);
    }
  }
  return keep;
}

function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const unionA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const unionB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (unionA + unionB - inter);
}

// ─── DRAW ─────────────────────────────────────────────────────────────────────
function drawDetections(canvas, detections) {
  const W = canvas.width, H = canvas.height;
  const oCanvas = overlayCanvas;
  oCanvas.width  = W;
  oCanvas.height = H;
  oCtx.clearRect(0, 0, W, H);

  for (const det of detections) {
    const x = det.x1 * W, y = det.y1 * H;
    const w = (det.x2 - det.x1) * W, h = (det.y2 - det.y1) * H;
    const color = CLASS_COLORS[det.label] || '#ffffff';
    const conf  = (det.conf * 100).toFixed(1);

    // Box
    oCtx.strokeStyle = color;
    oCtx.lineWidth   = 2.5;
    oCtx.strokeRect(x, y, w, h);

    // Fill glow
    oCtx.fillStyle = color + '18';
    oCtx.fillRect(x, y, w, h);

    // Label bg
    const label = `${det.label.toUpperCase()} ${conf}%`;
    oCtx.font = 'bold 13px Space Mono, monospace';
    const tw = oCtx.measureText(label).width;
    const lx = x, ly = y > 22 ? y - 22 : y + h + 2;
    oCtx.fillStyle = color;
    oCtx.fillRect(lx, ly, tw + 12, 22);

    // Label text
    oCtx.fillStyle = '#000';
    oCtx.fillText(label, lx + 6, ly + 15);

    // Corner accents
    const cs = 12, clw = 3;
    oCtx.lineWidth   = clw;
    oCtx.strokeStyle = color;
    // TL
    oCtx.beginPath(); oCtx.moveTo(x, y + cs); oCtx.lineTo(x, y); oCtx.lineTo(x + cs, y); oCtx.stroke();
    // TR
    oCtx.beginPath(); oCtx.moveTo(x + w - cs, y); oCtx.lineTo(x + w, y); oCtx.lineTo(x + w, y + cs); oCtx.stroke();
    // BL
    oCtx.beginPath(); oCtx.moveTo(x, y + h - cs); oCtx.lineTo(x, y + h); oCtx.lineTo(x + cs, y + h); oCtx.stroke();
    // BR
    oCtx.beginPath(); oCtx.moveTo(x + w - cs, y + h); oCtx.lineTo(x + w, y + h); oCtx.lineTo(x + w, y + h - cs); oCtx.stroke();
  }
}

// ─── UI RESULTS ───────────────────────────────────────────────────────────────
function updateResults(detections, elapsed) {
  resultsPanel.style.display = 'block';
  statsBar.style.display     = 'grid';
  statInference.textContent  = `${elapsed.toFixed(1)} ms`;
  statObjects.textContent    = detections.length;

  if (detections.length === 0) {
    resultsGrid.innerHTML = '<div class="no-detection">Tidak ada objek terdeteksi</div>';
    resultsSummary.textContent = '';
    return;
  }

  resultsGrid.innerHTML = detections.map(det => {
    const conf  = (det.conf * 100).toFixed(1);
    const cls   = det.label;
    const emoji = CLASS_EMOJI[cls] || '⬜';
    return `
      <div class="result-item">
        <span class="result-badge">${emoji}</span>
        <div class="result-info">
          <div class="result-class ${cls}">${cls.toUpperCase()}</div>
          <div class="result-conf">Confidence: ${conf}%</div>
        </div>
        <div class="result-bar-wrap">
          <div class="result-bar ${cls}" style="width:${conf}%"></div>
        </div>
      </div>
    `;
  }).join('');

  const counts = {};
  for (const d of detections) counts[d.label] = (counts[d.label] || 0) + 1;
  const summary = Object.entries(counts).map(([k, v]) => `${v}× ${k}`).join(' · ');
  resultsSummary.textContent = `Terdeteksi: ${summary} · ${elapsed.toFixed(0)}ms inference`;
}

// ─── IMAGE FLOW ───────────────────────────────────────────────────────────────
async function processImage(source) {
  placeholder.style.display = 'none';
  canvasWrapper.classList.add('active');

  // Show loading
  const loader = document.createElement('div');
  loader.className = 'loading-overlay';
  loader.innerHTML = `<div class="spinner"></div><div class="loading-text">Mendeteksi...</div>`;
  canvasWrapper.appendChild(loader);

  // Draw to canvas
  outputCanvas.style.display = 'block';
  const img = new Image();
  await new Promise(res => { img.onload = res; img.src = source; });

  outputCanvas.width  = img.naturalWidth  || img.width;
  outputCanvas.height = img.naturalHeight || img.height;
  ctx.drawImage(img, 0, 0);

  // Run inference
  const result = await runInference(outputCanvas);
  loader.remove();

  if (result) {
    drawDetections(outputCanvas, result.detections);
    updateResults(result.detections, result.elapsed);
  }
}

// ─── CAMERA FLOW ──────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    videoEl.srcObject = stream;
    videoEl.style.display = 'block';
    outputCanvas.style.display = 'none';
    placeholder.style.display = 'none';
    cameraControls.style.display = 'flex';
    btnCamera.style.display = 'none';
    btnUpload.disabled = true;
    canvasWrapper.classList.add('active');

    await videoEl.play();
    startLiveDetection();
    showToast('Kamera aktif', 'success');
  } catch (err) {
    showToast('Tidak bisa akses kamera: ' + err.message, 'error');
  }
}

function stopCamera() {
  cancelAnimationFrame(rafId);
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  videoEl.style.display = 'none';
  outputCanvas.style.display = 'none';
  overlayCanvas.width = 0;
  cameraControls.style.display = 'none';
  btnCamera.style.display = 'flex';
  btnUpload.disabled = false;
  canvasWrapper.classList.remove('active');
  placeholder.style.display = 'block';
  resultsPanel.style.display = 'none';
  statsBar.style.display = 'none';
}

let lastTime = 0;
const FPS_LIMIT = 8; // 8fps for inference (heavier models)

function startLiveDetection() {
  async function loop(ts) {
    rafId = requestAnimationFrame(loop);
    if (ts - lastTime < 1000 / FPS_LIMIT) return;
    if (videoEl.readyState < 2) return;
    lastTime = ts;

    // Draw video frame to offscreen canvas
    const tmp = document.createElement('canvas');
    tmp.width  = videoEl.videoWidth  || 640;
    tmp.height = videoEl.videoHeight || 480;
    tmp.getContext('2d').drawImage(videoEl, 0, 0);

    // Sync overlay to video display rect
    const rect = videoEl.getBoundingClientRect();
    overlayCanvas.style.width  = rect.width  + 'px';
    overlayCanvas.style.height = rect.height + 'px';
    overlayCanvas.width  = tmp.width;
    overlayCanvas.height = tmp.height;

    const result = await runInference(tmp);
    if (result) {
      drawDetections(tmp, result.detections);
      updateResults(result.detections, result.elapsed);
    }
  }
  rafId = requestAnimationFrame(loop);
}

function takeSnapshot() {
  const tmp = document.createElement('canvas');
  tmp.width  = videoEl.videoWidth  || 640;
  tmp.height = videoEl.videoHeight || 480;
  tmp.getContext('2d').drawImage(videoEl, 0, 0);

  // Merge overlay
  const merged = document.createElement('canvas');
  merged.width  = tmp.width;
  merged.height = tmp.height;
  const mCtx = merged.getContext('2d');
  mCtx.drawImage(tmp, 0, 0);
  mCtx.drawImage(overlayCanvas, 0, 0, tmp.width, tmp.height);

  const link = document.createElement('a');
  link.download = `monitor-snap-${Date.now()}.png`;
  link.href = merged.toDataURL('image/png');
  link.click();
  showToast('Snapshot disimpan!', 'success');
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function setStatus(text, cls) {
  const dot  = modelStatus.querySelector('.dot');
  const span = modelStatus.querySelector('span:last-child');
  span.textContent = text;
  modelStatus.className = 'status-pill' + (cls ? ' ' + cls : '');
}

function showToast(msg, type = '') {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const t = document.createElement('div');
  t.className = 'toast' + (type ? ' ' + type : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.style.opacity = '0', 2500);
  setTimeout(() => t.remove(), 3000);
}

// ─── EVENTS ───────────────────────────────────────────────────────────────────
btnCamera.addEventListener('click', startCamera);
btnStopCamera.addEventListener('click', stopCamera);
btnSnapshot.addEventListener('click', takeSnapshot);

btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  processImage(url);
  fileInput.value = '';
});

// Drag & drop
canvasWrapper.addEventListener('dragover', e => {
  e.preventDefault();
  canvasWrapper.style.borderColor = '#00ff88';
});
canvasWrapper.addEventListener('dragleave', () => {
  canvasWrapper.style.borderColor = '';
});
canvasWrapper.addEventListener('drop', e => {
  e.preventDefault();
  canvasWrapper.style.borderColor = '';
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    processImage(URL.createObjectURL(file));
  }
});

// ─── BOOTSTRAP ────────────────────────────────────────────────────────────────
btnCamera.disabled = true;
btnUpload.disabled = true;

initModel();
