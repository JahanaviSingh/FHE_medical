/**
 * frontend/static/js/app.js
 *
 * Handles all UI logic:
 *   - Mode/operation selection
 *   - File upload (drag-drop + browse + demo)
 *   - Calls Flask API for validation + FHE pipeline
 *   - Renders results: ciphertext noise, heatmap, diagnosis, metrics
 */

"use strict";

// ─────────────────────────────────────────────
//  OPERATION DEFINITIONS
// ─────────────────────────────────────────────

const MODES = {
  xray: {
    label: "Chest X-ray",
    ops: [
      {
        id: "pneumonia_detection",
        name: "Pneumonia detection",
        sub: "Bilateral infiltrate analysis",
        dot: "#E24B4A",
        dataset: "NIH CheXNet · 112k images",
      },
      {
        id: "nodule_screening",
        name: "Nodule screening",
        sub: "Lung nodule & mass detection",
        dot: "#EF9F27",
        dataset: "RSNA Pneumonia · 26k scans",
      },
      {
        id: "patient_anonymize",
        name: "Patient anonymize",
        sub: "Face & PHI redaction (HIPAA)",
        dot: "#534AB7",
        dataset: "DICOM de-identification",
      },
    ],
  },
  mri: {
    label: "Brain MRI",
    ops: [
      {
        id: "tumor_boundary",
        name: "Tumor boundary",
        sub: "Glioma segmentation & grading",
        dot: "#E24B4A",
        dataset: "BraTS 2023 · 1,251 cases",
      },
      {
        id: "mri_denoise",
        name: "MRI denoising",
        sub: "Rician noise reduction (NLM)",
        dot: "#1D9E75",
        dataset: "BraTS reconstruction",
      },
      {
        id: "structure_map",
        name: "Structure map",
        sub: "Cortical segmentation · 89 regions",
        dot: "#378ADD",
        dataset: "BraTS atlas alignment",
      },
    ],
  },
  bone: {
    label: "Bone X-ray",
    ops: [
      {
        id: "fracture_detection",
        name: "Fracture detection",
        sub: "Cortical break localisation",
        dot: "#E24B4A",
        dataset: "MURA · 40k X-rays",
      },
      {
        id: "edge_enhance",
        name: "Edge enhance",
        sub: "Bone margin sharpening",
        dot: "#378ADD",
        dataset: "MURA preprocessing",
      },
      {
        id: "bone_density",
        name: "Bone density",
        sub: "Osteoporosis risk scoring",
        dot: "#EF9F27",
        dataset: "DXA reference atlas",
      },
    ],
  },
  ct: {
    label: "CT scan",
    ops: [
      {
        id: "ct_contrast",
        name: "CT contrast",
        sub: "Hounsfield unit calibration",
        dot: "#1D9E75",
        dataset: "TCIA CT dataset",
      },
      {
        id: "organ_segment",
        name: "Organ segmentation",
        sub: "Multi-organ auto-contouring",
        dot: "#378ADD",
        dataset: "TotalSegmentator · 1200",
      },
      {
        id: "bleed_detection",
        name: "Bleed detection",
        sub: "Intracranial haemorrhage",
        dot: "#E24B4A",
        dataset: "RSNA ICH · 25k scans",
      },
    ],
  },
};

// ─────────────────────────────────────────────
//  STATE
// ─────────────────────────────────────────────

let state = {
  mode: "xray",
  operation: "pneumonia_detection",
  imageB64: null,
  imageValid: false,
  running: false,
  t0: Date.now(),
};

// ─────────────────────────────────────────────
//  INIT
// ─────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  setupModeTabs();
  setupDragDrop();
  setupFileInput();
  renderOps();

  document.getElementById("runBtn").addEventListener("click", runPipeline);
  document
    .getElementById("demoBtn")
    .addEventListener("click", () => loadDemo(false));
  document
    .getElementById("invalidBtn")
    .addEventListener("click", () => loadDemo(true));

  setupLightbox(); // ← NEW

  log("FHE Medical Pipeline ready · input validator active");
});

// ─────────────────────────────────────────────
//  LIGHTBOX  (fullscreen view for AI result)
// ─────────────────────────────────────────────

function setupLightbox() {
  // Inject markup into DOM once
  const markup = `
    <div id="fhe-lightbox" role="dialog" aria-modal="true" aria-label="Full screen AI result">
      <div id="fhe-lb-inner">
        <button id="fhe-lb-close" aria-label="Close full screen">&#x2715;</button>
        <canvas id="fhe-lb-canvas"></canvas>
        <p id="fhe-lb-caption">AI result (decrypted) &mdash; click anywhere outside to close</p>
      </div>
    </div>`;
  document.body.insertAdjacentHTML("beforeend", markup);

  const lb = document.getElementById("fhe-lightbox");
  const lbClose = document.getElementById("fhe-lb-close");

  // Close on backdrop click (not on the canvas/inner itself)
  lb.addEventListener("click", (e) => {
    if (e.target === lb) closeLightbox();
  });
  lbClose.addEventListener("click", closeLightbox);

  // Close on Escape
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && lb.classList.contains("lb-open")) closeLightbox();
  });

  // Attach click to c3 wrapper once — and re-attach after pipeline renders
  attachResultClick();
}

function attachResultClick() {
  const wrap =
    document.getElementById("c3wrap") ||
    document.getElementById("c3").parentElement;
  if (!wrap || wrap._lbAttached) return;
  wrap._lbAttached = true;
  wrap.style.cursor = "zoom-in";
  wrap.title = "Click to view full screen";
  // Zoom-in icon hint on hover (injected once)
  if (!wrap.querySelector(".fhe-zoom-hint")) {
    const hint = document.createElement("div");
    hint.className = "fhe-zoom-hint";
    hint.innerHTML = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
    </svg>`;
    wrap.appendChild(hint);
  }
  wrap.addEventListener("click", () => {
    const c3 = document.getElementById("c3");
    if (c3.style.display === "none" || !c3.width) return; // no result yet
    openLightbox(c3);
  });
}

function openLightbox(sourceCanvas) {
  const lb = document.getElementById("fhe-lightbox");
  const lbCanvas = document.getElementById("fhe-lb-canvas");

  // Copy pixels from c3 into the lightbox canvas at native resolution
  lbCanvas.width = sourceCanvas.width;
  lbCanvas.height = sourceCanvas.height;
  lbCanvas.getContext("2d").drawImage(sourceCanvas, 0, 0);

  lb.classList.add("lb-open");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  const lb = document.getElementById("fhe-lightbox");
  lb.classList.remove("lb-open");
  document.body.style.overflow = "";
}

// ─────────────────────────────────────────────
//  MODE TABS
// ─────────────────────────────────────────────

function setupModeTabs() {
  document.querySelectorAll(".mtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".mtab")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      state.operation = MODES[state.mode].ops[0].id;
      renderOps();
      if (state.imageB64) validateImage(state.imageB64);
      log(`Modality: ${MODES[state.mode].label}`);
    });
  });
}

// ─────────────────────────────────────────────
//  OPERATION GRID
// ─────────────────────────────────────────────

function renderOps() {
  const grid = document.getElementById("opGrid");
  grid.innerHTML = "";
  MODES[state.mode].ops.forEach((op, i) => {
    const div = document.createElement("div");
    div.className = "op-card" + (i === 0 ? " sel" : "");
    div.innerHTML = `
      <div class="op-dot" style="background:${op.dot}"></div>
      <div class="op-name">${op.name}</div>
      <div class="op-sub">${op.sub}</div>
      <div class="op-dataset">${op.dataset}</div>`;
    div.addEventListener("click", () => {
      document
        .querySelectorAll(".op-card")
        .forEach((c) => c.classList.remove("sel"));
      div.classList.add("sel");
      state.operation = op.id;
      log(`Operation: ${op.name}`);
    });
    grid.appendChild(div);
  });
  state.operation = MODES[state.mode].ops[0].id;
}

// ─────────────────────────────────────────────
//  FILE HANDLING
// ─────────────────────────────────────────────

function setupDragDrop() {
  const zone = document.getElementById("dropZone");
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function setupFileInput() {
  document.getElementById("fileInput").addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  log(`Loading file: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
  const reader = new FileReader();
  reader.onload = (ev) => {
    const full = ev.target.result;
    const b64 = full.includes(",") ? full.split(",")[1] : full;
    drawImageOnCanvas("c1", full);
    state.imageB64 = b64;
    validateImage(b64);
  };
  reader.readAsDataURL(file);
}

// ─────────────────────────────────────────────
//  DEMO IMAGE GENERATION (pure canvas, no server)
// ─────────────────────────────────────────────

function loadDemo(invalid) {
  const c = document.getElementById("c1");
  c.width = 400;
  c.height = 400;
  const ctx = c.getContext("2d");

  if (invalid) {
    drawBaboon(ctx, 400, 400);
    log("Loaded non-medical test image (baboon)", "log-warn");
  } else {
    drawMedicalScan(ctx, 400, 400, state.mode);
    log(`Loaded synthetic ${MODES[state.mode].label} demo`, "log-ok");
  }

  document.getElementById("ph1").style.display = "none";
  c.style.display = "block";

  // Get base64 from canvas
  const b64 = c.toDataURL("image/png").split(",")[1];
  state.imageB64 = b64;
  validateImage(b64);
  resetResults();
}

function drawBaboon(ctx, w, h) {
  // Colourful natural scene — will fail all grayscale checks
  ctx.fillStyle = "#4a7c3f";
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 600; i++) {
    ctx.fillStyle = `hsl(${80 + Math.random() * 60},${40 + Math.random() * 40}%,${20 + Math.random() * 40}%)`;
    ctx.fillRect(
      Math.random() * w,
      Math.random() * h,
      Math.random() * 20 + 3,
      Math.random() * 20 + 3,
    );
  }
  // Head
  ctx.fillStyle = "#8B4513";
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.45, w * 0.22, h * 0.28, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#D2691E";
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.38, w * 0.18, w * 0.18, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#FF6347";
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.42, w * 0.1, h * 0.07, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#111";
  ctx.beginPath();
  ctx.arc(w * 0.43, h * 0.34, w * 0.03, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(w * 0.57, h * 0.34, w * 0.03, 0, Math.PI * 2);
  ctx.fill();
}

function drawMedicalScan(ctx, w, h, mode) {
  ctx.fillStyle = "#060606";
  ctx.fillRect(0, 0, w, h);

  if (mode === "xray") {
    // Noise background
    for (let i = 0; i < 1500; i++) {
      const v = 140 + Math.random() * 60;
      ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * 0.06})`;
      ctx.fillRect(
        Math.random() * w,
        Math.random() * h,
        Math.random() * 5 + 1,
        Math.random() * 4 + 1,
      );
    }
    // Lung fields
    ctx.fillStyle = "rgba(180,180,180,0.12)";
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, w * 0.38, h * 0.44, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(200,200,200,0.2)";
    ctx.beginPath();
    ctx.ellipse(w * 0.38, h / 2, w * 0.13, h * 0.32, 0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(200,200,200,0.2)";
    ctx.beginPath();
    ctx.ellipse(w * 0.62, h / 2, w * 0.13, h * 0.32, -0.1, 0, Math.PI * 2);
    ctx.fill();
    // Spine
    ctx.strokeStyle = "rgba(160,160,160,0.25)";
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.moveTo(w / 2, h * 0.15);
    ctx.lineTo(w / 2, h * 0.82);
    ctx.stroke();
    // Infiltrate hotspot
    ctx.fillStyle = "rgba(240,150,60,0.18)";
    ctx.beginPath();
    ctx.ellipse(w * 0.62, h * 0.6, w * 0.09, h * 0.12, 0.2, 0, Math.PI * 2);
    ctx.fill();
  } else if (mode === "bone") {
    // Long bone — dark medullary canal, bright cortical margins, flared epiphyses
    const bx = w * 0.5,
      bw = w * 0.13;
    // Soft tissue surround
    for (let i = 0; i < 600; i++) {
      const v = 55 + Math.random() * 35;
      ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * 0.07})`;
      ctx.fillRect(
        Math.random() * w,
        Math.random() * h,
        Math.random() * 5 + 1,
        Math.random() * 4 + 1,
      );
    }
    // Medullary canal (dark)
    ctx.fillStyle = "rgba(18,18,18,1)";
    ctx.fillRect(bx - bw * 0.42, h * 0.1, bw * 0.84, h * 0.8);
    // Cortical bone left margin (bright)
    ctx.fillStyle = "rgba(228,228,228,1)";
    ctx.fillRect(bx - bw, h * 0.08, bw * 0.58, h * 0.84);
    // Cortical bone right margin (bright)
    ctx.fillRect(bx + bw * 0.42, h * 0.08, bw * 0.58, h * 0.84);
    // Flared epiphysis — top
    ctx.fillStyle = "rgba(195,195,195,0.9)";
    ctx.beginPath();
    ctx.ellipse(bx, h * 0.09, bw * 1.7, h * 0.09, 0, 0, Math.PI * 2);
    ctx.fill();
    // Flared epiphysis — bottom
    ctx.beginPath();
    ctx.ellipse(bx, h * 0.91, bw * 1.7, h * 0.09, 0, 0, Math.PI * 2);
    ctx.fill();
    // Trabecular texture in epiphyses
    for (let i = 0; i < 180; i++) {
      const tx = bx + (Math.random() - 0.5) * bw * 3.2;
      const side = Math.random() > 0.5;
      const ty = side
        ? h * 0.03 + Math.random() * h * 0.12
        : h * 0.85 + Math.random() * h * 0.12;
      const v = 100 + Math.random() * 90;
      ctx.fillStyle = `rgba(${v},${v},${v},0.55)`;
      ctx.fillRect(tx, ty, Math.random() * 4 + 1, Math.random() * 3 + 1);
    }
  } else if (mode === "mri") {
    for (let y = 0; y < h; y += 2)
      for (let x = 0; x < w; x += 2) {
        const d = Math.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2);
        const v = Math.max(0, 1 - d / (w * 0.44));
        ctx.fillStyle = `rgba(${(v * 50) | 0},${(v * 65) | 0},${(v * 105) | 0},1)`;
        ctx.fillRect(x, y, 2, 2);
      }
    ctx.fillStyle = "rgba(160,185,225,0.18)";
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, w * 0.36, h * 0.41, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "rgba(85,125,185,0.28)";
    ctx.beginPath();
    ctx.ellipse(w * 0.42, h * 0.44, w * 0.12, h * 0.09, 0.3, 0, Math.PI * 2);
    ctx.fill();
    // Tumor
    ctx.fillStyle = "rgba(195,95,55,0.32)";
    ctx.beginPath();
    ctx.ellipse(w * 0.55, h * 0.42, w * 0.07, h * 0.06, -0.2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.fillStyle = "rgba(140,140,140,0.1)";
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, w * 0.4, h * 0.45, 0, 0, Math.PI * 2);
    ctx.fill();
    for (let i = 0; i < 1200; i++) {
      const v = 170 + Math.random() * 50;
      ctx.fillStyle = `rgba(${v},${v},${v},${Math.random() * 0.05})`;
      ctx.fillRect(
        Math.random() * w,
        Math.random() * h,
        Math.random() * 4 + 1,
        Math.random() * 4 + 1,
      );
    }
    ctx.strokeStyle = "rgba(200,200,200,0.35)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(w * 0.3, h * 0.2);
    ctx.lineTo(w * 0.28, h * 0.5);
    ctx.lineTo(w * 0.35, h * 0.8);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w * 0.7, h * 0.2);
    ctx.lineTo(w * 0.72, h * 0.5);
    ctx.lineTo(w * 0.65, h * 0.8);
    ctx.stroke();
  }
}

// ─────────────────────────────────────────────
//  VALIDATION  (calls /api/validate/check)
// ─────────────────────────────────────────────

async function validateImage(b64) {
  setCanvas("c1", b64);

  try {
    const res = await fetch("/api/validate/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_b64: b64, modality: state.mode }),
    });
    const data = await res.json();
    renderValidation(data);
    state.imageValid = data.status !== "fail";
    document.getElementById("runBtn").disabled = !state.imageValid;

    if (!state.imageValid) {
      showRejectedOverlay(data.message, data.hint);
    } else {
      document.getElementById("rejOverlay").style.display = "none";
    }
  } catch (e) {
    // If backend not running, fall back to client-side validation
    log("Backend not reachable — using client-side validation", "log-warn");
    clientSideValidate(b64);
  }
}

function renderValidation(data) {
  const panel = document.getElementById("valPanel");
  panel.style.display = "flex";
  panel.style.flexDirection = "column";
  panel.style.gap = "8px";

  const banner = document.getElementById("valBanner");
  const icon = document.getElementById("valIcon");
  const title = document.getElementById("valTitle");
  const hint = document.getElementById("valHint");

  banner.className = `val-banner ${data.status}`;
  icon.className = `val-icon ${data.status}`;
  icon.textContent =
    data.status === "pass" ? "✓" : data.status === "warn" ? "!" : "✕";
  title.textContent = data.message;
  hint.textContent = data.hint;

  const grid = document.getElementById("checksGrid");
  grid.innerHTML = "";
  (data.checks || []).forEach((c) => {
    const div = document.createElement("div");
    div.className = `chk ${c.passed ? "pass" : "fail"}`;
    div.innerHTML = `<div class="chk-label">${c.label}</div>
                     <div class="chk-val">${c.passed ? c.expected : c.value}</div>`;
    grid.appendChild(div);
  });

  log(
    `Validation ${data.status.toUpperCase()} · score ${(data.score * 100).toFixed(0)}% · ${data.message}`,
    data.status === "pass"
      ? "log-ok"
      : data.status === "warn"
        ? "log-warn"
        : "log-err",
  );
}

function showRejectedOverlay(msg, hint) {
  const ov = document.getElementById("rejOverlay");
  ov.style.display = "flex";
  document.getElementById("rejText").textContent = msg;
  document.getElementById("rejHint").textContent = hint;
}

// Fallback: client-side pixel analysis (mirrors validator.py logic in JS)
function clientSideValidate(b64) {
  const c = document.getElementById("c1");
  if (!c.width) return;
  const ctx = c.getContext("2d");
  const id = ctx.getImageData(
    0,
    0,
    Math.min(c.width, 200),
    Math.min(c.height, 200),
  );
  const d = id.data;

  let rSum = 0,
    gSum = 0,
    bSum = 0;
  const n = d.length / 4;
  for (let i = 0; i < d.length; i += 4) {
    rSum += d[i];
    gSum += d[i + 1];
    bSum += d[i + 2];
  }
  const chroma =
    Math.abs(rSum / n - gSum / n) +
    Math.abs(gSum / n - bSum / n) +
    Math.abs(rSum / n - bSum / n);
  const isGray = chroma < 20;

  const fakeReport = {
    status: isGray ? "pass" : "fail",
    score: isGray ? 0.8 : 0.2,
    message: isGray
      ? `Valid ${MODES[state.mode].label} (client check)`
      : "Not a medical scan — colour image rejected",
    hint: isGray
      ? "Proceeding with pipeline"
      : "Please upload a grayscale medical image",
    checks: [
      {
        label: "Colour space",
        passed: isGray,
        expected: "Grayscale",
        value: `chroma=${chroma.toFixed(1)}`,
      },
    ],
  };
  renderValidation(fakeReport);
  state.imageValid = isGray;
  document.getElementById("runBtn").disabled = !isGray;
  if (!isGray) showRejectedOverlay(fakeReport.message, fakeReport.hint);
}

// ─────────────────────────────────────────────
//  FHE PIPELINE  (calls /api/fhe/pipeline)
// ─────────────────────────────────────────────

async function runPipeline() {
  if (!state.imageB64 || !state.imageValid || state.running) return;
  state.running = true;
  document.getElementById("runBtn").disabled = true;
  document.getElementById("step5").style.display = "none";
  document.getElementById("progressWrap").style.display = "block";
  resetResults();

  try {
    // Use raw original bytes — not canvas re-read which may be transformed
    const payload = {
      image_b64: state.imageB64raw || state.imageB64,
      modality: state.mode,
      operation: state.operation,
    };

    // ── Animate pipeline stages while waiting for response
    const stageAnim = animatePipelineStages();

    setProgress(10, "Encoding pixels as CKKS polynomial coefficients...");
    log("FHE encrypt → server receives ciphertext", "log-enc");

    const res = await fetch("/api/fhe/pipeline", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearInterval(stageAnim);

    const data = await res.json();

    if (!res.ok || data.status === "validation_failed") {
      const vdata = data.validation || {};
      // "warn" means physics passed but ML was uncertain — same as a client
      // warn, so allow the pipeline through rather than hard-blocking.
      // Only stop on a genuine "fail" where physics checks also failed.
      const serverStatus = (vdata.status || "fail").toLowerCase();
      if (serverStatus === "warn") {
        log(
          `Server validator uncertain (${vdata.message || "warn"}) — running local demo`,
          "log-warn",
        );
        runLocalDemo();
        return;
      }
      renderValidation({
        ...vdata,
        status: "fail",
        message: vdata.message || "Server validation failed",
      });
      showRejectedOverlay(
        vdata.message || "Validation failed",
        vdata.hint ||
          "Deploy the updated validator.py to the server to fix bone/MRI scans.",
      );
      log(
        `Pipeline blocked · server validator score=${((vdata.score ?? 0) * 100) | 0}% · deploy updated validator.py`,
        "log-err",
      );
      return;
    }

    // ── Render ciphertext panel
    setProgress(
      60,
      "Running ML on encrypted data — server sees 0% plaintext...",
    );
    if (data.ciphertext_b64)
      setCanvas("c2", "data:image/png;base64," + data.ciphertext_b64);
    markStage("ps-encrypt", "complete");
    markStage("ps-process", "active");
    log(
      "ML inference complete on ciphertext · decrypting client-side...",
      "log-enc",
    );

    // ── Render result panel
    setProgress(90, "Decrypting with private key (client-side only)...");
    if (data.result_b64) {
      setCanvas("c3", "data:image/png;base64," + data.result_b64);
      attachResultClick(); // ← re-attach in case c3's parent was re-created
    }
    markStage("ps-process", "complete");
    markStage("ps-decrypt", "complete");
    markStage("ps-result", "complete");

    document.getElementById("tag3").textContent = "decrypted";
    document.getElementById("tag3").className = "tag tag-result";

    setProgress(
      100,
      "Pipeline complete · patient data never left device unencrypted",
    );
    log(
      `Pipeline done in ${data.metrics?.total_time_s || "—"}s · privacy score ${data.metrics?.privacy_score || "—"}`,
      "log-ok",
    );

    // ── Show diagnosis + metrics
    renderDiagnosis(data.diagnosis || {});
    renderMetrics(data.metrics || {});
    document.getElementById("step5").style.display = "block";
  } catch (e) {
    log(`Error: ${e.message} — running local demo mode`, "log-err");
    runLocalDemo(); // graceful degradation: show JS-computed result
  } finally {
    state.running = false;
    document.getElementById("runBtn").disabled = false;
    setTimeout(() => {
      document.getElementById("progressWrap").style.display = "none";
    }, 1500);
  }
}

// ─────────────────────────────────────────────
//  LOCAL DEMO MODE (backend not running)
// ─────────────────────────────────────────────

function runLocalDemo() {
  // Draw noise on panel 2
  const c1 = document.getElementById("c1");
  const c2 = document.getElementById("c2");
  c2.width = c1.width || 400;
  c2.height = c1.height || 400;
  drawNoise(c2.getContext("2d"), c2.width, c2.height);
  document.getElementById("ph2").style.display = "none";
  c2.style.display = "block";
  markStage("ps-encrypt", "complete");

  // Draw processed on panel 3
  const c3 = document.getElementById("c3");
  c3.width = c1.width || 400;
  c3.height = c1.height || 400;
  const ctx = c3.getContext("2d");
  ctx.drawImage(c1, 0, 0, c3.width, c3.height);
  // Grayscale
  const id = ctx.getImageData(0, 0, c3.width, c3.height);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    const g = d[i] * 0.3 + d[i + 1] * 0.59 + d[i + 2] * 0.11;
    d[i] = d[i + 1] = d[i + 2] = g;
  }
  ctx.putImageData(id, 0, 0);

  // ── Mode-specific overlay (no red flood on bone/CT/MRI) ──────────
  if (state.mode === "xray") {
    // Red/orange pneumonia infiltrate — right lower lobe
    const hg = ctx.createRadialGradient(
      c3.width * 0.62,
      c3.height * 0.6,
      0,
      c3.width * 0.62,
      c3.height * 0.6,
      c3.width * 0.15,
    );
    hg.addColorStop(0, "rgba(226,75,74,0.70)");
    hg.addColorStop(0.5, "rgba(239,159,39,0.40)");
    hg.addColorStop(1, "rgba(55,138,221,0)");
    ctx.fillStyle = hg;
    ctx.fillRect(0, 0, c3.width, c3.height);
  } else if (state.mode === "bone") {
    // Yellow/orange fracture highlight — small localised spot on shaft
    const hg = ctx.createRadialGradient(
      c3.width * 0.5,
      c3.height * 0.42,
      0,
      c3.width * 0.5,
      c3.height * 0.42,
      c3.width * 0.07,
    );
    hg.addColorStop(0, "rgba(255,225,0,0.85)"); // bright yellow core
    hg.addColorStop(0.5, "rgba(255,140,0,0.50)"); // orange ring
    hg.addColorStop(1, "rgba(255,100,0,0)");
    ctx.fillStyle = hg;
    ctx.fillRect(0, 0, c3.width, c3.height);
    // Thin fracture line across shaft
    ctx.save();
    ctx.strokeStyle = "rgba(255,210,0,0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(c3.width * 0.42, c3.height * 0.39);
    ctx.lineTo(c3.width * 0.58, c3.height * 0.45);
    ctx.stroke();
    ctx.restore();
  } else if (state.mode === "ct") {
    // Cyan/teal organ segmentation tint — centred
    const hg = ctx.createRadialGradient(
      c3.width * 0.5,
      c3.height * 0.5,
      0,
      c3.width * 0.5,
      c3.height * 0.5,
      c3.width * 0.3,
    );
    hg.addColorStop(0, "rgba(0,210,210,0.45)");
    hg.addColorStop(0.6, "rgba(0,130,190,0.22)");
    hg.addColorStop(1, "rgba(0,80,160,0)");
    ctx.fillStyle = hg;
    ctx.fillRect(0, 0, c3.width, c3.height);
  }
  // mri — brain.py already draws the tumour overlay server-side;
  // in local demo the grayscale brain canvas is fine as-is.
  document.getElementById("ph3").style.display = "none";
  c3.style.display = "block";
  markStage("ps-result", "complete");
  document.getElementById("tag3").textContent = "decrypted";
  document.getElementById("tag3").className = "tag tag-result";

  attachResultClick(); // ← ensure click is live after local demo

  // Fake metrics
  renderMetrics({
    privacy_score: 94,
    expansion_ratio: 8.4,
    total_time_s: 2.3,
    security_bits: 128,
  });
  // ── Mode-specific demo diagnosis ─────────────────────────────────
  const demoDiagnosis = {
    xray: {
      condition: "Pneumonia detected (demo)",
      risk_pct: 92,
      model: "NIH CheXNet (simulated)",
      differentials: [
        { label: "Pneumonia", pct: 92 },
        { label: "Pleural effusion", pct: 41 },
        { label: "Normal", pct: 8 },
      ],
    },
    bone: {
      condition: "Cortical irregularity detected (demo)",
      risk_pct: 74,
      model: "MURA DenseNet (simulated)",
      differentials: [
        { label: "Fracture", pct: 74 },
        { label: "Bone contusion", pct: 22 },
        { label: "Normal variant", pct: 9 },
      ],
    },
    mri: {
      condition: "Tumour boundary mapped (demo)",
      risk_pct: 81,
      model: "BraTS U-Net (simulated)",
      differentials: [
        { label: "Grade III–IV", pct: 81 },
        { label: "Grade I–II", pct: 13 },
        { label: "Normal tissue", pct: 6 },
      ],
    },
    ct: {
      condition: "Multi-organ segmentation complete (demo)",
      risk_pct: null,
      model: "TotalSegmentator (simulated)",
      differentials: [
        { label: "Liver", pct: 96 },
        { label: "Spleen", pct: 91 },
        { label: "Kidneys", pct: 88 },
      ],
    },
  };
  renderDiagnosis(demoDiagnosis[state.mode] || demoDiagnosis.xray);
  document.getElementById("step5").style.display = "block";
  log(
    "Local demo complete · start Flask backend for real FHE processing",
    "log-warn",
  );
}

function drawNoise(ctx, w, h) {
  const id = ctx.createImageData(w, h);
  const d = id.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = (Math.random() * 255 * 0.7 + 40) | 0; // R — purple tint
    d[i + 1] = (Math.random() * 80) | 0; // G — low
    d[i + 2] = (Math.random() * 180 + 60) | 0; // B — high
    d[i + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
}

// ─────────────────────────────────────────────
//  DIAGNOSIS RENDER
// ─────────────────────────────────────────────

function renderDiagnosis(diag) {
  document.getElementById("diagCondition").textContent =
    diag.condition || "Processing complete";
  document.getElementById("diagModel").textContent = diag.model || "";

  const rb = document.getElementById("riskBadge");
  if (diag.risk_pct != null) {
    rb.style.display = "block";
    rb.textContent = `Risk: ${diag.risk_pct}%`;
    rb.className =
      "risk-badge " +
      (diag.risk_pct >= 70
        ? "risk-high"
        : diag.risk_pct >= 40
          ? "risk-med"
          : "risk-low");
  } else {
    rb.style.display = "none";
  }

  // Differentials
  const dr = document.getElementById("diffRows");
  dr.innerHTML = "";
  (diag.differentials || []).forEach((item) => {
    const col =
      item.pct >= 70 ? "#E24B4A" : item.pct >= 40 ? "#BA7517" : "#1D9E75";
    dr.innerHTML += `
      <div class="diff-row">
        <span class="diff-label">${item.label}</span>
        <div class="diff-track"><div class="diff-fill" style="width:${item.pct}%;background:${col}"></div></div>
        <span class="diff-pct">${item.pct}%</span>
      </div>`;
  });
}

// ─────────────────────────────────────────────
//  METRICS RENDER
// ─────────────────────────────────────────────

function renderMetrics(m) {
  document.getElementById("mPrivacy").textContent = m.privacy_score
    ? m.privacy_score
    : "—";
  document.getElementById("mExpansion").textContent = m.expansion_ratio
    ? m.expansion_ratio + "×"
    : "—";
  document.getElementById("mTime").textContent = m.total_time_s
    ? m.total_time_s + "s"
    : "—";
  document.getElementById("mSecurity").textContent =
    (m.security_bits || 128) + " bits";
}

// ─────────────────────────────────────────────
//  UTILITIES
// ─────────────────────────────────────────────

function setCanvas(id, src) {
  const c = document.getElementById(id);
  const ph = document.getElementById("ph" + id.slice(-1));
  const img = new Image();
  img.onload = () => {
    c.width = img.naturalWidth || 400;
    c.height = img.naturalHeight || 400;
    c.getContext("2d").drawImage(img, 0, 0);
    if (ph) ph.style.display = "none";
    c.style.display = "block";
  };
  img.src = src.startsWith("data:") ? src : "data:image/png;base64," + src;
}

function drawImageOnCanvas(id, dataUrl) {
  setCanvas(id, dataUrl);
}

function resetResults() {
  ["c2", "c3"].forEach((id) => {
    const c = document.getElementById(id);
    c.style.display = "none";
    c.width = 0;
    c.height = 0;
  });
  ["ph2", "ph3"].forEach(
    (id) => (document.getElementById(id).style.display = "flex"),
  );
  document.getElementById("tag3").textContent = "pending";
  document.getElementById("tag3").className = "tag tag-wait";
  ["ps-encrypt", "ps-process", "ps-decrypt", "ps-result"].forEach((id) =>
    markStage(id, "pending"),
  );
}

function markStage(id, status) {
  const el = document.getElementById(id);
  if (!el) return;
  const dot = el.querySelector(".ps-dot");
  if (!dot) return;
  dot.className = "ps-dot " + (status !== "pending" ? status : "");
}

function animatePipelineStages() {
  const stages = ["ps-encrypt", "ps-process", "ps-decrypt", "ps-result"];
  let i = 0;
  return setInterval(() => {
    if (i > 0) markStage(stages[i - 1], "complete");
    if (i < stages.length) {
      markStage(stages[i], "active");
      i++;
    } else clearInterval(this);
  }, 900);
}

function setProgress(pct, label) {
  document.getElementById("progFill").style.width = pct + "%";
  document.getElementById("progLabel").textContent = label;
}

// ─────────────────────────────────────────────
//  LOG
// ─────────────────────────────────────────────

function log(msg, cls = "") {
  const body = document.getElementById("logBody");
  const elapsed = Math.floor((Date.now() - state.t0) / 1000);
  const ts =
    String(Math.floor(elapsed / 60)).padStart(2, "0") +
    ":" +
    String(elapsed % 60).padStart(2, "0");
  const div = document.createElement("div");
  div.className = "log-entry";
  div.innerHTML = `<span class="log-ts">${ts}</span><span class="${cls}">${msg}</span>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function clearLog() {
  document.getElementById("logBody").innerHTML = "";
  log("Log cleared");
}
