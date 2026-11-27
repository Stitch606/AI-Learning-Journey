import { PoseRenderer } from "./renderer.js?v=4";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const emptyState = document.querySelector("#empty-state");
const cameraButton = document.querySelector("#camera-button");
const videoInput = document.querySelector("#video-input");
const downloadVideo = document.querySelector("#download-video");
const statusText = document.querySelector("#runtime-status span");
const statusDot = document.querySelector("#runtime-status i");
const notice = document.querySelector("#notice");
const modelProgress = document.querySelector("#model-progress");
const modelProgressBar = document.querySelector("#model-progress-bar");
const modelProgressText = document.querySelector("#model-progress-text");
const perfLine = document.querySelector("#perf");

let lastResultAt = 0;
let fpsEma = 0;
let lastCaptureMs = 0;
let gpuLabel = "YOLO provider: initializing";
if (navigator.gpu) {
  navigator.gpu.requestAdapter()
    .then(adapter => { gpuLabel = adapter ? "WebGPU adapter detected" : "WebGPU adapter unavailable"; })
    .catch(() => { gpuLabel = "WebGPU adapter error"; });
} else {
  gpuLabel = "WebGPU: not supported";
}

function updatePerf(message) {
  const now = performance.now();
  if (lastResultAt) {
    const fps = 1000 / (now - lastResultAt);
    fpsEma = fpsEma ? fpsEma * 0.8 + fps * 0.2 : fps;
  }
  lastResultAt = now;
  perfLine.textContent =
    `${fpsEma.toFixed(1)} FPS · capture ${lastCaptureMs.toFixed(0)} ms · ` +
    `prep ${message.prepMs.toFixed(0)} ms · YOLO ${message.inferenceMs.toFixed(0)} ms · ` +
    `decode ${message.decodeMs.toFixed(1)} ms · ${message.provider || gpuLabel}`;
}

const renderer = new PoseRenderer(canvas, {
  keypointThreshold: 0.25,
  smoothing: 0.38,
  lineWidth: 2.4,
  pointRadius: 3.1
});

const actionStatus = document.querySelector("#action-status");
const actionResult = document.querySelector("#action-result");
const actionConfidence = document.querySelector("#action-confidence");
let actionWorker = null;
let actionWorkerReady = false;
let actionWorkerStarting = false;
let pendingActionFrames = [];
let latestDetections = [];
let latestAction = null;
let latestWidth = 0;
let latestHeight = 0;

function ensureActionWorker() {
  if (actionWorker || actionWorkerStarting) return;
  actionWorkerStarting = true;
  actionStatus.textContent = "Loading action model on demand";
  actionWorker = new Worker("./action-worker.js?v=2", { type: "module" });
  actionWorker.addEventListener("message", event => {
    const message = event.data;
    if (message.type === "progress") actionStatus.textContent = message.message;
    if (message.type === "ready") {
      actionWorkerReady = true;
      actionWorkerStarting = false;
      actionStatus.textContent = message.provider + " · collecting 100 frames";
      actionResult.textContent = "Waiting for 100 frames";
      for (const frame of pendingActionFrames) actionWorker.postMessage(frame);
      pendingActionFrames = [];
    }
    if (message.type === "action") {
      latestAction = message.action;
      actionResult.textContent = message.action.label;
      actionConfidence.textContent = `${(message.action.confidence * 100).toFixed(1)}%`;
      actionStatus.textContent = `${(message.action.trackIds || []).length} people in the analyzed pair · ${message.action.inferenceMs.toFixed(1)} ms`;
      renderer.draw(latestDetections, latestWidth, latestHeight, latestAction);
    }
    if (message.type === "error" || message.type === "inference-error") {
      actionStatus.textContent = "Action model error";
      actionResult.textContent = message.message;
      actionWorkerStarting = false;
    }
  });
  actionWorker.addEventListener("error", event => {
    actionStatus.textContent = "Action runtime error";
    actionResult.textContent = event.message || "Could not load the action model.";
    actionWorkerStarting = false;
  });
  actionWorker.postMessage({ type: "init" });
}

function sendActionFrame(frame) {
  if (actionWorkerReady) {
    actionWorker.postMessage(frame);
    return;
  }
  pendingActionFrames.push(frame);
  if (pendingActionFrames.length > 100) pendingActionFrames.shift();
}

function resetActionWorker() {
  pendingActionFrames = [];
  latestAction = null;
  if (actionWorker) actionWorker.postMessage({ type: "reset" });
}

async function prepareThreadIsolation() {
  if (!("serviceWorker" in navigator)) return false;
  try {
    await navigator.serviceWorker.register("./coi-serviceworker.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) {
      await new Promise(resolve => {
        const timeout = setTimeout(resolve, 5000);
        navigator.serviceWorker.addEventListener("controllerchange", () => {
          clearTimeout(timeout);
          resolve();
        }, { once: true });
      });
      if (navigator.serviceWorker.controller) {
        window.location.reload();
        return false;
      }
    }
    return Boolean(window.crossOriginIsolated &&
      typeof SharedArrayBuffer !== "undefined");
  } catch (error) {
    console.warn("Cross-origin isolation unavailable; ONNX Runtime will use its compatible fallback.", error);
    return false;
  }
}

let threadIsolationReady = false;

let cameraStream = null;
let cameraRunning = false;
let workerReady = false;
let inferenceBusy = false;
let animationFrame = 0;
let lastSubmittedAt = 0;
let pendingBitmap = null;
let selectedVideoUrl = null;
let selectedVideoStarted = false;
let worker = null;
let workerInitTimer = 0;
let workerStage = "Starting ONNX Runtime WebGPU";
let usingSingleThreadFallback = false;

cameraButton.disabled = false;
videoInput.disabled = false;

function launchWorker(forceSingleThread = false) {
  clearTimeout(workerInitTimer);
  worker?.terminate();
  workerReady = false;
  inferenceBusy = false;
  worker = new Worker("./inference-worker.js?v=7", { type: "module" });

  worker.addEventListener("message", event => {
    const message = event.data;

    if (message.type === "progress") {
      workerStage = message.status;
      setStatus(message.status);
      notice.textContent = message.detail;
      if (message.loaded !== undefined) {
        modelProgress.hidden = false;
        modelProgressText.textContent = message.total
          ? `${(message.loaded / 1048576).toFixed(2)} / ${(message.total / 1048576).toFixed(2)} MB · ${Math.min(100, Math.round(message.loaded / message.total * 100))}%`
          : `${(message.loaded / 1048576).toFixed(2)} MB downloaded`;
        if (message.total) {
          modelProgressBar.max = message.total;
          modelProgressBar.value = message.loaded;
          modelProgressBar.removeAttribute("data-indeterminate");
        } else {
          modelProgressBar.removeAttribute("value");
          modelProgressBar.setAttribute("data-indeterminate", "true");
        }
      } else {
        modelProgress.hidden = false;
        modelProgressText.textContent = message.detail || message.status;
        modelProgressBar.removeAttribute("value");
        modelProgressBar.setAttribute("data-indeterminate", "true");
      }
      armWorkerTimeout();
      return;
    }

    if (message.type === "ready") {
      clearTimeout(workerInitTimer);
      workerReady = true;
      cameraButton.disabled = cameraRunning;
      videoInput.disabled = false;
      gpuLabel = message.provider || "YOLO runtime ready";
      modelProgress.hidden = false;
      modelProgressBar.max = 1;
      modelProgressBar.value = 1;
      modelProgressBar.removeAttribute("data-indeterminate");
      modelProgressText.textContent = "YOLO model ready";
      setStatus(`${message.provider} · 512×512 · model ready`, true);
      if (cameraRunning) {
        if (!animationFrame) animationFrame = requestAnimationFrame(captureCameraFrame);
        notice.textContent = selectedVideoUrl
          ? "YOLO26n-Pose is ready. Local video analysis is running."
          : "YOLO26n-Pose is ready. Live pose inference is running.";
      } else {
        notice.textContent = "YOLO26n-Pose is loaded locally. Start the camera or choose a video.";
      }
      return;
    }

    if (message.type === "result") {
      if (message.provider && message.provider !== gpuLabel) {
        gpuLabel = message.provider;
        setStatus(`${message.provider} · 512×512 · model ready`, true);
      }
      latestDetections = message.detections;
      latestWidth = message.sourceWidth;
      latestHeight = message.sourceHeight;
      updatePerf(message);
      renderer.draw(latestDetections, latestWidth, latestHeight, latestAction);
      sendActionFrame({
        type: "pose-frame",
        detections: message.detections,
        sourceWidth: message.sourceWidth,
        sourceHeight: message.sourceHeight
      });
      inferenceBusy = false;
      processPendingFrame();
      return;
    }

    if (message.type === "frame-error") {
      inferenceBusy = false;
      setStatus("Inference error");
      notice.textContent = message.message;
      processPendingFrame();
      return;
    }

    if (message.type === "error") {
      if (!usingSingleThreadFallback && message.stage !== "frame" && message.stage !== "model") {
        retrySingleThread(message.message);
        return;
      }
      clearTimeout(workerInitTimer);
      inferenceBusy = false;
      setStatus("Model loading failed");
      notice.textContent = "YOLO ONNX could not start: " + message.message;
      videoInput.disabled = true;
      processPendingFrame();
    }
  });

  worker.addEventListener("error", event => {
    if (!usingSingleThreadFallback) {
      retrySingleThread(event.message || "Threaded runtime crashed.");
      return;
    }
    clearTimeout(workerInitTimer);
    videoInput.disabled = true;
    inferenceBusy = false;
    setStatus("Runtime error");
    notice.textContent = event.message || "Could not load the YOLO ONNX runtime.";
  });

  workerStage = "Starting ONNX Runtime WebGPU";
  armWorkerTimeout();
  worker.postMessage({ type: "init", forceSingleThread });
}

function armWorkerTimeout() {
  clearTimeout(workerInitTimer);
  workerInitTimer = setTimeout(() => {
    if (workerReady) return;
    if (!usingSingleThreadFallback) {
      retrySingleThread(workerStage + " timed out in the threaded runtime.");
      return;
    }
    setStatus("Model loading timed out");
    notice.textContent = workerStage + " is taking longer than 250 seconds. Check the connection, reload the page, and try again.";
    videoInput.disabled = true;
    worker?.terminate();
  }, 250000);
}

function retrySingleThread(reason) {
  if (usingSingleThreadFallback) return;
  usingSingleThreadFallback = true;
  clearTimeout(workerInitTimer);
  videoInput.disabled = true;
  setStatus("Retrying compatible ONNX runtime");
  notice.textContent = reason + " Switching to the compatible single-thread runtime…";
  launchWorker(true);
}

function setStatus(text, ready = false) {
  statusText.textContent = text;
  statusDot.style.background = ready ? "#5ee7a0" : "#e7b65a";
}

function supportsWasmSimd() {
  try {
    return WebAssembly.validate(new Uint8Array([
      0,97,115,109,1,0,0,0,
      1,5,1,96,0,1,123,
      3,2,1,0,
      10,22,1,20,0,253,12,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      11
    ]));
  } catch {
    return false;
  }
}

function submitBitmap(bitmap, priority = false) {
  if (!workerReady) {
    bitmap.close();
    return;
  }
  if (inferenceBusy) {
    if (priority) {
      pendingBitmap?.close();
      pendingBitmap = bitmap;
    } else {
      bitmap.close();
    }
    return;
  }
  inferenceBusy = true;
  worker.postMessage({ type: "frame", bitmap }, [bitmap]);
}

function processPendingFrame() {
  if (!pendingBitmap || inferenceBusy) return;
  const nextBitmap = pendingBitmap;
  pendingBitmap = null;
  submitBitmap(nextBitmap, true);
}

async function captureCameraFrame(timestamp) {
  if (!cameraRunning) return;

  if (!video.paused && timestamp - lastSubmittedAt >= 33 && !inferenceBusy && workerReady &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    lastSubmittedAt = timestamp;
    try {
      const captureStart = performance.now();
      const bitmap = await createImageBitmap(video);
      lastCaptureMs = performance.now() - captureStart;
      if (cameraRunning) submitBitmap(bitmap);
      else bitmap.close();
    } catch (error) {
      notice.textContent = "Could not capture camera frame: " + error.message;
    }
  }

  if (cameraRunning) {
    animationFrame = requestAnimationFrame(captureCameraFrame);
  }
}

function stopCamera() {
  cameraRunning = false;
  cancelAnimationFrame(animationFrame);
  animationFrame = 0;
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  video.pause();
  video.srcObject = null;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    notice.textContent = "Camera access requires HTTPS and a browser with camera support.";
    return;
  }

  try {
    stopCamera();
    ensureActionWorker();
    video.pause();
    video.removeAttribute("src");
    video.controls = false;
    video.load();
    if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
    selectedVideoUrl = null;
    downloadVideo.hidden = true;
    downloadVideo.removeAttribute("href");
    latestAction = null;
    latestDetections = [];
    actionResult.textContent = "Waiting for 100 frames";
    actionConfidence.textContent = "--";
    resetActionWorker();
    renderer.clear();

    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30, max: 30 }
      }
    });

    video.removeAttribute("src");
    video.controls = false;
    video.load();
    video.srcObject = cameraStream;
    video.style.display = "block";
    emptyState.style.display = "none";
    await video.play();
    cameraRunning = true;
    cameraButton.textContent = "Camera running";
    cameraButton.disabled = true;
    notice.textContent = "Live pose inference is running on this device.";
    if (workerReady && !animationFrame) animationFrame = requestAnimationFrame(captureCameraFrame);
  } catch (error) {
    notice.textContent = "Could not start camera: " + error.message;
  }
}

cameraButton.addEventListener("click", startCamera);

videoInput.addEventListener("change", async () => {
  const file = videoInput.files?.[0];
  if (!file) return;

  if (!file.type.startsWith("video/")) {
    notice.textContent = "Choose a supported video file.";
    videoInput.value = "";
    return;
  }
  stopCamera();
  ensureActionWorker();
  latestAction = null;
  latestDetections = [];
  latestWidth = 0;
  latestHeight = 0;
  actionResult.textContent = "Waiting for 100 frames";
  actionConfidence.textContent = "--";
  resetActionWorker();
  renderer.clear();

  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
  selectedVideoUrl = URL.createObjectURL(file);
  selectedVideoStarted = false;
  downloadVideo.href = selectedVideoUrl;
  downloadVideo.download = file.name || "analysis-video";
  downloadVideo.hidden = false;
  video.srcObject = null;
  video.src = selectedVideoUrl;
  video.controls = true;
  video.style.display = "block";
  emptyState.style.display = "none";
  cameraButton.disabled = true;
  cameraButton.textContent = "Video is playing";

  video.onended = () => {
    cameraRunning = false;
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    cameraButton.disabled = false;
    cameraButton.textContent = "Start camera";
    notice.textContent = "Video analysis finished.";
  };
  video.onplay = () => {
    if (selectedVideoStarted && !cameraRunning && video.src && !cameraStream) {
      cameraRunning = true;
      lastSubmittedAt = 0;
      animationFrame = requestAnimationFrame(captureCameraFrame);
      cameraButton.disabled = true;
      cameraButton.textContent = "Video is playing";
    }
  };
  video.onerror = () => {
    cameraRunning = false;
    cancelAnimationFrame(animationFrame);
    animationFrame = 0;
    cameraButton.disabled = false;
    cameraButton.textContent = "Start camera";
    notice.textContent = "This video format could not be played by the browser.";
  };

  videoInput.value = "";
  try {
    await video.play();
    selectedVideoStarted = true;
    cameraRunning = true;
    lastSubmittedAt = 0;
    notice.textContent = "Analyzing the selected video locally. Use the video controls to pause or seek.";
    if (workerReady && !animationFrame) animationFrame = requestAnimationFrame(captureCameraFrame);
  } catch (error) {
    cameraRunning = false;
    cameraButton.disabled = false;
    cameraButton.textContent = "Start camera";
    notice.textContent = "Could not play the selected video: " + error.message;
  }
});

window.addEventListener("pagehide", () => {
  stopCamera();
  pendingBitmap?.close();
  worker?.terminate();
  actionWorker?.terminate();
  if (selectedVideoUrl) URL.revokeObjectURL(selectedVideoUrl);
});

setStatus("Loading YOLO ONNX model");
notice.textContent = "Preparing the FP16 WebGPU model. A FP32 WASM fallback is available if WebGPU cannot initialize.";
Promise.race([
  prepareThreadIsolation(),
  new Promise(resolve => setTimeout(() => resolve(false), 8000))
]).then(ready => {
  threadIsolationReady = ready;
  window.__ncnnIsolationReady = ready;
  usingSingleThreadFallback = !ready;
  launchWorker(!ready);
}).catch(error => {
  console.warn("Thread isolation setup failed; using ONNX Runtime compatibility fallback.", error);
  threadIsolationReady = false;
  window.__ncnnIsolationReady = false;
  usingSingleThreadFallback = true;
  launchWorker(true);
});
