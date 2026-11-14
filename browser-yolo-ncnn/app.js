import { PoseRenderer } from "./renderer.js";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const emptyState = document.querySelector("#empty-state");
const cameraButton = document.querySelector("#camera-button");
const imageInput = document.querySelector("#image-input");
const statusText = document.querySelector("#runtime-status span");
const statusDot = document.querySelector("#runtime-status i");
const notice = document.querySelector("#notice");

const renderer = new PoseRenderer(canvas, {
  keypointThreshold: 0.25,
  smoothing: 0.38,
  lineWidth: 2.4,
  pointRadius: 3.1
});

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
    console.warn("Threaded NCNN unavailable; using SIMD single-thread fallback.", error);
    return false;
  }
}

const threadIsolationReady = await prepareThreadIsolation();
window.__ncnnIsolationReady = threadIsolationReady;

let cameraStream = null;
let cameraRunning = false;
let workerReady = false;
let inferenceBusy = false;
let animationFrame = 0;
let lastSubmittedAt = 0;
let pendingBitmap = null;
let selectedImageUrl = null;
let worker = null;
let workerInitTimer = 0;
let workerStage = "Starting NCNN";
let usingSingleThreadFallback = false;

cameraButton.disabled = true;
imageInput.disabled = true;

function launchWorker(forceSingleThread = false) {
  clearTimeout(workerInitTimer);
  worker?.terminate();
  workerReady = false;
  inferenceBusy = false;
  worker = new Worker("./inference-worker.js", { type: "module" });

  worker.addEventListener("message", event => {
    const message = event.data;

    if (message.type === "progress") {
      workerStage = message.status;
      setStatus(message.status);
      notice.textContent = message.detail;
      armWorkerTimeout();
      return;
    }

    if (message.type === "ready") {
      clearTimeout(workerInitTimer);
      workerReady = true;
      cameraButton.disabled = false;
      imageInput.disabled = false;
      setStatus(message.runtimeMode === "simd-threads"
        ? `NCNN · SIMD · ${message.numThreads} CPU threads · model ready`
        : "NCNN · SIMD · single thread · model ready", true);
      notice.textContent = "YOLO26n-Pose is loaded locally. Start the camera or choose an image.";
      return;
    }

    if (message.type === "result") {
      renderer.draw(message.detections, message.sourceWidth, message.sourceHeight);
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
      if (!usingSingleThreadFallback && message.stage !== "frame") {
        retrySingleThread(message.message);
        return;
      }
      clearTimeout(workerInitTimer);
      inferenceBusy = false;
      setStatus("Model loading failed");
      notice.textContent = "NCNN could not start: " + message.message;
      cameraButton.disabled = true;
      imageInput.disabled = true;
      processPendingFrame();
    }
  });

  worker.addEventListener("error", event => {
    if (!usingSingleThreadFallback) {
      retrySingleThread(event.message || "Threaded runtime crashed.");
      return;
    }
    clearTimeout(workerInitTimer);
    cameraButton.disabled = true;
    imageInput.disabled = true;
    inferenceBusy = false;
    setStatus("Runtime error");
    notice.textContent = event.message || "Could not load the NCNN WebAssembly runtime.";
  });

  workerStage = "Starting NCNN";
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
    notice.textContent = workerStage + " is taking longer than 180 seconds. Check the connection, reload the page, and try again.";
    cameraButton.disabled = true;
    imageInput.disabled = true;
    worker?.terminate();
  }, 180000);
}

function retrySingleThread(reason) {
  if (usingSingleThreadFallback) return;
  usingSingleThreadFallback = true;
  clearTimeout(workerInitTimer);
  cameraButton.disabled = true;
  imageInput.disabled = true;
  setStatus("Retrying with single-thread NCNN");
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

  if (timestamp - lastSubmittedAt >= 66 && !inferenceBusy && workerReady &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    lastSubmittedAt = timestamp;
    try {
      const bitmap = await createImageBitmap(video);
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
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  video.pause();
  video.srcObject = null;
}

async function startCamera() {
  if (!workerReady) {
    notice.textContent = "Wait until the NCNN model status says model ready.";
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    notice.textContent = "Camera access requires HTTPS and a browser with camera support.";
    return;
  }

  try {
    stopCamera();
    document.querySelector("#image-preview")?.remove();
    if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
    selectedImageUrl = null;
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

    video.srcObject = cameraStream;
    video.style.display = "block";
    emptyState.style.display = "none";
    await video.play();
    cameraRunning = true;
    cameraButton.textContent = "Camera running";
    cameraButton.disabled = true;
    notice.textContent = "Live pose inference is running on this device.";
    animationFrame = requestAnimationFrame(captureCameraFrame);
  } catch (error) {
    notice.textContent = "Could not start camera: " + error.message;
  }
}

cameraButton.addEventListener("click", startCamera);

imageInput.addEventListener("change", async () => {
  const file = imageInput.files?.[0];
  if (!file || !workerReady) return;

  stopCamera();
  renderer.clear();
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);

  const image = new Image();
  image.onload = async () => {
    video.style.display = "none";
    emptyState.style.display = "none";
    const stage = document.querySelector(".stage");
    let preview = document.querySelector("#image-preview");
    if (!preview) {
      preview = document.createElement("img");
      preview.id = "image-preview";
      preview.alt = "Selected image";
      preview.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain";
      stage.insertBefore(preview, canvas);
    }
    preview.src = selectedImageUrl;
    cameraButton.disabled = false;
    cameraButton.textContent = "Start camera";

    try {
      const bitmap = await createImageBitmap(file);
      submitBitmap(bitmap, true);
      notice.textContent = "Running pose inference on the selected image…";
    } catch (error) {
      notice.textContent = "Could not decode the selected image: " + error.message;
    }
  };
  image.onerror = () => {
    notice.textContent = "The selected image could not be decoded.";
  };
  image.src = selectedImageUrl;
});

window.addEventListener("pagehide", () => {
  stopCamera();
  pendingBitmap?.close();
  worker.terminate();
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
});

if (!supportsWasmSimd()) {
  setStatus("SIMD unsupported");
  notice.textContent = "Camera preview is available, but this browser does not support WebAssembly SIMD, so local pose inference cannot start.";
} else {
  setStatus("Loading NCNN model");
  notice.textContent = "Loading the local NCNN model. Camera controls unlock when the model is ready.";
  usingSingleThreadFallback = !threadIsolationReady;
  launchWorker(!threadIsolationReady);
}
