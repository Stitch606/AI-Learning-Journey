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

let cameraStream = null;
let selectedImage = null;
let animationFrame = 0;
let latestDetections = [];

function supportsWasmSimd() {
  try {
    const bytes = new Uint8Array([
      0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,
      3,2,1,0,10,22,1,20,0,253,12,
      0,0,0,0,0,0,0,0,0,0,0,0,11
    ]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}

function setStatus(text, ready = false) {
  statusText.textContent = text;
  statusDot.style.background = ready ? "#5ee7a0" : "#e7b65a";
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  video.pause();
  video.srcObject = null;
  cancelAnimationFrame(animationFrame);
}

function drawLoop() {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.videoWidth && video.videoHeight) {
    renderer.draw(latestDetections, video.videoWidth, video.videoHeight);
  }
  animationFrame = requestAnimationFrame(drawLoop);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    notice.textContent = "Camera access requires HTTPS and a browser with camera support.";
    return;
  }

  try {
    stopCamera();
    selectedImage = null;
    document.querySelector("#image-preview")?.remove();
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
    cameraButton.textContent = "Camera running";
    cameraButton.disabled = true;
    drawLoop();
  } catch (error) {
    notice.textContent = "Could not start camera: " + error.message;
  }
}

cameraButton.addEventListener("click", startCamera);

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  stopCamera();
  latestDetections = [];
  renderer.clear();
  const url = URL.createObjectURL(file);
  const image = new Image();

  image.onload = () => {
    selectedImage = image;
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
    preview.src = url;
    renderer.draw(latestDetections, image.naturalWidth, image.naturalHeight);
    cameraButton.disabled = false;
    cameraButton.textContent = "Start camera";
    notice.textContent = "Image loaded locally. Pose inference is not connected yet.";
  };

  image.onerror = () => {
    URL.revokeObjectURL(url);
    notice.textContent = "The selected image could not be decoded.";
  };
  image.src = url;
});

window.addEventListener("pagehide", stopCamera);

if (supportsWasmSimd()) {
  setStatus("SIMD ready · model missing");
  notice.textContent = "WebAssembly SIMD is supported, but NCNN runtime, exported model files, and inference bridge are not present yet. The live pose renderer is ready for detections.";
} else {
  setStatus("SIMD unsupported");
  notice.textContent = "This browser does not support WebAssembly SIMD. The selected NCNN SIMD runtime cannot run here.";
}

/*
 * Inference bridge contract:
 * window.publishPoseDetections({
 *   sourceWidth: 416,
 *   sourceHeight: 416,
 *   detections: [{
 *     trackId: 0,
 *     label: "Person",
 *     score: 0.94,
 *     box: [x1, y1, x2, y2],
 *     keypoints: [[x, y, confidence], ...17 COCO points]
 *   }]
 * });
 *
 * Coordinates must be in the source image/video coordinate system.
 * The NCNN bridge should call this once per completed inference frame.
 */
window.publishPoseDetections = ({ sourceWidth, sourceHeight, detections }) => {
  latestDetections = Array.isArray(detections) ? detections : [];
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    renderer.draw(latestDetections, sourceWidth || video.videoWidth, sourceHeight || video.videoHeight);
  } else if (selectedImage) {
    renderer.draw(latestDetections, sourceWidth || selectedImage.naturalWidth, sourceHeight || selectedImage.naturalHeight);
  }
};
