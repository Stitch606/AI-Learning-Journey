const video = document.querySelector("#camera");
const canvas = document.querySelector("#overlay");
const emptyState = document.querySelector("#empty-state");
const cameraButton = document.querySelector("#camera-button");
const imageInput = document.querySelector("#image-input");
const statusText = document.querySelector("#runtime-status span");
const statusDot = document.querySelector("#runtime-status i");
const notice = document.querySelector("#notice");

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

function setStatus(text, ready) {
  statusText.textContent = text;
  statusDot.style.background = ready ? "#5ee7a0" : "#e7b65a";
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    notice.textContent = "Camera access is unavailable. Open this page over HTTPS and use a supported browser.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    video.style.display = "block";
    emptyState.style.display = "none";
    cameraButton.textContent = "Camera running";
    cameraButton.disabled = true;
    await video.play();
    resizeOverlay();
  } catch (error) {
    notice.textContent = "Could not start camera: " + error.message;
  }
}

cameraButton.addEventListener("click", startCamera);
window.addEventListener("resize", resizeOverlay);

imageInput.addEventListener("change", function () {
  const file = imageInput.files && imageInput.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = function () {
    video.pause();
    video.srcObject = null;
    video.style.display = "none";
    emptyState.style.display = "flex";
    emptyState.querySelector("strong").textContent = file.name;
    emptyState.querySelector("span").textContent = image.naturalWidth + " × " + image.naturalHeight + " image selected";
    URL.revokeObjectURL(url);
  };
  image.src = url;
});

if (supportsWasmSimd()) {
  setStatus("SIMD supported", true);
  notice.textContent = "This browser supports WebAssembly SIMD. The NCNN SIMD runtime and pose inference bridge are the next integration step.";
} else {
  setStatus("SIMD unavailable", false);
  notice.textContent = "This browser does not report WebAssembly SIMD support. The SIMD build cannot run here.";
}
