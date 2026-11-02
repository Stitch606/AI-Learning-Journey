import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

const $ = (id) => document.getElementById(id);
const liveMode = $("liveMode");
const videoMode = $("videoMode");
const modeIndicator = $("modeIndicator");
const viewport = $("viewport");
const cameraPreview = $("cameraPreview");
const filePreview = $("filePreview");
const emptyTitle = $("emptyTitle");
const emptyDescription = $("emptyDescription");
const startCamera = $("startCamera");
const videoInput = $("videoInput");
const chooseFile = $("chooseFile");
const analyzeButton = $("analyzeButton");
const notice = $("notice");
const noticeText = $("noticeText");

let currentMode = "live";
let cameraStream = null;
let selectedFile = null;
let clientPromise = null;

function setNotice(message, isError = false) {
  noticeText.textContent = message;
  notice.classList.toggle("error", isError);
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraPreview.srcObject = null;
  viewport.classList.remove("has-camera");
}

function setMode(mode) {
  currentMode = mode;
  liveMode.classList.toggle("active", mode === "live");
  videoMode.classList.toggle("active", mode === "video");
  liveMode.setAttribute("aria-pressed", String(mode === "live"));
  videoMode.setAttribute("aria-pressed", String(mode === "video"));
  modeIndicator.classList.toggle("video", mode === "video");
  $("monitorTitle").textContent = mode === "live" ? "LIVE INPUT" : "VIDEO INPUT";
  $("monitorCode").textContent = mode === "live" ? "CAM / 01" : "FILE / 02";
  $("monitorLed").style.background = mode === "live" ? "var(--cyan)" : "var(--violet)";
  chooseFile.style.display = mode === "video" ? "inline-flex" : "none";
  $("dropHint").style.display = mode === "live" ? "block" : "none";

  if (mode === "live") {
    filePreview.pause();
    filePreview.removeAttribute("src");
    filePreview.load();
    viewport.classList.remove("has-file");
    $("sourceName").textContent = cameraStream ? "Live camera connected" : "No camera source selected";
    $("viewportResolution").textContent = cameraStream ? "LIVE FEED" : "AWAITING SOURCE";
    emptyTitle.textContent = "Camera is standing by";
    emptyDescription.textContent = "Start your camera to preview a live feed";
    startCamera.style.display = "inline-flex";
    analyzeButton.disabled = true;
  } else {
    stopCamera();
    viewport.classList.remove("has-camera");
    $("sourceName").textContent = selectedFile ? selectedFile.name : "No video file selected";
    $("viewportResolution").textContent = selectedFile ? "FILE LOADED" : "AWAITING SOURCE";
    emptyTitle.textContent = "Add a video to begin";
    emptyDescription.textContent = "Choose a video file from your device";
    startCamera.style.display = "none";
    analyzeButton.disabled = !selectedFile;
  }
}

liveMode.addEventListener("click", () => setMode("live"));
videoMode.addEventListener("click", () => setMode("video"));

startCamera.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    setNotice("Camera access is unavailable in this browser context. Open the Space over HTTPS or use video upload.", true);
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false
    });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    viewport.classList.add("has-camera");
    $("sourceName").textContent = "Live camera connected";
    $("viewportResolution").textContent = "LIVE FEED";
    startCamera.style.display = "none";
    setNotice("Camera preview is active. The detection model is not connected yet.");
  } catch (error) {
    setNotice("Camera permission was not granted or the camera is unavailable.", true);
  }
});

videoInput.addEventListener("change", () => {
  const file = videoInput.files?.[0];
  if (!file) return;
  stopCamera();
  selectedFile = file;
  filePreview.src = URL.createObjectURL(file);
  filePreview.load();
  viewport.classList.add("has-file");
  $("sourceName").textContent = file.name;
  $("viewportResolution").textContent = "FILE LOADED";
  analyzeButton.disabled = false;
  setNotice("Video loaded successfully. The detection pipeline still needs to be connected.");
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Connecting…";
  setNotice("Sending the video to the Python backend…");
  try {
    clientPromise ??= Client.connect(window.location.origin);
    const client = await clientPromise;
    const result = await client.predict("/analyze_video", {
      video_path: handle_file(selectedFile)
    });
    const payload = result.data?.[0];
    setNotice(payload?.message || "Backend received the request.");
  } catch (error) {
    setNotice("Could not reach the Gradio backend: " + (error?.message || "unknown error"), true);
  } finally {
    analyzeButton.disabled = !selectedFile;
    analyzeButton.innerHTML = 'Analyze input <span class="arrow">→</span>';
  }
});

window.addEventListener("beforeunload", stopCamera);
setMode("live");
