

const INPUT_SIZE = 416;
const VALUES_PER_DETECTION = 56;
let moduleInstance = null;
let runtimeMode = "simd-single";
let ready = false;
let canvas = null;
let context = null;
let pixelPointer = 0;
let outputPointer = 0;
let config = null;
let frameIndex = 0;
let nextTrackId = 1;
let tracks = [];
const MODEL_CACHE_NAME = "yolo26n-pose-assets-v1";

function allocateCString(value) {
  const size = moduleInstance.lengthBytesUTF8(value) + 1;
  const pointer = moduleInstance._malloc(size);
  moduleInstance.stringToUTF8(value, pointer, size);
  return pointer;
}

async function downloadBytes(url, label, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const absoluteUrl = new URL(url, self.location.href).href;
    const request = new Request(absoluteUrl, { method: "GET" });
    let cache = null;
    try { cache = "caches" in self ? await caches.open(MODEL_CACHE_NAME) : null; } catch {}
    const cached = cache ? await cache.match(request).catch(() => null) : null;
    if (cached) {
      const bytes = new Uint8Array(await cached.arrayBuffer());
      if (bytes.length) {
        self.postMessage({
          type: "progress",
          status: "Using cached " + label,
          detail: label + ": loaded from this device's model cache (" + (bytes.length / 1048576).toFixed(2) + " MB)."
        });
        return bytes;
      }
    }
    self.postMessage({ type: "progress", status: "Downloading " + label, detail: "Connecting to the model file…" });
    const response = await fetch(request, { signal: controller.signal, cache: "default" });
    if (!response.ok) {
      const error = new Error(label + " download failed (HTTP " + response.status + ").");
      error.stage = "model";
      throw error;
    }
    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (!bytes.length) throw new Error(label + " downloaded as an empty file.");
      try { if (cache) await cache.put(request, new Response(bytes, {
        headers: { "Content-Type": response.headers.get("content-type") || "application/octet-stream" }
      })); } catch {}
      return bytes;
    }

    const reader = response.body.getReader();
    const chunks = [];
    const total = Number(response.headers.get("content-length")) || 0;
    let loaded = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      const received = (loaded / 1048576).toFixed(2);
      const expected = total ? " / " + (total / 1048576).toFixed(2) + " MB" : "";
      const percent = total ? " (" + Math.min(100, Math.round(loaded / total * 100)) + "%)" : "";
      self.postMessage({
        type: "progress",
        status: "Downloading " + label,
        detail: label + ": " + received + " MB" + expected + percent
      });
    }
    if (!loaded) throw new Error(label + " downloaded as an empty file.");
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    try { if (cache) await cache.put(request, new Response(bytes, {
      headers: { "Content-Type": response.headers.get("content-type") || "application/octet-stream" }
    })); } catch {
      self.postMessage({
        type: "progress",
        status: "Model downloaded",
        detail: label + " is ready; persistent browser storage was unavailable, so it may download again next time."
      });
    }
    return bytes;
  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error(label + " download timed out after " + Math.round(timeoutMs / 1000) + " seconds.");
      timeoutError.stage = "model";
      throw timeoutError;
    }
    if (!error.stage) error.stage = "model";
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function initialize(forceSingleThread = false) {
  self.postMessage({ type: "progress", status: "Loading model configuration", detail: "Reading the model input and output settings…" });
  let configBytes;
  try {
    configBytes = await downloadBytes("./model/model-config.json", "model configuration", 30000);
    config = JSON.parse(new TextDecoder().decode(configBytes));
  } catch (error) {
    error.stage = "model";
    throw error;
  }

  const [paramBytes, binBytes] = await Promise.all([
    downloadBytes("./model/model.param", "model.param", 60000),
    downloadBytes("./model/model.bin", "model.bin", 250000)
  ]);

  self.postMessage({ type: "progress", status: "Loading NCNN runtime", detail: "Both model files downloaded. Preparing the browser inference engine…" });
  const canUseThreads = !forceSingleThread && self.crossOriginIsolated &&
    typeof SharedArrayBuffer !== "undefined";
  const runtimeUrl = canUseThreads
    ? "./runtime/ncnn-threaded.js"
    : "./runtime/ncnn.js";
  const { default: createNcnnModule } = await import(runtimeUrl);
  runtimeMode = canUseThreads ? "simd-threads" : "simd-single";
  const numThreads = canUseThreads
    ? Math.max(1, Math.min(self.navigator.hardwareConcurrency || 2, 8))
    : 1;
  moduleInstance = await createNcnnModule({
    locateFile: file => new URL("./runtime/" + file, import.meta.url).href
  });
  self.postMessage({ type: "progress", status: "Initializing YOLO model", detail: "NCNN runtime loaded. Initializing YOLO26n-Pose…" });
  moduleInstance.FS.writeFile("/model.param", new Uint8Array(paramBytes));
  moduleInstance.FS.writeFile("/model.bin", new Uint8Array(binBytes));

  const paramPath = allocateCString("/model.param");
  const binPath = allocateCString("/model.bin");
  const inputName = allocateCString(config.input);
  const outputName = allocateCString(config.output);
  const result = moduleInstance._init_model(paramPath, binPath, inputName, outputName, numThreads);
  [paramPath, binPath, inputName, outputName].forEach(pointer => moduleInstance._free(pointer));
  if (result !== 0) throw new Error("NCNN model initialization failed (" + result + ").");

  canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  context = canvas.getContext("2d", { willReadFrequently: true });
  pixelPointer = moduleInstance._malloc(INPUT_SIZE * INPUT_SIZE * 4);
  outputPointer = moduleInstance._malloc(50 * VALUES_PER_DETECTION * 4);
  if (!pixelPointer || !outputPointer) throw new Error("Could not allocate inference buffers.");

  ready = true;
  self.postMessage({ type: "ready", runtimeMode, numThreads });
}


function boxIoU(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / (areaA + areaB - intersection + 1e-6);
}

function attachTrackIds(detections) {
  frameIndex += 1;
  const claimed = new Set();

  for (const detection of detections) {
    let bestTrack = null;
    let bestOverlap = 0.20;

    for (const track of tracks) {
      if (claimed.has(track.id)) continue;
      const overlap = boxIoU(detection.box, track.box);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestTrack = track;
      }
    }

    if (bestTrack) {
      bestTrack.box = detection.box;
      bestTrack.lastSeen = frameIndex;
      detection.trackId = bestTrack.id;
      claimed.add(bestTrack.id);
    } else {
      const id = nextTrackId++;
      tracks.push({ id, box: detection.box, lastSeen: frameIndex });
      detection.trackId = id;
      claimed.add(id);
    }
  }

  tracks = tracks.filter(track => frameIndex - track.lastSeen <= 12);
  return detections;
}

function decodeResults(bitmap) {
  const sourceWidth = bitmap.width;
  const sourceHeight = bitmap.height;
  const scale = Math.min(INPUT_SIZE / sourceWidth, INPUT_SIZE / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const padX = (INPUT_SIZE - drawWidth) * 0.5;
  const padY = (INPUT_SIZE - drawHeight) * 0.5;

  const prepStart = performance.now();
  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  context.drawImage(bitmap, padX, padY, drawWidth, drawHeight);
  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  moduleInstance.HEAPU8.set(pixels, pixelPointer);
  const prepMs = performance.now() - prepStart;

  const inferenceStart = performance.now();
  const result = moduleInstance._run_inference(pixelPointer, INPUT_SIZE, INPUT_SIZE);
  const inferenceMs = performance.now() - inferenceStart;
  if (result < 0) throw new Error("NCNN inference failed (" + result + ").");

  const decodeStart = performance.now();
  const count = moduleInstance._get_detection_count();
  if (count > 0) {
    moduleInstance._copy_detections(outputPointer);
  }
  const start = outputPointer >> 2;
  const values = moduleInstance.HEAPF32.subarray(
    start, start + count * VALUES_PER_DETECTION
  );

  const detections = [];
  for (let i = 0; i < count; i++) {
    const offset = i * VALUES_PER_DETECTION;
    const x = value => Math.max(0, Math.min(sourceWidth, (value - padX) / scale));
    const y = value => Math.max(0, Math.min(sourceHeight, (value - padY) / scale));
    const box = [x(values[offset]), y(values[offset + 1]),
                 x(values[offset + 2]), y(values[offset + 3])];
    const keypoints = [];
    for (let k = 0; k < 17; k++) {
      const p = offset + 5 + k * 3;
      keypoints.push([
        x(values[p]),
        y(values[p + 1]),
        values[p + 2]
      ]);
    }
    detections.push({
      label: "Person",
      score: values[offset + 4],
      box,
      keypoints
    });
  }

  const tracked = attachTrackIds(detections);
  const decodeMs = performance.now() - decodeStart;
  return { sourceWidth, sourceHeight, detections: tracked, inferenceMs, prepMs, decodeMs };
}

self.onmessage = async event => {
  const message = event.data;
  if (message.type === "init") {
    try {
      await initialize(Boolean(message.forceSingleThread));
    } catch (error) {
      self.postMessage({ type: "error", stage: error.stage || "init", message: error.message });
    }
    return;
  }

  if (message.type === "frame" && ready) {
    const bitmap = message.bitmap;
    try {
      const result = decodeResults(bitmap);
      self.postMessage({ type: "result", ...result });
    } catch (error) {
      self.postMessage({ type: "frame-error", message: error.message });
    } finally {
      bitmap.close();
    }
  }
};
