import createNcnnModule from "./runtime/ncnn.js";

const INPUT_SIZE = 416;
const VALUES_PER_DETECTION = 56;
let moduleInstance = null;
let ready = false;
let canvas = null;
let context = null;
let pixelPointer = 0;
let outputPointer = 0;
let config = null;

function allocateCString(value) {
  const size = moduleInstance.lengthBytesUTF8(value) + 1;
  const pointer = moduleInstance._malloc(size);
  moduleInstance.stringToUTF8(value, pointer, size);
  return pointer;
}

async function initialize() {
  const configResponse = await fetch("./model/model-config.json");
  if (!configResponse.ok) throw new Error("Model configuration was not found.");
  config = await configResponse.json();

  const [paramResponse, binResponse] = await Promise.all([
    fetch("./model/model.param"),
    fetch("./model/model.bin")
  ]);
  if (!paramResponse.ok || !binResponse.ok) {
    throw new Error("The exported YOLO26n-Pose NCNN files are missing.");
  }

  const [paramBytes, binBytes] = await Promise.all([
    paramResponse.arrayBuffer(),
    binResponse.arrayBuffer()
  ]);

  moduleInstance = await createNcnnModule({
    locateFile: file => new URL("./runtime/" + file, import.meta.url).href
  });
  moduleInstance.FS.writeFile("/model.param", new Uint8Array(paramBytes));
  moduleInstance.FS.writeFile("/model.bin", new Uint8Array(binBytes));

  const paramPath = allocateCString("/model.param");
  const binPath = allocateCString("/model.bin");
  const inputName = allocateCString(config.input);
  const outputName = allocateCString(config.output);
  const result = moduleInstance._init_model(paramPath, binPath, inputName, outputName);
  [paramPath, binPath, inputName, outputName].forEach(pointer => moduleInstance._free(pointer));
  if (result !== 0) throw new Error("NCNN model initialization failed (" + result + ").");

  canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  context = canvas.getContext("2d", { willReadFrequently: true });
  pixelPointer = moduleInstance._malloc(INPUT_SIZE * INPUT_SIZE * 4);
  outputPointer = moduleInstance._malloc(50 * VALUES_PER_DETECTION * 4);
  if (!pixelPointer || !outputPointer) throw new Error("Could not allocate inference buffers.");

  ready = true;
  self.postMessage({ type: "ready" });
}

function decodeResults(bitmap) {
  const sourceWidth = bitmap.width;
  const sourceHeight = bitmap.height;
  const scale = Math.min(INPUT_SIZE / sourceWidth, INPUT_SIZE / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const padX = (INPUT_SIZE - drawWidth) * 0.5;
  const padY = (INPUT_SIZE - drawHeight) * 0.5;

  context.fillStyle = "rgb(114,114,114)";
  context.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  context.drawImage(bitmap, padX, padY, drawWidth, drawHeight);
  const pixels = context.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  moduleInstance.HEAPU8.set(pixels, pixelPointer);

  const result = moduleInstance._run_inference(pixelPointer, INPUT_SIZE, INPUT_SIZE);
  if (result < 0) throw new Error("NCNN inference failed (" + result + ").");

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

  return { sourceWidth, sourceHeight, detections };
}

self.onmessage = async event => {
  const message = event.data;
  if (message.type === "init") {
    try {
      await initialize();
    } catch (error) {
      self.postMessage({ type: "error", message: error.message });
    }
    return;
  }

  if (message.type === "frame" && ready) {
    const bitmap = message.bitmap;
    try {
      const result = decodeResults(bitmap);
      self.postMessage({ type: "result", ...result });
    } catch (error) {
      self.postMessage({ type: "error", message: error.message });
    } finally {
      bitmap.close();
    }
  }
};
