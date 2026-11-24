import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const port = 4173;
const server = spawn("python", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
  stdio: "ignore",
  cwd: "browser-yolo-ncnn"
});
let browser;
const pageErrors = [];

try {
  await delay(1200);
  browser = await chromium.launch({
    headless: true,
    args: [
      "--enable-webassembly-simd",
      "--use-fake-device-for-media-stream",
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required"
    ]
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle" });
  if (!(await page.locator("#video-input").isEnabled()))
    throw new Error("Video upload should be available while the pose model is loading.");

  await page.waitForFunction(() => window.__ncnnIsolationReady === true ||
    document.querySelector("#runtime-status span")?.textContent.includes("model ready"),
    null, { timeout: 20000 });

  const runtimeConfig = await page.evaluate(async () => {
    const response = await fetch("./model/model-config.json");
    if (!response.ok) throw new Error("NCNN model config missing.");
    return response.json();
  });
  if (runtimeConfig.precision !== "FP16" || runtimeConfig.input_size !== 416)
    throw new Error("Expected FP16 NCNN model at 416x416.");

  const isolation = await page.evaluate(() => ({
    crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer !== "undefined"
  }));
  const initialRuntimeStatus = await page.locator("#runtime-status span").textContent();
  const threadedRuntimeAvailable = isolation.crossOriginIsolated && isolation.sharedArrayBuffer;
  if (!threadedRuntimeAvailable && !initialRuntimeStatus.includes("single thread"))
    throw new Error("The app did not select its single-thread fallback when SharedArrayBuffer was unavailable.");
  console.log(`RUNTIME_MODE ${JSON.stringify({ ...isolation, selected: threadedRuntimeAvailable ? "simd-threads" : "simd-single" })}`);

  const simd = await page.evaluate(() => WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,22,1,20,0,253,12,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11
  ])));
  if (!simd) throw new Error("Chromium does not validate WebAssembly SIMD.");

  await page.waitForFunction(
    () => document.querySelector("#runtime-status span")?.textContent.includes("model ready"),
    null,
    { timeout: 120000 }
  );
  const cameraButton = page.locator("#camera-button");
  await cameraButton.click();
  await page.waitForFunction(() => {
    const video = document.querySelector("#camera");
    return video && video.videoWidth > 0 && video.videoHeight > 0;
  }, null, { timeout: 15000 });
  const camera = await page.evaluate(() => ({
    width: document.querySelector("#camera").videoWidth,
    height: document.querySelector("#camera").videoHeight
  }));
  console.log(`CAMERA_OK ${JSON.stringify(camera)}`);

  await page.reload({ waitUntil: "networkidle" });
  await page.waitForFunction(
    () => document.querySelector("#runtime-status span")?.textContent.includes("model ready"),
    null,
    { timeout: 120000 }
  );
  const videoFixtureBase64 = await page.evaluate(async () => {
    const image = new Image();
    image.src = "./.ci/bus.jpg";
    await image.decode();

    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);

    const stream = canvas.captureStream(15);
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
      ? "video/webm;codecs=vp8"
      : "video/webm";
    const recorder = new MediaRecorder(stream, { mimeType });
    const chunks = [];
    recorder.addEventListener("dataavailable", event => {
      if (event.data.size) chunks.push(event.data);
    });
    const stopped = new Promise(resolve => recorder.addEventListener("stop", resolve, { once: true }));
    recorder.start(200);

    let frames = 0;
    const drawTimer = setInterval(() => {
      context.drawImage(image, 0, 0);
      frames += 1;
      if (frames >= 120) {
        clearInterval(drawTimer);
        recorder.stop();
      }
    }, 1000 / 15);

    await stopped;
    stream.getTracks().forEach(track => track.stop());
    const blob = new Blob(chunks, { type: mimeType });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    }
    return btoa(binary);
  });
  if (!videoFixtureBase64) throw new Error("Chromium did not create the local WebM test video.");
  await page.locator("#video-input").setInputFiles({
    name: "bus.webm",
    mimeType: "video/webm",
    buffer: Buffer.from(videoFixtureBase64, "base64")
  });
  await page.waitForFunction(() => {
    const video = document.querySelector("#camera");
    const download = document.querySelector("#download-video");
    return video && video.videoWidth > 0 && video.videoHeight > 0 &&
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      download && !download.hidden && download.href.startsWith("blob:") && Boolean(download.download);
  }, null, { timeout: 30000 });
  await page.waitForFunction(() => {
    const canvas = document.querySelector("#overlay");
    if (!canvas || !canvas.width || !canvas.height) return false;
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) return true;
    return false;
  }, null, { timeout: 30000 });
  const appVideoPipeline = await page.evaluate(() => {
    const video = document.querySelector("#camera");
    const canvas = document.querySelector("#overlay");
    const download = document.querySelector("#download-video");
    return {
      previewLoaded: video.videoWidth > 0 && video.videoHeight > 0,
      localObjectUrl: video.currentSrc.startsWith("blob:"),
      downloadAvailable: !download.hidden && download.href.startsWith("blob:"),
      downloadFilename: download.download,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
      renderedPixels: true
    };
  });
  if (!appVideoPipeline.previewLoaded || !appVideoPipeline.localObjectUrl || !appVideoPipeline.downloadAvailable)
    throw new Error("The local video preview or download action did not initialize.");
  const [downloadedVideo] = await Promise.all([
    page.waitForEvent("download"),
    page.locator("#download-video").click()
  ]);
  if (downloadedVideo.suggestedFilename() !== "bus.webm")
    throw new Error("The downloaded video did not preserve its original filename.");
  appVideoPipeline.downloadTriggered = true;
  const videoLifecycle = await page.evaluate(async () => {
    const video = document.querySelector("#camera");
    const button = document.querySelector("#camera-button");
    video.pause();
    video.currentTime = 0.5;
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 250));
    const progressed = video.currentTime > 0.5;
    video.pause();
    const pausedAt = video.currentTime;
    await new Promise(resolve => setTimeout(resolve, 200));
    const heldWhilePaused = Math.abs(video.currentTime - pausedAt) < 0.05;
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 250));
    const resumed = video.currentTime > pausedAt;
    video.pause();
    video.currentTime = 0;
    await video.play();
    await new Promise(resolve => {
      if (video.ended) return resolve();
      video.addEventListener("ended", resolve, { once: true });
    });
    const ended = video.ended && !button.disabled;
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 250));
    const replayed = !video.paused && button.disabled && video.currentTime > 0;
    return { progressed, heldWhilePaused, resumed, ended, replayed };
  });
  if (Object.values(videoLifecycle).some(value => value !== true))
    throw new Error("Video pause, resume, end, or replay lifecycle failed: " + JSON.stringify(videoLifecycle));
  console.log(`APP_VIDEO_PIPELINE_OK ${JSON.stringify({ ...appVideoPipeline, lifecycle: videoLifecycle })}`);

  const benchmark = await page.evaluate(async () => {
    const worker = new Worker("./inference-worker.js", { type: "module" });
    const waitFor = (type, timeout = 120000) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        worker.removeEventListener("message", listener);
        reject(new Error(`Timed out waiting for worker message: ${type}`));
      }, timeout);
      const listener = event => {
        const data = event.data;
        if (data?.type === "error") {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          reject(new Error(data.message || "Unknown inference worker error."));
        } else if (data?.type === type) {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          resolve(data);
        }
      };
      worker.addEventListener("message", listener);
    });

    try {
      const readyPromise = waitFor("ready");
      worker.postMessage({ type: "init" });
      await readyPromise;

      const response = await fetch("./.ci/bus.jpg");
      if (!response.ok) throw new Error("Benchmark image download is missing.");
      const imageBlob = await response.blob();
      const endToEndTimes = [];
      const inferenceTimes = [];
      let finalResult;

      for (let i = 0; i < 33; i++) {
        const bitmap = await createImageBitmap(imageBlob);
        const resultPromise = waitFor("result", 30000);
        const start = performance.now();
        worker.postMessage({ type: "frame", bitmap }, [bitmap]);
        finalResult = await resultPromise;
        const endToEndMs = performance.now() - start;
        if (!Number.isFinite(finalResult.inferenceMs) || finalResult.inferenceMs <= 0)
          throw new Error("Worker did not return a valid inference-only timing.");
        if (i >= 3) {
          endToEndTimes.push(endToEndMs);
          inferenceTimes.push(finalResult.inferenceMs);
        }
      }

      const detections = finalResult.detections || [];
      if (detections.length < 2)
        throw new Error(`Expected multiple people in bus.jpg; got ${detections.length}.`);
      if (detections.some(d => !Array.isArray(d.keypoints) || d.keypoints.length !== 17))
        throw new Error("At least one detection does not contain 17 keypoints.");
      if (detections.some(d => !Number.isFinite(d.score) || d.score < 0 || d.score > 1))
        throw new Error("Invalid detection confidence.");
      if (detections.some(d => d.keypoints.some(p =>
        p.length !== 3 || !p.every(Number.isFinite) || p[2] < 0 || p[2] > 1
      ))) throw new Error("Invalid or non-finite keypoint values.");

      const stats = values => {
        const sorted = [...values].sort((a, b) => a - b);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const median = sorted.length % 2
          ? sorted[Math.floor(sorted.length / 2)]
          : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
        const p95 = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
        return {
          meanMs: Number(mean.toFixed(2)),
          medianMs: Number(median.toFixed(2)),
          p95Ms: Number(p95.toFixed(2)),
          fpsFromMean: Number((1000 / mean).toFixed(2))
        };
      };

      return {
        warmupIterations: 3,
        measuredIterations: inferenceTimes.length,
        detections: detections.length,
        keypointsPerPerson: 17,
        inferenceOnly: stats(inferenceTimes),
        endToEnd: stats(endToEndTimes),
        image: { width: finalResult.sourceWidth, height: finalResult.sourceHeight }
      };
    } finally {
      worker.terminate();
    }
  });

  const actionInference = await page.evaluate(async () => {
    const worker = new Worker("./action-worker.js", { type: "module" });
    const waitFor = (type, timeout = 90000) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("Timed out waiting for " + type)), timeout);
      const listener = event => {
        if (event.data?.type === "error" || event.data?.type === "inference-error") {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          reject(new Error(event.data.message || "Action worker failed."));
        } else if (event.data?.type === type) {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          resolve(event.data);
        }
      };
      worker.addEventListener("message", listener);
    });
    try {
      const readyPromise = waitFor("ready", 90000);
      worker.postMessage({ type: "init" });
      const ready = await readyPromise;
      const actionPromise = waitFor("action", 90000);
      for (let frame = 0; frame < 105; frame++) {
        const makePerson = (trackId, baseX) => ({
          trackId, score: 0.95,
          box: [baseX, 30, baseX + 90, 300],
          keypoints: Array.from({ length: 17 }, (_, joint) => [
            baseX + 12 + joint * 2 + Math.sin(frame / 5 + joint) * 8,
            45 + joint * 9 + Math.cos(frame / 6 + joint) * 7,
            0.9
          ])
        });
        worker.postMessage({
          type: "pose-frame",
          sourceWidth: 320,
          sourceHeight: 480,
          detections: [makePerson(1, 25), makePerson(2, 190)]
        });
      }
      const result = await actionPromise;
      const action = result.action;
      if (!Number.isInteger(action.classId) || action.classId < 0 || action.classId >= 60)
        throw new Error("ST-GCN++ returned an invalid class ID.");
      if (!Number.isFinite(action.confidence) || action.confidence < 0 || action.confidence > 1)
        throw new Error("ST-GCN++ returned an invalid confidence.");
      if (!Number.isFinite(action.inferenceMs) || action.inferenceMs <= 0)
        throw new Error("ST-GCN++ returned an invalid inference time.");
      return {
        provider: ready.provider,
        classes: ready.classes,
        inputShape: ready.inputShape,
        classId: action.classId,
        label: action.label,
        confidence: action.confidence,
        inferenceMs: Number(action.inferenceMs.toFixed(2)),
        pairBox: action.pairBox
      };
    } finally {
      worker.terminate();
    }
  });

  const rendererCheck = await page.evaluate(async () => {
    const { PoseRenderer } = await import("./renderer.js");
    const canvas = document.querySelector("#overlay");
    const renderer = new PoseRenderer(canvas);
    renderer.draw([{
      trackId: 1,
      label: "Person",
      score: 0.9,
      box: [20, 20, 200, 300],
      keypoints: Array.from({ length: 17 }, (_, i) => [40 + i * 4, 50 + i * 7, 0.95])
    }], 320, 480);
    const ctx = canvas.getContext("2d");
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let painted = false;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) { painted = true; break; }
    }
    renderer.destroy();
    if (!painted) throw new Error("Pose renderer did not paint skeleton/box pixels.");
    return { painted };
  });

  if (pageErrors.length) throw new Error("Browser page errors: " + pageErrors.join(" | "));
  console.log(`INFERENCE_BENCHMARK ${JSON.stringify(benchmark)}`);
  console.log(`STGCN_ACTION_INFERENCE ${JSON.stringify(actionInference)}`);
  console.log(`RENDERER_OK ${JSON.stringify(rendererCheck)}`);
  console.log("BROWSER_MODEL_TEST_PASSED");
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}
