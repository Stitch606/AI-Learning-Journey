import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const port = 4173;
const server = spawn("python", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
  stdio: "ignore",
  cwd: "browser-yolo-ncnn"
});
let browser;
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
  page.on("pageerror", error => { throw error; });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "networkidle" });

  const simd = await page.evaluate(() => WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,22,1,20,0,253,12,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11
  ])));
  if (!simd) throw new Error("Chromium does not validate WebAssembly SIMD.");

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

  const benchmark = await page.evaluate(async () => {
    const worker = new Worker("./inference-worker.js", { type: "module" });
    const waitFor = (type, timeout = 60000) => new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`Timed out waiting for worker message: ${type}`)), timeout);
      const listener = event => {
        if (event.data.type === "error") {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          reject(new Error(event.data.message));
        }
        if (event.data.type === type) {
          clearTimeout(timer);
          worker.removeEventListener("message", listener);
          resolve(event.data);
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
      const times = [];
      let finalResult;
      for (let i = 0; i < 12; i++) {
        const bitmap = await createImageBitmap(imageBlob);
        const resultPromise = waitFor("result", 30000);
        const start = performance.now();
        worker.postMessage({ type: "frame", bitmap }, [bitmap]);
        finalResult = await resultPromise;
        times.push(performance.now() - start);
      }

      const detections = finalResult.detections || [];
      if (detections.length < 1) throw new Error("YOLO returned zero person detections on bus.jpg.");
      if (detections.some(d => !Array.isArray(d.keypoints) || d.keypoints.length !== 17))
        throw new Error("At least one detection does not contain 17 keypoints.");
      if (detections.some(d => !Number.isFinite(d.score) || d.score < 0 || d.score > 1))
        throw new Error("Invalid detection confidence.");
      if (detections.some(d => d.keypoints.some(p => p.length !== 3 || !p.every(Number.isFinite))))
        throw new Error("Invalid or non-finite keypoint values.");

      const sorted = [...times].sort((a, b) => a - b);
      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const median = sorted[Math.floor(sorted.length / 2)];
      const p95 = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)];
      return {
        iterations: times.length,
        detections: detections.length,
        keypointsPerPerson: 17,
        meanMs: Number(mean.toFixed(2)),
        medianMs: Number(median.toFixed(2)),
        p95Ms: Number(p95.toFixed(2)),
        meanFps: Number((1000 / mean).toFixed(2)),
        lastFrame: { width: finalResult.sourceWidth, height: finalResult.sourceHeight }
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
    for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) { painted = true; break; }
    renderer.destroy();
    if (!painted) throw new Error("Pose renderer did not paint skeleton/box pixels.");
    return { painted };
  });

  console.log(`INFERENCE_BENCHMARK ${JSON.stringify(benchmark)}`);
  console.log(`RENDERER_OK ${JSON.stringify(rendererCheck)}`);
  console.log("BROWSER_MODEL_TEST_PASSED");
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
}
