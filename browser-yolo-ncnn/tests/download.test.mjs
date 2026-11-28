// Run with:  node --test browser-yolo-ncnn/tests/download.test.mjs
// Starts a local HTTP server that can cut or freeze a download, and checks that
// downloadBytes() finishes with identical bytes by resuming with Range requests.
import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import crypto from "node:crypto";
import { downloadBytes } from "../resumable-download.js";

const FILE = crypto.randomBytes(2 * 1024 * 1024);
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");

/** mode(requestIndex, hasRange) -> "ok" | "cut" | "freeze" | "ignore-range" | "404" */
async function withServer(mode, run) {
  const requests = [];
  const open = new Set();
  const server = http.createServer((req, res) => {
    const range = /bytes=(\d+)-/.exec(req.headers.range || "");
    const index = requests.length;
    requests.push({ range: range ? Number(range[1]) : null });
    const action = mode(index, Boolean(range));
    if (action === "404") { res.writeHead(404); res.end(); return; }
    const start = range && action !== "ignore-range" ? Number(range[1]) : 0;
    const body = FILE.subarray(start);
    const headers = { "Content-Type": "application/octet-stream", "Content-Length": body.length, "Accept-Ranges": "bytes" };
    if (start) headers["Content-Range"] = `bytes ${start}-${FILE.length - 1}/${FILE.length}`;
    res.writeHead(start ? 206 : 200, headers);
    if (action === "cut" || action === "freeze") {
      const part = Math.floor(body.length * 0.9);
      res.write(body.subarray(0, part));
      if (action === "cut") setTimeout(() => req.socket.destroy(), 20);
      else open.add(res);                               // never finish: the connection goes quiet
      return;
    }
    res.end(body);
  });
  await new Promise(resolve => server.listen(0, resolve));
  try {
    await run(`http://127.0.0.1:${server.address().port}/model.onnx`, requests);
  } finally {
    for (const res of open) res.destroy();
    server.closeAllConnections?.();
    await new Promise(resolve => server.close(resolve));
  }
}

test("a clean download returns the exact bytes in one request", async () => {
  await withServer(() => "ok", async (url, requests) => {
    const { bytes, resumes } = await downloadBytes(url, { label: "t" });
    assert.equal(sha(bytes), sha(FILE));
    assert.equal(resumes, 0);
    assert.equal(requests.length, 1);
  });
});

test("a connection cut at 90% resumes from the received byte, not from zero", async () => {
  await withServer(i => (i === 0 ? "cut" : "ok"), async (url, requests) => {
    const { bytes, resumes } = await downloadBytes(url, { label: "t" });
    assert.equal(sha(bytes), sha(FILE));
    assert.equal(resumes, 1);
    assert.equal(requests.length, 2);
    assert.ok(requests[1].range > FILE.length * 0.85, "second request must start near the cut, got " + requests[1].range);
  });
});

test("three cuts in a row still end with identical bytes", async () => {
  await withServer(i => (i < 3 ? "cut" : "ok"), async (url) => {
    const { bytes, resumes } = await downloadBytes(url, { label: "t" });
    assert.equal(sha(bytes), sha(FILE));
    assert.equal(resumes, 3);
  });
});

test("a connection that goes silent is aborted and resumed", async () => {
  await withServer(i => (i === 0 ? "freeze" : "ok"), async (url) => {
    const started = Date.now();
    const { bytes } = await downloadBytes(url, { label: "t", stallMs: 300 });
    assert.equal(sha(bytes), sha(FILE));
    assert.ok(Date.now() - started < 5000);
  });
});

test("a server that ignores Range makes the download start over, still correct", async () => {
  await withServer(i => (i === 0 ? "cut" : "ignore-range"), async (url) => {
    const { bytes } = await downloadBytes(url, { label: "t" });
    assert.equal(sha(bytes), sha(FILE));
  });
});

test("HTTP 404 fails at once without retrying", async () => {
  await withServer(() => "404", async (url, requests) => {
    await assert.rejects(downloadBytes(url, { label: "t" }), error => error.stage === "download");
    assert.equal(requests.length, 1);
  });
});

test("it gives up after the attempt limit and says how much arrived", async () => {
  await withServer(() => "cut", async (url) => {
    await assert.rejects(
      downloadBytes(url, { label: "t", attempts: 3 }),
      error => error.stage === "download" && /3 attempts/.test(error.message)
    );
  });
});
