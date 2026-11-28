// Downloads a file into memory and survives a dropped or stalled connection.
//
// A plain fetch() that loses the connection at 90% throws, and the caller had
// to start again from byte 0. Here the bytes already received are kept, and the
// next attempt asks the server for the rest with an HTTP Range request. A
// connection that goes quiet for `stallMs` is aborted and resumed too, instead
// of hanging until an outer watchdog restarts everything.

const MB = 1048576;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function fatal(message) {
  const error = new Error(message);
  error.fatal = true;
  error.stage = "download";
  return error;
}

async function waitUntilOnline(maxMs) {
  if (globalThis.navigator?.onLine !== false) return;
  await new Promise(resolve => {
    const timer = setTimeout(resolve, maxMs);
    globalThis.addEventListener("online", () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

/**
 * @param {string} url
 * @param {{label:string, onProgress?:(p:{loaded:number,total:number,note?:string})=>void,
 *          attempts?:number, stallMs?:number}} options
 * @returns {Promise<{bytes:Uint8Array,total:number,resumes:number}>}
 */
export async function downloadBytes(url, { label, onProgress = () => {}, attempts = 12, stallMs = 25000 }) {
  const chunks = [];
  let loaded = 0;
  let total = 0;
  let resumable = true;          // false when the server compresses: byte offsets no longer match
  let resumes = 0;
  let lastError = null;
  let finished = false;

  for (let attempt = 1; attempt <= attempts && !finished; attempt++) {
    const controller = new AbortController();
    let timer = 0;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(() => controller.abort(new Error("no data for " + Math.round(stallMs / 1000) + " s")), stallMs);
    };

    try {
      arm();
      const headers = new Headers();
      if (loaded > 0) headers.set("Range", `bytes=${loaded}-`);
      const response = await fetch(url, { headers, signal: controller.signal, cache: "no-cache" });

      if (response.status === 404 || response.status === 403 || response.status === 410) {
        throw fatal(`${label} download failed (HTTP ${response.status}).`);
      }
      if (response.status === 416) {                 // asked for bytes past the end
        if (total && loaded === total) { finished = true; break; }
        chunks.length = 0; loaded = 0; total = 0;
        throw new Error("server rejected the resume request");
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const encoding = response.headers.get("content-encoding");
      if (encoding && encoding !== "identity") resumable = false;

      if (response.status === 206) {
        const match = /bytes\s+(\d+)-(\d+)\/(\d+|\*)/i.exec(response.headers.get("content-range") || "");
        if (!match || Number(match[1]) !== loaded) {
          chunks.length = 0; loaded = 0; total = 0;
          throw new Error("server answered a resume request with the wrong range");
        }
        if (match[3] !== "*") total = Number(match[3]);
      } else {
        // 200: the whole file. If we already had part of it, the server ignored Range.
        chunks.length = 0;
        loaded = 0;
        total = Number(response.headers.get("content-length")) || 0;
      }

      if (!response.body) {
        const bytes = new Uint8Array(await response.arrayBuffer());
        chunks.push(bytes);
        loaded += bytes.length;
        onProgress({ loaded, total: total || loaded });
      } else {
        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          arm();
          chunks.push(value);
          loaded += value.byteLength;
          onProgress({ loaded, total });
        }
      }

      if (total && loaded < total) throw new Error("connection closed early");
      finished = true;
    } catch (error) {
      if (error.fatal) throw error;
      lastError = error;
      if (!resumable) { chunks.length = 0; loaded = 0; }
      if (attempt < attempts) {
        resumes++;
        onProgress({
          loaded,
          total,
          note: `${label}: connection interrupted at ${(loaded / MB).toFixed(2)} MB ` +
                `(${error.message}). Resuming, attempt ${attempt + 1} of ${attempts}…`
        });
        await waitUntilOnline(60000);
        await sleep(Math.min(5000, 600 * attempt));
      }
    } finally {
      clearTimeout(timer);
    }
  }

  if (!finished) {
    const error = new Error(
      `${label} could not be downloaded after ${attempts} attempts (${(loaded / MB).toFixed(2)} MB received): ` +
      (lastError?.message || "unknown network error")
    );
    error.stage = "download";
    throw error;
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (!bytes.length) {
    const error = new Error(`${label} downloaded as an empty file.`);
    error.stage = "download";
    throw error;
  }
  return { bytes, total: total || bytes.length, resumes };
}
