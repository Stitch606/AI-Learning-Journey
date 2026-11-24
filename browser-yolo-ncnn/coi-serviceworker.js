self.addEventListener("install", event => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", event => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", event => {
  const request = event.request;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith((async () => {
    const runtimeAsset = request.method === "GET" &&
      /\/runtime\/ncnn(?:-threaded)?\.(?:js|wasm)$/.test(url.pathname);
    let response;

    if (runtimeAsset) {
      try {
        const cache = await caches.open("ncnn-runtime-v1");
        response = await cache.match(request);
        if (!response) {
          response = await fetch(request);
          if (response.ok && response.type !== "opaque") {
            try { await cache.put(request, response.clone()); } catch {}
          }
        }
      } catch {
        response = await fetch(request);
      }
    } else {
      response = await fetch(request);
    }

    if (response.type === "opaque" || response.status === 0) return response;
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("Cross-Origin-Resource-Policy", "same-origin");
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers
    });
  })());
});
