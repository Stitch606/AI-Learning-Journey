let ort = null;

const MODEL_FP16_URL = new URL("./model/stgcnpp_ntu60-fp16.onnx", import.meta.url).href;
const MODEL_FP32_URL = new URL("./model/stgcnpp_ntu60-fp32.onnx", import.meta.url).href;
const LABELS_URL = new URL("./model/ntu60_labels.txt", import.meta.url).href;
const ACTION_CACHE_NAME = "stgcnpp-ntu60-fp16-v2";
const T = 100, V = 17, M = 2, C = 3, STRIDE = 5;
let session = null, modelBytes = null, labels = [], frames = [], count = 0, busy = false, provider = "WASM CPU";
let slotTrackIds = [null, null];
let slotMisses = [0, 0];

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function choosePeople(detections) {
  const valid = (detections || [])
    .filter(d => d.trackId != null && Array.isArray(d.keypoints) && d.keypoints.length === V)
    .sort((a, b) => (b.score || 0) - (a.score || 0));
  const assigned = [null, null];
  let changed = false;

  for (let slot = 0; slot < M; slot++) {
    const id = slotTrackIds[slot];
    if (id == null) continue;
    const match = valid.find(d => d.trackId === id);
    if (match) {
      assigned[slot] = match;
      slotMisses[slot] = 0;
    } else {
      slotMisses[slot]++;
      if (slotMisses[slot] > 12) {
        slotTrackIds[slot] = null;
        slotMisses[slot] = 0;
        changed = true;
      }
    }
  }

  for (let slot = 0; slot < M; slot++) {
    if (slotTrackIds[slot] != null) continue;
    const candidate = valid.find(d => !slotTrackIds.includes(d.trackId));
    if (candidate) {
      slotTrackIds[slot] = candidate.trackId;
      assigned[slot] = candidate;
      slotMisses[slot] = 0;
      changed = true;
    }
  }

  if (changed && frames.length) {
    frames = [];
    count = 0;
  }
  return assigned;
}

function makeInput() {
  const normalized = Array.from({length:M}, () =>
    Array.from({length:T}, () => Array.from({length:V}, () => [0,0,0])));
  for (let t=0;t<T;t++) {
    const f=frames[t];
    for (let p=0;p<M;p++) {
      const person=f.people[p];
      if (!person) continue;
      const w=f.width, h=f.height;
      for (let j=0;j<V;j++) {
        const k=person.keypoints[j] || [0,0,0];
        const s=clamp(Number(k[2]) || 0,0,1);
        if (s<=0.01) continue;
        normalized[p][t][j]=[(Number(k[0])-w/2)/(w/2),(Number(k[1])-h/2)/(h/2),s];
      }
    }
  }
  const data=new Float32Array(M*T*V*C);
  for(let p=0;p<M;p++) for(let t=0;t<T-1;t++) for(let j=0;j<V;j++) {
    const a=normalized[p][t][j], b=normalized[p][t+1][j];
    const i=(((p*T+t)*V+j)*C);
    data[i]=b[0]-a[0]; data[i+1]=b[1]-a[1]; data[i+2]=(a[2]+b[2])*0.5;
  }
  return data;
}

function softmax(x) {
  const m=Math.max(...x), e=x.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0)||1;
  return e.map(v=>v/s);
}

async function loadActionModel(modelUrl, label) {
  const request = new Request(modelUrl, { method: "GET" });
  let cache = null;
  try { cache = "caches" in self ? await caches.open(ACTION_CACHE_NAME) : null; } catch {}
  const cached = cache ? await cache.match(request).catch(() => null) : null;
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    if (bytes.length) {
      self.postMessage({type:"progress",message:label+" loaded from device cache ("+(bytes.length/1048576).toFixed(2)+" MB)"});
      return bytes;
    }
  }

  self.postMessage({type:"progress",message:"Downloading "+label});
  const response = await fetch(request, {cache:"default"});
  if (!response.ok) throw new Error(label+" download failed (HTTP "+response.status+").");
  let bytes;
  if (!response.body) {
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    const reader = response.body.getReader(), chunks = [];
    const total = Number(response.headers.get("content-length")) || 0;
    let loaded = 0;
    while (true) {
      const {done,value} = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      const mb = (loaded/1048576).toFixed(2);
      const pct = total ? " · "+Math.min(100,Math.round(loaded/total*100))+"%" : "";
      self.postMessage({type:"progress",message:"ST-GCN++ "+mb+" MB"+(total?" / "+(total/1048576).toFixed(2)+" MB":"")+pct});
    }
    bytes = new Uint8Array(loaded);
    let offset=0;
    for (const chunk of chunks) { bytes.set(chunk,offset); offset+=chunk.byteLength; }
  }
  if (!bytes.length) throw new Error(label+" downloaded as an empty file.");
  try {
    if (cache) await cache.put(request,new Response(bytes,{
      headers:{"Content-Type":response.headers.get("content-type")||"application/octet-stream"}
    }));
  } catch {
    self.postMessage({type:"progress",message:"Weights loaded; persistent cache unavailable on this browser."});
  }
  return bytes;
}

async function init() {
  self.postMessage({type:"progress",message:"Loading action runtime"});
  const ortModule = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs");
  ort = ortModule;
  self.postMessage({type:"progress",message:"Loading original NTU60 ST-GCN++"});
  labels=(await (await fetch(LABELS_URL, {cache:"default"})).text()).trim().split(/\r?\n/);
  let modelBytes = await loadActionModel(MODEL_FP16_URL, "ST-GCN++ FP16");
  ort.env.logLevel="error";
  ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.numThreads=self.crossOriginIsolated ? Math.max(1,Math.min(self.navigator.hardwareConcurrency||2,4)) : 1;
  try {
    if(!self.navigator.gpu) throw new Error("WebGPU unavailable");
    const adapter = await self.navigator.gpu.requestAdapter({powerPreference:"high-performance"});
    if(!adapter) throw new Error("No WebGPU adapter was returned.");
    session=await ort.InferenceSession.create(modelBytes,{executionProviders:["webgpu"],graphOptimizationLevel:"all"});
    provider="WebGPU GPU · FP16";
  } catch(e) {
    self.postMessage({type:"progress",message:"ST-GCN++ WebGPU FP16 unavailable ("+e.message+"); loading FP32 CPU compatibility model"});
    modelBytes = await loadActionModel(MODEL_FP32_URL, "ST-GCN++ FP32 compatibility");
    session=await ort.InferenceSession.create(modelBytes,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
    provider="WASM CPU · FP32 fallback";
  }
  self.postMessage({type:"ready",provider,classes:labels.length,inputShape:[1,1,2,100,17,3]});
}

async function infer() {
  if(busy || !session || frames.length<T) return;
  busy=true;
  try {
    if (frames.filter(f => f.people.some(Boolean)).length < 70) return;
    const start=performance.now();
    const input=new ort.Tensor("float32",makeInput(),[1,1,M,T,V,C]);
    let output;
    try {
      output=await session.run({keypoint:input});
    } catch (gpuError) {
      if (!provider.startsWith("WebGPU")) throw gpuError;
      const fallbackBytes = await loadActionModel(MODEL_FP32_URL, "ST-GCN++ FP32 compatibility");
      session=await ort.InferenceSession.create(fallbackBytes,{executionProviders:["wasm"],graphOptimizationLevel:"all"});
      provider="WASM CPU · FP32 fallback";
      output=await session.run({keypoint:input});
    }
    const probs=softmax(Array.from(output.logits.data));
    let id=0; for(let i=1;i<probs.length;i++) if(probs[i]>probs[id]) id=i;
    const people=frames[T-1].people;
    const selected=people.filter(Boolean);
    const boxes=selected.map(p=>p.box).filter(Boolean);
    const pairBox=boxes.length ? [
      Math.min(...boxes.map(b=>b[0])),Math.min(...boxes.map(b=>b[1])),
      Math.max(...boxes.map(b=>b[2])),Math.max(...boxes.map(b=>b[3]))
    ] : null;
    self.postMessage({type:"action",action:{
      classId:id,label:labels[id]||("class_"+id),confidence:probs[id],
      trackIds:selected.map(p=>p.trackId).filter(x=>x!=null),pairBox,
      inferenceMs:performance.now()-start,provider
    }});
  } catch(e) {
    self.postMessage({type:"inference-error",message:e.message});
  } finally { busy=false; }
}

self.onmessage=async ({data:m})=>{
  if(m.type==="init"){try{await init()}catch(e){self.postMessage({type:"error",message:e.message})}return}
  if(m.type==="reset"){frames=[];count=0;slotTrackIds=[null,null];slotMisses=[0,0];return}
  if(m.type==="pose-frame" && session){
    frames.push({people:choosePeople(m.detections),width:m.sourceWidth,height:m.sourceHeight});
    if(frames.length>T) frames.shift();
    count++;
    if(frames.length===T && count%STRIDE===0) await infer();
  }
};
