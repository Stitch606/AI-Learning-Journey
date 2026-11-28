import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs";
import { downloadBytes } from "./resumable-download.js";

const INPUT_SIZE = 512;
const VALUES_PER_DETECTION = 56;
const MODEL_URL = "./model/yolo26n-pose-512-fp16.onnx";
const MODEL_FP32_URL = "./model/yolo26n-pose-512-fp32.onnx";
const CACHE_NAME = "yolo26n-pose-onnx-fp16-512-v1";
const SCORE_THRESHOLD = 0.30;
const NMS_THRESHOLD = 0.45;
let session = null;
let modelBytes = null;
let provider = "uninitialized";
let inputName = null;
let outputName = null;
let canvas = null;
let context = null;
let frameIndex = 0;
let nextTrackId = 1;
let tracks = [];

async function fetchModel(url, label) {
  const request = new Request(new URL(url, self.location.href));
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch {}
  const cached = cache ? await cache.match(request).catch(() => null) : null;
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    if (bytes.length) {
      self.postMessage({type:"progress",status:"Using cached "+label,detail:label+" loaded from device cache ("+(bytes.length/1048576).toFixed(2)+" MB)."});
      return bytes;
    }
  }
  self.postMessage({type:"progress",status:"Downloading "+label,detail:"Connecting to download "+label+"…",loaded:0,total:0,model:label});
  const {bytes,total} = await downloadBytes(request.url, {
    label,
    onProgress: ({loaded,total,note}) => self.postMessage({type:"progress",status:"Downloading "+label,
      detail: note || label+" · "+(loaded/1048576).toFixed(2)+" MB"+(total?" / "+(total/1048576).toFixed(2)+" MB · "+Math.min(100,Math.round(loaded/total*100))+"%":" downloaded"),
      loaded,total,model:label})
  });
  self.postMessage({type:"progress",status:"Download complete",
    detail:label+" · "+(bytes.length/1048576).toFixed(2)+" MB downloaded. Initializing runtime…",
    loaded:bytes.length,total:total||bytes.length,model:label});
  try { if (cache) await cache.put(request,new Response(bytes,{headers:{"Content-Type":"application/onnx"}})); } catch {}
  return bytes;
}

async function createSession(bytes, executionProvider) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: [executionProvider],
    graphOptimizationLevel: "all",
    executionMode: "sequential"
  });
}

async function initialize() {
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(self.navigator.hardwareConcurrency || 2, 4)) : 1;
  canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE);
  context = canvas.getContext("2d", {willReadFrequently:true});

  // Look for a GPU first: without WebGPU the FP16 file would be downloaded for nothing.
  let adapter = null;
  if (self.navigator.gpu) {
    try { adapter = await self.navigator.gpu.requestAdapter({powerPreference:"high-performance"}); } catch {}
  }
  if (adapter) {
    try {
      modelBytes = await fetchModel(MODEL_URL, "YOLO26n-Pose FP16 512");
      session = await createSession(modelBytes, "webgpu");
      provider = "WebGPU GPU · FP16";
    } catch (error) {
      if (error.stage === "download") throw error;
      self.postMessage({type:"progress",status:"WebGPU unavailable",detail:"WebGPU FP16 could not initialize ("+error.message+"). Switching to the FP32 compatibility model."});
    }
  }
  if (!session) {
    modelBytes = await fetchModel(MODEL_FP32_URL, "YOLO26n-Pose FP32 compatibility");
    session = await createSession(modelBytes, "wasm");
    provider = "WASM CPU · FP32 fallback";
  }
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  self.postMessage({type:"ready",provider,inputSize:INPUT_SIZE,precision:provider.includes("FP16")?"FP16":"FP32"});
}

function iou(a,b) {
  const x1=Math.max(a[0],b[0]), y1=Math.max(a[1],b[1]);
  const x2=Math.min(a[2],b[2]), y2=Math.min(a[3],b[3]);
  const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
  const aa=Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]);
  const ab=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  return inter/(aa+ab-inter+1e-6);
}

function assignTracks(detections) {
  frameIndex++;
  const claimed=new Set();
  for (const d of detections) {
    let best=null, bestIou=0.20;
    for (const t of tracks) {
      if (claimed.has(t.id)) continue;
      const overlap=iou(d.box,t.box);
      if (overlap>bestIou) {best=t;bestIou=overlap;}
    }
    if (best) {
      best.box=d.box; best.lastSeen=frameIndex; d.trackId=best.id; claimed.add(best.id);
    } else {
      d.trackId=nextTrackId++;
      tracks.push({id:d.trackId,box:d.box,lastSeen:frameIndex});
      claimed.add(d.trackId);
    }
  }
  tracks=tracks.filter(t=>frameIndex-t.lastSeen<=12);
  return detections;
}

function decodeOutput(tensor, sourceWidth, sourceHeight, scale, padX, padY) {
  const dims=tensor.dims;
  const data=tensor.data;
  if (dims.length!==3 || dims[0]!==1) throw new Error("Unexpected YOLO ONNX output shape: "+dims.join("×"));
  let channels, candidates, channelsFirst;
  if (dims[1]===VALUES_PER_DETECTION) {channels=dims[1];candidates=dims[2];channelsFirst=true;}
  else if (dims[2]===VALUES_PER_DETECTION) {channels=dims[2];candidates=dims[1];channelsFirst=false;}
  else throw new Error("YOLO output must contain 56 values per pose candidate; got "+dims.join("×"));
  const at=(c,n)=>channelsFirst?data[c*candidates+n]:data[n*channels+c];
  const x=v=>Math.max(0,Math.min(sourceWidth,(v-padX)/scale));
  const y=v=>Math.max(0,Math.min(sourceHeight,(v-padY)/scale));
  const proposals=[];
  for(let n=0;n<candidates;n++) {
    const score=Number(at(4,n));
    if(!Number.isFinite(score)||score<SCORE_THRESHOLD) continue;
    const cx=Number(at(0,n)),cy=Number(at(1,n)),w=Number(at(2,n)),h=Number(at(3,n));
    if(![cx,cy,w,h].every(Number.isFinite)||w<=0||h<=0) continue;
    const box=[x(cx-w/2),y(cy-h/2),x(cx+w/2),y(cy+h/2)];
    if(box[2]<=box[0]||box[3]<=box[1]) continue;
    const keypoints=[];
    for(let k=0;k<17;k++) {
      const px=Number(at(5+k*3,n)),py=Number(at(6+k*3,n)),ps=Number(at(7+k*3,n));
      keypoints.push([Number.isFinite(px)?x(px):0,Number.isFinite(py)?y(py):0,
        Number.isFinite(ps)?Math.max(0,Math.min(1,ps)):0]);
    }
    proposals.push({label:"Person",score,box,keypoints});
  }
  proposals.sort((a,b)=>b.score-a.score);
  const selected=[];
  for(const d of proposals) {
    if(selected.every(k=>iou(d.box,k.box)<=NMS_THRESHOLD)) selected.push(d);
    if(selected.length>=50) break;
  }
  return assignTracks(selected);
}

function preprocess(bitmap) {
  const sourceWidth=bitmap.width, sourceHeight=bitmap.height;
  const scale=Math.min(INPUT_SIZE/sourceWidth,INPUT_SIZE/sourceHeight);
  const drawWidth=sourceWidth*scale,drawHeight=sourceHeight*scale;
  const padX=(INPUT_SIZE-drawWidth)/2,padY=(INPUT_SIZE-drawHeight)/2;
  context.fillStyle="rgb(114,114,114)";
  context.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
  context.drawImage(bitmap,padX,padY,drawWidth,drawHeight);
  const rgba=context.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data;
  const input=new Float32Array(3*INPUT_SIZE*INPUT_SIZE);
  const plane=INPUT_SIZE*INPUT_SIZE;
  for(let i=0;i<plane;i++) {
    const p=i*4;
    input[i]=rgba[p]/255;
    input[plane+i]=rgba[p+1]/255;
    input[plane*2+i]=rgba[p+2]/255;
  }
  return {input,sourceWidth,sourceHeight,scale,padX,padY};
}

async function fallbackToWasm(reason) {
  self.postMessage({type:"progress",status:"Switching to CPU compatibility model",detail:"WebGPU inference failed ("+reason+"). Loading the FP32 compatibility model."});
  try { await session?.release(); } catch {}
  modelBytes = await fetchModel(MODEL_FP32_URL, "YOLO26n-Pose FP32 compatibility");
  session = await createSession(modelBytes, "wasm");
  provider = "WASM CPU · FP32 fallback";
}

async function runFrame(bitmap) {
  const prepStart=performance.now();
  const prep=preprocess(bitmap);
  const prepMs=performance.now()-prepStart;
  const inferenceStart=performance.now();
  const feeds={[inputName]:new ort.Tensor("float32",prep.input,[1,3,INPUT_SIZE,INPUT_SIZE])};
  let outputs;
  try {
    outputs=await session.run(feeds);
  } catch(error) {
    if (!provider.startsWith("WebGPU")) throw error;
    await fallbackToWasm(error.message);
    outputs=await session.run(feeds);
  }
  const inferenceMs=performance.now()-inferenceStart;
  const decodeStart=performance.now();
  const detections=decodeOutput(outputs[outputName],prep.sourceWidth,prep.sourceHeight,prep.scale,prep.padX,prep.padY);
  const decodeMs=performance.now()-decodeStart;
  return {sourceWidth:prep.sourceWidth,sourceHeight:prep.sourceHeight,detections,prepMs,inferenceMs,decodeMs,provider,inputSize:INPUT_SIZE};
}

self.onmessage=async ({data:message})=>{
  if(message.type==="init") {
    try {await initialize();}
    catch(error) {self.postMessage({type:"error",stage:error.stage||"init",message:error.message});}
    return;
  }
  if(message.type==="frame"&&session) {
    const bitmap=message.bitmap;
    try {self.postMessage({type:"result",...(await runFrame(bitmap))});}
    catch(error) {self.postMessage({type:"frame-error",message:error.message});}
    finally {bitmap.close();}
  }
};
