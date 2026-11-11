import { state, setTheme } from "./state.js";
(() => {
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
setTheme(state.theme);
const clock=()=>$("#clock").textContent=new Date().toLocaleTimeString("en-GB",{hour12:false});clock();setInterval(clock,1000);
let runtimeStartedAt=null,runtimeElapsed=0,runtimeTicker=null;
const formatRuntime=s=>{const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60;return [h,m,sec].map(v=>String(v).padStart(2,"0")).join(":");};
function updateRuntime(){const elapsed=runtimeElapsed+(runtimeStartedAt===null?0:Math.floor((Date.now()-runtimeStartedAt)/1000));$("#runtimeValue").textContent=formatRuntime(elapsed);}
function startRuntime(){if(runtimeStartedAt!==null)return;runtimeStartedAt=Date.now();if(!runtimeTicker)runtimeTicker=setInterval(updateRuntime,1000);updateRuntime();}
function stopRuntime(reset=false){if(runtimeStartedAt!==null){runtimeElapsed+=Math.floor((Date.now()-runtimeStartedAt)/1000);runtimeStartedAt=null;}if(reset){runtimeElapsed=0;if(runtimeTicker){clearInterval(runtimeTicker);runtimeTicker=null;}}updateRuntime();}
function render(){const p=state.people;$("#resultsBody").innerHTML=p.map(x=>'<tr><td>'+x.id+'</td><td><i class="dot '+(x.color||"blue")+'"></i>'+x.person+'</td><td>'+x.action+'</td><td>'+ (state.settings.showConfidence?Number(x.confidence).toFixed(2):"—")+'</td><td>'+x.time+'</td></tr>').join("")||'<tr><td colspan="5" class="no-data">No detections yet</td></tr>';$("#peopleCount").textContent=state.settings.detectPeople?p.length:0;$("#actionCount").textContent=state.settings.recognizeActions?new Set(p.map(x=>x.action)).size:0;$("#totalActions").textContent=state.settings.recognizeActions?p.length:0;}
function log(message,color="blue"){const row=document.createElement("p");row.innerHTML='<i class="'+color+'"></i>'+new Date().toLocaleTimeString("en-GB")+"　"+message;$("#logList").prepend(row);while($("#logList").children.length>8)$("#logList").lastElementChild.remove();}
function showEmpty(){const video=$("#cameraVideo");video.pause();video.srcObject=null;video.removeAttribute("src");video.hidden=true;const overlay=$("#inferenceOverlay");overlay.hidden=true;overlay.removeAttribute("src");$("#videoEmpty").hidden=false;$("#liveBadge").hidden=true;$("#fpsBadge").hidden=true;$("#openCameraBtn").hidden=state.page==="Video Analysis";$("#chooseVideoBtn").hidden=state.page!=="Video Analysis";$("#videoEyebrow").textContent=state.page==="Dashboard"?"LIVE DETECTION":"VIDEO ANALYSIS";$("#videoTitle").textContent=state.page==="Dashboard"?"Ready to monitor":"Choose a video to analyse";$("#videoHelp").textContent=state.page==="Dashboard"?"Start your camera to begin real-time action recognition.":"Upload a video file to inspect actions and detected people.";$("#stopBtn").disabled=true;$("#pauseBtn").disabled=true;$("#cameraBtn").disabled=true;}
function showMedia(){startRuntime();$("#videoEmpty").hidden=true;$("#liveBadge").hidden=false;$("#fpsBadge").hidden=false;$("#stopBtn").disabled=false;$("#pauseBtn").disabled=false;$("#cameraBtn").disabled=!state.cameraStream;}
render();showEmpty();
let batchPlaybackTimer=null;
window.ActionUI={getState:()=>({...state,cameraStream:null,videoUrl:null,bridge:null}),playBatch(frames){if(!Array.isArray(frames)||!frames.length)return;if(batchPlaybackTimer)clearInterval(batchPlaybackTimer);let index=0;const showFrame=()=>{const frame=frames[index];if(frame)window.ActionUI.setState({annotated_frame:frame.annotated_frame,people:frame.people,fps:frame.fps});index++;if(index>=frames.length){clearInterval(batchPlaybackTimer);batchPlaybackTimer=null;}};showFrame();if(frames.length>1)batchPlaybackTimer=setInterval(showFrame,33);},setState(payload={}){const overlay=$("#inferenceOverlay");if(typeof payload.annotated_frame==="string"){overlay.src=payload.annotated_frame;overlay.hidden=false;}else{overlay.hidden=true;overlay.removeAttribute("src");}if(typeof payload.fps==="number"){$("#fpsValue").textContent=payload.fps.toFixed(1);$("#statFps").textContent=payload.fps.toFixed(1);}if(Array.isArray(payload.people)){state.people=payload.people.map((p,i)=>({id:p.id??i+1,person:p.person??"Person "+(i+1),action:p.action??"Unknown",confidence:Number(p.confidence??0),time:p.time??new Date().toLocaleTimeString("en-GB"),color:p.color??["green","blue","red","purple","cyan"][i%5]}));render();}if(typeof payload.running==="boolean"){state.running=payload.running;$("#liveBadge").hidden=!payload.running;}if(typeof payload.modelStatus==="string")if($("#engineStatusText"))$("#engineStatusText").textContent=payload.modelStatus;if(Array.isArray(payload.logs))$("#logList").replaceChildren(...payload.logs.map(x=>{const p=document.createElement("p");p.textContent=x.text??String(x);return p;}));if(payload.statistics)for(const [k,v] of Object.entries(payload.statistics)){const el=document.getElementById(k);if(el)el.textContent=v;}},setTheme,addLog:log};
async function connectBridge(){if(state.bridge)return state.bridge;if($("#engineStatusText"))$("#engineStatusText").textContent="Connecting to Gradio…";try{const {Client}=await import("https://esm.sh/@gradio/client@2.7.2");state.bridge=await Client.connect(location.origin);state.bridgeConnected=true;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge connected · awaiting model inference";log("Connected to Gradio data bridge","green");return state.bridge;}catch(e){state.bridgeConnected=false;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge offline · retry when ready";log("Gradio connection failed: "+e.message,"red");throw e;}}
const BATCH_SIZE=30;
const INITIAL_CAPTURE_INTERVAL_MS=300;
let captureIntervalMs=INITIAL_CAPTURE_INTERVAL_MS;
let frameCanvas=null,frameContext=null,captureTimer=null,batchFrames=[],pendingBatch=null,batchBusy=false,streamGeneration=0;
function getFrameCanvas(video){
  if(!frameCanvas){frameCanvas=document.createElement("canvas");frameContext=frameCanvas.getContext("2d",{alpha:false});}
  const scale=Math.min(1,640/video.videoWidth);
  const width=Math.max(1,Math.round(video.videoWidth*scale));
  const height=Math.max(1,Math.round(video.videoHeight*scale));
  if(frameCanvas.width!==width||frameCanvas.height!==height){frameCanvas.width=width;frameCanvas.height=height;}
  return {canvas:frameCanvas,context:frameContext};
}
async function encodeCurrentFrame(){
  const video=$("#cameraVideo");
  if(video.readyState<2||video.videoWidth===0)return null;
  const {canvas,context}=getFrameCanvas(video);
  context.drawImage(video,0,0,canvas.width,canvas.height);
  return await new Promise((resolve,reject)=>canvas.toBlob(value=>value?resolve(value):reject(new Error("JPEG encoding failed")),"image/jpeg",0.78));
}
async function packFrameBatch(blobs){
  const parts=[];
  const header=new Uint8Array(8);
  header.set([72,65,66,49],0);
  new DataView(header.buffer).setUint32(4,blobs.length,false);
  parts.push(header);
  const encoded=await Promise.all(blobs.map(blob=>blob.arrayBuffer()));
  for(const buffer of encoded){
    const bytes=new Uint8Array(buffer);
    const length=new Uint8Array(4);
    new DataView(length.buffer).setUint32(0,bytes.byteLength,false);
    parts.push(length,bytes);
  }
  return new Blob(parts,{type:"application/octet-stream"});
}
async function sendFrameBatch(blobs,generation){
  batchBusy=true;
  state.bridgeBusy=true;
  const started=performance.now();
  try{
    const packed=await packFrameBatch(blobs);
    const bridge=await connectBridge();
    const file=new File([packed],"frames.hab",{type:"application/octet-stream"});
    const result=await bridge.predict("/receive_yolo_only_batch",{batch:file});
    if(generation!==streamGeneration)return;
    const payload=Array.isArray(result.data)?result.data[0]:result.data;
    if(payload&&typeof payload==="object"){
      if(Array.isArray(payload.frames)&&payload.frames.length)window.ActionUI.playBatch(payload.frames);
      else if(Array.isArray(payload.people)||typeof payload.fps==="number")window.ActionUI.setState(payload);
      if(payload.message&&state.lastFrameAt===0)log(payload.message);
      const measured=payload.batch_timing?.inference_ms_per_frame;
      if(typeof measured==="number"&&Number.isFinite(measured)&&measured>0){
        captureIntervalMs=Math.max(80,Math.min(1000,measured*0.9));
      }
      log("Processed batch: "+(payload.batch_timing?.frame_count??blobs.length)+" frames · "+(payload.batch_timing?.inference_ms_per_frame??"—")+" ms/frame","green");
    }
    state.lastFrameAt=Date.now();
    console.table([{batch_frames:blobs.length,batch_roundtrip_ms:+(performance.now()-started).toFixed(2),server_decode_ms:payload?.batch_timing?.decode_ms??null,server_inference_ms:payload?.batch_timing?.inference_ms??null,server_inference_ms_per_frame:payload?.batch_timing?.inference_ms_per_frame??null,capture_interval_ms:+captureIntervalMs.toFixed(1),batch_kb:+(packed.size/1024).toFixed(1)}]);
  }catch(e){
    if(generation===streamGeneration){
      if(state.bridgeConnected){state.bridge=null;state.bridgeConnected=false;}
      log("Batch transfer error: "+e.message,"red");
    }
  }finally{
    if(generation===streamGeneration){
      batchBusy=false;
      state.bridgeBusy=false;
      if(state.running&&!state.paused){
        if(pendingBatch){
          const next=pendingBatch;
          pendingBatch=null;
          void sendFrameBatch(next,generation);
        }
        scheduleNextCapture(generation);
      }
    }
  }
}
async function captureNextFrame(generation){
  if(generation!==streamGeneration||!state.running||state.paused)return;
  const blob=await encodeCurrentFrame();
  if(!blob||generation!==streamGeneration)return;
  batchFrames.push(blob);
  if(batchFrames.length>=BATCH_SIZE){
    const fullBatch=batchFrames;
    batchFrames=[];
    if(!batchBusy)void sendFrameBatch(fullBatch,generation);
    else if(!pendingBatch)pendingBatch=fullBatch;
    else log("Capture backpressure: waiting for the current batch","yellow");
  }
}
function scheduleNextCapture(generation){
  if(generation!==streamGeneration||!state.running||state.paused||captureTimer||pendingBatch)return;
  captureTimer=setTimeout(async()=>{
    captureTimer=null;
    try{await captureNextFrame(generation);}
    catch(e){log("Frame capture error: "+e.message,"red");}
    scheduleNextCapture(generation);
  },captureIntervalMs);
}
function startFrameTransfer(){
  stopFrameTransfer();
  const generation=streamGeneration;
  scheduleNextCapture(generation);
}
function stopFrameTransfer(){
  streamGeneration++;
  if(captureTimer)clearTimeout(captureTimer);
  captureTimer=null;
  batchFrames=[];
  pendingBatch=null;
  batchBusy=false;
  state.bridgeBusy=false;
  state.bridgeTimer=null;
  state.lastFrameAt=0;
}
async function camera(){if(!navigator.mediaDevices?.getUserMedia){log("Camera requires HTTPS or localhost","red");return;}try{if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:state.cameraFacing},audio:false});const v=$("#cameraVideo");v.srcObject=state.cameraStream;v.hidden=false;v.classList.remove("portrait-source");v.style.width="";v.style.height="";v.onloadedmetadata=null;await v.play();state.running=true;showMedia();log("Camera stream started","green");startFrameTransfer();}catch(e){log("Unable to open camera: "+e.name,"red");}}
function stopMedia(){stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);state.videoUrl=null;state.running=false;state.paused=false;$("#pauseBtn").textContent="Ⅱ　Pause";showEmpty();}
$$("[data-nav]").forEach(b=>b.addEventListener("click",()=>{state.page=b.dataset.nav;$$(".nav").forEach(n=>n.classList.toggle("active",n===b));$("#sidebar").classList.remove("open");if(state.cameraStream||state.videoUrl)stopMedia();showEmpty();}));
$("#mobileMenu").addEventListener("click",()=>$("#sidebar").classList.toggle("open"));
$("#themeToggle").addEventListener("click",()=>setTheme(state.theme==="dark"?"light":"dark"));
$$("[data-setting]").forEach(i=>i.addEventListener("change",()=>{state.settings[i.dataset.setting]=i.checked;render();}));
$("#openCameraBtn").addEventListener("click",camera);
$("#chooseVideoBtn").addEventListener("click",()=>$("#videoUpload").click());
$("#cameraBtn").addEventListener("click",()=>{state.cameraFacing=state.cameraFacing==="environment"?"user":"environment";camera();});
$("#stopBtn").addEventListener("click",()=>{stopMedia();log("Stream stopped","red");});
$("#pauseBtn").addEventListener("click",()=>{const v=$("#cameraVideo");if(!state.cameraStream&&!state.videoUrl)return;state.paused=!state.paused;if(state.paused)v.pause();else{v.play();if(state.running)startFrameTransfer();}$("#pauseBtn").textContent=state.paused?"▶　Resume":"Ⅱ　Pause";});
$("#videoUpload").addEventListener("change",e=>{const f=e.target.files?.[0];if(!f)return;stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);state.videoUrl=URL.createObjectURL(f);const v=$("#cameraVideo");v.classList.remove("portrait-source");v.style.width="";v.style.height="";v.onloadedmetadata=null;v.srcObject=null;v.src=state.videoUrl;v.hidden=false;v.onloadeddata=()=>{v.play();state.running=true;showMedia();$("#cameraBtn").disabled=true;startFrameTransfer();};log("Video loaded: "+f.name,"green");e.target.value="";});
$("#fullscreenBtn").addEventListener("click",()=>document.fullscreenElement?document.exitFullscreen():$("#videoFrame").requestFullscreen?.());
$("#clearLog").addEventListener("click",()=>$("#logList").replaceChildren());
$("#viewAll").addEventListener("click",()=>log("Showing "+state.people.length+" detected people"));
window.addEventListener("beforeunload",()=>{stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);});
})();