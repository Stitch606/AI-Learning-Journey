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

window.ActionUI={getState:()=>({...state,cameraStream:null,videoUrl:null,bridge:null}),setState(payload={}){const overlay=$("#inferenceOverlay");if(typeof payload.annotated_frame==="string"){overlay.src=payload.annotated_frame;overlay.hidden=false;}else{overlay.hidden=true;overlay.removeAttribute("src");}if(typeof payload.fps==="number"){$("#fpsValue").textContent=payload.fps.toFixed(1);$("#statFps").textContent=payload.fps.toFixed(1);}if(Array.isArray(payload.people)){state.people=payload.people.map((p,i)=>({id:p.id??i+1,person:p.person??"Person "+(i+1),action:p.action??"Unknown",confidence:Number(p.confidence??0),time:p.time??new Date().toLocaleTimeString("en-GB"),color:p.color??["green","blue","red","purple","cyan"][i%5]}));render();}if(typeof payload.running==="boolean"){state.running=payload.running;$("#liveBadge").hidden=!payload.running;}if(typeof payload.modelStatus==="string")if($("#engineStatusText"))$("#engineStatusText").textContent=payload.modelStatus;if(Array.isArray(payload.logs))$("#logList").replaceChildren(...payload.logs.map(x=>{const p=document.createElement("p");p.textContent=x.text??String(x);return p;}));if(payload.statistics)for(const [k,v] of Object.entries(payload.statistics)){const el=document.getElementById(k);if(el)el.textContent=v;}},setTheme,addLog:log};
async function connectBridge(){if(state.bridge)return state.bridge;if($("#engineStatusText"))$("#engineStatusText").textContent="Connecting to Gradio…";try{const {Client}=await import("https://esm.sh/@gradio/client@2.7.2");state.bridge=await Client.connect(location.origin);state.bridgeConnected=true;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge connected · awaiting model inference";log("Connected to Gradio data bridge","green");return state.bridge;}catch(e){state.bridgeConnected=false;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge offline · retry when ready";log("Gradio connection failed: "+e.message,"red");throw e;}}
const CAPTURE_INTERVAL_MS=33;
let frameCanvas=null,frameContext=null,captureLoopTask=null;
let streamGeneration=0,activeStreamId=null,streamJob=null;
let nextFrameId=0,outgoingFrames=[],sendingFrames=false,lastRenderedFrameId=-1;

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
async function consumeYoloResults(job,generation){
  try{
    for await(const message of job){
      if(generation!==streamGeneration)break;
      if(message.type!=="data")continue;
      const payload=Array.isArray(message.data)?message.data[0]:message.data;
      if(!payload||typeof payload!=="object")continue;
      if(payload.status==="ok"){
        if(Number(payload.frame_id)>lastRenderedFrameId){
          lastRenderedFrameId=Number(payload.frame_id);
          window.ActionUI.setState(payload);
          state.lastFrameAt=Date.now();
        }
      }else if(payload.message){
        log(payload.message,"red");
      }
    }
  }catch(e){
    if(generation===streamGeneration)log("YOLO result stream error: "+e.message,"red");
  }
}
async function flushOutgoingFrames(streamId,generation){
  if(sendingFrames)return;
  sendingFrames=true;
  let currentItem=null;
  try{
    const bridge=await connectBridge();
    while(outgoingFrames.length&&generation===streamGeneration&&state.running){
      currentItem=outgoingFrames.shift();
      const file=new File([currentItem.blob],"frame-"+currentItem.id+".jpg",{type:"image/jpeg"});
      await bridge.predict("/push_yolo_frame",{
        stream_id:streamId,
        frame_id:currentItem.id,
        frame:file
      });
      currentItem=null;
    }
  }catch(e){
    if(currentItem)outgoingFrames.unshift(currentItem);
    if(generation===streamGeneration)log("Frame transfer error: "+e.message,"red");
    await new Promise(resolve=>setTimeout(resolve,250));
  }finally{
    sendingFrames=false;
    if(outgoingFrames.length&&generation===streamGeneration&&state.running){
      void flushOutgoingFrames(streamId,generation);
    }
  }
}
async function captureContinuously(streamId,generation){
  while(generation===streamGeneration&&state.running){
    const started=performance.now();
    if(!state.paused){
      try{
        const blob=await encodeCurrentFrame();
        if(blob&&generation===streamGeneration){
          const frameId=++nextFrameId;
          outgoingFrames.push({id:frameId,blob});
          void flushOutgoingFrames(streamId,generation);
        }
      }catch(e){
        if(generation===streamGeneration)log("Frame capture error: "+e.message,"red");
      }
    }
    const elapsed=performance.now()-started;
    await new Promise(resolve=>setTimeout(resolve,Math.max(0,CAPTURE_INTERVAL_MS-elapsed)));
  }
}
async function startFrameTransfer(){
  stopFrameTransfer();
  const generation=streamGeneration;
  try{
    const bridge=await connectBridge();
    const started=await bridge.predict("/start_yolo_stream",{});
    const startPayload=Array.isArray(started.data)?started.data[0]:started.data;
    if(!startPayload?.stream_id)throw new Error(startPayload?.message||"Could not create YOLO stream.");
    if(generation!==streamGeneration){
      await bridge.predict("/stop_yolo_stream",{stream_id:startPayload.stream_id}).catch(()=>{});
      return;
    }
    activeStreamId=startPayload.stream_id;
    nextFrameId=0;
    lastRenderedFrameId=-1;
    outgoingFrames=[];
    state.bridgeBusy=true;
    streamJob=bridge.submit("/run_yolo_stream",{stream_id:activeStreamId});
    void consumeYoloResults(streamJob,generation);
    captureLoopTask=captureContinuously(activeStreamId,generation);
    log("Continuous YOLO stream started","green");
  }catch(e){
    if(generation===streamGeneration){
      state.bridgeBusy=false;
      log("Unable to start YOLO stream: "+e.message,"red");
    }
  }
}
function stopFrameTransfer(){
  streamGeneration++;
  const oldStreamId=activeStreamId;
  const oldJob=streamJob;
  activeStreamId=null;
  streamJob=null;
  outgoingFrames=[];
  sendingFrames=false;
  captureLoopTask=null;
  state.bridgeBusy=false;
  state.bridgeTimer=null;
  state.lastFrameAt=0;
  if(oldStreamId&&state.bridge){
    const stopped=state.bridge.predict("/stop_yolo_stream",{stream_id:oldStreamId}).catch(()=>{});
    if(oldJob)void stopped.finally(()=>oldJob.cancel());
  }else if(oldJob){
    oldJob.cancel();
  }
}
async function camera(){if(!navigator.mediaDevices?.getUserMedia){log("Camera requires HTTPS or localhost","red");return;}try{if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:state.cameraFacing},audio:false});const v=$("#cameraVideo");v.srcObject=state.cameraStream;v.hidden=false;v.classList.remove("portrait-source");v.style.width="";v.style.height="";v.onloadedmetadata=null;await v.play();state.running=true;showMedia();log("Camera stream started","green");void startFrameTransfer();}catch(e){log("Unable to open camera: "+e.name,"red");}}
function stopMedia(){stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);state.videoUrl=null;state.running=false;state.paused=false;$("#pauseBtn").textContent="Ⅱ　Pause";showEmpty();}
$$("[data-nav]").forEach(b=>b.addEventListener("click",()=>{state.page=b.dataset.nav;$$(".nav").forEach(n=>n.classList.toggle("active",n===b));$("#sidebar").classList.remove("open");if(state.cameraStream||state.videoUrl)stopMedia();showEmpty();}));
$("#mobileMenu").addEventListener("click",()=>$("#sidebar").classList.toggle("open"));
$("#themeToggle").addEventListener("click",()=>setTheme(state.theme==="dark"?"light":"dark"));
$$("[data-setting]").forEach(i=>i.addEventListener("change",()=>{state.settings[i.dataset.setting]=i.checked;render();}));
$("#openCameraBtn").addEventListener("click",camera);
$("#chooseVideoBtn").addEventListener("click",()=>$("#videoUpload").click());
$("#cameraBtn").addEventListener("click",()=>{state.cameraFacing=state.cameraFacing==="environment"?"user":"environment";camera();});
$("#stopBtn").addEventListener("click",()=>{stopMedia();log("Stream stopped","red");});
$("#pauseBtn").addEventListener("click",()=>{const v=$("#cameraVideo");if(!state.cameraStream&&!state.videoUrl)return;state.paused=!state.paused;if(state.paused)v.pause();else v.play();$("#pauseBtn").textContent=state.paused?"▶　Resume":"Ⅱ　Pause";});
$("#videoUpload").addEventListener("change",e=>{const f=e.target.files?.[0];if(!f)return;stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);state.videoUrl=URL.createObjectURL(f);const v=$("#cameraVideo");v.classList.remove("portrait-source");v.style.width="";v.style.height="";v.onloadedmetadata=null;v.srcObject=null;v.src=state.videoUrl;v.hidden=false;v.onloadeddata=()=>{v.play();state.running=true;showMedia();$("#cameraBtn").disabled=true;void startFrameTransfer();};log("Video loaded: "+f.name,"green");e.target.value="";});
$("#fullscreenBtn").addEventListener("click",()=>document.fullscreenElement?document.exitFullscreen():$("#videoFrame").requestFullscreen?.());
$("#clearLog").addEventListener("click",()=>$("#logList").replaceChildren());
$("#viewAll").addEventListener("click",()=>log("Showing "+state.people.length+" detected people"));
window.addEventListener("beforeunload",()=>{stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);});
})();