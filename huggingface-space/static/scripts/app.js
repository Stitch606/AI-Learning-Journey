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
function showEmpty(){$("#analysisProgress").hidden=true;const video=$("#cameraVideo");video.pause();video.srcObject=null;video.removeAttribute("src");video.hidden=true;const overlay=$("#inferenceOverlay");overlay.hidden=true;overlay.removeAttribute("src");$("#videoEmpty").hidden=false;$("#liveBadge").hidden=true;$("#fpsBadge").hidden=true;$("#openCameraBtn").hidden=state.page==="Video Analysis";$("#chooseVideoBtn").hidden=state.page!=="Video Analysis";$("#videoEyebrow").textContent=state.page==="Dashboard"?"LIVE DETECTION":"VIDEO ANALYSIS";$("#videoTitle").textContent=state.page==="Dashboard"?"Ready to monitor":"Choose a video to analyse";$("#videoHelp").textContent=state.page==="Dashboard"?"Start your camera to begin real-time action recognition.":"Upload a video file to inspect actions and detected people.";$("#stopBtn").disabled=true;$("#pauseBtn").disabled=true;$("#cameraBtn").disabled=true;}
function showMedia(){$("#analysisProgress").hidden=true;startRuntime();$("#videoEmpty").hidden=true;$("#liveBadge").hidden=false;$("#fpsBadge").hidden=false;$("#stopBtn").disabled=false;$("#pauseBtn").disabled=false;$("#cameraBtn").disabled=!state.cameraStream;}
render();showEmpty();

window.ActionUI={getState:()=>({...state,cameraStream:null,videoUrl:null,bridge:null}),setState(payload={}){const overlay=$("#inferenceOverlay");if(typeof payload.annotated_frame==="string"){overlay.src=payload.annotated_frame;overlay.hidden=false;}else{overlay.hidden=true;overlay.removeAttribute("src");}if(typeof payload.fps==="number"){$("#fpsValue").textContent=payload.fps.toFixed(1);$("#statFps").textContent=payload.fps.toFixed(1);}if(Array.isArray(payload.people)){state.people=payload.people.map((p,i)=>({id:p.id??i+1,person:p.person??"Person "+(i+1),action:p.action??"Unknown",confidence:Number(p.confidence??0),time:p.time??new Date().toLocaleTimeString("en-GB"),color:p.color??["green","blue","red","purple","cyan"][i%5]}));render();}if(typeof payload.running==="boolean"){state.running=payload.running;$("#liveBadge").hidden=!payload.running;}if(typeof payload.modelStatus==="string")if($("#engineStatusText"))$("#engineStatusText").textContent=payload.modelStatus;if(Array.isArray(payload.logs))$("#logList").replaceChildren(...payload.logs.map(x=>{const p=document.createElement("p");p.textContent=x.text??String(x);return p;}));if(payload.statistics)for(const [k,v] of Object.entries(payload.statistics)){const el=document.getElementById(k);if(el)el.textContent=v;}},setTheme,addLog:log};
async function connectBridge(){if(state.bridge)return state.bridge;if($("#engineStatusText"))$("#engineStatusText").textContent="Connecting to Gradio…";try{const {Client}=await import("https://esm.sh/@gradio/client@2.7.2");state.bridge=await Client.connect(location.origin);state.bridgeConnected=true;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge connected · awaiting model inference";log("Connected to Gradio data bridge","green");return state.bridge;}catch(e){state.bridgeConnected=false;if($("#engineStatusText"))$("#engineStatusText").textContent="Bridge offline · retry when ready";log("Gradio connection failed: "+e.message,"red");throw e;}}
const CAPTURE_INTERVAL_MS=33;
let frameCanvas=null,frameContext=null,captureLoopTask=null;
let streamGeneration=0,nextFrameId=0,lastRenderedFrameId=-1;
let lastTransferErrorAt=0;

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
async function sendYoloFrame(blob,frameId,generation){
  try{
    const bridge=await connectBridge();
    if(generation!==streamGeneration||!state.running)return;
    const file=new File([blob],"frame-"+frameId+".jpg",{type:"image/jpeg"});
    const response=await bridge.predict("/receive_yolo_frame",{
      frame_id:frameId,
      frame:file
    });
    if(generation!==streamGeneration||!state.running)return;
    const payload=Array.isArray(response.data)?response.data[0]:response.data;
    if(payload?.status==="ok"&&Number(payload.frame_id)>lastRenderedFrameId){
      lastRenderedFrameId=Number(payload.frame_id);
      window.ActionUI.setState(payload);
      state.lastFrameAt=Date.now();
    }else if(payload?.message&&payload.status==="error"){
      log(payload.message,"red");
    }
  }catch(e){
    if(generation===streamGeneration&&Date.now()-lastTransferErrorAt>3000){
      lastTransferErrorAt=Date.now();
      log("Frame transfer error: "+e.message,"red");
    }
  }
}
async function captureContinuously(generation){
  while(generation===streamGeneration&&state.running){
    const started=performance.now();
    if(!state.paused){
      try{
        const blob=await encodeCurrentFrame();
        if(blob&&generation===streamGeneration){
          const frameId=++nextFrameId;
          void sendYoloFrame(blob,frameId,generation);
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
    await connectBridge();
    if(generation!==streamGeneration||!state.running)return;
    nextFrameId=0;
    lastRenderedFrameId=-1;
    state.bridgeBusy=true;
    captureLoopTask=captureContinuously(generation);
    log("Continuous YOLO frame sending started","green");
  }catch(e){
    if(generation===streamGeneration){
      stopFrameTransfer();
      log("Unable to start YOLO sending: "+e.message,"red");
    }
  }
}
function stopFrameTransfer(){
  streamGeneration++;
  captureLoopTask=null;
  state.bridgeBusy=false;
  state.bridgeTimer=null;
  state.lastFrameAt=0;
}
async function camera(){if(!navigator.mediaDevices?.getUserMedia){log("Camera requires HTTPS or localhost","red");return;}try{if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:state.cameraFacing},audio:false});const v=$("#cameraVideo");v.srcObject=state.cameraStream;v.hidden=false;v.classList.remove("portrait-source");v.style.width="";v.style.height="";v.onloadedmetadata=null;await v.play();state.running=true;showMedia();log("Camera stream started","green");void startFrameTransfer();}catch(e){log("Unable to open camera: "+e.name,"red");}}
function stopMedia(){if(videoAnalysisAbort)videoAnalysisAbort.abort();if(videoUploadRequest)videoUploadRequest.abort();stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());state.cameraStream=null;if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);state.videoUrl=null;state.running=false;state.paused=false;$("#pauseBtn").textContent="Ⅱ　Pause";showEmpty();}
$$("[data-nav]").forEach(b=>b.addEventListener("click",()=>{state.page=b.dataset.nav;$$(".nav").forEach(n=>n.classList.toggle("active",n===b));$("#sidebar").classList.remove("open");if(state.cameraStream||state.videoUrl)stopMedia();showEmpty();}));
$("#mobileMenu").addEventListener("click",()=>$("#sidebar").classList.toggle("open"));
$("#themeToggle").addEventListener("click",()=>setTheme(state.theme==="dark"?"light":"dark"));
$$("[data-setting]").forEach(i=>i.addEventListener("change",()=>{state.settings[i.dataset.setting]=i.checked;render();}));
$("#openCameraBtn").addEventListener("click",camera);
$("#chooseVideoBtn").addEventListener("click",()=>$("#videoUpload").click());
$("#cameraBtn").addEventListener("click",()=>{state.cameraFacing=state.cameraFacing==="environment"?"user":"environment";camera();});
$("#stopBtn").addEventListener("click",()=>{stopMedia();log("Stream stopped","red");});
$("#pauseBtn").addEventListener("click",()=>{const v=$("#cameraVideo");if(!state.cameraStream&&!state.videoUrl)return;state.paused=!state.paused;if(state.paused){v.pause();if(state.cameraStream)stopFrameTransfer();}else{v.play();if(state.cameraStream&&state.running)void startFrameTransfer();}$("#pauseBtn").textContent=state.paused?"▶　Resume":"Ⅱ　Pause";});
let videoAnalysisAbort=null,videoUploadRequest=null;
function updateAnalysisProgress(stage,percent,detail){
  const panel=$("#analysisProgress");
  panel.hidden=false;
  panel.classList.toggle("processing",stage==="processing");
  panel.classList.toggle("queued",stage==="queued");
  $("#analysisProgressTitle").textContent=stage==="upload"?"Uploading video":stage==="queued"?"Upload complete · waiting for GPU":"AI processing video";
  $("#analysisProgressValue").textContent=Number.isFinite(percent)?Math.round(percent)+"%":"…";
  $("#analysisProgressBar").style.width=Number.isFinite(percent)?Math.max(0,Math.min(100,percent))+"%":"32%";
  $("#analysisProgressDetail").textContent=detail;
  $("#uploadStage").className=stage==="upload"?"active":"done";
  $("#processingStage").className=stage==="processing"||stage==="queued"?"active":stage==="complete"?"done":"";
}
function uploadVideoFile(file){
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();
    videoUploadRequest=xhr;
    xhr.open("POST","/api/analyze-video");
    xhr.responseType="json";
    xhr.timeout=300000;
    xhr.upload.onprogress=event=>{
      if(event.lengthComputable){
        const percent=event.total?event.loaded/event.total*100:0;
        updateAnalysisProgress("upload",percent,"Uploading "+(file.size/1048576).toFixed(1)+" MB · "+Math.round(percent)+"% sent");
      }else updateAnalysisProgress("upload",null,"Uploading video…");
    };
    xhr.onload=()=>{
      videoUploadRequest=null;
      if(xhr.status>=200&&xhr.status<300&&xhr.response?.job_id)resolve(xhr.response);
      else reject(new Error(xhr.response?.detail||"Video upload failed ("+xhr.status+")"));
    };
    xhr.onerror=()=>{videoUploadRequest=null;reject(new Error("Network error while uploading the video."));};
    xhr.ontimeout=()=>{videoUploadRequest=null;reject(new Error("Video upload timed out. Please try again."));};
    xhr.onabort=()=>{videoUploadRequest=null;reject(new DOMException("Upload cancelled","AbortError"));};
    const form=new FormData();
    form.append("file",file,file.name);
    xhr.send(form);
  });
}
async function waitForVideoJob(jobId,signal){
  while(true){
    const response=await fetch("/api/analyze-video/"+encodeURIComponent(jobId),{signal});
    const status=await response.json();
    if(!response.ok)throw new Error(status.detail||"Could not read video analysis status.");
    if(status.status==="failed")throw new Error(status.error||"Video processing failed.");
    if(status.status==="completed")return status.result;
    if(status.status==="queued"){
      updateAnalysisProgress("queued",null,"Upload complete · waiting for the GPU worker…");
    }else{
      const percent=Number(status.progress);
      const count=status.total_frames?status.processed_frames+" / "+status.total_frames+" sampled frames":"Processing sampled frames…";
      updateAnalysisProgress("processing",Number.isFinite(percent)?percent:null,count);
    }
    await new Promise(resolve=>setTimeout(resolve,1200));
  }
}
function waitForVideoFrame(video, timeoutMs=20000){
  return new Promise((resolve,reject)=>{
    let settled=false;
    const cleanup=()=>{
      clearTimeout(timer);
      video.removeEventListener("loadeddata",check);
      video.removeEventListener("canplay",check);
      video.removeEventListener("error",onError);
    };
    const finish=(error)=>{
      if(settled)return;
      settled=true;
      cleanup();
      error?reject(error):resolve();
    };
    const check=()=>{
      if(video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA&&video.videoWidth>0&&video.videoHeight>0)finish();
    };
    const onError=()=>{
      const mediaError=video.error;
      finish(new Error("The analyzed video could not be decoded by this browser (media error "+(mediaError?.code??"unknown")+")."));
    };
    const timer=setTimeout(()=>finish(new Error("The analyzed video did not produce a visible frame within "+(timeoutMs/1000)+" seconds.")),timeoutMs);
    video.addEventListener("loadeddata",check);
    video.addEventListener("canplay",check);
    video.addEventListener("error",onError);
    check();
  });
}
async function analyzeUploadedVideo(file){
  stopFrameTransfer();
  if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());
  state.cameraStream=null;
  if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);
  state.videoUrl=null;
  state.running=false;
  state.paused=false;
  stopRuntime(true);
  const video=$("#cameraVideo");
  video.pause();
  video.srcObject=null;
  video.classList.remove("portrait-source");
  video.style.width="";
  video.style.height="";
  video.hidden=false;
  video.controls=true;
  video.muted=true;
  video.playsInline=true;
  video.preload="auto";
  const previewUrl=URL.createObjectURL(file);
  video.src=previewUrl;
  await video.play().catch(()=>{});
  video.pause();
  $("#videoEmpty").hidden=true;
  $("#liveBadge").hidden=false;
  $("#liveBadge").textContent="UPLOADING";
  $("#fpsBadge").hidden=true;
  $("#stopBtn").disabled=false;
  $("#pauseBtn").disabled=true;
  $("#cameraBtn").disabled=true;
  $("#videoTitle").textContent="Uploading video";
  $("#videoHelp").textContent="The upload and AI processing stages will be shown below.";
  updateAnalysisProgress("upload",0,"Preparing "+file.name+"…");
  log("Video selected: "+file.name+" · upload started","green");
  videoAnalysisAbort=new AbortController();
  try{
    const job=await uploadVideoFile(file);
    $("#liveBadge").textContent="PROCESSING";
    $("#videoTitle").textContent="AI analysis in progress";
    log("Upload complete · background analysis started","green");
    const result=await waitForVideoJob(job.job_id,videoAnalysisAbort.signal);
    updateAnalysisProgress("complete",100,"Analysis complete · preparing the video player…");
    state.videoUrl="/api/analyze-video/"+encodeURIComponent(job.job_id)+"/result";
    video.pause();
    video.src=state.videoUrl;
    video.load();
    await waitForVideoFrame(video);
    URL.revokeObjectURL(previewUrl);
    try{
      await video.play();
    }catch(playError){
      log("Video is ready; press Play in the video controls to start viewing ("+playError.name+").","blue");
    }
    state.running=!video.paused;
    showMedia();
    $("#liveBadge").textContent="ANALYZED";
    $("#fpsBadge").hidden=true;
    $("#pauseBtn").disabled=false;
    $("#pauseBtn").textContent=video.paused?"▶　Resume":"Ⅱ　Pause";
    $("#cameraBtn").disabled=true;
    if(Array.isArray(result?.people))window.ActionUI.setState({people:result.people});
    $("#videoTitle").textContent="Video analysis complete";
    $("#videoHelp").textContent=result.frames_analyzed+" frames analyzed · "+result.duration_seconds+" seconds of video · "+video.videoWidth+" × "+video.videoHeight;
    log("Video player ready · "+video.videoWidth+" × "+video.videoHeight+" · "+result.frames_analyzed+" frames analyzed","green");
  }catch(error){
    URL.revokeObjectURL(previewUrl);
    if(error.name!=="AbortError")log("Video analysis/playback error: "+error.message,"red");
    state.running=false;
    showEmpty();
    $("#videoTitle").textContent=error.name==="AbortError"?"Analysis cancelled":"Video playback failed";
    $("#videoHelp").textContent=error.name==="AbortError"?"The video analysis was cancelled.":error.message;
  }finally{
    videoAnalysisAbort=null;
    videoUploadRequest=null;
  }
}
$("#videoUpload").addEventListener("change",e=>{const f=e.target.files?.[0];if(f)void analyzeUploadedVideo(f);e.target.value="";});
$("#fullscreenBtn").addEventListener("click",()=>document.fullscreenElement?document.exitFullscreen():$("#videoFrame").requestFullscreen?.());
$("#clearLog").addEventListener("click",()=>$("#logList").replaceChildren());
$("#viewAll").addEventListener("click",()=>log("Showing "+state.people.length+" detected people"));
window.addEventListener("beforeunload",()=>{stopRuntime(true);stopFrameTransfer();if(state.cameraStream)state.cameraStream.getTracks().forEach(t=>t.stop());if(state.videoUrl)URL.revokeObjectURL(state.videoUrl);});
})();