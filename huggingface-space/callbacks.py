from video_panel import resolve_video_path, inspect_video
from charts import create_activity_chart, create_distribution_chart
import tempfile, csv, os
from datetime import datetime

def empty_result():
    return {"status":"Ready","people":0,"actions":0,"falls":0,"collisions":0,"walking":0,"standing":0,"lifting":0,"falling":0,"timestamps":[],"walking_history":[],"standing_history":[],"lifting_history":[],"falling_history":[],"events":[]}

def normalize_result(result):
    if not isinstance(result,dict): raise ValueError("AI processor must return a dictionary")
    data=empty_result(); data.update(result)
    for field in ("people","actions","falls","collisions"):
        try: data[field]=max(0,int(float(data[field])))
        except (ValueError,TypeError): data[field]=0
    for field in ("walking","standing","lifting","falling"):
        try: data[field]=max(0,min(100,float(data[field])))
        except (ValueError,TypeError): data[field]=0
    if not isinstance(data["events"],list): data["events"]=[]
    return data

def build_dashboard_outputs(result):
    data=normalize_result(result)
    chart=create_activity_chart(data["timestamps"],data["walking_history"],data["standing_history"],data["lifting_history"],data["falling_history"])
    distribution=create_distribution_chart({"Walking":data["walking"],"Standing":data["standing"],"Lifting":data["lifting"],"Fall":data["falling"]})
    rows=[]
    for event in data["events"]:
        if isinstance(event,dict):
            try: confidence=float(event.get("confidence",0))
            except (ValueError,TypeError): confidence=0
            rows.append([str(event.get("action","Unknown")),f"{confidence:.2f}",str(event.get("time","--:--:--"))])
    return (data["status"],100 if data["status"]=="Analysis completed" else 0,data["people"],data["actions"],data["falls"],data["collisions"],data["walking"],data["standing"],data["lifting"],data["falling"],chart,distribution,rows)

def run_analysis(video_input,threshold,model,device,processor=None):
    video_path=resolve_video_path(video_input)
    if video_path is None:
        result=empty_result(); result["status"]="Choose a video first, or select Webcam as the video source."
        return build_dashboard_outputs(result)
    metadata=inspect_video(video_path)
    if metadata is None:
        result=empty_result(); result["status"]="The selected video could not be opened. Try MP4 or another video."
        return build_dashboard_outputs(result)
    if processor is None:
        result=empty_result(); result["status"]=f"Video loaded: {metadata['filename']} · {metadata['resolution']} · {metadata['fps']} FPS. AI processor is waiting to be connected."
        return build_dashboard_outputs(result)
    try:
        ai_result=processor(video_path=video_path,threshold=threshold,model=model,device=device)
        result=normalize_result(ai_result); result["status"]="Analysis completed"
        return build_dashboard_outputs(result)
    except Exception as error:
        result=empty_result(); result["status"]=f"Analysis error: {error}"
        return build_dashboard_outputs(result)

def reset_dashboard():
    result=empty_result(); result["status"]="Dashboard reset. Upload a video or choose Webcam to begin."
    return build_dashboard_outputs(result)

def camera_help():
    return "📷 **To use your camera:** open the video source selector above the preview and choose **Webcam**. Allow camera permission when your browser asks."

def clear_video():
    return None,"Video cleared. Upload a new clip or choose Webcam.",0

def export_report(video,threshold,model,device):
    path=resolve_video_path(video); metadata=inspect_video(path) if path else None
    fd,report_path=tempfile.mkstemp(prefix="action_detector_",suffix=".csv"); os.close(fd)
    with open(report_path,"w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.writer(handle)
        writer.writerow(["Action Detector - Analysis Report"])
        writer.writerow(["Generated",datetime.now().isoformat(timespec="seconds")])
        writer.writerow(["Video",metadata["filename"] if metadata else "Not selected"])
        writer.writerow(["Resolution",metadata["resolution"] if metadata else "--"])
        writer.writerow(["FPS",metadata["fps"] if metadata else "--"])
        writer.writerow(["Confidence threshold",threshold]); writer.writerow(["Model",model]); writer.writerow(["Device",device])
        writer.writerow([]); writer.writerow(["Note","AI model is not connected yet; this report contains selected input and settings only."])
    return report_path
