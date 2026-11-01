import gradio as gr
from layout import create_layout
from callbacks import run_analysis, reset_dashboard, camera_help, clear_video, export_report

def build_app(processor=None):
    ui = create_layout()
    demo = ui["demo"]
    outputs = [ui["video_status"], ui["progress"], ui["people"], ui["actions"], ui["falls"], ui["collisions"], ui["walking"], ui["standing"], ui["lifting"], ui["falling"], ui["chart"], ui["distribution"], ui["events"]]
    def start(video, threshold, model, device):
        return run_analysis(video, threshold, model, device, processor=processor)
    ui["start_btn"].click(fn=start, inputs=[ui["video"], ui["threshold"], ui["model"], ui["device"]], outputs=outputs, show_progress="full")
    ui["stop_btn"].click(fn=reset_dashboard, inputs=None, outputs=outputs)
    ui["camera_btn"].click(fn=camera_help, inputs=None, outputs=ui["video_status"])
    ui["clear_btn"].click(fn=clear_video, inputs=None, outputs=[ui["video"], ui["video_status"], ui["progress"]])
    ui["video"].clear(fn=clear_video, inputs=None, outputs=[ui["video"], ui["video_status"], ui["progress"]], queue=False)
    ui["download_btn"].click(fn=export_report, inputs=[ui["video"], ui["threshold"], ui["model"], ui["device"]], outputs=ui["download_btn"])
    demo.queue()
    return demo

if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
