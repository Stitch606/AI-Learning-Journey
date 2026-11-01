
import gradio as gr

from styles import CSS
from layout import create_layout

from callbacks import (
    run_analysis,
    reset_dashboard
)


# ==========================================
# 1. CREATE APPLICATION
# ==========================================

def build_app(processor=None):

    """
    Build the complete Action Detector application.

    processor:
        External AI processing function.
        Will be connected when the AI engine is ready.
    """

    # Create all UI components.
    ui = create_layout()

    demo = ui["demo"]

    # ======================================
    # 2. START ANALYSIS
    # ======================================

    def start_analysis(video, threshold, model, device):

        return run_analysis(
            video_input=video,
            threshold=threshold,
            model=model,
            device=device,
            processor=processor
        )

    analysis_outputs = [

        ui["video_status"],

        ui["people"],
        ui["actions"],
        ui["falls"],
        ui["collisions"],

        ui["walking"],
        ui["standing"],
        ui["lifting"],
        ui["falling"],

        ui["chart"],

        ui["events"]
    ]

    ui["start_btn"].click(
        fn=start_analysis,

        inputs=[
            ui["upload"],
            ui["threshold"],
            ui["model"],
            ui["device"]
        ],

        outputs=analysis_outputs,

        show_progress="full"
    )

    # ======================================
    # 3. RESET DASHBOARD
    # ======================================

    # We will connect a dedicated reset button
    # when adding it to the layout.

    # ======================================
    # 4. APPLICATION CONFIGURATION
    # ======================================

    demo.queue()

    return demo


# ==========================================
# 5. LAUNCH APPLICATION
# ==========================================

def launch():

    demo = build_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS
    )


# ==========================================
# 6. ENTRY POINT
# ==========================================

if __name__ == "__main__":

    launch()
