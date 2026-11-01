from main import build_app
from styles import CSS


# Create application
demo = build_app()


# Hugging Face Spaces entry point
if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS
    )
    