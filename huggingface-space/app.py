from main import build_app
from styles import CSS
from pathlib import Path

demo = build_app()

# Inject ordinary browser-side interactions after the Gradio app mounts.
INTERACTIONS_JS = Path(__file__).with_name("interactions.js").read_text(encoding="utf-8")

# Gradio event listeners must be registered inside the Blocks context.
with demo:
    demo.load(fn=None, js=INTERACTIONS_JS)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
