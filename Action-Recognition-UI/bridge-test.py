import numpy as np
from gradio_client import Client

client = Client("http://127.0.0.1:7860/gradio")
result = client.predict(np.zeros((80, 120, 3), dtype=np.uint8), api_name="/receive_frame")
assert result["status"] == "ok", result
assert result["width"] == 120 and result["height"] == 80, result
assert result["model_connected"] is False, result
assert "process_frame()" in result["message"], result
print("Gradio frame bridge passed:", result)
