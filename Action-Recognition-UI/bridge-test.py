import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from gradio_client import Client

client = Client("http://127.0.0.1:7860/gradio")
test_array = np.zeros((80, 120, 3), dtype=np.uint8)

with tempfile.TemporaryDirectory() as temp_dir:
    image_path = Path(temp_dir) / "test-frame.png"
    Image.fromarray(test_array, mode="RGB").save(image_path)
    result = client.predict(str(image_path), api_name="/receive_frame")

assert result["status"] == "ok", result
assert result["width"] == 120 and result["height"] == 80, result
assert result["frame_shape"] == [80, 120, 3], result
assert result["frame_dtype"] == "uint8", result
assert result["model_connected"] is False, result
assert "converted to NumPy" in result["message"], result
print("Gradio frame -> PIL -> NumPy bridge passed:", result)
