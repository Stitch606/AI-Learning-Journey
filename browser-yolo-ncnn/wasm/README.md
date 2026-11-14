# WebAssembly bridge

This folder will contain the small C++ bridge that exposes the NCNN pose model to JavaScript through Emscripten.

Required bridge behavior:
1. Load the matching NCNN .param and .bin files from the browser virtual filesystem.
2. Accept RGB image pixels from JavaScript.
3. Apply YOLO letterbox preprocessing.
4. Run the NCNN network.
5. Decode YOLO26 pose outputs, including person boxes and 17 COCO keypoints.
6. Return structured detections to JavaScript for drawing.

The output tensor names and exact decoder must be taken from the actual exported model, not guessed. That is why inference wiring follows model export and inspection.
