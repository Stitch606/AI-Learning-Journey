# Model assets

Place the exported NCNN model files in this folder.

Run from the project root:

    python -m pip install --upgrade ultralytics
    python tools/export_model.py

Expected files include an NCNN .param and matching .bin, plus Ultralytics metadata when generated. The selected source model is yolo26n-pose.pt; export precision is FP16 and initial input size is 416.
