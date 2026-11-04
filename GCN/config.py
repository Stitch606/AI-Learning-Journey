"""Default configuration for ST-GCN++ with YOLO COCO-17 keypoints."""

NUM_CLASSES = 10  # Replace with the number of action classes in your dataset.
NUM_PERSON = 1
NUM_KEYPOINTS = 17
IN_CHANNELS = 3  # x, y, confidence
SEQUENCE_LENGTH = 100
BASE_CHANNELS = 64
NUM_STAGES = 10
DROPOUT = 0.0
ADAPTIVE_GRAPH = True
