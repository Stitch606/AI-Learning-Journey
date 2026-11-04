"""Configuration matching PYSKL's official ST-GCN++ NTU60-XSub HRNet JM setup."""

# Target project classes. The published checkpoint itself has a 60-class head.
NUM_CLASSES = 10
PRETRAINED_NUM_CLASSES = 60

NUM_PERSON = 2
NUM_KEYPOINTS = 17
IN_CHANNELS = 3  # x, y, confidence
SEQUENCE_LENGTH = 100

BASE_CHANNELS = 64
NUM_STAGES = 10
CH_RATIO = 2
INFLATE_STAGES = (5, 8)
DOWN_STAGES = (5, 8)

GRAPH_LAYOUT = "coco"
GRAPH_MODE = "spatial"
GCN_ADAPTIVE = "init"
GCN_WITH_RES = True
TCN_TYPE = "mstcn"
TCN_DROPOUT = 0.0
DATA_BN_TYPE = "VC"

# Official PYSKL checkpoint: NTU60 XSub, HRNet 2D, Joint Motion.
PRETRAINED_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/"
    "stgcnpp_ntu60_xsub_hrnet/jm.pth"
)
