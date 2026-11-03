from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/"
    "ctrgcn_pyskl_ntu60_xsub_hrnet/jm.pth"
)

HERE = Path(__file__).resolve().parent
WEIGHTS_DIR = HERE / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
DESTINATION = WEIGHTS_DIR / "ctrgcn_coco17_joint_motion.pth"

if DESTINATION.exists() and DESTINATION.stat().st_size > 0:
    print(f"Checkpoint already exists: {DESTINATION}")
else:
    print("Downloading official CTR-GCN COCO-layout Joint Motion checkpoint...")
    urlretrieve(URL, DESTINATION)
    print(f"Saved: {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")
