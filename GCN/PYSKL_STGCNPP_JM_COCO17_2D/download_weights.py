from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://download.openmmlab.com/mmaction/pyskl/ckpt/"
    "stgcnpp/stgcnpp_ntu60_xsub_hrnet/jm.pth"
)

HERE = Path(__file__).resolve().parent
DEST = HERE / "jm.pth"

if DEST.exists() and DEST.stat().st_size > 1_000_000:
    print(f"Checkpoint already exists: {DEST} ({DEST.stat().st_size:,} bytes)")
else:
    print(f"Downloading PYSKL ST-GCN++ 2D COCO-17 Joint Motion to {DEST}")
    urlretrieve(URL, DEST)
    size = DEST.stat().st_size
    if size < 1_000_000:
        DEST.unlink(missing_ok=True)
        raise RuntimeError(f"Download looks incomplete ({size} bytes).")
    print(f"Downloaded: {DEST} ({size:,} bytes)")
