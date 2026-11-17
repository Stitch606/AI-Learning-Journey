from __future__ import annotations

import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "CHAD_Meta"
BASE_CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
OUTPUT_DIR = ROOT / "chad_pose_outputs"
BEST_CHECKPOINT = OUTPUT_DIR / "stgcnpp_chad_weak_person_anomaly_best.pth"
REPORT_PATH = OUTPUT_DIR / "chad_weak_person_anomaly_report.json"

CLIP_LEN = 100
WINDOW_STRIDE = 50
MAX_PERSONS = 10
MAX_PER_VIDEO_PER_CLASS = 8
MAX_TRAIN_PER_CLASS = 800
MAX_TEST_PER_CLASS = 300
BATCH_SIZE = 4
EPOCHS = 4
SEED = 255


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def locate_dir(root: Path, name: str) -> Path:
    candidates = [p for p in root.rglob(name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find directory {name!r} under {root}")
    return candidates[0]


def video_stem(line: str) -> str:
    return Path(line.strip()).stem


def load_split_ids(split_dir: Path, prefix: str) -> set[str]:
    preferred = split_dir / f"{prefix}_split_2.txt"
    if not preferred.exists():
        preferred = split_dir / f"{prefix}_split_1.txt"
    if not preferred.exists():
        return set()
    return {
        video_stem(line)
        for line in preferred.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def numeric_frame_key(key) -> int:
    match = re.search(r"\d+", str(key))
    if not match:
        raise ValueError(f"Cannot parse frame number from {key!r}")
    return int(match.group(0))


def parse_person_pose(payload) -> np.ndarray | None:
    if isinstance(payload, dict):
        bbox = payload.get("bbox")
        keypoints = payload.get("keypoints", payload.get("pose"))
        if keypoints is None and isinstance(payload.get("data"), (list, tuple)):
            values = payload["data"]
            keypoints = values[1] if len(values) > 1 else None
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        keypoints = payload[1]
    else:
        return None

    if keypoints is None:
        return None
    points = np.asarray(keypoints, dtype=np.float32).squeeze()
    if points.shape == (17, 2):
        points = np.concatenate(
            [points, np.ones((17, 1), dtype=np.float32)], axis=1
        )
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return None
    points[:, 2] = np.clip(points[:, 2], 0.0, 1.0)
    return points


def frame_to_label_index(frame_number: int, minimum_frame: int,
                         maximum_frame: int, label_count: int,
                         sequential_index: int) -> int:
    if minimum_frame == 0 and 0 <= frame_number < label_count:
        return frame_number
    if minimum_frame >= 1 and 0 <= frame_number - minimum_frame < label_count:
        return frame_number - minimum_frame
    if 0 <= frame_number < label_count:
        return frame_number
    if 0 <= frame_number - 1 < label_count:
        return frame_number - 1
    return sequential_index


def joint_motion(raw: np.ndarray, width: float, height: float) -> np.ndarray:
    xy = raw[..., :2].copy()
    score = np.clip(raw[..., 2], 0.0, 1.0)
    xy[..., 0] = 2.0 * xy[..., 0] / width - 1.0
    xy[..., 1] = 2.0 * xy[..., 1] / height - 1.0
    xy[score <= 0.01] = 0.0

    motion = np.zeros((CLIP_LEN, 17, 3), dtype=np.float32)
    motion[:-1, :, :2] = np.diff(xy, axis=0)
    motion[:-1, :, 2] = (score[:-1] + score[1:]) / 2.0
    return motion


def build_video_windows(annotation_path: Path, label_path: Path,
                        per_video_cap: int, seed: int) -> dict[int, list[tuple[np.ndarray, int]]]:
    with annotation_path.open("rb") as handle:
        frames = pickle.load(handle, encoding="latin1")
    frame_labels = np.asarray(np.load(label_path), dtype=np.int64).reshape(-1)
    if frame_labels.size < CLIP_LEN:
        return {0: [], 1: []}

    frame_items = sorted(
        ((numeric_frame_key(key), key, value) for key, value in frames.items()),
        key=lambda item: item[0],
    )
    if not frame_items:
        return {0: [], 1: []}

    frame_numbers = [item[0] for item in frame_items]
    minimum_frame = min(frame_numbers)
    maximum_frame = max(frame_numbers)
    tracks = defaultdict(dict)

    for sequence_index, (frame_number, original_key, people) in enumerate(frame_items):
        label_index = frame_to_label_index(
            frame_number, minimum_frame, maximum_frame,
            len(frame_labels), sequence_index
        )
        if not 0 <= label_index < len(frame_labels):
            continue
        if not isinstance(people, dict):
            continue
        for person_id, payload in people.items():
            pose = parse_person_pose(payload)
            if pose is not None:
                tracks[str(person_id)][label_index] = pose

    camera = annotation_path.stem.split("_")[0]
    width, height = (1280.0, 720.0) if camera == "4" else (1920.0, 1080.0)

    starts = list(range(0, len(frame_labels) - CLIP_LEN + 1, WINDOW_STRIDE))
    rng = random.Random(seed + sum(ord(ch) for ch in annotation_path.stem))
    rng.shuffle(starts)
    selected = {0: [], 1: []}

    for start in starts:
        end = start + CLIP_LEN
        label = int(np.any(frame_labels[start:end] > 0))
        if len(selected[label]) >= per_video_cap:
            if len(selected[0]) >= per_video_cap and len(selected[1]) >= per_video_cap:
                break
            continue

        person_sequences = []
        for track in tracks.values():
            present = [
                frame_index for frame_index in range(start, end)
                if frame_index in track
            ]
            if len(present) < 8:
                continue
            raw = np.zeros((CLIP_LEN, 17, 3), dtype=np.float32)
            for frame_index in present:
                raw[frame_index - start] = track[frame_index]
            visible_frames = int(
                np.count_nonzero((raw[..., 2] > 0.05).sum(axis=1) >= 5)
            )
            if visible_frames < 5:
                continue
            quality = float(raw[..., 2].mean())
            person_sequences.append((visible_frames, quality, joint_motion(raw, width, height)))

        if not person_sequences:
            continue

        person_sequences.sort(key=lambda item: (item[0], item[1]), reverse=True)
        people_array = np.stack(
            [item[2] for item in person_sequences[:MAX_PERSONS]], axis=0
        )
        selected[label].append((people_array, label))

    return selected


def cap_samples(samples: list[tuple[np.ndarray, int]], limit: int,
                seed: int) -> list[tuple[np.ndarray, int]]:
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:limit]


class CHADWindowDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        people, label = self.samples[index]
        return torch.from_numpy(people), torch.tensor(label, dtype=torch.long)


def collate(items):
    max_people = min(MAX_PERSONS, max(x[0].shape[0] for x in items))
    batch = np.zeros(
        (len(items), 1, max_people, CLIP_LEN, 17, 3), dtype=np.float32
    )
    mask = np.zeros((len(items), max_people), dtype=bool)
    labels = np.zeros((len(items),), dtype=np.int64)
    for i, (people, label) in enumerate(items):
        count = min(max_people, people.shape[0])
        batch[i, 0, :count] = people[:count]
        mask[i, :count] = True
        labels[i] = int(label.item())
    return (
        torch.from_numpy(batch),
        torch.from_numpy(mask),
        torch.from_numpy(labels),
    )


def build_split_samples(annotation_dir: Path, label_dir: Path,
                        video_ids: set[str], per_video_cap: int,
                        global_cap: int, seed: int):
    files = {
        p.stem: p for p in annotation_dir.glob("*.pkl")
        if "__MACOSX" not in str(p)
    }
    label_files = {p.stem: p for p in label_dir.glob("*.npy")}
    ids = sorted(video_ids & files.keys() & label_files.keys())
    if not ids:
        raise RuntimeError(
            f"No matching annotation/label files. Split IDs={len(video_ids)}, "
            f"annotations={len(files)}, labels={len(label_files)}"
        )

    by_class = {0: [], 1: []}
    for index, stem in enumerate(ids):
        windows = build_video_windows(
            files[stem], label_files[stem], per_video_cap, seed + index
        )
        by_class[0].extend(windows[0])
        by_class[1].extend(windows[1])

    by_class[0] = cap_samples(by_class[0], global_cap, seed)
    by_class[1] = cap_samples(by_class[1], global_cap, seed + 1)
    result = by_class[0] + by_class[1]
    random.Random(seed + 2).shuffle(result)
    return result, {
        "videos_used": len(ids),
        "normal_windows": len(by_class[0]),
        "anomaly_windows": len(by_class[1]),
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    tp = fp = fn = tn = 0
    per_person_scores = {0: [], 1: []}

    for inputs, mask, labels in loader:
        inputs = inputs.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        person_logits = model._person_logits(inputs)[:, :, 0, :]
        person_scores = person_logits[:, :, 1] - person_logits[:, :, 0]
        person_scores = person_scores.masked_fill(~mask, -1e4)
        scene_score = person_scores.max(dim=1).values
        scene_logits = torch.stack((torch.zeros_like(scene_score), scene_score), dim=1)
        predictions = scene_logits.argmax(dim=1)

        total += int(labels.numel())
        correct += int((predictions == labels).sum().item())
        tp += int(((predictions == 1) & (labels == 1)).sum().item())
        fp += int(((predictions == 1) & (labels == 0)).sum().item())
        fn += int(((predictions == 0) & (labels == 1)).sum().item())
        tn += int(((predictions == 0) & (labels == 0)).sum().item())

        for i, target in enumerate(labels.cpu().tolist()):
            valid_scores = person_scores[i][mask[i]].detach().cpu().tolist()
            if valid_scores:
                per_person_scores[int(target)].append(float(max(valid_scores)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    normal_accuracy = tn / max(tn + fp, 1)
    anomaly_accuracy = tp / max(tp + fn, 1)
    return {
        "window_accuracy": correct / max(total, 1),
        "balanced_accuracy": (normal_accuracy + anomaly_accuracy) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "windows": total,
        "max_person_score_mean_normal_windows": (
            float(np.mean(per_person_scores[0])) if per_person_scores[0] else None
        ),
        "max_person_score_mean_anomaly_windows": (
            float(np.mean(per_person_scores[1])) if per_person_scores[1] else None
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    annotation_dir = locate_dir(DATA_ROOT, "annotations")
    label_dir = locate_dir(DATA_ROOT, "anomaly_labels")
    split_dir = locate_dir(DATA_ROOT, "splits")

    annotation_files = [
        p for p in annotation_dir.glob("*.pkl")
        if "__MACOSX" not in str(p)
    ]
    label_files = list(label_dir.glob("*.npy"))
    print(
        f"CHAD annotation files={len(annotation_files)}; "
        f"anomaly label files={len(label_files)}",
        flush=True,
    )

    train_ids = load_split_ids(split_dir, "train")
    test_ids = load_split_ids(split_dir, "test")
    if not train_ids or not test_ids:
        all_ids = sorted(p.stem for p in annotation_files)
        random.Random(SEED).shuffle(all_ids)
        cut = int(0.6 * len(all_ids))
        train_ids, test_ids = set(all_ids[:cut]), set(all_ids[cut:])
    if train_ids & test_ids:
        raise RuntimeError("CHAD video-level train/test split overlaps")

    train_samples, train_stats = build_split_samples(
        annotation_dir, label_dir, train_ids,
        MAX_PER_VIDEO_PER_CLASS, MAX_TRAIN_PER_CLASS, SEED,
    )
    test_samples, test_stats = build_split_samples(
        annotation_dir, label_dir, test_ids,
        max(2, MAX_PER_VIDEO_PER_CLASS // 2), MAX_TEST_PER_CLASS, SEED + 10,
    )
    if len(train_samples) < 100 or len(test_samples) < 40:
        raise RuntimeError(
            f"Insufficient CHAD windows: train={len(train_samples)}, test={len(test_samples)}"
        )

    train_counts = Counter(label for _, label in train_samples)
    test_counts = Counter(label for _, label in test_samples)
    print(
        f"Train windows={len(train_samples)} {dict(train_counts)}; "
        f"test windows={len(test_samples)} {dict(test_counts)}",
        flush=True,
    )
    print(
        f"Train videos={train_stats['videos_used']}; "
        f"test videos={test_stats['videos_used']}",
        flush=True,
    )

    train_loader = DataLoader(
        CHADWindowDataset(train_samples), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, collate_fn=collate,
    )
    test_loader = DataLoader(
        CHADWindowDataset(test_samples), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson

    model = STGCNPPMultiPerson(
        checkpoint_path=None,
        num_classes=2,
        device=str(device),
        strict=True,
        max_persons=MAX_PERSONS,
        class_names=["normal", "anomalous"],
    )

    base = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    state = base.get("state_dict", base) if isinstance(base, dict) else base
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("No backbone weights found in original ST-GCN++ checkpoint")
    model.backbone.load_state_dict(backbone_state, strict=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.cls_head.parameters(), "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_f1 = -1.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for step, (inputs, mask, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            person_logits = model._person_logits(inputs)[:, :, 0, :]
            person_scores = person_logits[:, :, 1] - person_logits[:, :, 0]
            person_scores = person_scores.masked_fill(~mask, -1e4)
            scene_score = person_scores.max(dim=1).values
            scene_logits = torch.stack((torch.zeros_like(scene_score), scene_score), dim=1)
            loss = F.cross_entropy(scene_logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            steps += 1

            if step % 50 == 0:
                print(
                    f"epoch={epoch}/{EPOCHS} step={step}/{len(train_loader)} "
                    f"loss={total_loss / steps:.5f}",
                    flush=True,
                )

        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "test": metrics,
        }
        history.append(record)
        print("EPOCH_RESULT=" + json.dumps(record), flush=True)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "num_classes": 2,
                    "class_names": ["normal", "anomalous"],
                    "dataset": "CHAD precomputed COCO-17 person tracks",
                    "supervision": "weak video-frame labels with person-level MIL",
                    "epoch": epoch,
                    "test_metrics": metrics,
                },
                BEST_CHECKPOINT,
            )
            print(f"BEST_CHECKPOINT_SAVED={BEST_CHECKPOINT}", flush=True)

    report = {
        "dataset": "CHAD metadata-only precomputed person skeleton annotations",
        "annotation_files": len(annotation_files),
        "label_files": len(label_files),
        "train": train_stats,
        "test": test_stats,
        "train_class_counts": dict(train_counts),
        "test_class_counts": dict(test_counts),
        "best_test_f1": best_f1,
        "training_history": history,
        "checkpoint": str(BEST_CHECKPOINT),
        "supervision_note": (
            "CHAD provides COCO-17 keypoints and persistent Person_ID tracks per frame, "
            "but its released anomaly labels are frame-level binary labels rather than "
            "individual action labels. Training therefore uses multiple-instance learning: "
            "the maximum per-person anomaly score predicts the frame-window label. "
            "Per-person outputs are weakly supervised anomaly scores, not verified "
            "person-specific action classes."
        ),
        "input_note": (
            "Only the 246 MB CHAD_Meta archive was used: precomputed skeletons, "
            "frame anomaly labels and split lists. No RGB videos or pose extraction."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report), flush=True)
    print(f"REPORT_PATH={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
