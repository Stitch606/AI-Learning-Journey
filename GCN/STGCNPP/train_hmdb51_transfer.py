from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data" / "hmdb51_hrnet.pkl"
BASE_CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
OUTPUT_DIR = ROOT / "hmdb51_transfer_outputs"
BEST_CHECKPOINT = OUTPUT_DIR / "stgcnpp_hmdb51_pose_transfer_best.pth"
REPORT_PATH = OUTPUT_DIR / "hmdb51_pose_transfer_report.json"

HMDB_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
    "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat",
    "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit",
    "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
    "pour", "pullup", "punch", "push", "pushup", "ride_bike",
    "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow",
    "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
    "stand", "swing_baseball", "sword", "sword_exercise", "talk",
    "throw", "turn", "walk", "wave",
]

TARGET_CLASSES = [
    "fall_floor", "hit", "kick", "punch", "push", "throw",
    "run", "walk", "smoke", "shoot_gun", "sword", "fencing",
    "sit", "stand", "shake_hands",
]

CLIP_LEN = 100
BATCH_SIZE = 8
EPOCHS = 6
MAX_TRAIN_PER_CLASS = 60
MAX_TEST_PER_CLASS = 30
SEED = 255


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def find_split(split: dict, prefix: str) -> str:
    keys = list(split)
    preferred = f"{prefix}1"
    if preferred in split:
        return preferred
    candidates = sorted(k for k in keys if k.lower().startswith(prefix))
    if candidates:
        return candidates[0]
    raise KeyError(f"No {prefix} split found. Available splits: {keys}")


def normalize_label_ids(annotations: list[dict]) -> None:
    labels = sorted({int(a["label"]) for a in annotations})
    if labels and labels[0] == 1 and labels[-1] == 51 and 0 not in labels:
        for item in annotations:
            item["label"] = int(item["label"]) - 1
    bad = sorted({int(a["label"]) for a in annotations} - set(range(51)))
    if bad:
        raise ValueError(f"HMDB51 labels outside 0..50: {bad}")


def choose_primary_person(annotation: dict) -> tuple[np.ndarray, np.ndarray]:
    keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
    if keypoint.ndim == 3:
        keypoint = keypoint[None, ...]
    if keypoint.ndim != 4 or keypoint.shape[-2:] != (17, 2):
        raise ValueError(f"Expected [M,T,17,2], got {keypoint.shape}")

    score_value = annotation.get("keypoint_score")
    if score_value is None:
        score = np.ones(keypoint.shape[:-1], dtype=np.float32)
    else:
        score = np.asarray(score_value, dtype=np.float32)
        if score.ndim == 2:
            score = score[None, ...]
    if score.shape != keypoint.shape[:-1]:
        raise ValueError(
            f"keypoint_score shape {score.shape} does not match {keypoint.shape[:-1]}"
        )

    if keypoint.shape[0] == 0:
        raise ValueError("No detected person in skeleton sample")

    visible = score > 0.01
    coverage = visible.sum(axis=(1, 2))
    mean_conf = np.divide(
        (score * visible).sum(axis=(1, 2)),
        np.maximum(coverage, 1),
    )
    quality = coverage * mean_conf
    person_index = int(np.argmax(quality))
    return keypoint[person_index], np.clip(score[person_index], 0.0, 1.0)


def make_joint_motion(annotation: dict, training: bool) -> np.ndarray:
    keypoint, score = choose_primary_person(annotation)
    frames = keypoint.shape[0]
    if frames < 1:
        raise ValueError("Empty temporal sequence")

    h, w = annotation.get("img_shape", (256, 340))
    h, w = float(h), float(w)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid img_shape: {(h, w)}")

    xy = keypoint.copy()
    xy[..., 0] = (xy[..., 0] - w / 2.0) / (w / 2.0)
    xy[..., 1] = (xy[..., 1] - h / 2.0) / (h / 2.0)
    xy[score <= 0.01] = 0.0

    motion = np.zeros((frames, 17, 3), dtype=np.float32)
    if frames > 1:
        motion[:-1, :, :2] = np.diff(xy, axis=0)
        motion[:-1, :, 2] = (score[:-1] + score[1:]) / 2.0

    if frames >= CLIP_LEN:
        edges = np.linspace(0, frames, CLIP_LEN + 1).astype(np.int64)
        indices = []
        for i in range(CLIP_LEN):
            low = int(edges[i])
            high = max(low + 1, int(edges[i + 1]))
            if training:
                indices.append(np.random.randint(low, min(high, frames)))
            else:
                indices.append(min((low + high - 1) // 2, frames - 1))
        indices = np.asarray(indices, dtype=np.int64)
    else:
        indices = np.arange(CLIP_LEN, dtype=np.int64) % frames

    return motion[indices]


class HMDBPersonDataset(Dataset):
    def __init__(self, annotations: list[dict], class_to_index: dict[str, int],
                 training: bool):
        self.annotations = annotations
        self.class_to_index = class_to_index
        self.training = training

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        item = self.annotations[index]
        motion = make_joint_motion(item, self.training)
        class_name = HMDB_CLASSES[int(item["label"])]
        label = self.class_to_index[class_name]
        return torch.from_numpy(motion), torch.tensor(label, dtype=torch.long)


def collate(items):
    motion = torch.stack([x[0] for x in items], dim=0)
    labels = torch.stack([x[1] for x in items], dim=0)
    # [B, clips=1, persons=1, T, V, C]
    return motion[:, None, None], labels


def cap_per_class(items: list[dict], target_names: list[str],
                  limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for name in target_names:
        class_id = HMDB_CLASSES.index(name)
        rows = [x for x in items if int(x["label"]) == class_id]
        rng.shuffle(rows)
        result.extend(rows[:limit])
    rng.shuffle(result)
    return result


@torch.inference_mode()
def evaluate(model, loader, device, class_names):
    model.eval()
    total = correct = 0
    class_total = Counter()
    class_correct = Counter()
    loss_sum = 0.0

    for motion, labels in loader:
        motion = motion.to(device)
        labels = labels.to(device)
        logits = model._person_logits(motion)[:, 0, 0, :]
        loss_sum += float(F.cross_entropy(logits, labels).item())
        predictions = logits.argmax(dim=1)
        total += int(labels.numel())
        correct += int((predictions == labels).sum().item())
        for target, prediction in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            class_total[int(target)] += 1
            class_correct[int(target)] += int(target == prediction)

    per_class = {
        name: {
            "samples": class_total[i],
            "accuracy": class_correct[i] / max(class_total[i], 1),
        }
        for i, name in enumerate(class_names)
    }
    macro = float(np.mean([
        value["accuracy"] for value in per_class.values()
        if value["samples"] > 0
    ]))
    return {
        "loss": loss_sum / max(len(loader), 1),
        "top1_accuracy": correct / max(total, 1),
        "macro_class_accuracy": macro,
        "samples": total,
        "per_class": per_class,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)

    seed_everything(SEED)
    print(f"Dataset file: {DATA_PATH} ({DATA_PATH.stat().st_size:,} bytes)", flush=True)

    with DATA_PATH.open("rb") as handle:
        dataset = pickle.load(handle, encoding="latin1")

    annotations = dataset["annotations"]
    normalize_label_ids(annotations)
    split = dataset["split"]
    train_key = find_split(split, "train")
    test_key = find_split(split, "test")
    train_ids = set(split[train_key])
    test_ids = set(split[test_key])
    if train_ids & test_ids:
        raise RuntimeError("Official HMDB51 train/test splits overlap")

    target_ids = {HMDB_CLASSES.index(name) for name in TARGET_CLASSES}
    train_rows = [
        a for a in annotations
        if a["frame_dir"] in train_ids and int(a["label"]) in target_ids
    ]
    test_rows = [
        a for a in annotations
        if a["frame_dir"] in test_ids and int(a["label"]) in target_ids
    ]

    train_rows = cap_per_class(train_rows, TARGET_CLASSES, MAX_TRAIN_PER_CLASS, SEED)
    test_rows = cap_per_class(test_rows, TARGET_CLASSES, MAX_TEST_PER_CLASS, SEED + 1)

    train_counts = Counter(HMDB_CLASSES[int(a["label"])] for a in train_rows)
    test_counts = Counter(HMDB_CLASSES[int(a["label"])] for a in test_rows)
    missing = [
        name for name in TARGET_CLASSES
        if train_counts[name] == 0 or test_counts[name] == 0
    ]
    if missing:
        raise RuntimeError(
            f"Target classes missing from official split {train_key}/{test_key}: {missing}; "
            f"train={dict(train_counts)}, test={dict(test_counts)}"
        )

    class_to_index = {name: i for i, name in enumerate(TARGET_CLASSES)}
    train_dataset = HMDBPersonDataset(train_rows, class_to_index, training=True)
    test_dataset = HMDBPersonDataset(test_rows, class_to_index, training=False)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device}; official splits={train_key}/{test_key}; "
        f"train={len(train_rows)}; test={len(test_rows)}; "
        f"classes={len(TARGET_CLASSES)}",
        flush=True,
    )
    print("TRAIN_CLASS_COUNTS=" + json.dumps(dict(train_counts)), flush=True)
    print("TEST_CLASS_COUNTS=" + json.dumps(dict(test_counts)), flush=True)

    from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson

    model = STGCNPPMultiPerson(
        checkpoint_path=None,
        num_classes=len(TARGET_CLASSES),
        device=str(device),
        strict=True,
        max_persons=1,
        class_names=TARGET_CLASSES,
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
        raise RuntimeError("No backbone weights found in the original ST-GCN++ checkpoint")
    model.backbone.load_state_dict(backbone_state, strict=True)
    print(
        f"Loaded {len(backbone_state)} pretrained backbone tensors; "
        "initialized a new 15-class head.",
        flush=True,
    )

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

    best_accuracy = -1.0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for step, (motion, labels) in enumerate(train_loader, start=1):
            motion = motion.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model._person_logits(motion)[:, 0, 0, :]
            loss = F.cross_entropy(logits, labels)
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
        validation = evaluate(model, test_loader, device, TARGET_CLASSES)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "test": validation,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(record)
        print("EPOCH_RESULT=" + json.dumps(record), flush=True)

        if validation["top1_accuracy"] > best_accuracy:
            best_accuracy = validation["top1_accuracy"]
            torch.save(
                {
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "num_classes": len(TARGET_CLASSES),
                    "class_names": TARGET_CLASSES,
                    "dataset": "PYSKL HMDB51 HRNet COCO-17 precomputed skeletons",
                    "base_checkpoint": BASE_CHECKPOINT.name,
                    "epoch": epoch,
                    "test_metrics": validation,
                },
                BEST_CHECKPOINT,
            )
            print(f"BEST_CHECKPOINT_SAVED={BEST_CHECKPOINT}", flush=True)

    report = {
        "dataset": "PYSKL HMDB51 HRNet 2D precomputed skeleton annotations",
        "dataset_url": (
            "https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl"
        ),
        "official_train_split": train_key,
        "official_test_split": test_key,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "target_classes": TARGET_CLASSES,
        "train_class_counts": dict(train_counts),
        "test_class_counts": dict(test_counts),
        "best_test_top1_accuracy": best_accuracy,
        "best_checkpoint": str(BEST_CHECKPOINT),
        "training_history": history,
        "supervision_note": (
            "HMDB51 provides clip-level labels, not independent ground-truth labels "
            "for every detected person. For each clip, the primary skeleton track was "
            "selected by visible-keypoint coverage and confidence, then trained with "
            "the clip label. This is actor-focused pose transfer, not dense multi-person "
            "individual-action ground truth."
        ),
        "input_note": (
            "Precomputed HRNet COCO-17 2D keypoints and scores; Joint Motion was "
            "calculated from the supplied skeleton coordinates. No videos or pose "
            "estimator were used."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report), flush=True)
    print(f"REPORT_PATH={REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
