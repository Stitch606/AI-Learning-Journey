from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# Running this file by path puts GCN/STGCNPP, not the repository root, on
# sys.path. Add the root explicitly so the repository package imports work
# the same way in GitHub Actions and from a local checkout.
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson


DATA_PATH = ROOT / "data" / "nturgbd" / "ntu60_hrnet.pkl"
CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
REPORT = ROOT / "ntu60_per_person_report.json"
CLIP_LEN = 100
NUM_PERSONS = 2
BATCH_SIZE = 8


def test_sample_indices(num_frames: int, motion: np.ndarray) -> np.ndarray:
    """Mirror PYSKL UniformSampleFrames(clip_len=100, num_clips=1, seed=255)."""
    rng = np.random.RandomState(255)
    rng.rand()
    rng.randint(1)  # p_interval=1, matching PYSKL's offset draw

    if num_frames < CLIP_LEN:
        indices = np.arange(CLIP_LEN)
    elif num_frames < CLIP_LEN * 2:
        basic = np.arange(CLIP_LEN)
        chosen = rng.choice(CLIP_LEN + 1, num_frames - CLIP_LEN, replace=False)
        offsets = np.zeros(CLIP_LEN + 1, dtype=np.int64)
        offsets[chosen] = 1
        indices = basic + np.cumsum(offsets)[:-1]
    else:
        boundaries = np.array(
            [i * num_frames // CLIP_LEN for i in range(CLIP_LEN + 1)]
        )
        sizes = np.diff(boundaries)
        indices = boundaries[:CLIP_LEN] + rng.randint(sizes)

    indices = indices % num_frames

    # Match PYSKL UniformSampleFrames' person-count transition correction.
    person_counts = np.zeros(num_frames, dtype=np.int64)
    for frame in range(num_frames):
        count = motion.shape[0]
        while count > 0 and np.all(np.abs(motion[count - 1, frame]) < 1e-5):
            count -= 1
        person_counts[frame] = count

    transitional = np.zeros(num_frames, dtype=bool)
    for frame in range(1, num_frames - 1):
        if person_counts[frame] != person_counts[frame - 1]:
            transitional[frame] = transitional[frame - 1] = True
        if person_counts[frame] != person_counts[frame + 1]:
            transitional[frame] = transitional[frame + 1] = True

    indices_int = indices.astype(np.int64)
    coeff = transitional[indices_int]
    indices = (coeff * indices_int + (1 - coeff) * indices).astype(np.int64)
    return indices


def preprocess_annotation(annotation: dict):
    """PYSKL PreNormalize2D -> GenSkeFeat(jm) -> UniformSample -> FormatGCNInput."""
    keypoint = np.asarray(annotation["keypoint"], dtype=np.float32).copy()
    score = np.asarray(annotation["keypoint_score"], dtype=np.float32).copy()
    if keypoint.ndim != 4 or keypoint.shape[-2:] != (17, 2):
        raise ValueError(f"Unexpected keypoint shape: {keypoint.shape}")
    if score.shape != keypoint.shape[:-1]:
        raise ValueError(f"Unexpected keypoint_score shape: {score.shape}")

    persons, frames = keypoint.shape[:2]
    if persons < 1 or frames < 1:
        raise ValueError("Empty skeleton annotation")

    h, w = annotation.get("img_shape", (1080, 1920))
    visible = score > 0.01

    # Exact PYSKL PreNormalize2D(mode='fix', threshold=0.01).
    keypoint[..., 0] = (keypoint[..., 0] - (w / 2.0)) / (w / 2.0)
    keypoint[..., 1] = (keypoint[..., 1] - (h / 2.0)) / (h / 2.0)
    keypoint[~visible] = 0.0

    # Exact PYSKL ToMotion(dataset='coco'): coordinate differences and
    # average confidence for adjacent frames.
    motion = np.zeros((persons, frames, 17, 3), dtype=np.float32)
    joined = np.concatenate([keypoint, score[..., None]], axis=-1)
    if frames > 1:
        motion[:, :-1] = np.diff(joined, axis=1)
        motion[:, :-1, :, 2] = (
            score[:, :-1] + score[:, 1:]
        ) / 2.0

    indices = test_sample_indices(frames, motion)
    sampled = motion[:, indices]
    sampled_scores = score[:, indices]

    # Exact FormatGCNInput(num_person=2, mode='zero'): truncate/pad on M.
    actual_persons = min(persons, NUM_PERSONS)
    valid = np.any(sampled_scores[:actual_persons] > 0.01, axis=(1, 2))
    packed = np.zeros((NUM_PERSONS, CLIP_LEN, 17, 3), dtype=np.float32)
    packed[:actual_persons] = sampled[:actual_persons]
    return packed, valid


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing PYSKL annotation file: {DATA_PATH}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing official checkpoint: {CHECKPOINT}")

    torch.set_num_threads(2)
    torch.manual_seed(255)
    with DATA_PATH.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    split = dataset["split"]["xsub_val"]
    split_ids = set(split)
    annotations = [
        item for item in dataset["annotations"]
        if item["frame_dir"] in split_ids
    ]
    if not annotations:
        raise RuntimeError("No annotations matched the official xsub_val split")

    # Keep a stable, reproducible ordering and ensure we evaluated the complete
    # official validation split rather than a random subset.
    annotations.sort(key=lambda item: item["frame_dir"])
    if len(annotations) != len(split_ids):
        raise RuntimeError(
            f"xsub_val mismatch: split has {len(split_ids)} IDs, "
            f"matched {len(annotations)} annotations"
        )

    model = STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        num_classes=60,
        device="cpu",
        strict=True,
        max_persons=NUM_PERSONS,
    )
    model.eval()

    total_people = 0
    correct_people = 0
    by_person_count = Counter()
    by_person_correct = Counter()
    samples_used = 0

    for start in range(0, len(annotations), BATCH_SIZE):
        chunk = annotations[start:start + BATCH_SIZE]
        batch = np.zeros(
            (len(chunk), 1, NUM_PERSONS, CLIP_LEN, 17, 3),
            dtype=np.float32,
        )
        labels = []
        masks = []

        for row, annotation in enumerate(chunk):
            prepared, valid = preprocess_annotation(annotation)
            batch[row, 0] = prepared
            labels.append(int(annotation["label"]))
            masks.append(valid)

        tensor = torch.from_numpy(batch)
        with torch.inference_mode():
            # This is the same per-person logits path used by predict().
            logits = model._person_logits(tensor)[:, :, 0, :]
            predictions = logits.argmax(dim=-1).cpu().numpy()

        for row, valid in enumerate(masks):
            label = labels[row]
            count = int(valid.sum())
            if count == 0:
                continue
            samples_used += 1
            by_person_count[count] += count
            for person_index in range(NUM_PERSONS):
                if person_index >= len(valid) or not valid[person_index]:
                    continue
                total_people += 1
                is_correct = int(predictions[row, person_index] == label)
                correct_people += is_correct
                by_person_correct[count] += is_correct

        if (start // BATCH_SIZE + 1) % 100 == 0:
            print(
                f"Processed {min(start + BATCH_SIZE, len(annotations))}/"
                f"{len(annotations)} xsub_val clips",
                flush=True,
            )

    if total_people == 0:
        raise RuntimeError("No valid person tracks were evaluated")

    report = {
        "dataset": "PYSKL NTU60 HRNet 2D annotations",
        "split": "xsub_val",
        "split_protocol": "cross-subject",
        "checkpoint": CHECKPOINT.name,
        "architecture": "ST-GCN++ per-person head; no person-axis averaging",
        "samples_in_split": len(split_ids),
        "samples_evaluated": samples_used,
        "person_predictions": total_people,
        "per_person_top1_accuracy": correct_people / total_people,
        "correct_person_predictions": correct_people,
        "person_count_breakdown": {
            str(count): {
                "person_predictions": by_person_count[count],
                "correct": by_person_correct[count],
                "top1_accuracy": (
                    by_person_correct[count] / by_person_count[count]
                    if by_person_count[count] else None
                ),
            }
            for count in sorted(by_person_count)
        },
        "metric_note": (
            "Each real person's prediction is compared independently with "
            "the clip action label. This is a per-person diagnostic metric, "
            "not the official PYSKL scene-level benchmark, because the "
            "modified head no longer averages person features."
        ),
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT_PATH={REPORT}")


if __name__ == "__main__":
    main()
