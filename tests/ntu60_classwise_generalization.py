from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson
from GCN.STGCNPP.evaluate_ntu60_per_person import (
    DATA_PATH,
    NUM_PERSONS,
    BATCH_SIZE,
    preprocess_annotation,
)

ORIGINAL = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
FINETUNED = ROOT / "huggingface-space" / "process" / "weights" / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
REPORT = ROOT / "ntu60_classwise_generalization_report.json"


def evaluate_checkpoint(name: str, checkpoint: Path, annotations: list[dict], class_names: list[str]):
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
        raise FileNotFoundError(f"{name} checkpoint missing/incomplete: {checkpoint}")

    model = STGCNPPMultiPerson(
        checkpoint_path=str(checkpoint),
        num_classes=60,
        device="cpu",
        strict=True,
        max_persons=NUM_PERSONS,
    )
    model.eval()

    class_total = np.zeros(60, dtype=np.int64)
    class_correct = np.zeros(60, dtype=np.int64)
    confusion = np.zeros((60, 60), dtype=np.int64)
    top2_for_22 = defaultdict(int)
    total_people = correct_people = 0

    for start in range(0, len(annotations), BATCH_SIZE):
        chunk = annotations[start:start + BATCH_SIZE]
        batch = np.zeros((len(chunk), 1, NUM_PERSONS, 100, 17, 3), dtype=np.float32)
        labels = []
        masks = []
        for row, annotation in enumerate(chunk):
            packed, valid = preprocess_annotation(annotation)
            batch[row, 0] = packed
            labels.append(int(annotation["label"]))
            masks.append(valid)

        with torch.inference_mode():
            logits = model._person_logits(torch.from_numpy(batch))[:, :, 0, :]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        predictions = probs.argmax(axis=-1)
        for row, valid in enumerate(masks):
            label = labels[row]
            for person_index in range(NUM_PERSONS):
                if not valid[person_index]:
                    continue
                pred = int(predictions[row, person_index])
                total_people += 1
                correct_people += int(pred == label)
                class_total[label] += 1
                class_correct[label] += int(pred == label)
                confusion[label, pred] += 1
                if label == 22:
                    top2 = np.argsort(probs[row, person_index])[-2:][::-1]
                    top2_for_22[(int(top2[0]), int(top2[1]))] += 1

        if (start // BATCH_SIZE + 1) % 100 == 0:
            print(f"{name}: {min(start + BATCH_SIZE, len(annotations))}/{len(annotations)} clips", flush=True)

    per_class = []
    for class_id in range(60):
        count = int(class_total[class_id])
        if count == 0:
            continue
        row = confusion[class_id]
        strongest = np.argsort(row)[::-1]
        mistakes = [
            {"predicted_class_id": int(pred_id), "predicted_action": class_names[int(pred_id)], "count": int(row[pred_id])}
            for pred_id in strongest if pred_id != class_id and row[pred_id] > 0
        ][:5]
        per_class.append({
            "class_id": class_id,
            "action": class_names[class_id],
            "samples": count,
            "correct": int(class_correct[class_id]),
            "top1_accuracy": float(class_correct[class_id] / count),
            "top_confusions": mistakes,
        })

    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "person_predictions": total_people,
        "overall_per_person_top1_accuracy": correct_people / max(total_people, 1),
        "class_21_cheer_up": next(x for x in per_class if x["class_id"] == 21),
        "class_22_hand_waving": next(x for x in per_class if x["class_id"] == 22),
        "per_class": per_class,
        "hand_waving_top2_pairs": [
            {"top1_class_id": pair[0], "top1_action": class_names[pair[0]], "top2_class_id": pair[1], "top2_action": class_names[pair[1]], "count": count}
            for pair, count in sorted(top2_for_22.items(), key=lambda item: item[1], reverse=True)[:10]
        ],
    }


def main():
    torch.set_num_threads(2)
    torch.manual_seed(255)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing official PYSKL annotations: {DATA_PATH}")

    import pickle
    with DATA_PATH.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    val_ids = set(dataset["split"]["xsub_val"])
    annotations = [item for item in dataset["annotations"] if item["frame_dir"] in val_ids]
    annotations.sort(key=lambda item: item["frame_dir"])
    if len(annotations) != len(val_ids):
        raise RuntimeError(f"xsub_val split mismatch: expected {len(val_ids)}, got {len(annotations)}")

    labels_path = ROOT / "GCN" / "STGCNPP" / "ntu60_labels.txt"
    class_names = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(class_names) != 60 or class_names[21] != "cheer up" or class_names[22] != "hand waving":
        raise RuntimeError("NTU60 class map does not match expected class IDs 21/22")

    print(f"Validation clips: {len(annotations)}", flush=True)
    original = evaluate_checkpoint("original", ORIGINAL, annotations, class_names)
    finetuned = evaluate_checkpoint("fine_tuned", FINETUNED, annotations, class_names)

    report = {
        "dataset": "PYSKL NTU60 HRNet 2D annotations",
        "split": "xsub_val",
        "split_protocol": "cross-subject",
        "samples_in_split": len(val_ids),
        "person_slots": NUM_PERSONS,
        "class_ids": {"cheer_up": 21, "hand_waving": 22},
        "evaluation": "Each valid person's logits are evaluated independently; no person-axis averaging.",
        "original": original,
        "fine_tuned": finetuned,
        "fine_tuning_delta": {
            "overall_top1": finetuned["overall_per_person_top1_accuracy"] - original["overall_per_person_top1_accuracy"],
            "cheer_up_top1": finetuned["class_21_cheer_up"]["top1_accuracy"] - original["class_21_cheer_up"]["top1_accuracy"],
            "hand_waving_top1": finetuned["class_22_hand_waving"]["top1_accuracy"] - original["class_22_hand_waving"]["top1_accuracy"],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"REPORT_PATH={REPORT}", flush=True)


if __name__ == "__main__":
    main()
