from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from GCN.STGCNPP.model_multi_person import STGCNPPMultiPerson
from GCN.STGCNPP.evaluate_ntu60_per_person import (
    DATA_PATH,
    NUM_PERSONS,
    preprocess_annotation,
)

CHECKPOINT = ROOT / "GCN" / "checkpoints" / "stgcnpp_jm_ntu60_xsub_hrnet.pth"
OUTPUT_DIR = ROOT / "finetune_outputs"
OUTPUT_CHECKPOINT = OUTPUT_DIR / "stgcnpp_jm_ntu60_xsub_hrnet_finetuned.pth"
RESUME_STATE = OUTPUT_DIR / "stgcnpp_jm_ntu60_xsub_resume_state.pth"
REPORT_PATH = OUTPUT_DIR / "finetune_report.json"

BATCH_SIZE = 16
TOTAL_EPOCHS = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 255


class NTU60PersonDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        packed, valid = preprocess_annotation(annotation)
        return (
            torch.from_numpy(packed),
            torch.from_numpy(valid.astype(np.bool_)),
            int(annotation["label"]),
        )


def collate_batch(items):
    inputs = torch.stack([item[0] for item in items], dim=0).unsqueeze(1)
    valid = torch.stack([item[1] for item in items], dim=0)
    labels = torch.tensor([item[2] for item in items], dtype=torch.long)
    return inputs, valid, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    person_total = person_correct = 0
    loss_sum = 0.0
    batches = 0

    for inputs, valid, labels in loader:
        inputs = inputs.to(device)
        valid = valid.to(device)
        labels = labels.to(device)

        logits = model._person_logits(inputs)[:, :, 0, :]
        predictions = logits.argmax(dim=-1)
        if valid.any():
            expanded_labels = labels[:, None].expand_as(predictions)
            person_logits = logits[valid]
            person_targets = expanded_labels[valid]
            person_loss = F.cross_entropy(person_logits, person_targets)
            loss_sum += float(person_loss.item())
            batches += 1
            person_correct += int((predictions[valid] == person_targets).sum().item())
            person_total += int(valid.sum().item())

    return {
        "loss": loss_sum / max(batches, 1),
        "per_person_top1_accuracy": person_correct / max(person_total, 1),
        "person_predictions": person_total,
    }


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_best_checkpoint(model, epoch, validation, best_metric, resume_source):
    torch.save(
        {
            "state_dict": cpu_state_dict(model),
            "fine_tuned": epoch > 0,
            "base_checkpoint": CHECKPOINT.name,
            "epoch": epoch,
            "validation": validation,
            "best_validation_accuracy": best_metric,
            "resume_source": resume_source,
            "training": {
                "dataset": "PYSKL NTU60 HRNet annotations",
                "train_split": "xsub_train",
                "validation_split": "xsub_val",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "total_epochs": TOTAL_EPOCHS,
                "person_loss_only": True,
            },
        },
        OUTPUT_CHECKPOINT,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing official NTU60 HRNet annotations: {DATA_PATH}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing original pretrained checkpoint: {CHECKPOINT}")

    resume_path_value = os.environ.get("RESUME_CHECKPOINT", "").strip()
    resume_path = Path(resume_path_value) if resume_path_value else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)

    with DATA_PATH.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    split = dataset["split"]
    train_ids = set(split["xsub_train"])
    val_ids = set(split["xsub_val"])
    if train_ids & val_ids:
        raise RuntimeError("xsub_train and xsub_val overlap")

    annotations = dataset["annotations"]
    train_annotations = [a for a in annotations if a["frame_dir"] in train_ids]
    val_annotations = [a for a in annotations if a["frame_dir"] in val_ids]
    train_annotations.sort(key=lambda a: a["frame_dir"])
    val_annotations.sort(key=lambda a: a["frame_dir"])

    if len(train_annotations) != len(train_ids):
        raise RuntimeError(
            f"xsub_train mismatch: expected {len(train_ids)}, found {len(train_annotations)}"
        )
    if len(val_annotations) != len(val_ids):
        raise RuntimeError(
            f"xsub_val mismatch: expected {len(val_ids)}, found {len(val_annotations)}"
        )

    print(
        f"Official split loaded: train={len(train_annotations)}, "
        f"validation={len(val_annotations)}",
        flush=True,
    )

    train_loader = DataLoader(
        NTU60PersonDataset(train_annotations),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        NTU60PersonDataset(val_annotations),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
        drop_last=False,
    )

    model = STGCNPPMultiPerson(
        checkpoint_path=str(CHECKPOINT),
        num_classes=60,
        device=str(device),
        strict=True,
        max_persons=NUM_PERSONS,
    )

    print("Evaluating original checkpoint before fine-tuning...", flush=True)
    original_baseline = evaluate(model, val_loader, device)
    print("ORIGINAL_BASELINE=" + json.dumps(original_baseline), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7
    )

    start_epoch = 0
    resume_source = "original pretrained checkpoint"
    if resume_path is not None:
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        resumed_weights = resumed.get("state_dict", resumed)
        model.load_state_dict(resumed_weights, strict=True)
        start_epoch = int(resumed.get("epoch", 0))
        resume_source = str(resume_path)
        if start_epoch >= TOTAL_EPOCHS:
            raise RuntimeError(
                f"Checkpoint is already at epoch {start_epoch}; total target is {TOTAL_EPOCHS}"
            )
        if "optimizer_state_dict" in resumed:
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            print("Optimizer state restored from resume checkpoint.", flush=True)
        else:
            print(
                "Resume weights restored; previous artifact has no optimizer state, "
                "so AdamW starts fresh at the configured learning rate.",
                flush=True,
            )
        if "scheduler_state_dict" in resumed:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        print(
            f"RESUMED_FROM={resume_path} completed_epoch={start_epoch}",
            flush=True,
        )

    starting_validation = evaluate(model, val_loader, device)
    print("RESUME_START_VALIDATION=" + json.dumps(starting_validation), flush=True)

    best_metric = starting_validation["per_person_top1_accuracy"]
    best_epoch = start_epoch
    best_validation = starting_validation
    save_best_checkpoint(
        model, best_epoch, best_validation, best_metric, resume_source
    )

    history = []
    for epoch in range(start_epoch + 1, TOTAL_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        skipped = 0

        for step, (inputs, valid, labels) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            valid = valid.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model._person_logits(inputs)[:, :, 0, :]
            if valid.any():
                person_labels = labels[:, None].expand(-1, logits.shape[1])
                loss = F.cross_entropy(logits[valid], person_labels[valid])
            else:
                skipped += 1
                continue

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.detach().item())
            batches += 1

            if step % 250 == 0:
                print(
                    f"epoch={epoch}/{TOTAL_EPOCHS} step={step}/{len(train_loader)} "
                    f"loss={running_loss / batches:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        validation = evaluate(model, val_loader, device)
        scheduler.step(validation["per_person_top1_accuracy"])

        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(batches, 1),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "skipped_empty_person_batches": skipped,
        }
        history.append(record)
        print("EPOCH_RESULT=" + json.dumps(record), flush=True)

        metric = validation["per_person_top1_accuracy"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_validation = validation
            save_best_checkpoint(
                model, best_epoch, best_validation, best_metric, resume_source
            )
            print(f"BEST_CHECKPOINT_SAVED={OUTPUT_CHECKPOINT}", flush=True)

        # Save the latest epoch separately with optimizer and scheduler state.
        # This makes any later continuation an actual training resume.
        torch.save(
            {
                "state_dict": cpu_state_dict(model),
                "epoch": epoch,
                "validation": validation,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "resume_source": resume_source,
                "training": {
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "total_epochs": TOTAL_EPOCHS,
                },
            },
            RESUME_STATE,
        )
        print(f"RESUME_STATE_SAVED={RESUME_STATE}", flush=True)

    report = {
        "dataset": "PYSKL NTU60 HRNet annotations",
        "protocol": "cross-subject",
        "train_split": "xsub_train",
        "validation_split": "xsub_val",
        "train_samples": len(train_annotations),
        "validation_samples": len(val_annotations),
        "original_checkpoint_baseline": original_baseline,
        "resume_start_validation": starting_validation,
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "per_person_gain_vs_original": (
            best_validation["per_person_top1_accuracy"]
            - original_baseline["per_person_top1_accuracy"]
        ),
        "training": {
            "completed_epochs_this_run": history,
            "total_target_epochs": TOTAL_EPOCHS,
            "resumed_from_epoch": start_epoch,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "all_model_parameters_trainable": True,
            "objective": (
                "Cross-entropy over each valid person's independent logits. "
                "No person-logit averaging and no scene-level loss or metrics."
            ),
            "supervision_note": (
                "NTU60 supplies a clip-level action label; the same label is "
                "applied to each valid person in that annotated clip."
            ),
        },
        "best_checkpoint": str(OUTPUT_CHECKPOINT),
        "latest_resume_state": str(RESUME_STATE),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL_REPORT=" + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
