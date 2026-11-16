#!/usr/bin/env python3
"""Create leakage-resistant train/validation/test manifests for PIP-175k.

The split is grouped by person/collection identity (never by individual clip),
approximately stratified by activity class, and stores paths/metadata only.
It does not copy or duplicate the 54 GB video archive.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

GROUP_KEYS = (
    "collector_id",
    "subject_id",
    "person_id",
    "collection_id",
    "recording_id",
    "source_video_id",
    "video_id",
)


def _metadata(video):
    try:
        value = video.metadata()
    except Exception:
        value = {}
    return value if isinstance(value, dict) else {}


def _group_id(video, meta):
    for key in GROUP_KEYS:
        value = meta.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    raise ValueError(
        "No subject/collector/session identity found in metadata. "
        "Refusing to guess groups from clip filenames because that can leak "
        "the same person across train/validation/test. Inspect the PIP "
        "metadata schema and add its verified identity field to GROUP_KEYS."
    )


def _video_path(video):
    try:
        return str(video.filename())
    except Exception:
        return ""


def _category(video):
    try:
        value = video.category()
    except Exception:
        value = None
    if not value:
        raise ValueError("A PIP clip has no activity category.")
    return str(value)


def grouped_stratified_split(records, ratios=(0.8, 0.1, 0.1), seed=606):
    names = ("train", "validation", "test")
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")
    by_group = defaultdict(list)
    for row in records:
        by_group[row["group_id"]].append(row)

    if len(by_group) < 3:
        raise ValueError("At least three independent groups are required.")

    total = Counter(row["label"] for row in records)
    labels = sorted(total)
    targets = {
        name: {label: total[label] * ratios[i] for label in labels}
        for i, name in enumerate(names)
    }
    target_sizes = {
        name: len(records) * ratios[i] for i, name in enumerate(names)
    }
    counts = {name: Counter() for name in names}
    sizes = Counter()
    assignments = {name: [] for name in names}

    rng = random.Random(seed)
    groups = list(by_group.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)

    # Greedily place whole identity groups where they best preserve class
    # proportions and the requested overall split sizes.
    for group_id, rows in groups:
        group_counts = Counter(row["label"] for row in rows)
        best_name, best_cost = None, None
        for name in names:
            next_size = sizes[name] + len(rows)
            size_cost = ((next_size - target_sizes[name]) /
                         max(target_sizes[name], 1.0)) ** 2
            class_cost = 0.0
            for label in labels:
                target = max(targets[name][label], 1.0)
                after = counts[name][label] + group_counts[label]
                class_cost += ((after - targets[name][label]) / target) ** 2
            # Penalize filling a split far beyond its target.
            overflow = max(0.0, next_size - target_sizes[name])
            cost = class_cost + 2.0 * size_cost + 4.0 * (
                overflow / max(target_sizes[name], 1.0)
            ) ** 2
            if best_cost is None or cost < best_cost:
                best_name, best_cost = name, cost
        assignments[best_name].extend(rows)
        sizes[best_name] += len(rows)
        counts[best_name].update(group_counts)

    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="Path to the extracted pip_175k.json")
    parser.add_argument("--output", default="pip175k_splits")
    parser.add_argument("--seed", type=int, default=606)
    args = parser.parse_args()

    try:
        import vipy
    except ImportError as exc:
        raise SystemExit("Install dataset reader first: pip install 'vipy[all]'") from exc

    dataset = vipy.util.load(args.annotations)
    records = []
    for index, video in enumerate(dataset):
        meta = _metadata(video)
        records.append({
            "clip_index": index,
            "path": _video_path(video),
            "label": _category(video),
            "group_id": _group_id(video, meta),
        })

    splits = grouped_stratified_split(records, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {"seed": args.seed, "ratios": {"train": 0.8,
               "validation": 0.1, "test": 0.1}, "splits": {}}
    for name, rows in splits.items():
        destination = output / f"{name}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][name] = {
            "clips": len(rows),
            "groups": len({row["group_id"] for row in rows}),
            "classes": dict(sorted(Counter(r["label"] for r in rows).items())),
            "manifest": str(destination),
        }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
