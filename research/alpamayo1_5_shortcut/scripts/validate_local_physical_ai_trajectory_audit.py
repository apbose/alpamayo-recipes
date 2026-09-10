#!/usr/bin/env python3
"""Independently validate a finalized local PhysicalAI-AV trajectory audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
EXPECTED_T0_US = tuple(range(2_000_000, 13_000_001, 1_000_000))
EXPECTED_IMAGE_SHAPE = [4, 4, 3, 1080, 1920]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    summary = read_json(audit_dir / "summary.json")

    manifests: dict[str, list[dict[str, Any]]] = {
        split: read_json(audit_dir / f"trajectory_{split}.json")
        for split in SPLITS
    }
    manifest_keys: set[tuple[str, int]] = set()
    split_clip_sets: dict[str, set[str]] = {}
    rows_per_clip: Counter[str] = Counter()
    t0_per_clip: defaultdict[str, set[int]] = defaultdict(set)
    clip_metadata: dict[str, tuple[int, str]] = {}
    rows_by_split: dict[str, int] = {}

    for split, rows in manifests.items():
        rows_by_split[split] = len(rows)
        split_clip_sets[split] = set()
        for row in rows:
            clip_id = str(row["clip_id"])
            t0_us = int(row["t0_relative"])
            key = (clip_id, t0_us)
            require(key not in manifest_keys, f"Duplicate manifest key: {key}")
            manifest_keys.add(key)
            require(row["split"] == split, f"Wrong split for {key}")
            require(t0_us in EXPECTED_T0_US, f"Unexpected t0 for {key}")
            metadata = (int(row["chunk"]), split)
            require(
                clip_id not in clip_metadata or clip_metadata[clip_id] == metadata,
                f"Inconsistent metadata for clip {clip_id}",
            )
            clip_metadata[clip_id] = metadata
            split_clip_sets[split].add(clip_id)
            rows_per_clip[clip_id] += 1
            t0_per_clip[clip_id].add(t0_us)

    require(
        not (split_clip_sets["train"] & split_clip_sets["val"]),
        "Train/validation clip overlap",
    )
    require(
        not (split_clip_sets["train"] & split_clip_sets["test"]),
        "Train/test clip overlap",
    )
    require(
        not (split_clip_sets["val"] & split_clip_sets["test"]),
        "Validation/test clip overlap",
    )
    for clip_id in clip_metadata:
        require(rows_per_clip[clip_id] == 12, f"Clip {clip_id} does not have 12 rows")
        require(
            t0_per_clip[clip_id] == set(EXPECTED_T0_US),
            f"Clip {clip_id} does not cover the expected timestamps",
        )

    journal_keys: set[tuple[str, int]] = set()
    journal_rows = 0
    for journal in sorted(audit_dir.glob("worker-*-of-*.jsonl")):
        with journal.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                key = (str(record["clip_id"]), int(record["t0_relative"]))
                require(key not in journal_keys, f"Duplicate journal key: {key}")
                journal_keys.add(key)
                journal_rows += 1
                require(record["passed"] is True, f"Failed audit record: {key}")
                require(record["errors"] == [], f"Errors recorded for {key}")
                require(
                    record["image_shape"] == EXPECTED_IMAGE_SHAPE,
                    f"Wrong image shape for {key}",
                )
                image_range = record["image_value_range"]
                require(
                    0 <= image_range[0] < image_range[1] <= 255,
                    f"Invalid image range for {key}",
                )
                for field in (
                    "median_image_dt_seconds",
                    "future_path_distance_m",
                    "future_endpoint_yaw_rad",
                    "decode_seconds",
                ):
                    require(math.isfinite(float(record[field])), f"Non-finite {field} for {key}")
                endpoint = record["future_endpoint_xyz"]
                require(
                    len(endpoint) == 3 and all(math.isfinite(float(value)) for value in endpoint),
                    f"Invalid future endpoint for {key}",
                )

    require(journal_keys == manifest_keys, "Journal and manifest keys differ")
    require(read_json(audit_dir / "failed_rows.json") == [], "failed_rows.json is non-empty")

    with (audit_dir / "clips.csv").open(newline="", encoding="utf-8") as handle:
        clip_rows = list(csv.DictReader(handle))
    require(len(clip_rows) == len(clip_metadata), "clips.csv clip count differs")
    require(
        all(
            int(row["passed_keyframes"]) == 12
            and int(row["failed_keyframes"]) == 0
            and row["all_keyframes_passed"] == "True"
            for row in clip_rows
        ),
        "clips.csv contains an incomplete clip",
    )

    require(summary["audit_complete"] is True, "Summary is not complete")
    require(summary["all_rows_passed"] is True, "Summary reports a failed row")
    require(summary["expected_rows"] == journal_rows, "Summary expected-row mismatch")
    require(summary["completed_rows"] == journal_rows, "Summary completed-row mismatch")
    require(summary["failed_rows"] == 0, "Summary failed-row mismatch")
    require(summary["local_valid_clips"] == len(clip_metadata), "Summary clip mismatch")
    require(
        summary["t0_relative_values"] == list(EXPECTED_T0_US),
        "Summary timestamp set differs",
    )

    result = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_dir": str(audit_dir),
        "validation_passed": True,
        "journal_files": len(list(audit_dir.glob("worker-*-of-*.jsonl"))),
        "validated_rows": journal_rows,
        "validated_clips": len(clip_metadata),
        "rows_by_split": rows_by_split,
        "clips_by_split": {
            split: len(split_clip_sets[split]) for split in SPLITS
        },
        "timestamps_per_clip": len(EXPECTED_T0_US),
        "clip_overlap_counts": {
            "train_val": len(split_clip_sets["train"] & split_clip_sets["val"]),
            "train_test": len(split_clip_sets["train"] & split_clip_sets["test"]),
            "val_test": len(split_clip_sets["val"] & split_clip_sets["test"]),
        },
        "all_journal_rows_passed": True,
        "journal_manifest_keys_identical": True,
        "all_json_decoded": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
