#!/usr/bin/env python3
"""Build deterministic route-less manifests from a completed trajectory audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-t0-relative", type=int, default=7_000_000)
    parser.add_argument("--val-clips", type=int, default=32)
    parser.add_argument("--test-clips", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_key(row: dict[str, Any], *, split: str, seed: int) -> str:
    value = f"{seed}:{split}:{row['clip_id']}".encode()
    return hashlib.sha256(value).hexdigest()


def stratified_sample(
    rows: list[dict[str, Any]], *, split: str, count: int, seed: int
) -> list[dict[str, Any]]:
    """Select one clip per chunk first, then fill by a stable hash order."""
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} {split} clips from {len(rows)} rows")
    by_chunk: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_chunk[int(row["chunk"])].append(row)
    if count < len(by_chunk):
        raise ValueError(
            f"{split} count {count} cannot cover all {len(by_chunk)} chunks"
        )

    chosen: list[dict[str, Any]] = []
    chosen_clips: set[str] = set()
    for chunk in sorted(by_chunk):
        candidate = min(
            by_chunk[chunk], key=lambda row: stable_key(row, split=split, seed=seed)
        )
        chosen.append(candidate)
        chosen_clips.add(str(candidate["clip_id"]))

    remaining = sorted(
        (row for row in rows if str(row["clip_id"]) not in chosen_clips),
        key=lambda row: stable_key(row, split=split, seed=seed),
    )
    chosen.extend(remaining[: count - len(chosen)])
    return sorted(chosen, key=lambda row: (int(row["chunk"]), str(row["clip_id"])))


def normalized_rows(path: Path, split: str) -> list[dict[str, Any]]:
    source_rows = json.loads(path.read_text())
    rows = [
        {
            "clip_id": str(row["clip_id"]),
            "t0_relative": int(row["t0_relative"]),
            "chunk": int(row["chunk"]),
            "split": str(row["split"]),
        }
        for row in source_rows
    ]
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"Source {split} manifest is empty or contains another split")
    keys = [(row["clip_id"], row["t0_relative"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Source {split} manifest has duplicate clip/timestamp keys")
    if any("nav_text" in row for row in source_rows):
        raise ValueError(f"Source {split} manifest unexpectedly contains nav_text")
    return sorted(rows, key=lambda row: (row["chunk"], row["clip_id"], row["t0_relative"]))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    one_per_clip: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        source = args.audit_dir / f"trajectory_{split}.json"
        source_hashes[split] = sha256(source)
        all_rows[split] = normalized_rows(source, split)
        one_per_clip[split] = [
            row
            for row in all_rows[split]
            if row["t0_relative"] == args.pilot_t0_relative
        ]
        clip_ids = [row["clip_id"] for row in one_per_clip[split]]
        if not clip_ids or len(clip_ids) != len(set(clip_ids)):
            raise ValueError(f"{split} pilot t0 is not exactly one row per clip")

    selected = {
        "train": all_rows["train"],
        "val": stratified_sample(
            one_per_clip["val"], split="val", count=args.val_clips, seed=args.seed
        ),
        "test": stratified_sample(
            one_per_clip["test"], split="test", count=args.test_clips, seed=args.seed
        ),
    }

    full_clip_sets = {
        split: {row["clip_id"] for row in rows}
        for split, rows in all_rows.items()
    }
    overlap_counts = {
        "train_val": len(full_clip_sets["train"] & full_clip_sets["val"]),
        "train_test": len(full_clip_sets["train"] & full_clip_sets["test"]),
        "val_test": len(full_clip_sets["val"] & full_clip_sets["test"]),
    }
    if any(overlap_counts.values()):
        raise ValueError(f"Source manifests overlap by clip: {overlap_counts}")

    args.output_dir.mkdir(parents=True)
    for split in SPLITS:
        write_json(args.output_dir / f"{split}.json", selected[split])
        write_json(args.output_dir / f"{split}_all_timestamps.json", all_rows[split])
        write_json(args.output_dir / f"{split}_one_per_clip.json", one_per_clip[split])

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "route-less Stage-2 trajectory shortcut",
        "policy": (
            "Train on all 12 audited timestamps per official train clip; use "
            "one timestamp from deterministic chunk-stratified validation/test clips"
        ),
        "source_audit_dir": str(args.audit_dir.resolve()),
        "source_manifest_sha256": source_hashes,
        "pilot_t0_relative": args.pilot_t0_relative,
        "selection_seed": args.seed,
        "annotation_rows": {
            split: len(selected[split]) for split in SPLITS
        },
        "unique_clips": {
            split: len({row["clip_id"] for row in selected[split]})
            for split in SPLITS
        },
        "available_rows_all_timestamps": {
            split: len(all_rows[split]) for split in SPLITS
        },
        "available_rows_one_per_clip": {
            split: len(one_per_clip[split]) for split in SPLITS
        },
        "chunks": {
            split: sorted({int(row["chunk"]) for row in selected[split]})
            for split in SPLITS
        },
        "overlap_counts": overlap_counts,
        "navigation_conditioning": False,
        "nav_text_rows": 0,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
