#!/usr/bin/env python3
"""Resumably validate multiple trajectory keyframes for every local PAI clip."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


DEFAULT_T0_US = (2_000_000, 4_800_000, 7_600_000, 10_400_000, 13_200_000)
EXPECTED_CAMERA_INDICES = [0, 1, 2, 6]
EXPECTED_IMAGE_SHAPE = (4, 4, 3, 1080, 1920)
EXPECTED_SHAPES = {
    "camera_indices": (4,),
    "absolute_timestamps": (4, 4),
    "relative_timestamps": (4, 4),
    "ego_history_xyz": (1, 16, 3),
    "ego_history_rot": (1, 16, 3, 3),
    "ego_future_xyz": (1, 64, 3),
    "ego_future_rot": (1, 64, 3, 3),
}
REQUIRED_ARCHIVE_PATTERNS = (
    "labels/egomotion/egomotion.chunk_{chunk:04d}.zip",
    "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{chunk:04d}.zip",
    "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{chunk:04d}.zip",
    "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{chunk:04d}.zip",
    "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk:04d}.zip",
)
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--t0-us", type=int, nargs="+", default=list(DEFAULT_T0_US))
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def local_chunks(dataset_dir: Path) -> list[int]:
    chunks: set[int] = set()
    pattern = re.compile(r"chunk_(\d+)\.zip$")
    for path in (dataset_dir / "labels/egomotion").glob("egomotion.chunk_*.zip"):
        match = pattern.search(path.name)
        if match:
            chunks.add(int(match.group(1)))
    if not chunks:
        raise FileNotFoundError("No local ego-motion chunk archives were found")

    missing: list[str] = []
    for chunk in sorted(chunks):
        for archive_pattern in REQUIRED_ARCHIVE_PATTERNS:
            relative = archive_pattern.format(chunk=chunk)
            if not (dataset_dir / relative).is_file():
                missing.append(relative)
    if missing:
        raise FileNotFoundError(
            "Local chunks are incomplete; missing required archives: "
            + ", ".join(missing[:20])
        )
    return sorted(chunks)


def local_clip_table(dataset_dir: Path, chunks: Iterable[int]) -> pd.DataFrame:
    table = pd.read_parquet(dataset_dir / "clip_index.parquet")
    if table.index.name != "clip_id":
        if "clip_id" not in table.columns:
            raise ValueError("clip_index.parquet lacks a clip_id index or column")
        table = table.set_index("clip_id")
    table = table[table["chunk"].isin(chunks) & table["clip_is_valid"]].copy()
    table.index = table.index.map(str)
    table = table.sort_values(["split", "chunk"], kind="stable")
    unexpected = set(table["split"].astype(str)) - set(SPLITS)
    if unexpected:
        raise ValueError(f"Unexpected official splits: {sorted(unexpected)}")
    if table.index.has_duplicates:
        raise ValueError("clip_index.parquet contains duplicate clip IDs")
    return table


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def rotation_errors(name: str, rotations: torch.Tensor, errors: list[str]) -> None:
    flattened = rotations.reshape(-1, 3, 3).float()
    identity = torch.eye(3, dtype=flattened.dtype).expand_as(flattened)
    gram = flattened.transpose(-1, -2) @ flattened
    orthogonality_error = float((gram - identity).abs().max())
    determinant_error = float((torch.linalg.det(flattened) - 1.0).abs().max())
    require(
        orthogonality_error < 1e-3,
        f"{name} orthogonality error is {orthogonality_error:.6g}",
        errors,
    )
    require(
        determinant_error < 1e-3,
        f"{name} determinant error is {determinant_error:.6g}",
        errors,
    )


def validate_sample(
    sample: dict[str, Any],
    *,
    clip_id: str,
    chunk: int,
    split: str,
    t0_us: int,
) -> dict[str, Any]:
    errors: list[str] = []
    required = {"image_frames", "clip_id", "t0_us", *EXPECTED_SHAPES}
    missing = sorted(required.difference(sample))
    require(not missing, f"missing fields: {missing}", errors)
    if missing:
        return {
            "clip_id": clip_id,
            "chunk": chunk,
            "split": split,
            "t0_relative": t0_us,
            "passed": False,
            "errors": errors,
        }

    require(str(sample["clip_id"]) == clip_id, "loader returned a different clip_id", errors)
    require(int(sample["t0_us"]) == t0_us, "loader returned a different t0_us", errors)

    images = sample["image_frames"]
    require(isinstance(images, torch.Tensor), "image_frames is not a tensor", errors)
    image_range: list[int] | None = None
    if isinstance(images, torch.Tensor):
        require(tuple(images.shape) == EXPECTED_IMAGE_SHAPE, f"image shape is {tuple(images.shape)}", errors)
        require(images.dtype == torch.uint8, f"image dtype is {images.dtype}", errors)
        image_min = int(images.min())
        image_max = int(images.max())
        image_range = [image_min, image_max]
        require(image_min < image_max, "decoded images are constant", errors)

    for name, expected_shape in EXPECTED_SHAPES.items():
        value = sample[name]
        require(isinstance(value, torch.Tensor), f"{name} is not a tensor", errors)
        if isinstance(value, torch.Tensor):
            require(tuple(value.shape) == expected_shape, f"{name} shape is {tuple(value.shape)}", errors)
            if value.is_floating_point():
                require(bool(torch.isfinite(value).all()), f"{name} has non-finite values", errors)

    camera_indices = sample["camera_indices"]
    if isinstance(camera_indices, torch.Tensor):
        require(camera_indices.tolist() == EXPECTED_CAMERA_INDICES, f"camera indices are {camera_indices.tolist()}", errors)

    median_image_dt_seconds: float | None = None
    timestamps = sample["absolute_timestamps"]
    if isinstance(timestamps, torch.Tensor) and tuple(timestamps.shape) == (4, 4):
        deltas = timestamps[:, 1:] - timestamps[:, :-1]
        require(bool((deltas > 0).all()), "camera timestamps are not increasing", errors)
        median_image_dt_seconds = float(deltas.float().median() * 1e-6)
        require(abs(median_image_dt_seconds - 0.1) < 0.02, f"median camera dt is {median_image_dt_seconds:.6f}s", errors)

    history_xyz = sample["ego_history_xyz"]
    history_rot = sample["ego_history_rot"]
    if isinstance(history_xyz, torch.Tensor) and tuple(history_xyz.shape) == (1, 16, 3):
        require(float(history_xyz[0, -1].abs().max()) < 1e-4, "history does not end at origin", errors)
    if isinstance(history_rot, torch.Tensor) and tuple(history_rot.shape) == (1, 16, 3, 3):
        identity_error = float((history_rot[0, -1].float() - torch.eye(3)).abs().max())
        require(identity_error < 1e-4, "history does not end at identity rotation", errors)

    for name in ("ego_history_rot", "ego_future_rot"):
        value = sample[name]
        if isinstance(value, torch.Tensor) and value.shape[-2:] == (3, 3) and bool(torch.isfinite(value).all()):
            rotation_errors(name, value, errors)

    future_xyz = sample["ego_future_xyz"]
    future_rot = sample["ego_future_rot"]
    endpoint: list[float] | None = None
    path_distance: float | None = None
    endpoint_yaw: float | None = None
    if isinstance(future_xyz, torch.Tensor) and tuple(future_xyz.shape) == (1, 64, 3):
        points = future_xyz[0].float()
        endpoint = [float(value) for value in points[-1]]
        segments = torch.diff(torch.cat((torch.zeros_like(points[:1]), points)), dim=0)
        path_distance = float(torch.linalg.vector_norm(segments[:, :2], dim=-1).sum())
    if isinstance(future_rot, torch.Tensor) and tuple(future_rot.shape) == (1, 64, 3, 3):
        final_rotation = future_rot[0, -1].float()
        endpoint_yaw = math.atan2(float(final_rotation[1, 0]), float(final_rotation[0, 0]))

    return {
        "clip_id": clip_id,
        "chunk": chunk,
        "split": split,
        "t0_relative": t0_us,
        "passed": not errors,
        "errors": errors,
        "image_shape": list(images.shape) if isinstance(images, torch.Tensor) else None,
        "image_value_range": image_range,
        "median_image_dt_seconds": median_image_dt_seconds,
        "future_endpoint_xyz": endpoint,
        "future_path_distance_m": path_distance,
        "future_endpoint_yaw_rad": endpoint_yaw,
    }


def load_completed(journal: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    if not journal.exists():
        return records
    with journal.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed journal line {line_number} in {journal}") from exc
            key = (str(record["clip_id"]), int(record["t0_relative"]))
            if key in records:
                raise ValueError(f"Duplicate journal key {key} in {journal}")
            records[key] = record
    return records


def worker(args: argparse.Namespace) -> None:
    if args.worker_index is None:
        raise ValueError("--worker-index is required unless --finalize is used")
    if args.num_workers <= 0 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")
    if len(set(args.t0_us)) != len(args.t0_us):
        raise ValueError("--t0-us values must be unique")
    if any(not 1_600_000 <= value <= 13_600_000 for value in args.t0_us):
        raise ValueError("Every t0 must preserve the 1.6s history and 6.4s future margins")

    torch.set_num_threads(1)
    chunks = local_chunks(args.dataset_dir)
    table = local_clip_table(args.dataset_dir, chunks)
    rows = [
        (str(clip_id), int(row["chunk"]), str(row["split"]))
        for clip_id, row in table.iterrows()
    ]
    assigned = [row for position, row in enumerate(rows) if position % args.num_workers == args.worker_index]
    if args.max_clips is not None:
        assigned = assigned[: args.max_clips]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / f"worker-{args.worker_index:02d}-of-{args.num_workers:02d}.jsonl"
    status_path = args.output_dir / f"worker-{args.worker_index:02d}-of-{args.num_workers:02d}.status.json"
    completed = load_completed(journal)
    expected = len(assigned) * len(args.t0_us)
    started = time.monotonic()
    initial_completed = len(completed)

    avdi = PhysicalAIAVDatasetLocalInterface(local_dir=args.dataset_dir, chunk_ids=chunks)
    with journal.open("a", encoding="utf-8", buffering=1) as output:
        for clip_position, (clip_id, chunk, split) in enumerate(assigned):
            for t0_us in args.t0_us:
                key = (clip_id, t0_us)
                if key in completed:
                    continue
                sample_started = time.monotonic()
                try:
                    sample = load_physical_aiavdataset(
                        clip_id,
                        t0_us=t0_us,
                        avdi=avdi,
                        num_history_steps=16,
                        num_future_steps=64,
                        time_step=0.1,
                    )
                    for name in list(sample):
                        if name.startswith("ego_"):
                            sample[name] = sample[name].squeeze(0)
                    record = validate_sample(
                        sample,
                        clip_id=clip_id,
                        chunk=chunk,
                        split=split,
                        t0_us=t0_us,
                    )
                except Exception as exc:  # Continue auditing after a corrupt sample.
                    record = {
                        "clip_id": clip_id,
                        "chunk": chunk,
                        "split": split,
                        "t0_relative": t0_us,
                        "passed": False,
                        "errors": [f"{type(exc).__name__}: {str(exc)[:2000]}"],
                    }
                record["decode_seconds"] = round(time.monotonic() - sample_started, 3)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                completed[key] = record
                del record
                if "sample" in locals():
                    del sample

                processed = len(completed)
                if processed % 10 == 0 or processed == expected:
                    elapsed = time.monotonic() - started
                    new_processed = processed - initial_completed
                    rate = new_processed / elapsed if elapsed > 0 and new_processed else 0.0
                    atomic_json(
                        status_path,
                        {
                            "status": "running" if processed < expected else "complete",
                            "updated_at_utc": utc_now(),
                            "worker_index": args.worker_index,
                            "num_workers": args.num_workers,
                            "assigned_clips": len(assigned),
                            "expected_rows": expected,
                            "completed_rows": processed,
                            "passed_rows": sum(bool(value["passed"]) for value in completed.values()),
                            "failed_rows": sum(not bool(value["passed"]) for value in completed.values()),
                            "new_rows_per_second": rate,
                        },
                    )
            if clip_position % 20 == 0:
                gc.collect()

    if len(completed) != expected:
        raise RuntimeError(f"Worker completed {len(completed)} rows, expected {expected}")
    atomic_json(
        status_path,
        {
            "status": "complete",
            "updated_at_utc": utc_now(),
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
            "assigned_clips": len(assigned),
            "expected_rows": expected,
            "completed_rows": len(completed),
            "passed_rows": sum(bool(value["passed"]) for value in completed.values()),
            "failed_rows": sum(not bool(value["passed"]) for value in completed.values()),
            "elapsed_seconds_this_invocation": round(time.monotonic() - started, 3),
        },
    )


def finalize(args: argparse.Namespace) -> None:
    chunks = local_chunks(args.dataset_dir)
    table = local_clip_table(args.dataset_dir, chunks)
    expected = {
        (str(clip_id), t0_us)
        for clip_id in table.index
        for t0_us in args.t0_us
    }
    records: dict[tuple[str, int], dict[str, Any]] = {}
    journals = sorted(args.output_dir.glob("worker-*-of-*.jsonl"))
    if len(journals) != args.num_workers:
        raise RuntimeError(f"Found {len(journals)} worker journals; expected {args.num_workers}")
    for journal in journals:
        for key, record in load_completed(journal).items():
            if key in records:
                raise RuntimeError(f"Duplicate result across worker journals: {key}")
            records[key] = record
    missing = sorted(expected - set(records))
    unexpected = sorted(set(records) - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Audit coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    ordered = [records[key] for key in sorted(records)]
    passed = [record for record in ordered if record["passed"]]
    failed = [record for record in ordered if not record["passed"]]
    by_split: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    by_chunk: dict[str, Counter[str]] = defaultdict(Counter)
    by_t0: dict[str, Counter[str]] = defaultdict(Counter)
    clip_counts: dict[str, Counter[str]] = defaultdict(Counter)
    error_types: Counter[str] = Counter()
    for record in ordered:
        outcome = "passed" if record["passed"] else "failed"
        by_split[record["split"]][outcome] += 1
        by_chunk[str(record["chunk"])][outcome] += 1
        by_t0[str(record["t0_relative"])][outcome] += 1
        clip_counts[record["clip_id"]][outcome] += 1
        for error in record["errors"]:
            error_types[error.split(":", 1)[0]] += 1

    manifests: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        manifests[split] = [
            {
                "clip_id": record["clip_id"],
                "t0_relative": record["t0_relative"],
                "chunk": record["chunk"],
                "split": record["split"],
            }
            for record in passed
            if record["split"] == split
        ]
        atomic_json(args.output_dir / f"trajectory_{split}.json", manifests[split])

    atomic_json(args.output_dir / "failed_rows.json", failed)
    clips_csv = args.output_dir / "clips.csv"
    with clips_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clip_id", "chunk", "split", "passed_keyframes", "failed_keyframes", "all_keyframes_passed"]
        )
        for clip_id, row in table.sort_index().iterrows():
            counts = clip_counts[str(clip_id)]
            writer.writerow(
                [
                    clip_id,
                    int(row["chunk"]),
                    str(row["split"]),
                    counts["passed"],
                    counts["failed"],
                    counts["failed"] == 0,
                ]
            )

    split_clip_sets = {
        split: {row["clip_id"] for row in manifests[split]} for split in SPLITS
    }
    overlap = {
        "train_val": len(split_clip_sets["train"] & split_clip_sets["val"]),
        "train_test": len(split_clip_sets["train"] & split_clip_sets["test"]),
        "val_test": len(split_clip_sets["val"] & split_clip_sets["test"]),
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "audit_complete": True,
        "all_rows_passed": not failed,
        "local_chunks": chunks,
        "local_valid_clips": len(table),
        "t0_relative_values": args.t0_us,
        "expected_rows": len(expected),
        "completed_rows": len(ordered),
        "passed_rows": len(passed),
        "failed_rows": len(failed),
        "fully_valid_clips": sum(counts["failed"] == 0 for counts in clip_counts.values()),
        "rows_by_split": {split: dict(by_split[split]) for split in SPLITS},
        "rows_by_chunk": {key: dict(value) for key, value in sorted(by_chunk.items(), key=lambda item: int(item[0]))},
        "rows_by_t0": {key: dict(value) for key, value in sorted(by_t0.items(), key=lambda item: int(item[0]))},
        "error_type_counts": dict(error_types),
        "clip_overlap_counts": overlap,
        "expected_contract": {
            "camera_indices": EXPECTED_CAMERA_INDICES,
            "image_shape": list(EXPECTED_IMAGE_SHAPE),
            "history_steps": 16,
            "future_steps": 64,
            "trajectory_hz": 10,
        },
        "annotation_status": {
            "navigation_text_included": False,
            "directly_consumable_by_PAIDatasetWithNav": False,
            "reason": (
                "This audit validates raw camera and ego-motion rows only. "
                "Navigation text must be sourced or designed separately."
            ),
        },
        "artifacts": {
            "trajectory_train": str((args.output_dir / "trajectory_train.json").resolve()),
            "trajectory_val": str((args.output_dir / "trajectory_val.json").resolve()),
            "trajectory_test": str((args.output_dir / "trajectory_test.json").resolve()),
            "failed_rows": str((args.output_dir / "failed_rows.json").resolve()),
            "clips_csv": str(clips_csv.resolve()),
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.finalize:
        finalize(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
