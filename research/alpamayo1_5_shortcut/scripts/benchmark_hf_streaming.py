#!/usr/bin/env python3
"""Benchmark fixed PhysicalAI-AV samples through the official HF range reader."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from alpamayo.data.pai_trajectory import PAITrajectoryDataset

EXPECTED_SHAPES = {
    "image_frames": (4, 4, 3, 1080, 1920),
    "camera_indices": (4,),
    "ego_history_xyz": (1, 16, 3),
    "ego_history_rot": (1, 16, 3, 3),
    "ego_future_xyz": (1, 64, 3),
    "ego_future_rot": (1, 64, 3, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


class ValidatingTimedDataset(Dataset):
    def __init__(self, source: PAITrajectoryDataset, sample_count: int) -> None:
        self.source = source
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        started = time.perf_counter()
        sample = self.source[index]
        elapsed = time.perf_counter() - started
        for key, expected_shape in EXPECTED_SHAPES.items():
            value = sample[key]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{key} shape {tuple(value.shape)} != {expected_shape}"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} contains a non-finite value")
        return {
            "index": index,
            "clip_id": sample["clip_id"],
            "t0_us": int(sample["t0_us"]),
            "load_seconds": elapsed,
        }


def identity_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[0]


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.samples > int(summary["annotation_rows"]["train"]):
        raise ValueError("samples exceeds training manifest length")
    if any(worker < 0 for worker in args.workers):
        raise ValueError("worker counts must be non-negative")

    source = PAITrajectoryDataset(
        annotations_path=str(args.manifest),
        access_mode="hf_stream",
        hf_revision=summary["revision"],
        hf_cache_dir=str(args.cache_dir),
        chunk_ids=summary["selected_chunks"],
    )
    dataset = ValidatingTimedDataset(source, args.samples)
    results: dict[str, Any] = {}
    for workers in args.workers:
        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": False,
            "collate_fn": identity_collate,
        }
        if workers:
            loader_kwargs["prefetch_factor"] = 1
            loader_kwargs["persistent_workers"] = False
        loader = DataLoader(**loader_kwargs)
        started = time.perf_counter()
        rows = list(loader)
        wall_seconds = time.perf_counter() - started
        loads = [float(row["load_seconds"]) for row in rows]
        results[str(workers)] = {
            "samples": len(rows),
            "wall_seconds": wall_seconds,
            "samples_per_second": len(rows) / wall_seconds,
            "per_sample_load_seconds": {
                "mean": statistics.fmean(loads),
                "median": statistics.median(loads),
                "p90": percentile(loads, 0.9),
                "max": max(loads),
            },
            "clip_ids": [row["clip_id"] for row in rows],
        }
        print(
            f"workers={workers} samples={len(rows)} wall={wall_seconds:.3f}s "
            f"throughput={len(rows) / wall_seconds:.4f} samples/s",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": summary["repo_id"],
        "revision": summary["revision"],
        "manifest": str(args.manifest.resolve()),
        "source_coverage_bytes": summary["selected_source_bytes"],
        "sample_count_per_worker_setting": args.samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
