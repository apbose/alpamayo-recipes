#!/usr/bin/env python3
"""Build deterministic route-less manifests spanning measured HF source shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from physical_ai_av import PhysicalAIAVDatasetInterface

DEFAULT_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
FEATURE_PATHS = {
    "egomotion": "labels/egomotion/egomotion.chunk_{chunk:04d}.zip",
    "camera_cross_left_120fov": (
        "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{chunk:04d}.zip"
    ),
    "camera_front_wide_120fov": (
        "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{chunk:04d}.zip"
    ),
    "camera_cross_right_120fov": (
        "camera/camera_cross_right_120fov/"
        "camera_cross_right_120fov.chunk_{chunk:04d}.zip"
    ),
    "camera_front_tele_30fov": (
        "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{chunk:04d}.zip"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-source-gb", type=float, default=300.0)
    parser.add_argument("--t0-relative", type=int, default=7_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--exclude-chunks", type=int, nargs="*", default=[])
    parser.add_argument("--query-chunks-at-once", type=int, default=64)
    return parser.parse_args()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_fixed_eval_rows(
    path: Path, expected_split: str, clip_index
) -> list[dict[str, Any]]:
    source = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source_row in source:
        clip_id = str(source_row["clip_id"])
        t0_relative = int(source_row["t0_relative"])
        key = (clip_id, t0_relative)
        if key in seen:
            raise ValueError(f"Duplicate row in {path}: {key}")
        seen.add(key)
        if str(clip_index.at[clip_id, "split"]) != expected_split:
            raise ValueError(f"{clip_id} is not in official {expected_split} split")
        rows.append(
            {
                "chunk": int(clip_index.at[clip_id, "chunk"]),
                "clip_id": clip_id,
                "split": expected_split,
                "t0_relative": t0_relative,
            }
        )
    if not rows:
        raise ValueError(f"Evaluation manifest is empty: {path}")
    return rows


def query_chunk_sizes(
    api: HfApi,
    *,
    chunks: list[int],
    revision: str,
) -> list[dict[str, Any]]:
    requested_paths = [
        template.format(chunk=chunk)
        for chunk in chunks
        for template in FEATURE_PATHS.values()
    ]
    path_info = api.get_paths_info(
        REPO_ID,
        requested_paths,
        expand=True,
        revision=revision,
        repo_type="dataset",
    )
    sizes = {item.path: item.size for item in path_info}
    missing = sorted(set(requested_paths).difference(sizes))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required feature archives: {missing[:5]}"
        )

    result = []
    for chunk in chunks:
        feature_sizes = {
            feature: int(sizes[template.format(chunk=chunk)])
            for feature, template in FEATURE_PATHS.items()
        }
        result.append(
            {
                "chunk": chunk,
                "bytes": sum(feature_sizes.values()),
                "feature_bytes": feature_sizes,
                "feature_paths": {
                    feature: template.format(chunk=chunk)
                    for feature, template in FEATURE_PATHS.items()
                },
            }
        )
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.target_source_gb <= 0:
        raise ValueError("target-source-gb must be positive")
    if args.query_chunks_at_once <= 0:
        raise ValueError("query-chunks-at-once must be positive")
    if not 1_600_000 < args.t0_relative <= 13_600_000:
        raise ValueError("t0-relative must leave 1.6 s history and 6.4 s future")

    dataset = PhysicalAIAVDatasetInterface(
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    clip_index = dataset.clip_index
    feature_presence = dataset.feature_presence
    required_features = list(FEATURE_PATHS)
    if not bool(feature_presence[required_features].all(axis=None)):
        raise ValueError("At least one clip lacks a required Alpamayo Stage-2 feature")

    excluded = set(args.exclude_chunks)
    train_index = clip_index.loc[
        clip_index["clip_is_valid"] & clip_index["split"].eq("train")
    ]
    candidates = sorted(
        (
            int(chunk)
            for chunk in train_index["chunk"].unique()
            if int(chunk) not in excluded
        ),
        key=lambda chunk: stable_key(args.seed, f"chunk:{chunk}"),
    )

    target_bytes = round(args.target_source_gb * 1_000_000_000)
    selected_inventory: list[dict[str, Any]] = []
    selected_bytes = 0
    api = HfApi()
    for offset in range(0, len(candidates), args.query_chunks_at_once):
        batch = candidates[offset : offset + args.query_chunks_at_once]
        for inventory in query_chunk_sizes(api, chunks=batch, revision=args.revision):
            selected_inventory.append(inventory)
            selected_bytes += int(inventory["bytes"])
            if selected_bytes >= target_bytes:
                break
        if selected_bytes >= target_bytes:
            break
    if selected_bytes < target_bytes:
        raise RuntimeError(
            f"Available train shards total only {selected_bytes / 1e9:.3f} GB, "
            f"below requested {args.target_source_gb:.3f} GB"
        )

    selected_chunks = [int(row["chunk"]) for row in selected_inventory]
    selected_index = train_index.loc[train_index["chunk"].isin(selected_chunks)]
    train_rows = [
        {
            "chunk": int(row.chunk),
            "clip_id": str(clip_id),
            "split": "train",
            "t0_relative": args.t0_relative,
        }
        for clip_id, row in selected_index.iterrows()
    ]
    train_rows.sort(key=lambda row: stable_key(args.seed, f"clip:{row['clip_id']}"))

    val_rows = load_fixed_eval_rows(args.validation_manifest, "val", clip_index)
    test_rows = load_fixed_eval_rows(args.test_manifest, "test", clip_index)
    split_clip_ids = {
        "train": {row["clip_id"] for row in train_rows},
        "val": {row["clip_id"] for row in val_rows},
        "test": {row["clip_id"] for row in test_rows},
    }
    overlaps = {
        "train_val": len(split_clip_ids["train"] & split_clip_ids["val"]),
        "train_test": len(split_clip_ids["train"] & split_clip_ids["test"]),
        "val_test": len(split_clip_ids["val"] & split_clip_ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Clip-level split overlap: {overlaps}")

    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "train.json", train_rows)
    write_json(args.output_dir / "val.json", val_rows)
    write_json(args.output_dir / "test.json", test_rows)
    write_json(args.output_dir / "shard_inventory.json", selected_inventory)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "route-less Alpamayo 1.5 Stage-2 HF on-demand shortcut training",
        "repo_id": REPO_ID,
        "revision": args.revision,
        "access_mode": "hf_stream",
        "navigation_conditioning": False,
        "nav_text_rows": 0,
        "selection_seed": args.seed,
        "t0_relative": args.t0_relative,
        "required_features": required_features,
        "target_source_bytes": target_bytes,
        "selected_source_bytes": selected_bytes,
        "selected_source_gb_decimal": selected_bytes / 1_000_000_000,
        "selected_source_gib": selected_bytes / (1024**3),
        "selected_chunks": selected_chunks,
        "selected_chunk_count": len(selected_chunks),
        "excluded_local_chunks": sorted(excluded),
        "annotation_rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "unique_clips": {
            "train": len(split_clip_ids["train"]),
            "val": len(split_clip_ids["val"]),
            "test": len(split_clip_ids["test"]),
        },
        "overlap_counts": overlaps,
        "fixed_eval_manifest_sha256": {
            "val": file_sha256(args.validation_manifest),
            "test": file_sha256(args.test_manifest),
        },
        "measurement": (
            "Sum of the exact Hugging Face file sizes for egomotion and the four "
            "camera ZIP archives used by Alpamayo Stage-2 in every selected chunk"
        ),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
