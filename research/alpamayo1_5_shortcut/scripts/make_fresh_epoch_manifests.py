#!/usr/bin/env python3
"""Select two new train sets after A6's original set; only HF metadata is read."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from alpamayo1_5_sft.models.paper_data_plan import (
    BASE_MANIFEST_SHA, HF_REVISION, audit_phases, file_sha256,
    heldout_clip_ids, heldout_manifest_paths,
)

PROJECT = Path(__file__).resolve().parents[1]


def stable_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_phases(clip_index, feature_presence, required_features, first, heldout,
                  excluded_chunks, clips_per_phase=5295, seed=42):
    """Chunk-grouped deterministic selection with official split/feature checks."""
    audit_phases([first], heldout, clips_per_phase)
    used = {row["clip_id"] for row in first}
    for row in first:
        clip = row["clip_id"]
        official = clip_index.loc[clip]
        if (official["split"] != "train" or not bool(official["clip_is_valid"])
                or int(official["chunk"]) != row["chunk"]
                or not bool(feature_presence.loc[clip, required_features].all())):
            raise ValueError(f"Original A6 clip failed official metadata checks: {clip}")
    banned_chunks = set(excluded_chunks) | {row["chunk"] for row in first}
    candidates = clip_index.loc[clip_index["clip_is_valid"] & clip_index["split"].eq("train")]
    candidates = candidates.loc[feature_presence.loc[candidates.index, required_features].all(axis=1)]
    phases = [first]
    for phase in (2, 3):
        available = candidates.loc[~candidates["chunk"].isin(banned_chunks)]
        chunks = sorted((int(chunk) for chunk in available["chunk"].unique()),
                        key=lambda chunk: stable_key(seed, f"phase:{phase}:chunk:{chunk}"))
        rows = []
        for chunk in chunks:
            clips = sorted((str(clip) for clip in available.index[available["chunk"].eq(chunk)]
                            if clip not in used and clip not in heldout),
                           key=lambda clip: stable_key(seed, f"phase:{phase}:clip:{clip}"))
            take = clips[:clips_per_phase - len(rows)]
            if not take:
                continue
            rows.extend(dict(chunk=chunk, clip_id=clip, split="train", t0_relative=7_000_000) for clip in take)
            banned_chunks.add(chunk)
            if len(rows) == clips_per_phase:
                break
        if len(rows) != clips_per_phase:
            raise ValueError("Insufficient eligible, non-overlapping official training clips")
        rows.sort(key=lambda row: stable_key(seed, f"phase:{phase}:order:{row['clip_id']}"))
        phases.append(rows)
        used.update(row["clip_id"] for row in rows)
    audit_phases(phases, heldout, clips_per_phase)
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-heldout-manifests", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    from huggingface_hub import HfApi
    from physical_ai_av import PhysicalAIAVDatasetInterface
    from make_hf_streaming_manifests import FEATURE_PATHS, query_chunk_sizes, write_json

    first_path = PROJECT / "manifests/hf_stream_300gb/train.json"
    if file_sha256(first_path) != BASE_MANIFEST_SHA:
        raise ValueError("Original A6 manifest hash changed")
    first = json.loads(first_path.read_text())
    heldout_paths = sorted(set(heldout_manifest_paths(PROJECT / "manifests") + args.extra_heldout_manifests))
    heldout, evidence = heldout_clip_ids(heldout_paths)
    original_summary = json.loads((first_path.parent / "summary.json").read_text())
    official = PhysicalAIAVDatasetInterface(revision=HF_REVISION, cache_dir=args.cache_dir)
    # Include calibration / motion inputs, not just the large ZIP archives.
    required = list(FEATURE_PATHS) + ["camera_intrinsics", "sensor_extrinsics", "vehicle_dimensions"]
    phases = select_phases(official.clip_index, official.feature_presence, required, first, heldout,
                           original_summary["excluded_local_chunks"], seed=args.seed)
    audit = audit_phases(phases, heldout, 5295)
    chunks = sorted({row["chunk"] for phase in phases for row in phase})
    inventory, api = [], HfApi()
    for start in range(0, len(chunks), 32):
        inventory.extend(query_chunk_sizes(api, chunks=chunks[start:start + 32], revision=HF_REVISION))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Byte-for-byte preserve phase 1, so its hash and first 83 raw batches match A6.
    shutil.copyfile(first_path, args.output_dir / "phase_1.json")
    for number, rows in enumerate(phases[1:], 2):
        write_json(args.output_dir / f"phase_{number}.json", rows)
    flat = [row for phase in phases for row in phase]
    write_json(args.output_dir / "train.json", flat)
    write_json(args.output_dir / "shard_inventory.json", inventory)
    schedule = dict(schema_version=1, experiment="A8_fresh_data_A6", revision=HF_REVISION,
                    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", selection_seed=args.seed,
                    generated_utc=datetime.now(timezone.utc).isoformat(), clips_per_phase=5295,
                    phases=[dict(manifest=f"phase_{i}.json", sha256=file_sha256(args.output_dir / f"phase_{i}.json")) for i in (1, 2, 3)],
                    train_manifest_sha256=file_sha256(args.output_dir / "train.json"),
                    heldout_clip_ids=sorted(heldout), heldout_manifests=evidence, audit=audit,
                    excluded_local_chunks=original_summary["excluded_local_chunks"],
                    required_features=required, selected_chunks=chunks,
                    selected_source_bytes=sum(item["bytes"] for item in inventory),
                    measurement="Source ZIP coverage only, NOT bytes downloaded; metadata validated, payloads not exhaustively decoded")
    write_json(args.output_dir / "schedule.json", schedule)
    # Per-phase summaries reuse the existing real-sample validator interface.
    for number, rows in enumerate(phases, 1):
        phase_chunks = sorted({row["chunk"] for row in rows})
        write_json(args.output_dir / f"phase_{number}_summary.json", dict(
            revision=HF_REVISION, repo_id=schedule["repo_id"], selected_chunks=phase_chunks,
            annotation_rows=dict(train=len(rows)),
            selected_source_bytes=sum(item["bytes"] for item in inventory if item["chunk"] in phase_chunks)))
    print(json.dumps(dict(output=str(args.output_dir), **audit,
                          source_gb=schedule["selected_source_bytes"] / 1e9), indent=2))


if __name__ == "__main__":
    main()
