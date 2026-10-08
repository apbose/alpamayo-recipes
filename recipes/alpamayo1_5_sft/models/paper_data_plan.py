# SPDX-License-Identifier: Apache-2.0
"""Audited, clip-disjoint fresh-data phases for the existing A6 target layout."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .paper_shortcut_targets import raw_batch_plan

HF_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
BASE_MANIFEST_SHA = "21e94e04f441bceebb8c31159b660ff012669b7e2cbf2fdbb45243da00d1da5e"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heldout_manifest_paths(root):
    """Include historical nav, 32/128-clip and all-timestamp held-out sets."""
    return sorted(path for path in Path(root).rglob("*.json")
                  if path.stem in {"val", "test", "nav_val", "nav_test"}
                  or path.stem.startswith(("val_", "test_")))


def heldout_clip_ids(paths):
    ids, evidence = set(), []
    for path in paths:
        path = Path(path)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Empty or malformed held-out manifest: {path}")
        current = {row["clip_id"] for row in rows}
        if any(not isinstance(clip, str) or not clip for clip in current):
            raise ValueError(f"Invalid held-out clip ID in {path}")
        ids.update(current)
        evidence.append(dict(path=str(path), sha256=file_sha256(path), unique_clips=len(current)))
    if not ids:
        raise ValueError("No held-out clips supplied; refusing an unchecked training split")
    return ids, evidence


def validate_training_rows(rows, expected_size):
    if not isinstance(rows, list) or len(rows) != expected_size:
        raise ValueError(f"Expected {expected_size} rows per fresh-data phase")
    ids = set()
    for row in rows:
        clip = row.get("clip_id")
        t0 = row.get("t0_relative")
        if not isinstance(clip, str) or not clip or clip in ids:
            raise ValueError("Duplicate or invalid training clip ID")
        if type(t0) is not int or not 1_600_000 < t0 <= 13_600_000:
            raise ValueError("Invalid training timestamp/history/future margin")
        if row.get("split") != "train" or "nav_text" in row:
            raise ValueError("Fresh-data phases require route-less official-training rows")
        if type(row.get("chunk")) is not int or row["chunk"] < 0:
            raise ValueError("Invalid training chunk")
        ids.add(clip)
    return ids


def audit_phases(phases, heldout, expected_size):
    """Reject overlap by clip ID, even if timestamps differ."""
    seen, counts = set(), []
    for index, rows in enumerate(phases, 1):
        ids = validate_training_rows(rows, expected_size)
        if ids & heldout:
            raise ValueError(f"Phase {index} overlaps held-out clips: {len(ids & heldout)}")
        if ids & seen:
            raise ValueError(f"Phase {index} overlaps earlier training phases")
        seen.update(ids)
        counts.append(len(ids))
    return dict(phase_unique_clips=counts, total_unique_clips=len(seen),
                heldout_unique_clips=len(heldout), train_heldout_overlap=0,
                cross_phase_overlap=0, overlap_unit="clip_id, not timestamp")


def fresh_batch_plan(phase_sizes, batch_size=64, seed=10):
    """Ceil-sized phases, padding only from the current phase; never cross phases.

    The first phase uses the exact first ceil(N/B) A6 batches. Subsequent
    phases have new rows and independent deterministic shuffle seeds.
    """
    if not phase_sizes or batch_size <= 0 or any(size <= 0 for size in phase_sizes):
        raise ValueError("Phase sizes and batch size must be positive")
    plan, update_phases, offset = [], [], 0
    for phase, size in enumerate(phase_sizes):
        updates = math.ceil(size / batch_size)
        local = raw_batch_plan(size, batch_size, updates, seed + phase)
        plan.extend([[offset + item for item in batch] for batch in local])
        update_phases.extend([phase + 1] * updates)
        offset += size
    return plan, update_phases


def load_fresh_schedule(path, heldout_paths):
    """Verify hashes, original phase one, phase separation and live holdouts."""
    path = Path(path)
    schedule = json.loads(path.read_text())
    if (schedule.get("schema_version") != 1 or schedule.get("revision") != HF_REVISION
            or schedule.get("clips_per_phase") != 5295 or len(schedule.get("phases", [])) != 3):
        raise ValueError("Expected the pinned three-phase, 5,295-clips/phase protocol")
    heldout, evidence = heldout_clip_ids(heldout_paths)
    saved_heldout = set(schedule["heldout_clip_ids"])
    if not saved_heldout:
        raise ValueError("Schedule has no recorded holdouts")
    heldout.update(saved_heldout)
    phases = []
    for entry in schedule["phases"]:
        # Bundles must be self-contained; disallow path traversal.
        relative = Path(entry["manifest"])
        if relative.name != entry["manifest"] or relative.is_absolute():
            raise ValueError("Phase manifest must be a bundle-local filename")
        phase_path = path.parent / relative
        if file_sha256(phase_path) != entry["sha256"]:
            raise ValueError(f"Phase manifest hash changed: {relative}")
        phases.append(json.loads(phase_path.read_text()))
    if schedule["phases"][0]["sha256"] != BASE_MANIFEST_SHA:
        raise ValueError("Phase one must preserve the original A6 training manifest")
    audit = audit_phases(phases, heldout, 5295)
    combined = path.parent / "train.json"
    rows = [row for phase in phases for row in phase]
    if file_sha256(combined) != schedule["train_manifest_sha256"] or json.loads(combined.read_text()) != rows:
        raise ValueError("Combined manifest differs from its ordered phases")
    return combined, rows, dict(audit, live_heldout_manifests=evidence,
                                schedule_sha256=file_sha256(path)), [len(phase) for phase in phases]
