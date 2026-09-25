#!/usr/bin/env python3
"""Validate a saved reference-partition shortcut checkpoint with an EMA teacher."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


EMA_UPDATES = "shortcut_ema_updates"
STUDENT_TO_EMA_PREFIX = {
    "action_in_proj.": "ema_action_in_proj.",
    "expert.": "ema_expert.",
    "action_out_proj.": "ema_action_out_proj.",
}
OBJECTIVE_RE = re.compile(
    r"Shortcut objective: .*teacher=(?P<teacher>online|ema), "
    r"ema_updates=(?P<updates>\d+)"
)
PROBE_KEYS = (
    "action_in_proj.step_size_adapter.2.weight",
    "action_out_proj.weight",
    "expert.layers.0.self_attn.q_proj.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-decay", type=float, default=0.999)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def ema_key_for(student_key: str) -> str | None:
    for student_prefix, ema_prefix in STUDENT_TO_EMA_PREFIX.items():
        if student_key.startswith(student_prefix):
            return ema_prefix + student_key.removeprefix(student_prefix)
    return None


def load_tensor(
    checkpoint: Path,
    weight_map: dict[str, str],
    key: str,
) -> torch.Tensor:
    shard = checkpoint / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def pair_stats(student: torch.Tensor, ema: torch.Tensor) -> dict[str, Any]:
    student_float = student.float()
    ema_float = ema.float()
    difference = student_float - ema_float
    return {
        "shape": list(student.shape),
        "student_dtype": str(student.dtype),
        "ema_dtype": str(ema.dtype),
        "student_finite": bool(torch.isfinite(student_float).all()),
        "ema_finite": bool(torch.isfinite(ema_float).all()),
        "student_l2_norm": float(torch.linalg.vector_norm(student_float)),
        "ema_l2_norm": float(torch.linalg.vector_norm(ema_float)),
        "student_ema_difference_l2_norm": float(
            torch.linalg.vector_norm(difference)
        ),
        "student_ema_difference_nonzero": int(torch.count_nonzero(difference)),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    required = [
        "config.json",
        "model.safetensors.index.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    rng_files = sorted(path.name for path in checkpoint.glob("rng_state*.pth"))
    if not rng_files:
        missing.append("rng_state*.pth")

    config = load_json(checkpoint / "config.json")
    trainer_state = load_json(checkpoint / "trainer_state.json")
    index = load_json(checkpoint / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    shard_paths = [checkpoint / name for name in sorted(set(weight_map.values()))]

    tensor_count = 0
    ema_dtype_counts: Counter[str] = Counter()
    for shard_path in shard_paths:
        if not shard_path.is_file():
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensor_count += len(keys)
            for key in keys:
                if key.startswith(tuple(STUDENT_TO_EMA_PREFIX.values())):
                    ema_dtype_counts[handle.get_slice(key).get_dtype()] += 1

    expected_pairs = {
        student_key: ema_key
        for student_key in weight_map
        if (ema_key := ema_key_for(student_key)) is not None
    }
    missing_ema_keys = sorted(
        ema_key for ema_key in expected_pairs.values() if ema_key not in weight_map
    )
    unexpected_ema_keys = sorted(
        key
        for key in weight_map
        if key.startswith(tuple(STUDENT_TO_EMA_PREFIX.values()))
        and key not in expected_pairs.values()
    )

    ema_updates = None
    if EMA_UPDATES in weight_map:
        ema_updates = int(load_tensor(checkpoint, weight_map, EMA_UPDATES).item())

    probe_results: dict[str, dict[str, Any]] = {}
    for student_key in PROBE_KEYS:
        ema_key = ema_key_for(student_key)
        if (
            ema_key is None
            or student_key not in weight_map
            or ema_key not in weight_map
        ):
            continue
        probe_results[student_key] = pair_stats(
            load_tensor(checkpoint, weight_map, student_key),
            load_tensor(checkpoint, weight_map, ema_key),
        )

    objective_entries = []
    for line in args.train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Shortcut objective:" not in line:
            continue
        match = OBJECTIVE_RE.search(line)
        if match is None:
            raise ValueError(f"Malformed shortcut objective log line: {line}")
        objective_entries.append(match.groupdict())
    logged_updates = [int(entry["updates"]) for entry in objective_entries]

    probes_finite = bool(probe_results) and all(
        item["student_finite"] and item["ema_finite"]
        for item in probe_results.values()
    )
    probes_diverged = all(
        item["student_ema_difference_nonzero"] > 0
        for item in probe_results.values()
    )
    probes_use_configured_dtype = bool(probe_results) and all(
        item["ema_dtype"] == "torch.float32"
        for item in probe_results.values()
    )
    checks = {
        "required_files_present": not missing,
        "all_model_shards_present": all(path.is_file() for path in shard_paths),
        "weight_index_complete": tensor_count == len(weight_map),
        "global_step": int(trainer_state.get("global_step", -1))
        == args.expected_steps,
        "ema_teacher_configured": config.get("shortcut_teacher_mode") == "ema",
        "ema_inference_configured": config.get("shortcut_inference_weights") == "ema",
        "ema_float32_configured": config.get("shortcut_ema_dtype") == "float32",
        "ema_decay": math.isclose(
            float(config.get("shortcut_ema_decay", math.nan)),
            args.expected_decay,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "ema_update_counter_present": ema_updates is not None,
        "ema_update_counter": ema_updates == args.expected_steps,
        "all_student_tensors_have_ema_pairs": bool(expected_pairs)
        and not missing_ema_keys,
        "no_unpaired_ema_tensors": not unexpected_ema_keys,
        "all_ema_tensors_are_float32": ema_dtype_counts == {"F32": len(expected_pairs)},
        "probe_pairs_present": len(probe_results) == len(PROBE_KEYS),
        "probe_pairs_finite": probes_finite,
        "probe_ema_tensors_are_float32": probes_use_configured_dtype,
        "probe_pairs_changed_after_training": probes_diverged,
        "objective_log_count": len(objective_entries) == args.expected_steps,
        "objective_logs_use_ema": all(
            entry["teacher"] == "ema" for entry in objective_entries
        ),
        "logged_update_sequence": logged_updates == list(range(args.expected_steps)),
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checkpoint": str(checkpoint),
        "checks": checks,
        "missing_required_files": missing,
        "rng_files": rng_files,
        "global_step": trainer_state.get("global_step"),
        "ema_updates": ema_updates,
        "student_ema_tensor_pairs": len(expected_pairs),
        "ema_tensor_dtype_counts": dict(sorted(ema_dtype_counts.items())),
        "missing_ema_keys": missing_ema_keys,
        "unexpected_ema_keys": unexpected_ema_keys,
        "probe_parameter_stats": probe_results,
        "logged_updates_before_each_optimizer_step": logged_updates,
        "config": {
            "shortcut_teacher_mode": config.get("shortcut_teacher_mode"),
            "shortcut_ema_decay": config.get("shortcut_ema_decay"),
            "shortcut_ema_dtype": config.get("shortcut_ema_dtype"),
            "shortcut_inference_weights": config.get("shortcut_inference_weights"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
