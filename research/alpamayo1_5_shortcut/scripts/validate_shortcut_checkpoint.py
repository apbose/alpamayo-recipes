#!/usr/bin/env python3
"""Validate a resumable Alpamayo 1.5 shortcut-training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


OBJECTIVE_RE = re.compile(
    r"flow_loss=(?P<flow>[-+0-9.eE]+), "
    r"shortcut_loss=(?P<shortcut>[-+0-9.eE]+), "
    r"d_mean=(?P<step>[-+0-9.eE]+), "
    r"level_indices=\[(?P<level>\d+)\], "
    r"cycle_cursor=(?P<cursor>\d+)"
)
ADAPTER_PREFIX = "action_in_proj.step_size_adapter."
SAMPLER_CURSOR = "shortcut_level_sampler.cursor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "l2_norm": float(torch.linalg.vector_norm(values)),
        "max_abs": float(values.abs().max()) if values.numel() else 0.0,
        "nonzero": int(torch.count_nonzero(values)),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    required = [
        "config.json",
        "model.safetensors.index.json",
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]

    index = load_json(checkpoint / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shard_paths = [checkpoint / name for name in shard_names]
    all_shards_present = all(path.is_file() for path in shard_paths)

    tensor_count = 0
    tensors: dict[str, torch.Tensor] = {}
    shard_tensor_counts: dict[str, int] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            shard_tensor_counts[shard_path.name] = len(keys)
            tensor_count += len(keys)
            for key in keys:
                if key.startswith(ADAPTER_PREFIX) or key == SAMPLER_CURSOR:
                    tensors[key] = handle.get_tensor(key)

    log_matches = [
        OBJECTIVE_RE.search(line)
        for line in args.train_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if "Shortcut objective:" in line
    ]
    if any(match is None for match in log_matches):
        raise ValueError("At least one shortcut-objective log line was malformed")
    parsed = [match.groupdict() for match in log_matches if match is not None]
    flow_losses = [float(item["flow"]) for item in parsed]
    shortcut_losses = [float(item["shortcut"]) for item in parsed]
    observed = [
        (float(item["step"]), int(item["level"]), int(item["cursor"]))
        for item in parsed
    ]
    expected_cycle = [(0.25, 0, 1), (0.5, 1, 2), (1.0, 2, 0)]
    expected = [expected_cycle[index % 3] for index in range(args.expected_steps)]
    coverage = Counter(f"{step:.6f}" for step, _, _ in observed)

    trainer_state = load_json(checkpoint / "trainer_state.json")
    config = load_json(checkpoint / "config.json")
    cursor_tensor = tensors.get(SAMPLER_CURSOR)
    cursor_value = int(cursor_tensor.item()) if cursor_tensor is not None else None
    adapter_stats = {
        name: tensor_stats(tensor)
        for name, tensor in sorted(tensors.items())
        if name.startswith(ADAPTER_PREFIX)
    }
    finite = all(math.isfinite(value) for value in flow_losses + shortcut_losses)
    expected_cursor = args.expected_steps % len(expected_cycle)
    passed = all(
        [
            not missing,
            all_shards_present,
            tensor_count == len(weight_map),
            int(trainer_state.get("global_step", -1)) == args.expected_steps,
            len(observed) == args.expected_steps,
            observed == expected,
            finite,
            cursor_value == expected_cursor,
            len(adapter_stats) == 4,
            (checkpoint / "optimizer.pt").stat().st_size > 0,
            (checkpoint / "scheduler.pt").stat().st_size > 0,
            (checkpoint / "rng_state.pth").stat().st_size > 0,
        ]
    )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checkpoint": str(checkpoint),
        "checkpoint_total_bytes": sum(
            path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
        ),
        "missing_required_files": missing,
        "all_shards_present": all_shards_present,
        "model_shards": shard_names,
        "shard_tensor_counts": shard_tensor_counts,
        "model_tensor_count": tensor_count,
        "weight_map_tensor_count": len(weight_map),
        "global_step": trainer_state.get("global_step"),
        "training_log_steps": len(observed),
        "training_losses_finite": finite,
        "flow_loss_mean": sum(flow_losses) / len(flow_losses),
        "shortcut_loss_mean": sum(shortcut_losses) / len(shortcut_losses),
        "sampled_shortcut_step_size_counts": dict(sorted(coverage.items())),
        "balanced_cycle_exact": observed == expected,
        "saved_sampler_cursor": cursor_value,
        "expected_sampler_cursor": expected_cursor,
        "adapter_parameter_stats": adapter_stats,
        "config": {
            "shortcut_flow_step_size": config.get("shortcut_flow_step_size"),
            "shortcut_step_sizes": config.get("shortcut_step_sizes"),
            "shortcut_level_sampling": config.get("shortcut_level_sampling"),
            "shortcut_loss_weight": config.get("shortcut_loss_weight"),
            "shortcut_teacher_mode": config.get("shortcut_teacher_mode"),
        },
        "optimizer_bytes": (checkpoint / "optimizer.pt").stat().st_size,
        "scheduler_present": (checkpoint / "scheduler.pt").is_file(),
        "rng_state_present": (checkpoint / "rng_state.pth").is_file(),
        "sha256": {
            name: sha256(checkpoint / name)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "trainer_state.json",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
