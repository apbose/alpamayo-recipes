#!/usr/bin/env python3
"""Validate a saved reference-partition Alpamayo shortcut checkpoint."""

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

OBJECTIVE_RE = re.compile(
    r"estimator=reference_partition, branch=(?P<branch>shortcut|flow), "
    r"flow_loss=(?P<flow>none|[-+0-9.eE]+), "
    r"shortcut_loss=(?P<shortcut>none|[-+0-9.eE]+), "
    r"d_mean=(?P<step>none|[-+0-9.eE]+), "
    r"level_indices=(?P<levels>None|\[[0-9, ]+\]), "
    r"level_cursor=(?P<level_cursor>\d+), branch_cursor=(?P<branch_cursor>\d+)"
)
ADAPTER_PREFIX = "action_in_proj.step_size_adapter."
LEVEL_CURSOR = "shortcut_level_sampler.cursor"
BRANCH_CURSOR = "shortcut_branch_sampler.cursor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--training-rows", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(values).all()),
        "l2_norm": float(torch.linalg.vector_norm(values)),
        "nonzero": int(torch.count_nonzero(values)),
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

    index = load_json(checkpoint / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shard_paths = [checkpoint / name for name in shard_names]
    selected_tensors: dict[str, torch.Tensor] = {}
    tensor_count = 0
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            tensor_count += len(keys)
            for key in keys:
                if key.startswith(ADAPTER_PREFIX) or key in {
                    LEVEL_CURSOR,
                    BRANCH_CURSOR,
                }:
                    selected_tensors[key] = handle.get_tensor(key)

    matches = []
    for line in args.train_log.read_text(errors="replace").splitlines():
        if "Shortcut objective:" not in line:
            continue
        match = OBJECTIVE_RE.search(line)
        if match is None:
            raise ValueError(f"Malformed shortcut objective log line: {line}")
        matches.append(match.groupdict())

    expected_shortcut_logs = args.expected_steps
    observed_steps = [float(item["step"]) for item in matches]
    expected_steps = [
        (0.25, 0.5, 1.0)[index % 3] for index in range(args.expected_steps)
    ]
    shortcut_losses = [float(item["shortcut"]) for item in matches]
    level_cursor = selected_tensors.get(LEVEL_CURSOR)
    branch_cursor = selected_tensors.get(BRANCH_CURSOR)
    adapter_stats = {
        name: tensor_stats(tensor)
        for name, tensor in selected_tensors.items()
        if name.startswith(ADAPTER_PREFIX)
    }
    config = load_json(checkpoint / "config.json")
    trainer_state = load_json(checkpoint / "trainer_state.json")
    global_samples = args.expected_steps * args.world_size
    expected_shortcut_samples = global_samples // int(
        config["shortcut_bootstrap_every"]
    )
    expected_flow_samples = global_samples - expected_shortcut_samples

    checks = {
        "required_files_present": not missing,
        "all_model_shards_present": all(path.is_file() for path in shard_paths),
        "weight_index_complete": tensor_count == len(weight_map),
        "global_step": int(trainer_state.get("global_step", -1)) == args.expected_steps,
        "reference_estimator": config.get("shortcut_loss_estimator")
        == "reference_partition",
        "bootstrap_every_eight": int(config.get("shortcut_bootstrap_every", -1)) == 8,
        "rank_zero_shortcut_log_count": len(matches) == expected_shortcut_logs,
        "rank_zero_only_shortcut": all(
            item["branch"] == "shortcut" for item in matches
        ),
        "rank_zero_has_no_flow_loss": all(item["flow"] == "none" for item in matches),
        "shortcut_losses_finite": bool(shortcut_losses)
        and all(math.isfinite(value) for value in shortcut_losses),
        "balanced_level_cycle": observed_steps == expected_steps,
        "saved_level_cursor": level_cursor is not None
        and int(level_cursor.item()) == args.expected_steps % 3,
        "saved_branch_cursor": branch_cursor is not None
        and int(branch_cursor.item()) == global_samples % 8,
        "adapter_tensors_present": len(adapter_stats) == 4,
        "adapter_tensors_finite": bool(adapter_stats)
        and all(stats["finite"] for stats in adapter_stats.values()),
        "adapter_output_layer_updated": adapter_stats.get(
            "action_in_proj.step_size_adapter.2.weight", {}
        ).get("nonzero", 0)
        > 0,
        "one_epoch_sample_accounting": global_samples >= args.training_rows
        and global_samples - args.training_rows < args.world_size,
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
        "world_size": args.world_size,
        "training_rows": args.training_rows,
        "global_samples_after_distributed_padding": global_samples,
        "expected_branch_allocation": {
            "shortcut": expected_shortcut_samples,
            "flow": expected_flow_samples,
            "shortcut_fraction": expected_shortcut_samples / global_samples,
        },
        "rank_zero_shortcut_logs": len(matches),
        "shortcut_loss_mean": sum(shortcut_losses) / len(shortcut_losses),
        "shortcut_step_counts": dict(
            Counter(f"{value:.2f}" for value in observed_steps)
        ),
        "adapter_parameter_stats": adapter_stats,
        "saved_level_cursor": None
        if level_cursor is None
        else int(level_cursor.item()),
        "saved_branch_cursor": None
        if branch_cursor is None
        else int(branch_cursor.item()),
        "config": {
            "shortcut_loss_estimator": config.get("shortcut_loss_estimator"),
            "shortcut_bootstrap_every": config.get("shortcut_bootstrap_every"),
            "shortcut_teacher_mode": config.get("shortcut_teacher_mode"),
            "shortcut_flow_step_size": config.get("shortcut_flow_step_size"),
            "shortcut_step_sizes": config.get("shortcut_step_sizes"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
