#!/usr/bin/env python3
"""Validate a matched flow-only Alpamayo Stage-2 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

TRAIN_LOG_RE = re.compile(
    r"\{'loss': (?P<loss>[-+0-9.eE]+), 'grad_norm': (?P<grad>[-+0-9.eE]+),"
)
FORBIDDEN_WEIGHT_PREFIXES = (
    "action_in_proj.step_size_adapter.",
    "shortcut_level_sampler.",
    "shortcut_branch_sampler.",
)


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


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
    indexed_weight_count = 0
    forbidden_weights = []
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            indexed_weight_count += len(keys)
            forbidden_weights.extend(
                key for key in keys if key.startswith(FORBIDDEN_WEIGHT_PREFIXES)
            )

    log_matches = [
        match.groupdict()
        for match in TRAIN_LOG_RE.finditer(
            args.train_log.read_text(encoding="utf-8", errors="replace")
        )
    ]
    losses = [float(match["loss"]) for match in log_matches]
    grad_norms = [float(match["grad"]) for match in log_matches]
    config = load_json(checkpoint / "config.json")
    trainer_state = load_json(checkpoint / "trainer_state.json")
    global_samples = args.expected_steps * args.world_size

    checks = {
        "required_files_present": not missing,
        "all_model_shards_present": all(path.is_file() for path in shard_paths),
        "weight_index_complete": indexed_weight_count == len(weight_map),
        "global_step": int(trainer_state.get("global_step", -1)) == args.expected_steps,
        "one_log_per_optimizer_step": len(log_matches) == args.expected_steps,
        "losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
        "grad_norms_finite": bool(grad_norms)
        and all(math.isfinite(value) for value in grad_norms),
        "base_expert_architecture": config.get("architectures")
        == ["TrainableAlpamayoR1"],
        "no_shortcut_or_adapter_weights": not forbidden_weights,
        "one_epoch_sample_accounting": global_samples >= args.training_rows
        and global_samples - args.training_rows < args.world_size,
        "one_rng_state_per_rank": len(rng_files) == args.world_size,
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checkpoint": str(checkpoint),
        "checks": checks,
        "missing_required_files": missing,
        "forbidden_weights": forbidden_weights,
        "model_architectures": config.get("architectures"),
        "global_step": trainer_state.get("global_step"),
        "world_size": args.world_size,
        "training_rows": args.training_rows,
        "global_samples_after_distributed_padding": global_samples,
        "logged_optimizer_steps": len(log_matches),
        "mean_logged_loss": sum(losses) / len(losses) if losses else None,
        "mean_logged_grad_norm": (
            sum(grad_norms) / len(grad_norms) if grad_norms else None
        ),
        "rng_files": rng_files,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
