#!/usr/bin/env python3
"""Turn Alpamayo shortcut objective logs into a compact learning-curve artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


OBJECTIVE_RE = re.compile(
    r"flow_loss=(?P<flow>[-+0-9.eE]+), "
    r"shortcut_loss=(?P<shortcut>[-+0-9.eE]+), "
    r"d_mean=(?P<step_size>[-+0-9.eE]+), "
    r"level_indices=\[(?P<level>\d+)\], "
    r"cycle_cursor=(?P<cursor>\d+)"
)
TRAINER_RE = re.compile(
    r"\{'loss': (?P<loss>[-+0-9.eE]+), "
    r"'grad_norm': (?P<grad_norm>[-+0-9.eE]+), "
    r"'learning_rate': (?P<learning_rate>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--shortcut-loss-weight", type=float, default=0.125)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "flow_loss_mean": mean([row["flow_loss"] for row in rows]),
        "shortcut_loss_mean": mean([row["shortcut_loss"] for row in rows]),
        "weighted_objective_mean": mean([row["weighted_objective"] for row in rows]),
        "trainer_loss_mean": mean([row["trainer_loss"] for row in rows]),
        "grad_norm_mean": mean([row["grad_norm"] for row in rows]),
        "learning_rate_mean": mean([row["learning_rate"] for row in rows]),
    }


def main() -> None:
    args = parse_args()
    if args.expected_steps <= 0:
        raise ValueError("expected-steps must be positive")
    if not 0.0 <= args.shortcut_loss_weight <= 1.0:
        raise ValueError("shortcut-loss-weight must be in [0, 1]")

    lines = args.train_log.read_text(encoding="utf-8", errors="replace").splitlines()
    objectives = [
        match.groupdict()
        for line in lines
        if "Shortcut objective:" in line
        for match in [OBJECTIVE_RE.search(line)]
        if match is not None
    ]
    trainer_records = [
        match.groupdict()
        for line in lines
        for match in [TRAINER_RE.search(line)]
        if match is not None
    ]
    if len(objectives) != args.expected_steps:
        raise ValueError(
            f"Found {len(objectives)} objective records; expected {args.expected_steps}"
        )
    if len(trainer_records) != args.expected_steps:
        raise ValueError(
            f"Found {len(trainer_records)} trainer records; expected {args.expected_steps}"
        )

    weight = args.shortcut_loss_weight
    rows: list[dict[str, Any]] = []
    for index, (objective, trainer) in enumerate(
        zip(objectives, trainer_records, strict=True), start=1
    ):
        flow_loss = float(objective["flow"])
        shortcut_loss = float(objective["shortcut"])
        row = {
            "training_step": index,
            "shortcut_step_size": float(objective["step_size"]),
            "level_index": int(objective["level"]),
            "cycle_cursor": int(objective["cursor"]),
            "flow_loss": flow_loss,
            "shortcut_loss": shortcut_loss,
            "weighted_objective": (1.0 - weight) * flow_loss + weight * shortcut_loss,
            "trainer_loss": float(trainer["loss"]),
            "grad_norm": float(trainer["grad_norm"]),
            "learning_rate": float(trainer["learning_rate"]),
        }
        rows.append(row)

    numeric_values = [
        float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"training_step", "level_index", "cycle_cursor"}
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Training log contains a non-finite numeric value")

    window = min(100, max(1, len(rows) // 10))
    decile_size = math.ceil(len(rows) / 10)
    deciles = []
    for start in range(0, len(rows), decile_size):
        part = rows[start : start + decile_size]
        deciles.append(
            {
                "start_step": part[0]["training_step"],
                "end_step": part[-1]["training_step"],
                **aggregate(part),
            }
        )

    rows_by_step_size: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_step_size[f"{row['shortcut_step_size']:.6f}"].append(row)
    per_step_size = {
        key: {"count": len(values), **aggregate(values)}
        for key, values in sorted(rows_by_step_size.items())
    }
    expected_cycle = [(0.25, 0, 1), (0.5, 1, 2), (1.0, 2, 0)]
    observed_cycle = [
        (row["shortcut_step_size"], row["level_index"], row["cycle_cursor"])
        for row in rows
    ]
    exact_cycle = observed_cycle == [
        expected_cycle[index % len(expected_cycle)] for index in range(len(rows))
    ]

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_log": str(args.train_log.resolve()),
        "training_steps": len(rows),
        "all_values_finite": True,
        "balanced_cycle_exact": exact_cycle,
        "shortcut_step_size_counts": dict(
            Counter(f"{row['shortcut_step_size']:.6f}" for row in rows)
        ),
        "shortcut_loss_weight": weight,
        "comparison_window_steps": window,
        "first_window": aggregate(rows[:window]),
        "last_window": aggregate(rows[-window:]),
        "relative_change_first_to_last": {
            key: aggregate(rows[-window:])[key] / aggregate(rows[:window])[key] - 1.0
            for key in ("flow_loss_mean", "shortcut_loss_mean", "weighted_objective_mean")
        },
        "per_shortcut_step_size": per_step_size,
        "deciles": deciles,
    }
    if not exact_cycle:
        raise ValueError("Training log does not follow the expected balanced cycle")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
