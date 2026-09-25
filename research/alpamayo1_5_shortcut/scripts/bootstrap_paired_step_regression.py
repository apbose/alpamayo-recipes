#!/usr/bin/env python3
"""Paired clip bootstrap for two solver counts from one benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("min_ade", "ade", "corner_distance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reference-step", default="10")
    parser.add_argument("--candidate-step", default="2")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--relative-threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def paired_values(
    result: dict[str, Any], reference_step: str, candidate_step: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    def indexed(step: str) -> dict[int, float]:
        values = {}
        for row in result["results"][step]["per_sample"]:
            metric_values = row["metrics"][metric]
            if len(metric_values) != 1:
                raise ValueError(f"Expected one {metric} value per clip")
            values[int(row["sample_index"])] = float(metric_values[0])
        return values

    reference = indexed(reference_step)
    candidate = indexed(candidate_step)
    if reference.keys() != candidate.keys():
        raise ValueError("Solver-step results do not contain the same sample indices")
    indices = sorted(reference)
    return (
        np.asarray([reference[index] for index in indices], dtype=np.float64),
        np.asarray([candidate[index] for index in indices], dtype=np.float64),
    )


def bootstrap_metric(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if len(reference) < 2:
        raise ValueError("paired bootstrap requires at least two clips")
    rng = np.random.default_rng(seed)
    relative_changes = np.empty(iterations, dtype=np.float64)
    mean_deltas = np.empty(iterations, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, len(reference), size=(stop - start, len(reference)))
        reference_means = reference[indices].mean(axis=1)
        candidate_means = candidate[indices].mean(axis=1)
        mean_deltas[start:stop] = candidate_means - reference_means
        relative_changes[start:stop] = candidate_means / reference_means - 1.0

    reference_mean = float(reference.mean())
    candidate_mean = float(candidate.mean())
    observed_relative = candidate_mean / reference_mean - 1.0
    return {
        "paired_clips": len(reference),
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "observed_mean_delta": candidate_mean - reference_mean,
        "observed_relative_change": observed_relative,
        "mean_delta_95_ci": [
            float(value) for value in np.quantile(mean_deltas, [0.025, 0.975])
        ],
        "relative_change_95_ci": [
            float(value) for value in np.quantile(relative_changes, [0.025, 0.975])
        ],
        "probability_of_regression": float(np.mean(relative_changes > 0.0)),
        "probability_regression_exceeds_threshold": float(
            np.mean(relative_changes > threshold)
        ),
        "threshold": threshold,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paired solver-step bootstrap",
        "",
        f"- Reference step count: {report['reference_step']}",
        f"- Candidate step count: {report['candidate_step']}",
        f"- Paired clips: {report['paired_clips']}",
        f"- Bootstrap iterations: {report['iterations']:,}",
        "",
        "| Metric | Reference | Candidate | Relative change | 95% CI | P(change > 10%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, values in report["metrics"].items():
        interval = values["relative_change_95_ci"]
        lines.append(
            f"| {metric} | {values['reference_mean']:.4f} | "
            f"{values['candidate_mean']:.4f} | "
            f"{100 * values['observed_relative_change']:+.2f}% | "
            f"[{100 * interval[0]:+.2f}%, {100 * interval[1]:+.2f}%] | "
            f"{values['probability_regression_exceeds_threshold']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The interval resamples paired clips. It quantifies this fixed "
                "open-loop sample only; it is not a vehicle-safety confidence interval."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark.resolve()
    result = load_json(benchmark)
    for step in (args.reference_step, args.candidate_step):
        if step not in result.get("results", {}):
            raise ValueError(f"Benchmark has no {step}-step result")

    metrics = {}
    paired_clips = None
    for metric_index, metric in enumerate(METRICS):
        reference, candidate = paired_values(
            result, args.reference_step, args.candidate_step, metric
        )
        paired_clips = len(reference)
        metrics[metric] = bootstrap_metric(
            reference,
            candidate,
            iterations=args.iterations,
            seed=args.seed + metric_index,
            threshold=args.relative_threshold,
        )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark),
        "benchmark_sha256": sha256(benchmark),
        "reference_step": args.reference_step,
        "candidate_step": args.candidate_step,
        "paired_clips": paired_clips,
        "iterations": args.iterations,
        "seed": args.seed,
        "relative_threshold": args.relative_threshold,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
