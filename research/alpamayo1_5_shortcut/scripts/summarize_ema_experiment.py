#!/usr/bin/env python3
"""Summarize the matched EMA-teacher training and evaluation experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = ("min_ade", "ade", "corner_distance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--controls-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def relative(candidate: float, reference: float) -> float:
    return candidate / reference - 1.0


def compact_step(result: dict[str, Any], step: str) -> dict[str, Any] | None:
    if step not in result["results"]:
        return None
    value = result["results"][step]
    return {
        "metrics": {name: float(value["metrics"][name]) for name in METRICS},
        "action_expert_ms": float(
            value["latency_ms"]["action_expert_diffusion"]["mean"]
        ),
        "end_to_end_ms": float(value["latency_ms"]["end_to_end_model"]["mean"]),
    }


def format_value(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Alpamayo 1.5 EMA-teacher shortcut experiment",
        "",
        f"- EMA checkpoint validation passed: **{summary['checkpoint_validation_passed']}**",
        f"- Sampler/accounting validation passed: **{summary['reference_validation_passed']}**",
        f"- Optimizer/EMA updates: {summary['ema_updates']}",
        f"- Fixed evaluation clips: {summary['protocol']['validation_clips']}",
        f"- Overall two-step gate passed: **{summary['gate']['overall_gate_passed']}**",
        "",
        "## minADE on the same 128 clips",
        "",
        "| Solver calls | EMA weights | Same checkpoint, student weights | Online-teacher shortcut | Flow-only | Released |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for step in ("10", "4", "2", "1"):
        row = summary["results_by_step"][step]
        lines.append(
            f"| {step} | {format_value(row['ema_min_ade'])} | "
            f"{format_value(row['student_min_ade'])} | "
            f"{format_value(row['online_min_ade'])} | "
            f"{format_value(row['flow_min_ade'])} | "
            f"{format_value(row['released_min_ade'])} |"
        )

    two_step = summary["two_step"]
    lines.extend(
        [
            "",
            "## Two-step comparisons",
            "",
            "| Comparison | minADE change | ADE change | Corner-distance change |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, values in two_step["comparisons"].items():
        lines.append(
            f"| {label} | {100 * values['min_ade']:+.2f}% | "
            f"{100 * values['ade']:+.2f}% | "
            f"{100 * values['corner_distance']:+.2f}% |"
        )

    bootstrap = summary["paired_bootstrap_ema_10_vs_2"]["metrics"]["min_ade"]
    interval = bootstrap["relative_change_95_ci"]
    runtime = summary["ema_runtime"]
    lines.extend(
        [
            "",
            "## Runtime and uncertainty",
            "",
            f"- EMA runtime state: decay `{runtime['decay']}`, "
            f"updates `{runtime['updates']}`, dtypes `{runtime['parameter_dtypes']}`, "
            f"inference weights `{runtime['inference_weights']}`.",
            f"- EMA 10-to-2 Action-Expert speedup: "
            f"{two_step['action_expert_speedup']:.2f}x.",
            f"- Paired EMA 10-to-2 minADE change: "
            f"{100 * bootstrap['observed_relative_change']:+.2f}% "
            f"(95% CI {100 * interval[0]:+.2f}% to {100 * interval[1]:+.2f}%).",
            "",
            "## Interpretation boundary",
            "",
            f"- Quality smoke gate passed: {summary['gate']['quality']['passed']}.",
            f"- Efficiency gate passed: {summary['gate']['efficiency']['passed']}.",
            f"- Safety evidence passed: {summary['gate']['safety']['passed']} "
            f"(`{summary['gate']['safety']['status']}`).",
            "- Lower displacement error is better. EMA weights versus the old online-teacher "
            "checkpoint combines training-target and evaluation-weight effects.",
            "- The EMA-trained student versus the online-teacher checkpoint isolates the "
            "teacher-target change; both rows evaluate student weights.",
            "- EMA versus student weights within the new checkpoint isolates the "
            "evaluation-weight swap.",
            "- These are open-loop displacement metrics, not collision, off-road, or "
            "traffic-rule safety evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    controls_dir = args.controls_dir.resolve()
    ema = load_json(suite_dir / "ema_weights_128" / "benchmark_results.json")
    student = load_json(
        suite_dir / "student_weights_128" / "benchmark_results.json"
    )
    online = load_json(controls_dir / "shortcut_128" / "benchmark_results.json")
    flow = load_json(controls_dir / "flow_control_128" / "benchmark_results.json")
    released = load_json(controls_dir / "released_128" / "benchmark_results.json")
    validation = load_json(suite_dir / "ema_checkpoint_validation.json")
    reference_validation = load_json(
        suite_dir / "reference_checkpoint_validation.json"
    )
    gate = load_json(suite_dir / "two_step_gate_vs_released.json")
    ema_bootstrap = load_json(suite_dir / "bootstrap_ema_10_vs_2.json")
    student_bootstrap = load_json(suite_dir / "bootstrap_student_10_vs_2.json")

    protocol_keys = (
        "validation_manifest_sha256",
        "validation_samples",
        "validation_unique_clips",
        "num_traj_samples",
        "seed_reset_for_each_step_count",
        "attention_backend",
    )
    protocol_sources = {
        "ema": ema["configuration"],
        "student": student["configuration"],
        "online": online["configuration"],
        "flow": flow["configuration"],
        "released": released["configuration"],
    }
    protocol_checks = {
        key: len({json.dumps(source[key], sort_keys=True) for source in protocol_sources.values()})
        == 1
        for key in protocol_keys
    }
    if not all(protocol_checks.values()):
        raise ValueError(f"Evaluation protocol mismatch: {protocol_checks}")

    compact = {
        label: {
            step: compact_step(result, step)
            for step in ("10", "4", "2", "1")
        }
        for label, result in {
            "ema": ema,
            "student": student,
            "online": online,
            "flow": flow,
            "released": released,
        }.items()
    }
    results_by_step = {}
    for step in ("10", "4", "2", "1"):
        results_by_step[step] = {
            f"{label}_min_ade": (
                None if compact[label][step] is None else compact[label][step]["metrics"]["min_ade"]
            )
            for label in compact
        }

    def metric_comparison(candidate: str, reference: str, step: str) -> dict[str, float]:
        candidate_step = compact[candidate][step]
        reference_step = compact[reference][step]
        if candidate_step is None or reference_step is None:
            raise ValueError(f"Missing {step}-step result for {candidate} or {reference}")
        return {
            metric: relative(
                candidate_step["metrics"][metric], reference_step["metrics"][metric]
            )
            for metric in METRICS
        }

    ema_two = compact["ema"]["2"]
    ema_ten = compact["ema"]["10"]
    assert ema_two is not None and ema_ten is not None
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_validation_passed": bool(validation["passed"]),
        "reference_validation_passed": bool(reference_validation["passed"]),
        "ema_updates": validation["ema_updates"],
        "protocol": {
            "checks": protocol_checks,
            "passed": all(protocol_checks.values()),
            "validation_clips": ema["configuration"]["validation_unique_clips"],
            "manifest_sha256": ema["configuration"]["validation_manifest_sha256"],
            "candidates": ema["configuration"]["num_traj_samples"],
            "seed": ema["configuration"]["seed_reset_for_each_step_count"],
            "attention_backend": ema["configuration"]["attention_backend"],
        },
        "ema_runtime": ema["configuration"]["shortcut_runtime"],
        "results_by_step": results_by_step,
        "compact_results": compact,
        "two_step": {
            "comparisons": {
                "EMA 2 vs same EMA checkpoint 10": {
                    metric: relative(
                        ema_two["metrics"][metric], ema_ten["metrics"][metric]
                    )
                    for metric in METRICS
                },
                "EMA 2 vs online-teacher 2": metric_comparison("ema", "online", "2"),
                "EMA-trained student 2 vs online-teacher 2": metric_comparison(
                    "student", "online", "2"
                ),
                "EMA 2 vs flow-only 2": metric_comparison("ema", "flow", "2"),
                "EMA weights 2 vs student weights 2": metric_comparison(
                    "ema", "student", "2"
                ),
            },
            "action_expert_speedup": ema_ten["action_expert_ms"]
            / ema_two["action_expert_ms"],
        },
        "gate": gate,
        "paired_bootstrap_ema_10_vs_2": ema_bootstrap,
        "paired_bootstrap_student_10_vs_2": student_bootstrap,
        "inputs": {
            "suite_dir": str(suite_dir),
            "controls_dir": str(controls_dir),
        },
    }
    output_json = suite_dir / "summary.json"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (suite_dir / "REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
