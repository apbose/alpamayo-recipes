#!/usr/bin/env python3
"""Apply the predeclared Alpamayo 1.5 two-step smoke quality/safety gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/two_step_gate.json"

# Predeclared before reading the trained results. All trajectory errors are
# lower-is-better. This is a smoke threshold, not an AV deployment criterion.
MAX_RELATIVE_QUALITY_REGRESSION = 0.10
MIN_ACTION_EXPERT_SPEEDUP = 4.0
QUALITY_METRICS = ("min_ade", "ade", "corner_distance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Released-checkpoint benchmark_results.json.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Trained-shortcut benchmark_results.json.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--safety-report",
        type=Path,
        help=(
            "Optional external two-step safety report. It must contain "
            "passed=true, inference_steps=2, and the same validation manifest hash."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def quality_checks(
    candidate_value: float,
    references: dict[str, float],
) -> list[dict[str, Any]]:
    checks = []
    for reference_name, reference_value in references.items():
        allowed = reference_value * (1.0 + MAX_RELATIVE_QUALITY_REGRESSION)
        checks.append(
            {
                "reference": reference_name,
                "candidate": candidate_value,
                "reference_value": reference_value,
                "maximum_allowed": allowed,
                "relative_change": (
                    (candidate_value - reference_value) / reference_value
                ),
                "passed": candidate_value <= allowed,
            }
        )
    return checks


def validate_protocol(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_config = baseline["configuration"]
    candidate_config = candidate["configuration"]
    comparisons = {
        "validation_manifest_sha256": (
            baseline_config["validation_manifest_sha256"],
            candidate_config["validation_manifest_sha256"],
        ),
        "validation_samples": (
            baseline_config["validation_samples"],
            candidate_config["validation_samples"],
        ),
        "validation_unique_clips": (
            baseline_config["validation_unique_clips"],
            candidate_config["validation_unique_clips"],
        ),
        "num_traj_samples": (
            baseline_config["num_traj_samples"],
            candidate_config["num_traj_samples"],
        ),
        "seed": (
            baseline_config["seed_reset_for_each_step_count"],
            candidate_config["seed_reset_for_each_step_count"],
        ),
        "attention_backend": (
            baseline_config["attention_backend"],
            candidate_config["attention_backend"],
        ),
    }
    return [
        {
            "name": name,
            "baseline": values[0],
            "candidate": values[1],
            "passed": values[0] == values[1],
        }
        for name, values in comparisons.items()
    ]


def safety_evidence(
    safety_report_path: Path | None,
    manifest_sha256: str,
) -> dict[str, Any]:
    if safety_report_path is None:
        return {
            "available": False,
            "passed": False,
            "status": "not_evaluated",
            "reason": (
                "NVIDIA's public Stage-2 recipe reports displacement distance "
                "only; it has no collision, off-road, or traffic-rule evaluator. "
                "Finite outputs are necessary but are not safety evidence."
            ),
            "required_next": (
                "Evaluate the two-step checkpoint in a safety-capable closed-loop "
                "simulator or an annotated collision/off-road benchmark."
            ),
        }

    report = load_json(safety_report_path)
    protocol_ok = (
        report.get("inference_steps") == 2
        and report.get("validation_manifest_sha256") == manifest_sha256
    )
    return {
        "available": True,
        "passed": bool(report.get("passed")) and protocol_ok,
        "status": "passed" if bool(report.get("passed")) and protocol_ok else "failed",
        "protocol_match": protocol_ok,
        "source": str(safety_report_path.resolve()),
        "reported_metrics": report.get("metrics"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Alpamayo 1.5 two-step gate",
        "",
        f"- Overall gate passed: **{report['overall_gate_passed']}**",
        f"- Decision: **{report['decision']}**",
        f"- Quality smoke checks passed: {report['quality']['passed']}",
        f"- Action-Expert latency check passed: {report['efficiency']['passed']}",
        f"- Safety evidence passed: {report['safety']['passed']}",
        "",
        f"Evidence scope: {report['evidence_scope']}. This is not statistically "
        "sufficient evidence for vehicle deployment.",
        "",
        "## Quality",
        "",
    ]
    for metric_name, metric in report["quality"]["metrics"].items():
        lines.append(
            f"- {metric_name}: two-step={metric['candidate_two_step']:.6f}; "
            f"passed={metric['passed']}"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- Status: {report['safety']['status']}",
            f"- Reason: {report['safety'].get('reason', 'See external report.')}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    required_steps = {"10", "4", "2"}
    if not required_steps.issubset(candidate.get("results", {})):
        raise ValueError("Candidate benchmark must contain 10-, 4-, and 2-step results")

    protocol = validate_protocol(baseline, candidate)
    protocol_passed = all(check["passed"] for check in protocol)
    candidate_two = candidate["results"]["2"]
    candidate_ten = candidate["results"]["10"]
    baseline_ten = baseline["results"]["10"]

    metric_results: dict[str, Any] = {}
    for metric_name in QUALITY_METRICS:
        value = float(candidate_two["metrics"][metric_name])
        checks = quality_checks(
            value,
            {
                "trained_10_step": float(candidate_ten["metrics"][metric_name]),
                "released_10_step": float(baseline_ten["metrics"][metric_name]),
            },
        )
        metric_results[metric_name] = {
            "candidate_two_step": value,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }
    quality_passed = protocol_passed and all(
        metric["passed"] for metric in metric_results.values()
    )

    expert_speedup = float(
        candidate_two["comparison_to_10_step"]["action_expert_speedup"]
    )
    efficiency_passed = expert_speedup >= MIN_ACTION_EXPERT_SPEEDUP
    safety = safety_evidence(
        args.safety_report,
        candidate["configuration"]["validation_manifest_sha256"],
    )
    overall = quality_passed and efficiency_passed and bool(safety["passed"])
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": (
            f"{candidate['configuration']['validation_unique_clips']}-clip-"
            "open-loop-pilot"
        ),
        "thresholds_predeclared": {
            "maximum_relative_quality_regression": (
                MAX_RELATIVE_QUALITY_REGRESSION
            ),
            "minimum_action_expert_speedup_vs_10_step": (
                MIN_ACTION_EXPERT_SPEEDUP
            ),
        },
        "inputs": {
            "released_baseline": str(args.baseline.resolve()),
            "trained_candidate": str(args.candidate.resolve()),
        },
        "protocol": {
            "checks": protocol,
            "passed": protocol_passed,
        },
        "quality": {
            "metrics": metric_results,
            "passed": quality_passed,
        },
        "efficiency": {
            "action_expert_speedup_vs_trained_10_step": expert_speedup,
            "minimum_required": MIN_ACTION_EXPERT_SPEEDUP,
            "passed": efficiency_passed,
        },
        "safety": safety,
        "overall_gate_passed": overall,
        "decision": "pursue_one_step" if overall else "do_not_pursue_one_step",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
