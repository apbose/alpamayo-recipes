#!/usr/bin/env python3
"""Summarize the matched flow-only, larger-set, and adapter-scale controls."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FULL_STEPS = ("10", "8", "4", "2", "1")
LARGE_STEPS = ("10", "4", "2")
METRICS = ("min_ade", "ade", "corner_distance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--prior-suite-dir", type=Path, required=True)
    parser.add_argument(
        "--publication-dir",
        type=Path,
        help="Compact output directory; defaults to the research results tree.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def step_values(result: dict[str, Any], step: str) -> dict[str, float]:
    values = result["results"][step]
    return {
        **{metric: float(values["metrics"][metric]) for metric in METRICS},
        "expert_ms": float(values["latency_ms"]["action_expert_diffusion"]["mean"]),
        "end_to_end_ms": float(values["latency_ms"]["end_to_end_model"]["mean"]),
    }


def relative_change(value: float, reference: float) -> float:
    """Return a signed fraction; positive means a larger, worse error."""
    return value / reference - 1.0


def validate_benchmark(
    label: str, result: dict[str, Any], expected_steps: tuple[str, ...]
) -> None:
    missing = set(expected_steps) - set(result.get("results", {}))
    if missing:
        raise ValueError(f"{label} is missing solver steps {sorted(missing)}")
    config = result["configuration"]
    if config["seed_reset_for_each_step_count"] != 42:
        raise ValueError(f"{label} did not use seed 42")
    if config["num_traj_samples"] != 6:
        raise ValueError(f"{label} did not use six trajectory candidates")
    if config["attention_backend"] != "eager":
        raise ValueError(f"{label} did not use eager attention")


def comparison(
    left: dict[str, Any], right: dict[str, Any], steps: tuple[str, ...]
) -> dict[str, Any]:
    output = {}
    for step in steps:
        left_step = step_values(left, step)
        right_step = step_values(right, step)
        output[step] = {
            "left": left_step,
            "right": right_step,
            "right_error_change_vs_left": {
                metric: relative_change(right_step[metric], left_step[metric])
                for metric in METRICS
            },
        }
    return output


def within_checkpoint(result: dict[str, Any], steps: tuple[str, ...]) -> dict[str, Any]:
    ten = step_values(result, "10")
    return {
        step: {
            "values": step_values(result, step),
            "error_change_vs_10": {
                metric: relative_change(step_values(result, step)[metric], ten[metric])
                for metric in METRICS
            },
            "expert_speedup_vs_10": ten["expert_ms"]
            / step_values(result, step)["expert_ms"],
        }
        for step in steps
    }


def build_summary(
    suite_dir: Path,
    prior_suite_dir: Path,
    current: dict[str, dict[str, Any]],
    prior: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    flow32 = current["flow_control_32"]
    shortcut32 = prior["shortcut_32"]
    released32 = prior["released_32"]
    shortcut128 = current["shortcut_128"]
    flow128 = current["flow_control_128"]
    released128 = current["released_128"]

    manifest32 = {
        value["configuration"]["validation_manifest_sha256"]
        for value in (flow32, shortcut32, released32)
    }
    manifest128 = {
        value["configuration"]["validation_manifest_sha256"]
        for value in (flow128, shortcut128, released128)
    }
    if len(manifest32) != 1 or len(manifest128) != 1:
        raise ValueError("A matched comparison used different manifest hashes")

    adapter_results = {
        "0": prior["adapter_scale0"],
        "1": shortcut32,
        "2": current["adapter_scale2"],
        "4": current["adapter_scale4"],
        "8": current["adapter_scale8"],
    }
    adapter_sweep = {}
    for scale, result in adapter_results.items():
        ten = step_values(result, "10")
        adapter_sweep[scale] = {
            "steps": {step: step_values(result, step) for step in ("10", "2", "1")},
            "two_step_min_ade_regression_vs_10": relative_change(
                step_values(result, "2")["min_ade"], ten["min_ade"]
            ),
            "one_step_min_ade_regression_vs_10": relative_change(
                step_values(result, "1")["min_ade"], ten["min_ade"]
            ),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "prior_suite_dir": str(prior_suite_dir),
        "protocol": {
            "training_rows": 5295,
            "optimizer_steps": 662,
            "world_size": 8,
            "trajectory_candidates": 6,
            "seed": 42,
            "attention_backend": "eager",
            "validation_32_manifest_sha256": next(iter(manifest32)),
            "validation_128_manifest_sha256": next(iter(manifest128)),
            "latency_execution": "sequential on one NVIDIA B300",
        },
        "flow_checkpoint_validation": load_json(
            suite_dir / "flow_checkpoint_validation.json"
        ),
        "matched_32": {
            "shortcut_vs_flow": comparison(shortcut32, flow32, FULL_STEPS),
            "released_vs_flow": comparison(released32, flow32, FULL_STEPS),
            "shortcut_step_reduction": within_checkpoint(shortcut32, FULL_STEPS),
            "flow_step_reduction": within_checkpoint(flow32, FULL_STEPS),
        },
        "matched_128": {
            "shortcut_vs_flow": comparison(shortcut128, flow128, LARGE_STEPS),
            "released_vs_shortcut": comparison(released128, shortcut128, LARGE_STEPS),
            "released_vs_flow": comparison(released128, flow128, LARGE_STEPS),
            "shortcut_step_reduction": within_checkpoint(shortcut128, LARGE_STEPS),
            "flow_step_reduction": within_checkpoint(flow128, LARGE_STEPS),
        },
        "adapter_scale_sweep": adapter_sweep,
        "paired_bootstrap": {
            "flow_32": load_json(suite_dir / "bootstrap_flow_32_10_vs_2.json"),
            "shortcut_128": load_json(
                suite_dir / "bootstrap_shortcut_128_10_vs_2.json"
            ),
            "flow_128": load_json(suite_dir / "bootstrap_flow_128_10_vs_2.json"),
        },
        "gates": {
            "flow_32": load_json(
                suite_dir / "flow_control_32" / "two_step_gate_vs_released.json"
            ),
            "shortcut_128": load_json(
                suite_dir / "shortcut_128" / "two_step_gate_vs_released.json"
            ),
            "flow_128": load_json(
                suite_dir / "flow_control_128" / "two_step_gate_vs_released.json"
            ),
        },
        "interpretation_limits": [
            "Open-loop displacement metrics are not closed-loop safety evidence.",
            "Adapter scaling is an inference intervention, not retraining.",
            "The 128-clip set is larger but still comes from the 19 locally available chunks.",
        ],
    }


def markdown_header(columns: list[str]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    validation = summary["flow_checkpoint_validation"]
    lines = [
        "# Alpamayo 1.5 matched flow-control follow-ups",
        "",
        f"- Flow-only checkpoint validation passed: **{validation['passed']}**",
        f"- Flow-only optimizer steps: {validation['global_step']}",
        f"- Mean logged flow loss: {validation['mean_logged_loss']:.6f}",
        "",
        "## Same-data 32-clip comparison",
        "",
        *markdown_header(
            [
                "Steps",
                "Shortcut minADE",
                "Flow-only minADE",
                "Flow change",
                "Shortcut regression vs 10",
                "Flow regression vs 10",
            ]
        ),
    ]
    matched32 = summary["matched_32"]
    for step in FULL_STEPS:
        compare_item = matched32["shortcut_vs_flow"][step]
        shortcut_reduction = matched32["shortcut_step_reduction"][step]
        flow_reduction = matched32["flow_step_reduction"][step]
        lines.append(
            f"| {step} | {compare_item['left']['min_ade']:.4f} | "
            f"{compare_item['right']['min_ade']:.4f} | "
            f"{100 * compare_item['right_error_change_vs_left']['min_ade']:+.2f}% | "
            f"{100 * shortcut_reduction['error_change_vs_10']['min_ade']:+.2f}% | "
            f"{100 * flow_reduction['error_change_vs_10']['min_ade']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Larger 128-clip comparison",
            "",
            *markdown_header(
                [
                    "Steps",
                    "Released minADE",
                    "Shortcut minADE",
                    "Flow-only minADE",
                    "Shortcut vs flow",
                ]
            ),
        ]
    )
    matched128 = summary["matched_128"]
    for step in LARGE_STEPS:
        shortcut_flow = matched128["shortcut_vs_flow"][step]
        released_shortcut = matched128["released_vs_shortcut"][step]
        released = released_shortcut["left"]["min_ade"]
        lines.append(
            f"| {step} | {released:.4f} | {shortcut_flow['left']['min_ade']:.4f} | "
            f"{shortcut_flow['right']['min_ade']:.4f} | "
            f"{100 * shortcut_flow['right_error_change_vs_left']['min_ade']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Adapter-strength sweep",
            "",
            *markdown_header(
                [
                    "Scale",
                    "10-step minADE",
                    "2-step minADE",
                    "2-step regression",
                    "1-step minADE",
                ]
            ),
        ]
    )
    for scale, item in summary["adapter_scale_sweep"].items():
        lines.append(
            f"| {scale} | {item['steps']['10']['min_ade']:.4f} | "
            f"{item['steps']['2']['min_ade']:.4f} | "
            f"{100 * item['two_step_min_ade_regression_vs_10']:+.2f}% | "
            f"{item['steps']['1']['min_ade']:.4f} |"
        )

    shortcut_bootstrap = summary["paired_bootstrap"]["shortcut_128"]["metrics"][
        "min_ade"
    ]
    flow_bootstrap = summary["paired_bootstrap"]["flow_128"]["metrics"]["min_ade"]
    lines.extend(
        [
            "",
            "## Paired 128-clip uncertainty",
            "",
            (
                f"- Shortcut 10-to-2 minADE change: "
                f"{100 * shortcut_bootstrap['observed_relative_change']:+.2f}% "
                f"(95% CI "
                f"{100 * shortcut_bootstrap['relative_change_95_ci'][0]:+.2f}% to "
                f"{100 * shortcut_bootstrap['relative_change_95_ci'][1]:+.2f}%)."
            ),
            (
                f"- Flow-only 10-to-2 minADE change: "
                f"{100 * flow_bootstrap['observed_relative_change']:+.2f}% "
                f"(95% CI {100 * flow_bootstrap['relative_change_95_ci'][0]:+.2f}% "
                f"to {100 * flow_bootstrap['relative_change_95_ci'][1]:+.2f}%)."
            ),
            "",
            "## Limits",
            "",
            "- These are open-loop displacement results, not collision/off-road safety.",
            "- Adapter scaling changes inference only; it does not optimize a new checkpoint.",
            "- Latency runs were sequential on one B300 to avoid contention.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(
    path: Path,
    current: dict[str, dict[str, Any]],
    prior: dict[str, dict[str, Any]],
) -> None:
    all_results = {**current, **prior}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "solver_steps",
                *METRICS,
                "action_expert_mean_ms",
                "end_to_end_mean_ms",
                "adapter_scale",
                "validation_samples",
                "manifest_sha256",
            ]
        )
        for label, result in all_results.items():
            config = result["configuration"]
            for step in result["results"]:
                values = step_values(result, step)
                writer.writerow(
                    [
                        label,
                        step,
                        *(values[metric] for metric in METRICS),
                        values["expert_ms"],
                        values["end_to_end_ms"],
                        config.get("step_size_adapter_scale", 1.0),
                        config["validation_samples"],
                        config["validation_manifest_sha256"],
                    ]
                )


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    prior_suite_dir = args.prior_suite_dir.resolve()
    current = {
        "flow_control_32": load_json(
            suite_dir / "flow_control_32" / "benchmark_results.json"
        ),
        "shortcut_128": load_json(
            suite_dir / "shortcut_128" / "benchmark_results.json"
        ),
        "released_128": load_json(
            suite_dir / "released_128" / "benchmark_results.json"
        ),
        "flow_control_128": load_json(
            suite_dir / "flow_control_128" / "benchmark_results.json"
        ),
        "adapter_scale2": load_json(
            suite_dir / "adapter_scale2" / "benchmark_results.json"
        ),
        "adapter_scale4": load_json(
            suite_dir / "adapter_scale4" / "benchmark_results.json"
        ),
        "adapter_scale8": load_json(
            suite_dir / "adapter_scale8" / "benchmark_results.json"
        ),
    }
    prior = {
        "shortcut_32": load_json(
            prior_suite_dir / "candidate_seed42" / "benchmark_results.json"
        ),
        "released_32": load_json(
            prior_suite_dir / "released_seed42" / "benchmark_results.json"
        ),
        "adapter_scale0": load_json(
            prior_suite_dir / "candidate_zero_adapter_seed42" / "benchmark_results.json"
        ),
    }
    for label, result in current.items():
        expected = LARGE_STEPS if label.endswith("128") else FULL_STEPS
        if label.startswith("adapter_scale"):
            expected = ("10", "2", "1")
        validate_benchmark(label, result, expected)
    for label, result in prior.items():
        validate_benchmark(label, result, FULL_STEPS)

    summary = build_summary(suite_dir, prior_suite_dir, current, prior)
    publication_dir = (
        args.publication_dir.resolve()
        if args.publication_dir is not None
        else Path(__file__).resolve().parents[1] / "results" / suite_dir.name
    )
    if publication_dir.exists():
        raise FileExistsError(f"Refusing existing publication: {publication_dir}")
    summary["compact_publication_dir"] = str(publication_dir.resolve())
    atomic_json(suite_dir / "summary.json", summary)
    write_markdown(suite_dir / "REPORT.md", summary)
    write_csv(suite_dir / "comparison.csv", current, prior)
    publication_dir.mkdir(parents=True)
    atomic_json(publication_dir / "summary.json", summary)
    write_markdown(publication_dir / "REPORT.md", summary)
    write_csv(publication_dir / "comparison.csv", current, prior)
    print(json.dumps({"status": "complete", "suite_dir": str(suite_dir)}, indent=2))


if __name__ == "__main__":
    main()
