#!/usr/bin/env python3
"""Summarize the controlled post-training reference-shortcut evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEPS = ("10", "8", "4", "2", "1")
METRICS = ("min_ade", "ade", "corner_distance")
RESULT_LABELS = (
    "candidate_seed42",
    "released_seed42",
    "candidate_seed43",
    "candidate_seed44",
    "candidate_zero_adapter_seed42",
    "v1_seed42",
    "candidate_test_seed42",
    "released_test_seed42",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
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


def pct_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def quality_gain(value: float, reference: float) -> float:
    """Positive means the lower trajectory error improved."""
    return -pct_change(value, reference)


def compact_step(result: dict[str, Any], step: str) -> dict[str, Any]:
    step_result = result["results"][step]
    return {
        "metrics": {
            metric: float(step_result["metrics"][metric]) for metric in METRICS
        },
        "latency_ms": {
            "action_expert_mean": float(
                step_result["latency_ms"]["action_expert_diffusion"]["mean"]
            ),
            "end_to_end_mean": float(
                step_result["latency_ms"]["end_to_end_model"]["mean"]
            ),
        },
        "peak_gpu_memory_mib": float(step_result["peak_gpu_memory_mib"]),
    }


def validate_protocol(results: dict[str, dict[str, Any]]) -> None:
    for label, result in results.items():
        missing = set(STEPS) - set(result.get("results", {}))
        if missing:
            raise ValueError(f"{label} is missing solver steps: {sorted(missing)}")
        config = result["configuration"]
        if config["num_traj_samples"] != 6:
            raise ValueError(f"{label} did not use six trajectory candidates")
        if config["attention_backend"] != "eager":
            raise ValueError(f"{label} did not use eager attention")

    validation_labels = (
        "candidate_seed42",
        "released_seed42",
        "candidate_seed43",
        "candidate_seed44",
        "candidate_zero_adapter_seed42",
        "v1_seed42",
    )
    validation_hashes = {
        results[label]["configuration"]["validation_manifest_sha256"]
        for label in validation_labels
    }
    if len(validation_hashes) != 1:
        raise ValueError("Validation experiments did not use one manifest hash")
    test_hashes = {
        results[label]["configuration"]["validation_manifest_sha256"]
        for label in ("candidate_test_seed42", "released_test_seed42")
    }
    if len(test_hashes) != 1:
        raise ValueError("Test experiments did not use one manifest hash")
    if validation_hashes == test_hashes:
        raise ValueError("Validation and test manifest hashes unexpectedly match")


def build_summary(
    suite_dir: Path, results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    candidate = results["candidate_seed42"]
    released = results["released_seed42"]
    zero_adapter = results["candidate_zero_adapter_seed42"]
    v1 = results["v1_seed42"]
    candidate_test = results["candidate_test_seed42"]
    released_test = results["released_test_seed42"]

    validation_comparison = {}
    step_reduction = {}
    adapter_ablation = {}
    v1_comparison = {}
    test_comparison = {}
    candidate_ten = compact_step(candidate, "10")
    for step in STEPS:
        candidate_step = compact_step(candidate, step)
        released_step = compact_step(released, step)
        zero_step = compact_step(zero_adapter, step)
        v1_step = compact_step(v1, step)
        candidate_test_step = compact_step(candidate_test, step)
        released_test_step = compact_step(released_test, step)
        validation_comparison[step] = {
            "candidate": candidate_step,
            "released": released_step,
            "candidate_quality_gain_pct": {
                metric: quality_gain(
                    candidate_step["metrics"][metric],
                    released_step["metrics"][metric],
                )
                for metric in METRICS
            },
        }
        step_reduction[step] = {
            "candidate": candidate_step,
            "quality_change_vs_candidate_10_pct": {
                metric: pct_change(
                    candidate_step["metrics"][metric],
                    candidate_ten["metrics"][metric],
                )
                for metric in METRICS
            },
            "action_expert_speedup_vs_candidate_10": (
                candidate_ten["latency_ms"]["action_expert_mean"]
                / candidate_step["latency_ms"]["action_expert_mean"]
            ),
            "end_to_end_speedup_vs_candidate_10": (
                candidate_ten["latency_ms"]["end_to_end_mean"]
                / candidate_step["latency_ms"]["end_to_end_mean"]
            ),
        }
        adapter_ablation[step] = {
            "trained_adapter": candidate_step,
            "zeroed_adapter": zero_step,
            "zeroed_quality_change_pct": {
                metric: pct_change(
                    zero_step["metrics"][metric], candidate_step["metrics"][metric]
                )
                for metric in METRICS
            },
        }
        v1_comparison[step] = {
            "reference_partition": candidate_step,
            "v1_per_sample_weighted": v1_step,
            "reference_quality_change_pct": {
                metric: pct_change(
                    candidate_step["metrics"][metric], v1_step["metrics"][metric]
                )
                for metric in METRICS
            },
        }
        test_comparison[step] = {
            "candidate": candidate_test_step,
            "released": released_test_step,
            "candidate_quality_gain_pct": {
                metric: quality_gain(
                    candidate_test_step["metrics"][metric],
                    released_test_step["metrics"][metric],
                )
                for metric in METRICS
            },
        }

    seed_results = [
        results["candidate_seed42"],
        results["candidate_seed43"],
        results["candidate_seed44"],
    ]
    seed_stability = {}
    for step in STEPS:
        seed_stability[step] = {}
        for metric in METRICS:
            values = [
                float(result["results"][step]["metrics"][metric])
                for result in seed_results
            ]
            seed_stability[step][metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir.resolve()),
        "protocol": {
            "solver_steps": [int(step) for step in STEPS],
            "trajectory_candidates": 6,
            "validation_seeds": [42, 43, 44],
            "attention_backend": "eager",
            "execution": "sequential on one NVIDIA B300 to avoid contention",
            "validation_clips": candidate["configuration"]["validation_unique_clips"],
            "test_clips": candidate_test["configuration"]["validation_unique_clips"],
            "validation_manifest_sha256": candidate["configuration"][
                "validation_manifest_sha256"
            ],
            "test_manifest_sha256": candidate_test["configuration"][
                "validation_manifest_sha256"
            ],
        },
        "validation_candidate_vs_released": validation_comparison,
        "candidate_step_reduction": step_reduction,
        "candidate_seed_stability": seed_stability,
        "step_size_adapter_ablation": adapter_ablation,
        "reference_vs_v1": {
            "warning": (
                "This is descriptive, not a causal loss ablation: the checkpoints "
                "used different training populations and update counts."
            ),
            "steps": v1_comparison,
        },
        "independent_test_candidate_vs_released": test_comparison,
        "gates": {
            "validation": read_json(
                suite_dir / "candidate_seed42" / "two_step_gate_vs_released.json"
            ),
            "test": read_json(
                suite_dir / "candidate_test_seed42" / "two_step_gate_vs_released.json"
            ),
        },
    }


def write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "experiment",
                "split",
                "seed",
                "zero_step_size_adapter",
                "solver_steps",
                *METRICS,
                "action_expert_mean_ms",
                "end_to_end_mean_ms",
            ]
        )
        for label, result in results.items():
            config = result["configuration"]
            for step in STEPS:
                compact = compact_step(result, step)
                writer.writerow(
                    [
                        label,
                        config.get("evaluation_split", "val"),
                        config["seed_reset_for_each_step_count"],
                        config.get("zero_step_size_adapter", False),
                        step,
                        *(compact["metrics"][metric] for metric in METRICS),
                        compact["latency_ms"]["action_expert_mean"],
                        compact["latency_ms"]["end_to_end_mean"],
                    ]
                )


def markdown_table_header(columns: list[str]) -> list[str]:
    return [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Alpamayo 1.5 reference-shortcut follow-up results",
        "",
        (
            "All latency runs were executed sequentially on one NVIDIA B300. Each "
            "row uses six stochastic trajectory candidates and eager attention."
        ),
        "",
        "## Primary 32-clip validation comparison",
        "",
        *markdown_table_header(
            [
                "Steps",
                "Released minADE",
                "Candidate minADE",
                "Gain",
                "Candidate ADE",
                "Candidate corner",
            ]
        ),
    ]
    for step in STEPS:
        item = summary["validation_candidate_vs_released"][step]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    f"{item['released']['metrics']['min_ade']:.4f}",
                    f"{item['candidate']['metrics']['min_ade']:.4f}",
                    f"{item['candidate_quality_gain_pct']['min_ade']:+.2f}%",
                    f"{item['candidate']['metrics']['ade']:.4f}",
                    f"{item['candidate']['metrics']['corner_distance']:.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Step reduction within the reference-partition checkpoint",
            "",
            *markdown_table_header(
                [
                    "Steps",
                    "minADE",
                    "Change vs 10",
                    "Expert speedup",
                    "End-to-end speedup",
                ]
            ),
        ]
    )
    for step in STEPS:
        item = summary["candidate_step_reduction"][step]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    f"{item['candidate']['metrics']['min_ade']:.4f}",
                    f"{item['quality_change_vs_candidate_10_pct']['min_ade']:+.2f}%",
                    f"{item['action_expert_speedup_vs_candidate_10']:.2f}x",
                    f"{item['end_to_end_speedup_vs_candidate_10']:.2f}x",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Seed repeatability (42, 43, 44)",
            "",
            *markdown_table_header(
                ["Steps", "minADE mean", "minADE std", "ADE mean", "Corner mean"]
            ),
        ]
    )
    for step in STEPS:
        item = summary["candidate_seed_stability"][step]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    f"{item['min_ade']['mean']:.4f}",
                    f"{item['min_ade']['sample_std']:.4f}",
                    f"{item['ade']['mean']:.4f}",
                    f"{item['corner_distance']['mean']:.4f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Step-size adapter ablation",
            "",
            (
                "The checkpoint is unchanged on disk. The ablation zeros only the "
                "adapter's final projection after loading."
            ),
            "",
            *markdown_table_header(
                [
                    "Steps",
                    "Trained minADE",
                    "Zero-adapter minADE",
                    "Zero-adapter change",
                ]
            ),
        ]
    )
    for step in STEPS:
        item = summary["step_size_adapter_ablation"][step]
        lines.append(
            "| "
            + " | ".join(
                [
                    step,
                    f"{item['trained_adapter']['metrics']['min_ade']:.4f}",
                    f"{item['zeroed_adapter']['metrics']['min_ade']:.4f}",
                    f"{item['zeroed_quality_change_pct']['min_ade']:+.2f}%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Independent 32-clip test comparison",
            "",
            *markdown_table_header(
                ["Steps", "Released minADE", "Candidate minADE", "Gain"]
            ),
        ]
    )
    for step in STEPS:
        item = summary["independent_test_candidate_vs_released"][step]
        lines.append(
            f"| {step} | {item['released']['metrics']['min_ade']:.4f} | "
            f"{item['candidate']['metrics']['min_ade']:.4f} | "
            f"{item['candidate_quality_gain_pct']['min_ade']:+.2f}% |"
        )

    validation_gate = summary["gates"]["validation"]
    test_gate = summary["gates"]["test"]
    lines.extend(
        [
            "",
            "## Gate status",
            "",
            f"- Validation quality passed: **{validation_gate['quality']['passed']}**",
            f"- Validation efficiency passed: **{validation_gate['efficiency']['passed']}**",
            f"- Independent-test quality passed: **{test_gate['quality']['passed']}**",
            f"- Independent-test efficiency passed: **{test_gate['efficiency']['passed']}**",
            (
                "- Safety remains unavailable in the public open-loop recipe; therefore "
                "the overall deployment gate remains closed regardless of displacement "
                "quality."
            ),
            "",
            "## Interpretation limits",
            "",
            "- This is open-loop trajectory displacement evaluation, not closed-loop AV safety.",
            (
                "- The v1-versus-reference comparison is descriptive because training "
                "data volume and update counts differ."
            ),
            (
                "- End-to-end latency is VLM-dominated; Action-Expert latency is the "
                "relevant solver-step scaling measurement."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    suite_dir = args.suite_dir.resolve()
    publish_dir = Path(__file__).resolve().parents[1] / "results" / suite_dir.name
    if publish_dir.exists():
        raise FileExistsError(f"Refusing existing compact publication: {publish_dir}")
    results = {
        label: read_json(suite_dir / label / "benchmark_results.json")
        for label in RESULT_LABELS
    }
    validate_protocol(results)
    summary = build_summary(suite_dir, results)
    summary["compact_publication_dir"] = str(publish_dir.resolve())
    atomic_json(suite_dir / "summary.json", summary)
    write_csv(suite_dir / "comparison.csv", results)
    write_markdown(suite_dir / "REPORT.md", summary)
    publish_dir.mkdir(parents=True)
    atomic_json(publish_dir / "summary.json", summary)
    write_csv(publish_dir / "comparison.csv", results)
    write_markdown(publish_dir / "REPORT.md", summary)
    print(json.dumps({"status": "complete", "suite_dir": str(suite_dir)}, indent=2))


if __name__ == "__main__":
    main()
