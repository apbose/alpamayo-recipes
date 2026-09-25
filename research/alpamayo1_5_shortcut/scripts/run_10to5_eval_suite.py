#!/usr/bin/env python3
"""Run the three matched 10/5-step evaluations and write a compact report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent
RECIPE_DIR = REPO_ROOT / "recipes/alpamayo1_5_sft"
ASSETS = Path("/home/scratch.abose_sw/alpamayo-assets")
DEFAULT_TRAIN_RUN = ASSETS / "runs/alpamayo15_hf300gb_10to5_reference_ema_5295clips_3epochs_20260919_r2"
EXPECTED = {
    "shortcut_flow_step_size": 0.1,
    "shortcut_step_sizes": [0.2],
    "shortcut_require_dyadic_steps": False,
    "shortcut_bootstrap_every": 4,
    "shortcut_loss_weight": 0.25,
    "shortcut_loss_estimator": "reference_partition",
    "shortcut_teacher_mode": "ema",
    "shortcut_ema_decay": 0.999,
    "shortcut_inference_weights": "ema",
}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def preflight(args):
    checkpoint = args.train_run / "trainer_output/checkpoint-1986"
    cfg = read_json(checkpoint / "config.json")
    state = read_json(checkpoint / "trainer_state.json")
    train_manifest = PROJECT_DIR / "manifests/hf_stream_300gb/train.json"
    train_rows = read_json(train_manifest)
    val_rows = read_json(args.manifest_dir / "val.json")
    train_ids = {row["clip_id"] for row in train_rows}
    val_ids = {row["clip_id"] for row in val_rows}
    status = dict(
        line.split("=", 1) for line in (args.train_run / "STATUS").read_text().splitlines()
        if "=" in line
    )
    checks = {
        **{key: cfg.get(key) == value for key, value in EXPECTED.items()},
        "training_complete": status.get("status") == "complete",
        "global_step_1986": state["global_step"] == 1986,
        "three_epochs": state["epoch"] == 3.0,
        "training_5295_unique_clips": len(train_rows) == len(train_ids) == 5295,
        "validation_128_unique_clips": len(val_rows) == len(val_ids) == 128,
        "training_manifest_unchanged": status["train_manifest_sha256"] == digest(train_manifest),
        "train_validation_disjoint": not (train_ids & val_ids),
        "route_less_validation": all("nav_text" not in row for row in val_rows),
        "local_dataset_present": (args.dataset / "clip_index.parquet").is_file(),
        "released_checkpoint_present": (args.released_checkpoint / "config.json").is_file(),
    }
    report = {
        "passed": all(checks.values()), "checks": checks,
        "checkpoint": str(checkpoint),
        "train_manifest_sha256": digest(train_manifest),
        "validation_manifest_sha256": digest(args.manifest_dir / "val.json"),
        "train_validation_overlap": len(train_ids & val_ids),
        "expected_config": EXPECTED,
    }
    write_json(args.output_dir / "preflight.json", report)
    if not report["passed"]:
        raise ValueError(f"Preflight failed: {[key for key, value in checks.items() if not value]}")
    return checkpoint


def summarize(args):
    labels = ("ema_weights_128", "student_weights_128", "released_128")
    benchmarks = {label: read_json(args.output_dir / label / "benchmark_results.json") for label in labels}
    match_keys = (
        "validation_manifest_sha256", "validation_samples", "num_traj_samples",
        "seed_reset_for_each_step_count", "warmup_samples_per_step_count",
        "attention_backend", "gpu", "inference_steps",
    )
    reference = benchmarks[labels[0]]["configuration"]
    for label, benchmark in benchmarks.items():
        for key in match_keys:
            if benchmark["configuration"][key] != reference[key]:
                raise ValueError(f"Unmatched benchmark setting: {label}: {key}")
        for step in ("10", "5"):
            if len(benchmark["results"][step]["per_sample"]) != 128:
                raise ValueError(f"Incomplete benchmark: {label}, {step}")
    rows = []
    comparisons = {}
    for label, benchmark in benchmarks.items():
        for step in ("10", "5"):
            result = benchmark["results"][step]
            e2e = result["latency_ms"]["end_to_end_model"]["mean"]
            expert = result["latency_ms"]["action_expert_diffusion"]["mean"]
            rows.append({"model": label, "steps": int(step), **result["metrics"],
                         "expert_ms": expert, "model_ms": e2e,
                         "expert_hz": 1000 / expert, "model_hz": 1000 / e2e})
        high, low = (benchmark["results"][step] for step in ("10", "5"))
        comparisons[label] = {
            key: 100 * (low["metrics"][key] / high["metrics"][key] - 1)
            for key in ("min_ade", "ade", "corner_distance")
        }
        comparisons[label]["expert_speedup"] = (
            high["latency_ms"]["action_expert_diffusion"]["mean"]
            / low["latency_ms"]["action_expert_diffusion"]["mean"]
        )
    bootstrap = {label: read_json(args.output_dir / f"bootstrap_{label}_10_vs_5.json") for label in labels}
    report = {"generated_at_utc": utc(), "protocol": {key: reference[key] for key in match_keys},
              "rows": rows, "five_vs_ten_percent_change": comparisons, "paired_bootstrap": bootstrap,
              "safety_evaluated": False}
    write_json(args.output_dir / "summary.json", report)
    lines = ["# Alpamayo 1.5: targeted 10-to-5 results", "", "128 held-out clips; six trajectory candidates; seed 42; eager attention; sequential GPU 0 evaluations.", "",
             "| Weights | Steps | minADE (m) | ADE (m) | Corner (m) | Expert (ms) | Model (ms) | Model Hz |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['steps']} | {row['min_ade']:.4f} | {row['ade']:.4f} | {row['corner_distance']:.4f} | {row['expert_ms']:.2f} | {row['model_ms']:.2f} | {row['model_hz']:.3f} |")
    lines += ["", "## Five steps versus ten steps within each model", "",
              "| Weights | minADE change | ADE change | Corner change | Expert speedup |",
              "|---|---:|---:|---:|---:|"]
    for label, values in comparisons.items():
        lines.append(f"| {label} | {values['min_ade']:+.2f}% | {values['ade']:+.2f}% | {values['corner_distance']:+.2f}% | {values['expert_speedup']:.2f}x |")
    lines += ["", "Positive error changes are worse. Paired clip bootstrap intervals are in the three bootstrap markdown/JSON files.", "",
              "The trained 5-vs-10 comparison measures step reduction within one checkpoint. The trained-vs-released comparison also includes three epochs of fine-tuning; there is no matched three-epoch flow-only control in this suite.", "",
              "These are open-loop trajectory and model-call latency results. Collision/off-road safety is not evaluated. Model-call timing excludes dataset loading and decoding. Hz is the reciprocal of mean latency, not a closed-loop driving rate.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, default=DEFAULT_TRAIN_RUN)
    parser.add_argument("--released-checkpoint", type=Path, default=ASSETS / "checkpoints/Alpamayo-1.5-10B-A1-format")
    parser.add_argument("--dataset", type=Path, default=ASSETS / "physical_ai_av")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_DIR / "manifests/route_less_19chunks_128eval")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs = args.output_dir / "logs"
    logs.mkdir()
    state = {"status": "running", "started_utc": utc(), "pid": os.getpid(),
             "execution": "sequential_single_gpu", "gpu": "0", "completed_evaluations": 0,
             "total_evaluations": 3, "phase": "preflight"}

    def status(**updates):
        state.update(updates, updated_utc=utc())
        write_json(args.output_dir / "status.json", state)
        temporary = args.output_dir / "STATUS.tmp"
        temporary.write_text("".join(f"{key}={value}\n" for key, value in state.items()))
        temporary.replace(args.output_dir / "STATUS")

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", HF_HOME="/home/abose_sw/.cache/huggingface",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1", DS_IGNORE_CUDA_DETECTION="1",
               OMP_NUM_THREADS="1", TRITON_CACHE_DIR=f"/tmp/alpamayo15-10to5-eval-{os.getpid()}")

    def run(label, command):
        status(phase=label)
        print(f"[{utc()}] {label}", flush=True)
        with (args.output_dir / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"phase": label, "argv": [str(x) for x in command], "cwd": str(RECIPE_DIR)}) + "\n")
        with (logs / f"{label}.log").open("w") as handle:
            subprocess.run([str(x) for x in command], cwd=RECIPE_DIR, env=env,
                           stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=4 * 3600, check=True)

    try:
        status()
        checkpoint = preflight(args)
        run("checkpoint_validation", [sys.executable, SCRIPT_DIR / "validate_ema_checkpoint.py",
            "--checkpoint", checkpoint, "--train-log", args.train_run / "train.log",
            "--output", args.output_dir / "ema_checkpoint_validation.json", "--expected-steps", "1986"])
        if args.preflight_only:
            status(status="complete", phase="preflight_complete", finished_utc=utc())
            return
        evaluations = (
            ("ema_weights_128", checkpoint, "sft_stage2_trajectory_shortcut_10to5_reference_ema", "ema"),
            ("student_weights_128", checkpoint, "sft_stage2_trajectory_shortcut_10to5_reference_ema", "student"),
            ("released_128", args.released_checkpoint, "sft_stage2_trajectory_shortcut", None),
        )
        for label, weights, config, inference_weights in evaluations:
            command = [RECIPE_DIR / "a1_5_sft/bin/torchrun", "--standalone", "--nproc_per_node=1",
                SCRIPT_DIR / "benchmark_inference_steps.py", "--checkpoint", weights,
                "--config-name", config, "--dataset", args.dataset, "--manifest-dir", args.manifest_dir,
                "--output-dir", args.output_dir / label, "--eval-split", "val",
                "--attention-backend", "eager", "--num-traj-samples", "6", "--seed", "42",
                "--warmup-samples", "1", "--steps", "10", "5"]
            if inference_weights:
                command += ["--shortcut-inference-weights", inference_weights,
                            "--expected-ema-updates", "1986", "--verify-checkpoint-shortcut-config"]
            run(label, command)
            status(completed_evaluations=state["completed_evaluations"] + 1)
            run(f"bootstrap_{label}", [sys.executable, SCRIPT_DIR / "bootstrap_paired_step_regression.py",
                "--benchmark", args.output_dir / label / "benchmark_results.json",
                "--reference-step", "10", "--candidate-step", "5", "--iterations", "100000",
                "--output", args.output_dir / f"bootstrap_{label}_10_vs_5.json"])
        status(phase="summarizing")
        summarize(args)
        status(status="complete", phase="complete", finished_utc=utc(), report=str(args.output_dir / "REPORT.md"))
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc())
        raise


if __name__ == "__main__":
    main()
