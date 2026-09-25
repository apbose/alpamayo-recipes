#!/usr/bin/env python3
"""Detached, fail-closed paper EMA smoke -> training -> evaluation pipeline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parent
PROJECT = SCRIPT.parent
ROOT = PROJECT.parent.parent
RECIPE = ROOT / "recipes/alpamayo1_5_sft"
ASSETS = Path("/home/scratch.abose_sw/alpamayo-assets")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=249)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    logs = args.run_dir / "logs"
    logs.mkdir()
    state = dict(status="running", phase="preflight", started_utc=utc(), pid=os.getpid())
    def status(**values):
        state.update(values, updated_utc=utc())
        write_json(args.run_dir / "status.json", state)
        (args.run_dir / "STATUS").write_text("".join(f"{k}={v}\n" for k, v in state.items()))
    env = dict(os.environ)
    # Training preserves the existing HF on-demand source. Only evaluation is
    # strictly network-offline; all phases are detached from the user's session.
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.update(HF_HOME="/home/abose_sw/.cache/huggingface", HF_TOKEN_PATH="/home/abose_sw/.cache/huggingface/token",
        TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1",
        DS_IGNORE_CUDA_DETECTION="1", OMP_NUM_THREADS="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        TRITON_CACHE_DIR=f"/tmp/alpamayo-paper-ema-{os.getpid()}")
    torchrun = RECIPE / "a1_5_sft/bin/torchrun"
    python = RECIPE / "a1_5_sft/bin/python"
    checkpoint_base = ASSETS / "checkpoints/Alpamayo-1.5-10B-A1-format"
    manifests = PROJECT / "manifests/route_less_19chunks_128eval"

    def run(phase, argv, *, training=False, timeout=48 * 3600):
        status(phase=phase)
        phase_env = dict(env, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" if training else "0")
        if not training:
            phase_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (args.run_dir / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(phase=phase, argv=[str(x) for x in argv], cwd=str(ROOT))) + "\n")
        print(f"[{utc()}] {phase}", flush=True)
        with (logs / f"{phase}.log").open("w") as handle:
            subprocess.run([str(x) for x in argv], cwd=ROOT, env=phase_env,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def train(phase, updates):
        run(phase, [torchrun, "--standalone", "--nproc_per_node=8", "-m", "alpamayo1_5_sft.train_paper_ema",
            "--output-dir", args.run_dir / phase, "--updates", str(updates), "--save-every", str(updates if phase == "smoke" else 83)], training=True)
        metrics = [json.loads(line) for line in (args.run_dir / phase / "metrics.jsonl").read_text().splitlines()]
        if len(metrics) != updates or any(row["flow_pairs"] != 48 or row["shortcut_pairs"] != 16 or row["ema_updates"] != i + 1 for i, row in enumerate(metrics)):
            raise RuntimeError("Training target/EMA accounting failed")
        return args.run_dir / phase / f"checkpoint-{updates}"

    def benchmark(phase, checkpoint, steps, *, manifest=manifests, expected=None):
        cmd = [torchrun, "--standalone", "--nproc_per_node=1", SCRIPT / "benchmark_inference_steps.py",
            "--checkpoint", checkpoint, "--config-name", "sft_stage2_trajectory_shortcut_paper_ema" if expected is not None else "sft_stage2_trajectory_shortcut",
            "--dataset", ASSETS / "physical_ai_av", "--manifest-dir", manifest,
            "--output-dir", args.run_dir / phase, "--eval-split", "val", "--attention-backend", "eager",
            "--num-traj-samples", "6", "--seed", "42", "--warmup-samples", "1", "--steps", *map(str, steps)]
        if expected is not None:
            cmd += ["--shortcut-inference-weights", "ema", "--expected-ema-updates", str(expected), "--verify-checkpoint-shortcut-config"]
        run(phase, cmd, timeout=8 * 3600)
        result = json.loads((args.run_dir / phase / "benchmark_results.json").read_text())
        if set(result["results"]) != set(map(str, steps)):
            raise RuntimeError("Incomplete benchmark")
        return result

    try:
        status()
        files = [RECIPE / "models/paper_shortcut_targets.py", RECIPE / "models/paper_shortcut_alpamayo.py", RECIPE / "train_paper_ema.py", SCRIPT / "benchmark_inference_steps.py"]
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        write_json(args.run_dir / "source_hashes.json", hashes)
        run("tests", [python, "-m", "pytest", "-q", RECIPE / "tests/test_paper_shortcut.py", RECIPE / "tests/test_shortcut_model_config.py", RECIPE / "tests/test_shortcut_modules.py"], timeout=900)
        checkpoint = train("smoke", 2)
        smoke_manifest = args.run_dir / "smoke_manifest"
        smoke_manifest.mkdir()
        rows = json.loads((manifests / "val.json").read_text())[:1]
        summary = json.loads((manifests / "summary.json").read_text())
        summary["annotation_rows"]["val"] = 1
        write_json(smoke_manifest / "val.json", rows)
        write_json(smoke_manifest / "summary.json", summary)
        # Checkpoint serialization and actual inference must pass before the
        # long run starts; EMA must restore at precisely two updates.
        benchmark("smoke_reload_eval", checkpoint, [10, 8], manifest=smoke_manifest, expected=2)
        write_json(args.run_dir / "SMOKE_PASSED.json", dict(passed=True, finished_utc=utc(), checkpoint=str(checkpoint)))
        checkpoint = train("training", args.updates)
        trained = benchmark("ema_eval_128", checkpoint, [128, 10, 8, 5, 4], expected=args.updates)
        released = benchmark("released_eval_128", checkpoint_base, [128, 10])
        for count in [10, 8, 5, 4]:
            run(f"bootstrap_128_vs_{count}", [python, SCRIPT / "bootstrap_paired_step_regression.py",
                "--benchmark", args.run_dir / "ema_eval_128/benchmark_results.json", "--reference-step", "128",
                "--candidate-step", str(count), "--output", args.run_dir / f"bootstrap_128_vs_{count}.json"], timeout=900)
        run("bootstrap_10_vs_5", [python, SCRIPT / "bootstrap_paired_step_regression.py",
            "--benchmark", args.run_dir / "ema_eval_128/benchmark_results.json", "--reference-step", "10",
            "--candidate-step", "5", "--output", args.run_dir / "bootstrap_10_vs_5.json"], timeout=900)
        lines = ["# Paper-target EMA: completed results", "", "Target construction follows the pinned no-CFG reference with paper 75/25 allocation. Alpamayo architecture/data and the short training budget differ from the image paper.", "",
            "| Model | Steps | minADE (m) | ADE (m) | Corner (m) | Expert ms | Model ms |", "|---|---:|---:|---:|---:|---:|---:|"]
        for label, result in [("Paper-target EMA", trained), ("Released", released)]:
            for step, values in result["results"].items():
                m, latency = values["metrics"], values["latency_ms"]
                lines.append(f"| {label} | {step} | {m['min_ade']:.4f} | {m['ade']:.4f} | {m['corner_distance']:.4f} | {latency['action_expert_diffusion']['mean']:.2f} | {latency['end_to_end_model']['mean']:.2f} |")
        lines += ["", "10 and 5 steps interpolate outside the trained power-of-two grid. 128, 8 and 4 steps are on-grid. No one-step success or collision/off-road safety claim is made.", "", "Compare to earlier results only with the same held-out clips/protocol. This run changes several training factors together; it is not a one-factor EMA ablation.", ""]
        (args.run_dir / "REPORT.md").write_text("\n".join(lines))
        write_json(args.run_dir / "summary.json", dict(trained=trained, released=released, safety_evaluated=False))
        status(status="complete", phase="complete", finished_utc=utc(), report=str(args.run_dir / "REPORT.md"))
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc())
        raise


if __name__ == "__main__":
    main()
