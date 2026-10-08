#!/usr/bin/env python3
"""A8 preflight -> phase streaming checks -> optional A6 smoke/full training.

Defaults to preparation only. --train explicitly enables GPU training. This
runner never overwrites an existing run directory or resumes optimizer state.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=PROJECT / "manifests/a8_fresh_data_3x5295/schedule.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Released A1-format checkpoint, NOT trained A6")
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--train", action="store_true", help="After checks, run two-update smoke and fresh 249-update training")
    args = parser.parse_args()
    if args.nproc_per_node < 1 or 64 % args.nproc_per_node:
        raise ValueError("nproc-per-node must divide 64")
    args.run_dir, args.schedule = args.run_dir.resolve(), args.schedule.resolve()
    args.checkpoint, args.hf_cache = args.checkpoint.resolve(), args.hf_cache.resolve()
    # Prevent silently warm-starting from a trained shortcut/EMA checkpoint.
    config = json.loads((args.checkpoint / "config.json").read_text())
    if any(key.startswith(("shortcut_", "paper_")) for key in config):
        raise ValueError("Use the released checkpoint: trained shortcut weights are not an A6-matched initialization")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "logs").mkdir()
    state = dict(status="preparing", pid=os.getpid(), training_requested=args.train)
    def status(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = args.run_dir / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(args.run_dir / "status.json")
    env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               DS_IGNORE_CUDA_DETECTION="1", HYDRA_FULL_ERROR="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    def run(phase, command):
        status(phase=phase)
        argv = [str(arg) for arg in command]
        with (args.run_dir / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(phase=phase, argv=argv, cwd=str(ROOT))) + "\n")
        print(phase, flush=True)
        with (args.run_dir / "logs" / f"{phase}.log").open("w") as handle:
            # No wall-clock deadline; failures stop before subsequent phases.
            subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
    common = ["--fresh-data-schedule", args.schedule, "--checkpoint", args.checkpoint,
              "--hf-cache", args.hf_cache, "--bootstrap-every", "4", "--supervision", "ema_bootstrap"]
    try:
        run("tests", [sys.executable, "-m", "pytest", "-q", ROOT / "recipes/alpamayo1_5_sft/tests/test_paper_fresh_data.py",
                      ROOT / "recipes/alpamayo1_5_sft/tests/test_paper_shortcut.py"])
        run("plan", [sys.executable, "-m", "alpamayo1_5_sft.train_paper_ema", *common,
                     "--plan-only", "--updates", "249", "--output-dir", args.run_dir / "plan"])
        for phase in (1, 2, 3):
            run(f"stream_phase_{phase}", [sys.executable, PROJECT / "scripts/benchmark_hf_streaming.py",
                "--manifest", args.schedule.parent / f"phase_{phase}.json",
                "--summary", args.schedule.parent / f"phase_{phase}_summary.json",
                "--cache-dir", args.hf_cache, "--output", args.run_dir / f"stream_phase_{phase}.json",
                "--samples", "2", "--workers", "0"])
        if not args.train:
            status(status="ready_not_training")
            return
        launch = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                  f"--nproc_per_node={args.nproc_per_node}", "-m", "alpamayo1_5_sft.train_paper_ema", *common]
        run("gpu_smoke", [*launch, "--updates", "2", "--no-save", "--output-dir", args.run_dir / "gpu_smoke"])
        metrics = [json.loads(line) for line in (args.run_dir / "gpu_smoke/metrics.jsonl").read_text().splitlines()]
        if len(metrics) != 2 or any(row["flow_pairs"] != 48 or row["shortcut_pairs"] != 16
                                    or row["ema_updates"] != i + 1 for i, row in enumerate(metrics)):
            raise RuntimeError("GPU smoke failed target/EMA accounting")
        # Start afresh from the released checkpoint; do not carry smoke updates.
        run("training", [*launch, "--updates", "249", "--save-every", "83", "--output-dir", args.run_dir / "training"])
        metrics = [json.loads(line) for line in (args.run_dir / "training/metrics.jsonl").read_text().splitlines()]
        if len(metrics) != 249 or any(row["flow_pairs"] != 48 or row["shortcut_pairs"] != 16
                                      or row["ema_updates"] != i + 1 or row["data_phase"] != i // 83 + 1
                                      for i, row in enumerate(metrics)):
            raise RuntimeError("Full run failed target/EMA/phase accounting")
        status(status="training_complete_evaluation_pending", checkpoint=str(args.run_dir / "training/checkpoint-249"))
    except BaseException as error:
        status(status="failed", error=type(error).__name__ + ": " + str(error))
        raise


if __name__ == "__main__":
    main()
