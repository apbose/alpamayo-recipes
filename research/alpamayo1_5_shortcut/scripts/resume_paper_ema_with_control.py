#!/usr/bin/env python3
"""Complete A6 evaluations, then smoke/train/evaluate the A7 target-only control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess

SCRIPT = Path(__file__).resolve().parent
PROJECT = SCRIPT.parent
ROOT = PROJECT.parent.parent
RECIPE = ROOT / "recipes/alpamayo1_5_sft"
ASSETS = Path("/home/scratch.abose_sw/alpamayo-assets")
PYTHON = RECIPE / "a1_5_sft/bin/python"
TORCHRUN = RECIPE / "a1_5_sft/bin/torchrun"
METRICS = ("min_ade", "ade", "corner_distance")
MATCH_KEYS = ("validation_manifest_sha256", "validation_samples", "num_traj_samples",
              "seed_reset_for_each_step_count", "warmup_samples_per_step_count", "attention_backend",
              "torch_version", "torch_cuda_version", "gpu", "resolved_attention_backends")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def validate_result(result, steps, count=128):
    if set(result["results"]) != set(map(str, steps)):
        raise ValueError("Missing or unexpected solver counts")
    for value in result["results"].values():
        rows = value["per_sample"]
        if len(rows) != count or sorted(r["sample_index"] for r in rows) != list(range(count)):
            raise ValueError("Missing or duplicate validation clips")
        for metric in METRICS:
            if not math.isfinite(value["metrics"][metric]):
                raise ValueError("Non-finite aggregate metric")
            if any(len(r["metrics"][metric]) != 1 or not math.isfinite(r["metrics"][metric][0]) for r in rows):
                raise ValueError("Invalid per-clip metric")


def preflight(source, manifest):
    checkpoint = source / "training/checkpoint-249"
    state = read(source / "training/status.json")
    if state["status"] != "complete" or state["updates"] != 249:
        raise ValueError("A6 training is not complete")
    if read(checkpoint / "COMPLETE.json")["global_step"] != 249:
        raise ValueError("A6 checkpoint save incomplete")
    if read(checkpoint / "trainer_state.json")["ema_updates"] != 249:
        raise ValueError("Wrong saved EMA count")
    for name in set(read(checkpoint / "model.safetensors.index.json")["weight_map"].values()):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing checkpoint shard: {name}")
    old = read(source / "ema_eval_128/benchmark_results.json")
    validate_result(old, [128, 10, 8, 5, 4])
    if old["configuration"]["validation_manifest_sha256"] != digest(manifest / "val.json"):
        raise ValueError("Validation manifest changed")
    if old["configuration"]["checkpoint_config_sha256"] != digest(checkpoint / "config.json"):
        raise ValueError("Trained checkpoint configuration changed")
    train = PROJECT / "manifests/hf_stream_300gb/train.json"
    if digest(train) != read(source / "training/protocol.json")["train_manifest_sha256"]:
        raise ValueError("Training manifest changed")
    if {r["clip_id"] for r in read(train)} & {r["clip_id"] for r in read(manifest / "val.json")}:
        raise ValueError("Train/validation overlap")
    if not (ASSETS / "physical_ai_av/clip_index.parquet").is_file():
        raise ValueError("Local validation dataset is missing")
    if shutil.disk_usage(ASSETS).free < 200 * 1024**3:
        raise ValueError("Need 200 GiB free for smoke and two control checkpoints")
    return checkpoint, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=ASSETS / "runs/alpamayo15_paper_ema_5295clips_b64_20260920_r1")
    args = parser.parse_args()
    if args.control_run.exists():
        raise ValueError("Refusing to reuse an existing control run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs = args.output_dir / "logs"
    logs.mkdir()
    manifest = PROJECT / "manifests/route_less_19chunks_128eval"
    state = dict(status="running", phase="preflight", started_utc=utc(), pid=os.getpid(),
                 host=socket.gethostname(), source_run=str(args.source_run), control_run=str(args.control_run))

    def status(**values):
        state.update(values, updated_utc=utc())
        write(args.output_dir / "status.json", state)
        (args.output_dir / "STATUS").write_text("".join(f"{k}={v}\n" for k, v in state.items()))

    env = dict(os.environ)
    env.update(HF_HOME="/home/abose_sw/.cache/huggingface", HF_TOKEN_PATH="/home/abose_sw/.cache/huggingface/token",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1",
               DS_IGNORE_CUDA_DETECTION="1", OMP_NUM_THREADS="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
               TRITON_CACHE_DIR=f"/tmp/alpamayo-a6-a7-{os.getpid()}")

    def run(phase, command, training=False, timeout=8*3600):
        status(phase=phase)
        local_env = dict(env, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" if training else "0")
        if training:
            local_env.pop("HF_HUB_OFFLINE", None)
            local_env.pop("TRANSFORMERS_OFFLINE", None)
        else:
            local_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        with (args.output_dir / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(phase=phase, argv=list(map(str, command)), host=socket.gethostname(),
                                        offline=not training, cwd=str(ROOT))) + "\n")
        print(f"[{utc()}] {phase}", flush=True)
        with (logs / f"{phase}.log").open("w") as handle:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=local_env, stdin=subprocess.DEVNULL,
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def benchmark(phase, checkpoint, steps, model="ema", expected=249, manifests=manifest, count=128):
        configs = dict(ema="sft_stage2_trajectory_shortcut_paper_ema",
                       empirical="sft_stage2_trajectory_paper_empirical_ema", released="sft_stage2_trajectory_shortcut")
        command = [TORCHRUN, "--standalone", "--nproc_per_node=1", SCRIPT / "benchmark_inference_steps.py",
                   "--checkpoint", checkpoint, "--config-name", configs[model], "--dataset", ASSETS / "physical_ai_av",
                   "--manifest-dir", manifests, "--output-dir", args.output_dir / phase,
                   "--eval-split", "val", "--attention-backend", "eager", "--num-traj-samples", "6",
                   "--seed", "42", "--warmup-samples", "1", "--steps", *map(str, steps)]
        if model != "released":
            saved = read(checkpoint / "config.json").get("paper_supervision", "ema_bootstrap")
            if saved != ("empirical_velocity" if model == "empirical" else "ema_bootstrap"):
                raise ValueError("Checkpoint supervision disagrees with evaluation recipe")
            command += ["--shortcut-inference-weights", "ema", "--expected-ema-updates", str(expected),
                        "--verify-checkpoint-shortcut-config"]
        run(phase, command)
        result = read(args.output_dir / phase / "benchmark_results.json")
        validate_result(result, steps, count)
        return result

    def control_train(phase, updates):
        run(phase, [TORCHRUN, "--standalone", "--nproc_per_node=8", "-m", "alpamayo1_5_sft.train_paper_ema",
                    "--output-dir", args.control_run / phase, "--updates", updates,
                    "--save-every", updates if phase == "smoke" else 125, "--supervision", "empirical_velocity"],
            training=True, timeout=48*3600)
        metrics = [json.loads(row) for row in (args.control_run / phase / "metrics.jsonl").read_text().splitlines()]
        if len(metrics) != updates or any(row["flow_pairs"] != 64 or row["shortcut_pairs"] != 0 or row["ema_updates"] != i+1
                                         or not math.isfinite(row["loss"]) or not math.isfinite(row["grad_norm"])
                                         for i, row in enumerate(metrics)):
            raise ValueError("Control training objective/EMA accounting failed")
        if phase == "training":
            if read(args.control_run / phase / "raw_batch_plan.json") != read(args.source_run / "training/raw_batch_plan.json"):
                raise ValueError("A7 and A6 raw batch plans differ")
        return args.control_run / phase / f"checkpoint-{updates}"

    def bootstrap(phase, path, reference, candidate):
        run(phase, [PYTHON, SCRIPT / "bootstrap_paired_step_regression.py", "--benchmark", path,
                    "--reference-step", reference, "--candidate-step", candidate,
                    "--output", args.output_dir / f"{phase}.json"], timeout=900)

    def report(filename, datasets, extra):
        rows = ["# Alpamayo paper-target EMA: resumed experiment results", "",
                "128 fixed validation clips; six candidates; seed 42; eager attention. Lower error is better.", "",
                "| Model | Steps | minADE m | ADE m | Corner m | Expert ms | Model ms |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name, data in datasets.items():
            for step in sorted(data["results"], key=int, reverse=True):
                value = data["results"][step]
                m, lat = value["metrics"], value["latency_ms"]
                rows.append(f"| {name} | {step} | {m['min_ade']:.4f} | {m['ade']:.4f} | {m['corner_distance']:.4f} | {lat['action_expert_diffusion']['mean']:.2f} | {lat['end_to_end_model']['mean']:.2f} |")
        rows += ["", *extra, "", "This is an open-loop pilot, not a collision/off-road safety validation. MinADE is best-of-six. Model timings exclude data decoding.", ""]
        (args.output_dir / filename).write_text("\n".join(rows))

    try:
        status()
        checkpoint, historical = preflight(args.source_run, manifest)
        args.control_run.mkdir(parents=True, exist_ok=False)
        old_hashes = read(args.source_run / "source_hashes.json")
        write(args.output_dir / "provenance.json", dict(
            previous_source_hashes=old_hashes,
            current_source_hashes={name:digest(ROOT/name) for name in old_hashes},
            changes="Opt-in empirical target mode, train CLI/metrics support, and evaluation config; legacy EMA target values regression-tested.",
            historical_ema_result_sha256=digest(args.source_run / "ema_eval_128/benchmark_results.json"),
            preserved_source_run=True, retraining_A6=False, host=socket.gethostname()))
        run("tests", [PYTHON, "-m", "pytest", "-q", RECIPE / "tests/test_paper_empirical_ablation.py",
                       RECIPE / "tests/test_paper_shortcut.py", RECIPE / "tests/test_shortcut_model_config.py",
                       RECIPE / "tests/test_shortcut_modules.py"], timeout=1200)
        # Finish original statistical work now; it does not require another GPU run.
        for count in [10, 8, 5, 4]:
            bootstrap(f"historical_bootstrap_128_vs_{count}", args.source_run / "ema_eval_128/benchmark_results.json", 128, count)
        bootstrap("historical_bootstrap_10_vs_5", args.source_run / "ema_eval_128/benchmark_results.json", 10, 5)
        released = benchmark("released_eval_128", ASSETS / "checkpoints/Alpamayo-1.5-10B-A1-format", [128, 10], model="released")
        matched = benchmark("a6_matched_ema_eval", checkpoint, [128, 10, 5])
        for key in MATCH_KEYS:
            if matched["configuration"][key] != released["configuration"][key]:
                raise ValueError(f"A6/released protocol mismatch: {key}")
        report("A6_COMPLETED_REPORT.md", {"A6 EMA, this host":matched, "Released, this host":released},
               ["The completed historical 128/10/8/5/4 results were preserved. 128/10/5 EMA timings were repeated on this host; do not pool latency samples across hosts.",
                "New same-host inference is not new training. A6 remains the completed 249-update checkpoint."])
        write(args.output_dir / "A6_COMPLETE.json", dict(completed_utc=utc(), historical=historical, matched=matched, released=released))
        smoke = control_train("smoke", 2)
        smoke_manifest = args.output_dir / "smoke_manifest"
        smoke_manifest.mkdir()
        summary = read(manifest / "summary.json")
        summary["annotation_rows"]["val"] = 1
        write(smoke_manifest / "val.json", read(manifest / "val.json")[:1])
        write(smoke_manifest / "summary.json", summary)
        benchmark("control_smoke_reload", smoke, [10, 5], model="empirical", expected=2, manifests=smoke_manifest, count=1)
        write(args.output_dir / "CONTROL_SMOKE_PASSED.json", dict(passed=True, completed_utc=utc()))
        control_checkpoint = control_train("training", 249)
        control = benchmark("a7_empirical_ema_eval", control_checkpoint, [128, 10, 5], model="empirical")
        for key in MATCH_KEYS:
            if matched["configuration"][key] != control["configuration"][key]:
                raise ValueError(f"A6/A7 protocol mismatch: {key}")
        for label, data in [("a6_matched_ema_eval",matched), ("a7_empirical_ema_eval",control)]:
            for candidate in (10,5):
                bootstrap(f"{label}_128_vs_{candidate}", args.output_dir/label/"benchmark_results.json",128,candidate)
            bootstrap(f"{label}_10_vs_5", args.output_dir/label/"benchmark_results.json",10,5)
        # Existing paired-bootstrap code can compare models by using explicit
        # model labels as keys; no solver count is relabeled in source results.
        comparison = {"configuration":matched["configuration"], "results":{}}
        for step in ("128","10","5"):
            comparison["results"][f"A6_{step}"] = matched["results"][step]
            comparison["results"][f"A7_{step}"] = control["results"][step]
        write(args.output_dir / "cross_model_pairs.json", comparison)
        for step in (128,10,5):
            bootstrap(f"A7_vs_A6_{step}", args.output_dir / "cross_model_pairs.json", f"A6_{step}", f"A7_{step}")
        report("REPORT.md", {"A6 EMA":matched, "A7 empirical-target EMA":control, "Released":released},
               ["A7 replaces the 16 EMA-bootstrap targets with empirical velocities, while retaining ALL 64 input/source/noise/time/d assignments and EMA updates. It is not adapter-free vanilla flow matching.",
                "Both fine-tunes start from the released checkpoint: 249 updates, seed 10, global target batch 64, identical raw batch plan and optimizer. A7 does not continue from A6.",
                "EMA still retains roughly 77.9% initialization weight after 249 updates; these are short pilots, not convergence claims.",
                "Cross-model paired clip confidence intervals: A7_vs_A6_*.md. Ten and five steps are off-grid interpolation diagnostics."])
        write(args.output_dir / "summary.json", dict(a6=matched, a7=control, released=released, safety_evaluated=False))
        status(status="complete", phase="complete", finished_utc=utc(), report=str(args.output_dir / "REPORT.md"))
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc())
        raise


if __name__ == "__main__":
    main()
