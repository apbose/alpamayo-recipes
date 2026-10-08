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
import time

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


def dependency_ready(directory):
    """Fail closed: a dead/failed gold evaluation never releases the training queue."""
    state = read(directory / "status.json")
    if state.get("host") != socket.gethostname():
        raise ValueError("Evaluation dependency belongs to a different host")
    if state["status"] in ("failed", "cancelled", "blocked"):
        raise RuntimeError(f"Evaluation dependency did not succeed: {state['status']}")
    if state["status"] != "complete":
        os.kill(int(state["pid"]), 0)
        return False
    from evaluate_gold_r0_a6 import validate_result as validate_gold
    for model in ("R0", "A6"):
        result = read(directory / f"{model}_merged_benchmark_results.json")
        if set(result.get("results", {})) != {"10", "5", "4", "2"}:
            raise ValueError("Evaluation dependency has incomplete solver counts")
        validate_gold(result, model, directory / "manifest/val.json", (10, 5, 4, 2))
    return True


def checkpoint_storage_budget(checkpoint):
    """One smoke save plus one final save, including optimizer; no deletion required."""
    size = sum(path.stat().st_size for path in checkpoint.iterdir() if path.is_file())
    return 2 * size + 20 * 1024**3


def validate_control_plan(reference_dir, candidate_dir):
    if read(reference_dir / "raw_batch_plan.json") != read(candidate_dir / "raw_batch_plan.json"):
        raise ValueError("A7 and A6 raw batch plans differ")
    original, control = (read(path / "protocol.json") for path in (reference_dir, candidate_dir))
    keys = ("train_manifest_sha256", "seed", "dt_base", "source_indices", "optimizer_updates",
            "base_checkpoint", "learning_rate", "adam_betas", "adam_epsilon", "weight_decay",
            "weight_decay_all_trainable_parameters", "schedule", "warmup", "gradient_clipping",
            "student_parameter_dtype", "ema_dtype", "autocast", "ema_decay", "attention", "hf_revision")
    for key in keys:
        if original[key] != control[key]:
            raise ValueError(f"A6/A7 training protocol mismatch: {key}")
    if control["supervision"] != "empirical_velocity" or control["teacher_target_pairs"] != 0 or control["empirical_target_pairs"] != 64:
        raise ValueError("A7 must use exactly 64 empirical targets and no teacher targets")


def preflight(source, manifest, storage_parent=ASSETS, minimum_free=200 * 1024**3):
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
    if shutil.disk_usage(storage_parent).free < minimum_free:
        raise ValueError(f"Need {minimum_free / 1024**3:.1f} GiB free for this checkpoint-save schedule")
    return checkpoint, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=ASSETS / "runs/alpamayo15_paper_ema_5295clips_b64_20260920_r1")
    parser.add_argument("--control-only", action="store_true",
                        help="Train the missing A7 first; then matched A6/A7 10/5/4/2 evaluation, without repeating R0/128-step work")
    parser.add_argument("--after-evaluation", type=Path,
                        help="Wait for successful gold evaluation and all GPUs to be free before smoke/training")
    parser.add_argument("--hf-stream-max-attempts", type=int, default=1)
    args = parser.parse_args()
    if args.after_evaluation and not args.control_only:
        parser.error("--after-evaluation requires --control-only")
    if args.hf_stream_max_attempts < 1:
        parser.error("--hf-stream-max-attempts must be positive")
    if args.control_run.exists():
        raise ValueError("Refusing to reuse an existing control run")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logs = args.output_dir / "logs"
    logs.mkdir()
    manifest = PROJECT / "manifests/route_less_19chunks_128eval"
    state = dict(status="running", phase="preflight", started_utc=utc(), pid=os.getpid(),
                 host=socket.gethostname(), source_run=str(args.source_run), control_run=str(args.control_run),
                 control_only=args.control_only, after_evaluation=str(args.after_evaluation) if args.after_evaluation else None)

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
        run(phase, command, timeout=None if args.control_only else 8*3600)
        result = read(args.output_dir / phase / "benchmark_results.json")
        validate_result(result, steps, count)
        rows = read(manifests / "val.json")
        if result["configuration"]["validation_manifest_sha256"] != digest(manifests / "val.json"):
            raise ValueError("Evaluation manifest changed")
        for value in result["results"].values():
            for item in value["per_sample"]:
                expected_row = rows[item["sample_index"]]
                if any(item[key] != expected_row[key] for key in ("clip_id", "t0_relative")):
                    raise ValueError("Evaluation clip/timestamp pairing changed")
        return result

    def control_train(phase, updates):
        extra = ["--extra-heldout-manifests", args.after_evaluation / "manifest/val.json"] if args.after_evaluation else []
        run(phase, [TORCHRUN, "--standalone", "--nproc_per_node=8", "-m", "alpamayo1_5_sft.train_paper_ema",
                    "--output-dir", args.control_run / phase, "--updates", updates,
                    "--save-every", updates if phase == "smoke" or args.control_only else 125,
                    "--supervision", "empirical_velocity", "--hf-stream-max-attempts", args.hf_stream_max_attempts, *extra],
            training=True, timeout=None if args.control_only else 48*3600)
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
                "| Model | Steps | minADE m | minADE vs own 10 | ADE m | Corner m | Expert ms | Model ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, data in datasets.items():
            for step in sorted(data["results"], key=int, reverse=True):
                value = data["results"][step]
                m, lat = value["metrics"], value["latency_ms"]
                change = 100 * (m['min_ade'] / data['results']['10']['metrics']['min_ade'] - 1)
                rows.append(f"| {name} | {step} | {m['min_ade']:.4f} | {change:+.2f}% | {m['ade']:.4f} | {m['corner_distance']:.4f} | {lat['action_expert_diffusion']['mean']:.2f} | {lat['end_to_end_model']['mean']:.2f} |")
        rows += ["", *extra, "", "This is an open-loop pilot, not a collision/off-road safety validation. MinADE is best-of-six. Model timings exclude data decoding.", ""]
        (args.output_dir / filename).write_text("\n".join(rows))

    try:
        status()
        if args.after_evaluation:
            status(status="waiting", phase="waiting_for_gold_evaluation")
            while not dependency_ready(args.after_evaluation):
                status(dependency=read(args.after_evaluation / "status.json")["status"])
                time.sleep(30)
            status(phase="waiting_for_free_gpus", dependency="complete")
            while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
                time.sleep(30)
                status(phase="waiting_for_free_gpus")
        budget = checkpoint_storage_budget(args.source_run / "training/checkpoint-249") if args.control_only else 200 * 1024**3
        status(status="running", phase="preflight", required_free_bytes=budget)
        checkpoint, historical = preflight(args.source_run, manifest, args.control_run.parent, budget)
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
        if args.control_only:
            # CPU-only preflight; no model allocation, data download, or optimizer update.
            extra = ["--extra-heldout-manifests", args.after_evaluation / "manifest/val.json"] if args.after_evaluation else []
            run("plan", [PYTHON, "-m", "alpamayo1_5_sft.train_paper_ema", "--plan-only",
                         "--output-dir", args.output_dir / "plan", "--updates", "249",
                         "--supervision", "empirical_velocity", *extra], timeout=1200)
            validate_control_plan(args.source_run / "training", args.output_dir / "plan")
            smoke = control_train("smoke", 2)
            smoke_manifest = args.output_dir / "smoke_manifest"
            smoke_manifest.mkdir()
            summary = read(manifest / "summary.json")
            summary["annotation_rows"]["val"] = 1
            write(smoke_manifest / "val.json", read(manifest / "val.json")[:1])
            write(smoke_manifest / "summary.json", summary)
            benchmark("control_smoke_reload", smoke, [10, 2], model="empirical", expected=2,
                      manifests=smoke_manifest, count=1)
            write(args.output_dir / "CONTROL_SMOKE_PASSED.json", dict(passed=True, completed_utc=utc()))
            # Full training starts fresh from R0, never from the two-update smoke.
            control_checkpoint = control_train("training", 249)
            validate_control_plan(args.source_run / "training", args.control_run / "training")
            steps = (10, 5, 4, 2)
            control = benchmark("a7_empirical_ema_eval", control_checkpoint, steps, model="empirical")
            matched = benchmark("a6_matched_ema_eval", checkpoint, steps)
            for key in MATCH_KEYS:
                if matched["configuration"][key] != control["configuration"][key]:
                    raise ValueError(f"A6/A7 protocol mismatch: {key}")
            comparison = {"configuration": matched["configuration"], "results": {}}
            for label, data in (("a6_matched_ema_eval", matched), ("a7_empirical_ema_eval", control)):
                for candidate in (5, 4, 2):
                    bootstrap(f"{label}_10_vs_{candidate}", args.output_dir / label / "benchmark_results.json", 10, candidate)
            for count in steps:
                comparison["results"][f"A6_{count}"] = matched["results"][str(count)]
                comparison["results"][f"A7_{count}"] = control["results"][str(count)]
            write(args.output_dir / "cross_model_pairs.json", comparison)
            for count in steps:
                bootstrap(f"A7_vs_A6_{count}", args.output_dir / "cross_model_pairs.json", f"A6_{count}", f"A7_{count}")
            report("REPORT.md", {"A6 EMA": matched, "A7 empirical-target EMA": control}, [
                "A7 uses the same A6 input/source/noise/time/d assignments, 249 optimizer updates, seed 10, and EMA policy; only the 16 bootstrap-layout targets change to empirical velocity.",
                "Both start from R0. A7 is a target-only control, NOT adapter-free ordinary flow matching.",
                "This is the existing 128-clip held-out population, NOT the gold644 population. Original gold results are preserved.",
                "Within-checkpoint 10-to-5/4/2 degradation and matched A7-vs-A6 paired bootstrap intervals are saved separately.",
                "Single-seed, short-budget experiment; no causal conclusion from absolute fine-tuning gains alone and no safety pass.",
            ])
            write(args.output_dir / "summary.json", dict(a6=matched, a7=control, safety_evaluated=False))
            status(status="complete", phase="complete", finished_utc=utc(), report=str(args.output_dir / "REPORT.md"))
            return
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
