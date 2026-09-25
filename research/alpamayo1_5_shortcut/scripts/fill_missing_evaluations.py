#!/usr/bin/env python3
"""Detached, resumable A2--A6 evaluation only; no benchmark wall-clock timeout.

One independent single-GPU process per checkpoint. Historical results are read
only. Reports use historical ten-step QUALITY baselines, never cross-run timing
speedups. A7 is not launched, and one-step inference is outside this suite.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
REPO = PROJECT.parent.parent
WORKSPACE = REPO.parent.parent
RECIPE = REPO / "recipes/alpamayo1_5_sft"
ASSETS = Path("/home/scratch.abose_sw/alpamayo-assets")
RUNS = ASSETS / "runs"
MANIFEST = PROJECT / "manifests/route_less_19chunks_128eval"
DATASET = ASSETS / "physical_ai_av"
VAL_SHA = "d7dc21959b746dacdfee2f559550b329783d4664db88d3cacb5590323fec9e62"
FLOW_RESULTS = WORKSPACE / "results/alpamayo15_hf300gb_flow_control_followups_20260914_r1"
EMA_RESULTS = WORKSPACE / "results/alpamayo15_hf300gb_reference_ema_followups_20260915_r1"
FIVE_RESULTS = WORKSPACE / "results/alpamayo15_10to5_eval_128clips_20260920_r1"
PAPER_RUN = RUNS / "alpamayo15_paper_ema_5295clips_b64_20260920_r1"
METRICS = ("min_ade", "ade", "corner_distance")
STEPS = (10, 5, 4, 2)
LABELS = {
    "R0": "Released", "A2": "Current-student shortcut / student",
    "A3": "Flow-only / student", "A4": "EMA shortcut / EMA",
    "A5": "Targeted 10-to-5 / EMA", "A6": "Paper-target hierarchy / EMA",
}
JOBS = [
    dict(id="A2", steps=[5], recipe="sft_stage2_trajectory_shortcut", weights=None, updates=None,
         checkpoint=RUNS / "alpamayo15_hf300gb_reference_5295clips_1epoch_20260913_r2/trainer_output/checkpoint-662",
         sources=[FLOW_RESULTS / "shortcut_128/benchmark_results.json"]),
    dict(id="A3", steps=[5], recipe="sft_stage2_trajectory_shortcut", weights=None, updates=None,
         checkpoint=RUNS / "alpamayo15_hf300gb_flow_control_5295clips_1epoch_20260914_r1/trainer_output/checkpoint-662",
         sources=[FLOW_RESULTS / "flow_control_128/benchmark_results.json"]),
    dict(id="A4", steps=[5], recipe="sft_stage2_trajectory_shortcut_reference_ema", weights="ema", updates=662,
         checkpoint=RUNS / "alpamayo15_hf300gb_reference_ema_5295clips_1epoch_20260915_r1/trainer_output/checkpoint-662",
         sources=[EMA_RESULTS / "ema_weights_128/benchmark_results.json"]),
    dict(id="A5", steps=[4, 2], recipe="sft_stage2_trajectory_shortcut_10to5_reference_ema", weights="ema", updates=1986,
         checkpoint=RUNS / "alpamayo15_hf300gb_10to5_reference_ema_5295clips_3epochs_20260919_r2/trainer_output/checkpoint-1986",
         sources=[FIVE_RESULTS / "ema_weights_128/benchmark_results.json"]),
    dict(id="A6", steps=[2], recipe="sft_stage2_trajectory_shortcut_paper_ema", weights="ema", updates=249,
         checkpoint=PAPER_RUN / "training/checkpoint-249",
         sources=[PAPER_RUN / "ema_eval_128/benchmark_results.json"]),
]
RELEASED = dict(id="R0", recipe="sft_stage2_trajectory_shortcut", weights=None, updates=None,
    checkpoint=ASSETS / "checkpoints/Alpamayo-1.5-10B-A1-format",
    sources=[FLOW_RESULTS / "released_128/benchmark_results.json", FIVE_RESULTS / "released_128/benchmark_results.json"])


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def validate_benchmark(benchmark, job):
    cfg = benchmark["configuration"]
    expected = dict(validation_manifest_sha256=VAL_SHA, validation_samples=128,
        validation_unique_clips=128, num_traj_samples=6, seed_reset_for_each_step_count=42,
        warmup_samples_per_step_count=1, attention_backend="eager", evaluation_split="val",
        zero_step_size_adapter=False, step_size_adapter_scale=1.0,
        recipe_config=job["recipe"], gpu="NVIDIA B300 SXM6 AC",
        torch_version="2.8.0+cu129", torch_cuda_version="12.9")
    expected["checkpoint_config_sha256"] = digest(job["checkpoint"] / "config.json")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"{job['id']}: unmatched {key}: {cfg.get(key)!r} != {value!r}")
    if Path(cfg["checkpoint"]).resolve() != job["checkpoint"].resolve():
        raise ValueError(f"{job['id']}: checkpoint identity changed")
    if job["weights"] == "ema":
        runtime = cfg.get("shortcut_runtime", {})
        if runtime.get("inference_weights") != "ema" or runtime.get("updates") != job["updates"]:
            raise ValueError(f"{job['id']}: wrong EMA weights or update count")
    for step, result in benchmark["results"].items():
        samples = result["per_sample"]
        if [sample["sample_index"] for sample in samples] != list(range(128)):
            raise ValueError(f"{job['id']}/{step}: incomplete or reordered evaluation")
        for metric in METRICS:
            if not math.isfinite(result["metrics"][metric]) or result["metrics"][metric] < 0:
                raise ValueError(f"{job['id']}/{step}: invalid {metric}")
        for key in ("end_to_end_model", "action_expert_diffusion"):
            if not math.isfinite(result["latency_ms"][key]["mean"]) or result["latency_ms"][key]["mean"] <= 0:
                raise ValueError(f"{job['id']}/{step}: invalid timing")


def historical_results():
    merged = {}
    for job in [RELEASED, *JOBS]:
        merged[job["id"]] = {}
        for path in job["sources"]:
            benchmark = read_json(path)
            validate_benchmark(benchmark, job)
            for step, value in benchmark["results"].items():
                if int(step) not in STEPS:
                    continue
                previous = merged[job["id"]].get(step)
                if previous:
                    for metric in METRICS:
                        if not math.isclose(previous["result"]["metrics"][metric], value["metrics"][metric], abs_tol=1e-8, rel_tol=1e-7):
                            raise ValueError(f"Conflicting historical quality for {job['id']}/{step}")
                    continue
                merged[job["id"]][step] = dict(result=value, source=str(path), origin="historical")
    return merged


def collect_results(output, historical):
    merged = {key: dict(value) for key, value in historical.items()}
    for job in JOBS:
        for path in sorted(output.glob(f"{job['id']}/attempt_*/benchmark_results.json")):
            benchmark = read_json(path)
            validate_benchmark(benchmark, job)
            for step, result in benchmark["results"].items():
                if int(step) not in job["steps"]:
                    raise ValueError(f"Unexpected new evaluation: {job['id']}/{step}")
                merged[job["id"]][step] = dict(result=result, source=str(path), origin="new")
    return merged


def write_report(output, doc_path, merged, state):
    completed = sum(str(step) in merged[job["id"]] for job in JOBS for step in job["steps"])
    lines = ["# Alpamayo 1.5: missing inference evaluations", "", f"Updated UTC: {utc()}", "",
        f"Suite: **{state['status']}**. New cells complete: **{completed}/6**.", "",
        "Evaluation only: unchanged saved checkpoints, fixed 128 clip-disjoint validation windows, "
        "six candidates, seed 42 reset per solver count, eager attention, one excluded warm-up. "
        "The existing full VLM-generation path is unchanged. No training, one-step evaluation, "
        "A7 launch, or GPU power-setting changes are performed.", "",
        "## minADE (metres; lower is better)", "",
        "Each bracket is percentage change from that checkpoint's existing ten-step minADE. "
        "Positive is worse. These are quality comparisons, not cross-host latency comparisons.", "",
        "| Model / weights | 10 steps | 5 steps | 4 steps | 2 steps |",
        "|---|---:|---:|---:|---:|"]
    rows = []
    for label in LABELS:
        values = merged[label]
        baseline = values["10"]["result"]["metrics"]
        cells = []
        for step in STEPS:
            record = values.get(str(step))
            if record is None:
                cells.append("Pending")
                continue
            metrics = record["result"]["metrics"]
            change = 100 * (metrics["min_ade"] / baseline["min_ade"] - 1)
            cells.append(f"{metrics['min_ade']:.4f}" + (f" ({change:+.2f}%)" if step != 10 else ""))
            row = dict(experiment=label, weights=LABELS[label], steps=step, **{m: metrics[m] for m in METRICS},
                **{m + "_change_vs_10_percent": 100 * (metrics[m] / baseline[m] - 1) for m in METRICS},
                expert_ms=record["result"]["latency_ms"]["action_expert_diffusion"]["mean"],
                model_ms=record["result"]["latency_ms"]["end_to_end_model"]["mean"],
                origin=record["origin"], source=record["source"])
            rows.append(row)
        lines.append(f"| {label}: {LABELS[label]} | " + " | ".join(cells) + " |")
    lines += ["| A7: empirical-target control | Not trained | Not trained | Not trained | Not trained |", "",
        "## Complete quality measurements", "",
        "| Model | Steps | minADE m | ADE m | Corner m | Source |", "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['steps']} | {row['min_ade']:.4f} | {row['ade']:.4f} | {row['corner_distance']:.4f} | [{row['origin']}]({row['source']}) |")
    lines += ["", "## New timing measurements (do not pool with historical timings)", "",
        "Concurrent single-GPU evaluations on one shared host. Power caps and host contention "
        "can affect timings. GPU 5 had a higher enforced limit at launch; other assigned GPUs "
        "reported 200 W. No claim of matched latency speedup across runs or GPUs is made. "
        "Model-call timing excludes data loading/decoding and metric calculation; each call "
        "already includes six candidates.", "",
        "| Model | Steps | Expert ms | Full model seconds |", "|---|---:|---:|---:|"]
    for row in rows:
        if row["origin"] == "new":
            lines.append(f"| {row['experiment']} | {row['steps']} | {row['expert_ms']:.2f} | {row['model_ms']/1000:.3f} |")
    lines += ["", "## Interpretation limits", "",
        "- Training budgets and architectures remain different; filling the table does not create a matched training ablation.",
        "- A5 was trained at d=0.1/0.2; its four/two-step results test extrapolation. A2/A4 five-step d=0.2 is off their training grid.",
        "- A2/A3 retain their historical evaluation wrapper. Its training-loss settings are not used for inference; A3's missing d adapter is zero-initialized by the existing loader.",
        "- Best-of-six minADE is not a deployed trajectory-selection policy. Collision/off-road/closed-loop safety is not evaluated.",
        "- No benchmark wall-clock timeout. Errors still fail the affected job; other independent jobs continue. Resume only retries incomplete solver settings.", "",
        "## Runtime", "", f"Output: `{output}`", "", f"Runner PID: {state.get('pid')}; host: {state.get('host')}", "",
        "| Job | Physical GPU | State |", "|---|---:|---|"]
    for job_id, status in state.get("jobs", {}).items():
        lines.append(f"| {job_id} | {status.get('gpu')} | {status.get('status')} |")
    text = "\n".join(lines) + "\n"
    atomic_text(output / "REPORT.md", text)
    atomic_text(doc_path, text)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "comparison.csv", buffer.getvalue())
    write_json(output / "summary.json", dict(updated_utc=utc(), completed_missing_cells=completed,
        total_missing_cells=6, rows=rows, safety_evaluated=False, cross_run_latency_speedups_computed=False))
    return completed


def gpu_snapshot():
    fields = "index,uuid,name,utilization.gpu,memory.used,clocks.current.sm,power.limit,enforced.power.limit,clocks_event_reasons.sw_power_cap"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=20)
    return [dict(zip(fields.split(","), [x.strip() for x in row])) for row in csv.reader(result.stdout.splitlines())]


def preflight(output):
    if digest(MANIFEST / "val.json") != VAL_SHA:
        raise ValueError("Validation manifest changed")
    training = read_json(PROJECT / "manifests/hf_stream_300gb/train.json")
    validation = read_json(MANIFEST / "val.json")
    train_ids = {row["clip_id"] for row in training}
    val_ids = {row["clip_id"] for row in validation}
    if len(training) != 5295 or len(train_ids) != 5295 or len(validation) != 128 or len(val_ids) != 128 or train_ids & val_ids:
        raise ValueError("Unexpected dataset sizes or train/validation overlap")
    if any("nav_text" in row for row in validation) or not (DATASET / "clip_index.parquet").is_file():
        raise ValueError("Invalid route-less local dataset")
    files = {}
    for job in JOBS:
        checkpoint = job["checkpoint"]
        index = read_json(checkpoint / "model.safetensors.index.json")
        shards = sorted(set(index["weight_map"].values()))
        sizes = {shard: (checkpoint / shard).stat().st_size for shard in shards}
        if not sizes or any(size <= 0 for size in sizes.values()):
            raise ValueError(f"Missing/empty weight shards for {job['id']}")
        files[job["id"]] = dict(checkpoint=checkpoint, config_sha256=digest(checkpoint / "config.json"), shards=sizes)
    historical = historical_results()
    write_json(output / "preflight.json", dict(passed=True, checked_utc=utc(), validation_manifest_sha256=VAL_SHA,
        train_validation_overlap=0, checkpoints=files,
        note="Shard existence/size and config identity checked; loading validates runtime EMA. No full weight rehash."))
    return historical


def command_for(job, steps, destination):
    command = [RECIPE / "a1_5_sft/bin/torchrun", "--standalone", "--nproc_per_node=1", "--max-restarts=0",
        SCRIPT_DIR / "benchmark_inference_steps.py", "--checkpoint", job["checkpoint"],
        "--config-name", job["recipe"], "--dataset", DATASET, "--manifest-dir", MANIFEST,
        "--output-dir", destination, "--eval-split", "val", "--attention-backend", "eager",
        "--num-traj-samples", "6", "--seed", "42", "--warmup-samples", "1", "--steps", *steps]
    if job["weights"]:
        command += ["--shortcut-inference-weights", job["weights"], "--expected-ema-updates", job["updates"]]
    if job["id"] in {"A5", "A6"}:
        command += ["--verify-checkpoint-shortcut-config"]
    return [str(value) for value in command]


def progress(log_path):
    if not log_path.exists():
        return "starting"
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 16384))
        tail = stream.read().decode(errors="replace")
    matches = re.findall(r"steps=(\d+):[^\r\n]*?\s(\d+)/128[^\r\n]*", tail)
    return f"steps={matches[-1][0]} clips={matches[-1][1]}/128" if matches else tail.strip().splitlines()[-1][-200:] if tail.strip() else "starting"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,5,3", help="One physical GPU per A2,A3,A4,A5,A6")
    parser.add_argument("--doc-path", type=Path, default=WORKSPACE / "docs/ALPAMAYO15_MISSING_EVAL_2026-09-24.md")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    gpus = args.gpu_ids.split(",")
    if len(gpus) != len(JOBS) or len(set(gpus)) != len(JOBS) or not all(x.isdigit() for x in gpus):
        raise ValueError("Specify five distinct physical GPU indices")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "logs").mkdir(exist_ok=True)
    state = dict(status="preflight", pid=os.getpid(), host=socket.gethostname(), started_utc=utc(),
        benchmark_wall_clock_timeout=None, jobs={})
    processes = {}

    def save_state():
        state["updated_utc"] = utc()
        write_json(output / "status.json", state)
        atomic_text(output / "STATUS", "\n".join(f"{key}={state[key]}" for key in ("status", "pid", "host", "started_utc", "updated_utc")) + "\n")

    def stop_signal(signum, _frame):
        raise KeyboardInterrupt(f"Runner received signal {signum}")

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        save_state()
        historical = preflight(output)
        snapshot = gpu_snapshot()
        with (output / "gpu_telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(utc=utc(), devices=snapshot)) + "\n")
        write_json(output / f"provenance_{os.getpid()}.json", dict(host=socket.gethostname(), utc=utc(),
            jobs=JOBS, physical_gpus=gpus, benchmark_wall_clock_timeout=None,
            code_sha256={str(path): digest(path) for path in [Path(__file__), SCRIPT_DIR / "benchmark_inference_steps.py",
                RECIPE / "models/shortcut_alpamayo_r1.py", RECIPE / "models/shortcut_modules.py",
                RECIPE / "models/paper_shortcut_alpamayo.py", RECIPE / "models/paper_shortcut_targets.py"]},
            legacy_recipe_note="A2/A3 intentionally use exactly their historical inference recipe; no loss forward is run."))
        merged = collect_results(output, historical)
        if args.preflight_only:
            state["status"] = "preflight_passed"
            save_state()
            write_report(output, args.doc_path, merged, state)
            print("Preflight passed; no GPU benchmark launched.", flush=True)
            return
        by_id = {gpu["index"]: gpu for gpu in snapshot}
        for gpu in gpus:
            if gpu not in by_id or int(by_id[gpu]["memory.used"]) > 2048 or int(by_id[gpu]["utilization.gpu"]) > 5:
                raise ValueError(f"Requested GPU {gpu} is missing or busy; refusing to interfere")
        state["status"] = "running"
        for job, gpu in zip(JOBS, gpus):
            remaining = [step for step in job["steps"] if str(step) not in merged[job["id"]]]
            if not remaining:
                state["jobs"][job["id"]] = dict(status="already_complete", gpu=gpu)
                continue
            directory = output / job["id"]
            directory.mkdir(exist_ok=True)
            attempt = 1 + max([int(path.name.split("_")[1]) for path in directory.glob("attempt_*")] or [0])
            destination = directory / f"attempt_{attempt:03d}"
            destination.mkdir()
            log_path = output / "logs" / f"{job['id']}_attempt_{attempt:03d}.log"
            command = command_for(job, remaining, destination)
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HOME="/home/abose_sw/.cache/huggingface",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1", DS_IGNORE_CUDA_DETECTION="1",
                OMP_NUM_THREADS="1", TRITON_CACHE_DIR=f"/tmp/alpamayo15-missing-{os.getpid()}-{job['id']}")
            with (output / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(utc=utc(), id=job["id"], gpu=gpu, argv=command,
                    cwd=str(RECIPE), output=str(destination), wall_clock_timeout=None)) + "\n")
            with log_path.open("x") as stream:
                process = subprocess.Popen(command, cwd=RECIPE, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes[job["id"]] = process
            state["jobs"][job["id"]] = dict(status="running", gpu=gpu, pid=process.pid,
                steps=remaining, log=str(log_path), output=str(destination), started_utc=utc())
            save_state()
            print(f"[{utc()}] Started {job['id']} steps={remaining} GPU={gpu} PID={process.pid}", flush=True)
        last_telemetry = 0.0
        while True:
            active = False
            for job_id, process in processes.items():
                item = state["jobs"][job_id]
                item["progress"] = progress(Path(item["log"]))
                code = process.poll()
                if code is None:
                    active = True
                elif item["status"] == "running":
                    item.update(status="complete" if code == 0 else "failed", returncode=code, finished_utc=utc())
                    print(f"[{utc()}] {job_id} exited: {code}", flush=True)
            merged = collect_results(output, historical)
            state["completed_missing_cells"] = write_report(output, args.doc_path, merged, state)
            save_state()
            if time.monotonic() - last_telemetry >= 60:
                try:
                    with (output / "gpu_telemetry.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(utc=utc(), devices=gpu_snapshot())) + "\n")
                except (OSError, subprocess.SubprocessError) as error:
                    state["telemetry_warning"] = str(error)
                last_telemetry = time.monotonic()
            if not active:
                break
            # Heartbeat only: no deadline and no timeout on any benchmark process.
            time.sleep(20)
        success = state["completed_missing_cells"] == 6 and all(
            job["status"] in {"complete", "already_complete"} for job in state["jobs"].values())
        state.update(status="complete" if success else "incomplete", finished_utc=utc())
        save_state()
        write_report(output, args.doc_path, merged, state)
        if not success:
            raise RuntimeError("Some evaluation cells failed; inspect logs and resume incomplete settings")
    except BaseException as error:
        # Do not leave the old runner's orphan-process failure mode on interruption.
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        state.update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc())
        save_state()
        if "historical" in locals():
            write_report(output, args.doc_path, collect_results(output, historical), state)
        raise


if __name__ == "__main__":
    main()
