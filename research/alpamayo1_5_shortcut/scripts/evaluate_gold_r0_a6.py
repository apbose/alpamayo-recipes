#!/usr/bin/env python3
"""Detached R0/A6 gold-644 evaluation; inference only, no wall-clock deadline.

Two independent single-GPU benchmarks; both smoke tests must pass before the
full sweeps. Failed clips fail the run, never silently shrink the population.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from bootstrap_paired_step_regression import bootstrap_metric, paired_values
from fill_missing_evaluations import (
    ASSETS, DATASET, PAPER_RUN, RELEASED, REPO, SCRIPT_DIR, METRICS,
    atomic_text, digest, read_json, utc, write_json,
)

STEPS = (10, 5, 4, 2)
T0_US = 5_100_000
HF_REVISION = "33f9bf447ed3bcb7d545ce13f4226f824214fafb"
GOLD = REPO / "recipes/alpamayo1_5_quant/1005_7cam_gold_eval_metadb_public.parquet"
JOBS = {
    "R0": dict(checkpoint=RELEASED["checkpoint"], recipe="sft_stage2_trajectory_shortcut"),
    "A6": dict(checkpoint=PAPER_RUN / "training/checkpoint-249",
               recipe="sft_stage2_trajectory_shortcut_paper_ema"),
}


def make_rows(clip_ids, clip_index):
    if len(clip_ids) != 644 or len(set(clip_ids)) != 644:
        raise ValueError("Gold parquet must contain exactly 644 unique clips")
    missing = set(clip_ids) - set(clip_index.index)
    if missing:
        raise ValueError(f"Gold clips absent from pinned HF index: {len(missing)}")
    if "clip_is_valid" in clip_index and not clip_index.loc[clip_ids, "clip_is_valid"].all():
        raise ValueError("Gold index contains invalid clips; do not drop them silently")
    return [dict(clip_id=cid, t0_relative=T0_US, chunk=int(clip_index.at[cid, "chunk"]))
            for cid in clip_ids]


def audit_training(gold_ids, protocol, manifest, batches):
    all_ids = {row["clip_id"] for row in manifest}
    used_ids = {manifest[batch[index]]["clip_id"] for batch in batches
                for index in protocol["source_indices"]}
    result = dict(manifest_unique_clips=len(all_ids), used_unique_clips=len(used_ids),
                  gold_manifest_overlap=len(gold_ids & all_ids),
                  gold_used_overlap=len(gold_ids & used_ids))
    if result["gold_manifest_overlap"]:
        raise ValueError(f"A6 training/gold overlap: {result}")
    return result


def write_manifest(path, rows):
    path.mkdir()
    write_json(path / "val.json", rows)
    # The benchmark constructs only val_dataset. An empty train manifest is
    # explicit: these evaluation clips must never be used for training.
    write_json(path / "train.json", [])
    write_json(path / "summary.json", dict(
        annotation_rows=dict(train=0, val=len(rows)),
        chunks=dict(train=[], val=sorted({row["chunk"] for row in rows})),
        purpose="evaluation_only", hf_revision=HF_REVISION, t0_relative=T0_US,
        source_parquet=str(GOLD), source_parquet_sha256=digest(GOLD),
    ))


def validate_result(result, model_id, manifest_path, requested_steps):
    rows = read_json(manifest_path)
    job, cfg = JOBS[model_id], result["configuration"]
    expected = dict(validation_samples=len(rows), validation_unique_clips=len(rows),
                    validation_manifest_sha256=digest(manifest_path), num_traj_samples=6,
                    seed_reset_for_each_step_count=42, attention_backend="eager",
                    warmup_samples_per_step_count=1, access_mode="hf_stream",
                    hf_revision=HF_REVISION, recipe_config=job["recipe"],
                    checkpoint_config_sha256=digest(job["checkpoint"] / "config.json"),
                    zero_step_size_adapter=(model_id == "R0"),
                    step_size_adapter_scale=0.0 if model_id == "R0" else 1.0,
                    inference_steps=list(requested_steps), evaluation_split="val")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"{model_id}: incorrect {key}: {cfg.get(key)!r} != {value!r}")
    if Path(cfg["checkpoint"]).resolve() != job["checkpoint"].resolve():
        raise ValueError("Checkpoint identity changed")
    if model_id == "A6":
        runtime = cfg["shortcut_runtime"]
        if runtime["updates"] != 249 or runtime["inference_weights"] != "ema":
            raise ValueError("A6 must evaluate EMA weights after 249 updates")
    for step, value in result["results"].items():
        if int(step) not in requested_steps:
            raise ValueError("Unexpected solver count")
        samples = value["per_sample"]
        if len(samples) != len(rows):
            raise ValueError("Incomplete gold population")
        for index, (sample, row) in enumerate(zip(samples, rows, strict=True)):
            if (sample["sample_index"], sample["clip_id"], sample["t0_relative"]) != (
                    index, row["clip_id"], row["t0_relative"]):
                raise ValueError("Per-clip pairing mismatch")
            for metric in METRICS:
                values = sample["metrics"][metric]
                if len(values) != 1 or not math.isfinite(values[0]):
                    raise ValueError("Invalid per-clip metric")
        for metric in METRICS:
            observed = sum(s["metrics"][metric][0] for s in samples) / len(samples)
            if not math.isclose(observed, value["metrics"][metric], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("Aggregate does not match per-clip values")


def missing_steps(result):
    return tuple(step for step in STEPS if str(step) not in result.get("results", {}))


def merge_completed(previous, current, source):
    """Merge whole solver sweeps only; preserve source provenance and timings."""
    previous = previous or {}
    if previous:
        old, new = previous["configuration"], current["configuration"]
        allowed = {"inference_steps", "validation_manifest", "hf_cache_dir", "hf_stream_retry"}
        for key in (set(old) | set(new)) - allowed:
            if old.get(key) != new.get(key):
                raise ValueError(f"Resume changed benchmark configuration: {key}")
    overlap = set(previous.get("results", {})) & set(current.get("results", {}))
    if overlap:
        raise ValueError(f"Refusing to overwrite completed solver counts: {sorted(overlap)}")
    merged = copy.deepcopy(previous or current)
    merged.setdefault("results", {}).update(copy.deepcopy(current.get("results", {})))
    provenance = merged.setdefault("step_sources", {})
    for step in current.get("results", {}):
        provenance[step] = dict(path=str(source), sha256=digest(source))
    merged["configuration"]["inference_steps"] = [s for s in STEPS if str(s) in merged["results"]]
    merged["merged_across_runs"] = len({p["path"] for p in provenance.values()}) > 1
    for step, result in merged["results"].items():
        if provenance.get(step, {}).get("path") != provenance.get("10", {}).get("path"):
            result["comparison_to_10_step"] = None
    return merged


def load_previous(directory, manifest):
    old_manifest = directory / "manifest/val.json"
    if digest(old_manifest) != digest(manifest):
        raise ValueError("Resume manifest differs from the original gold selection")
    previous = {}
    for model in JOBS:
        source = directory / f"{model}_merged_benchmark_results.json"
        if not source.exists():
            source = directory / f"{model}_evaluating/benchmark_results.json"
        result = read_json(source)
        validate_result(result, model, manifest, result["configuration"]["inference_steps"])
        if not {"10", "5"} <= set(result["results"]):
            raise ValueError("Gold recovery requires completed 10/5-step sweeps")
        if "step_sources" not in result:
            result["step_sources"] = {step: dict(path=str(source), sha256=digest(source))
                                      for step in result["results"]}
        previous[model] = result
    return previous


def report_text(results, state, output):
    lines = ["# R0 versus A6 on the public 644-clip gold manifest", "",
             f"Updated UTC: {utc()}", "", f"Status: **{state['status']}**", "",
             "## Protocol", "",
             "- 644 unique clips, original parquet order; one timestamp per clip at 5.1 s.",
             "- Four cameras, four frames each; 16 history poses; 64 future poses (6.4 s).",
             "- Six candidates per model call; seed 42 reset per solver count; one excluded warm-up.",
             "- Frozen released VLM, route-less input, generated VLM rollout; no human nav/CoC labels.",
             "- Eager attention; BF16 autocast; identical existing Stage-2 benchmark for both models.",
             "- R0: released A1-format wrapper with zero adapter; A6: final checkpoint-249 EMA weights.",
             "- HF on-demand range reads at a pinned revision; zero A6 train/gold clip overlap.",
             "- Two separate GPUs, one per checkpoint; no training or GPU power-setting changes.",
             "", "## minADE (metres; lower is better)", "",
             "Parentheses compare each model with its OWN ten-step quality.", "",
             "| Model | 10 steps | 5 steps | 4 steps | 2 steps |",
             "|---|---:|---:|---:|---:|"]
    table = []
    for model_id in JOBS:
        measured = results.get(model_id, {}).get("results", {})
        cells = []
        for step in map(str, STEPS):
            if step not in measured:
                cells.append("Pending")
                continue
            value = measured[step]["metrics"]["min_ade"]
            own = 100 * (value / measured["10"]["metrics"]["min_ade"] - 1) if "10" in measured else None
            cells.append(f"{value:.4f}" + (f" ({own:+.2f}%)" if step != "10" and own is not None else ""))
            table.append(dict(model=model_id, steps=int(step), **measured[step]["metrics"],
                              minade_vs_own_10_percent=own,
                              expert_ms=measured[step]["latency_ms"]["action_expert_diffusion"]["mean"],
                              model_ms=measured[step]["latency_ms"]["end_to_end_model"]["mean"]))
        lines.append(f"| {model_id} | " + " | ".join(cells) + " |")
    lines += ["", "## Detailed quality and latency", "",
              "Timing excludes data fetching/decoding and metrics. Each call already produces six candidates.",
              "| Model | Steps | ADE m | Corner m | Expert ms | Full call s |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['model']} | {row['steps']} | {row['ade']:.4f} | {row['corner_distance']:.4f} | "
                     f"{row['expert_ms']:.2f} | {row['model_ms']/1000:.3f} |")
    lines += ["", "## Progress", "", "```json", json.dumps(state, indent=2), "```", "",
              "## Limits and provenance", "",
              "- This is a new evaluation on the public gold manifest, not the old 128-clip set.",
              "- It is not an exact reproduction of NVIDIA quantization-recipe scores: our established "
              "BF16/Stage-2 path and per-solver seed reset differ from that script's FP16/per-clip reset.",
              "- The manifest's event_t0s are not used: 5.1 s follows the public eval.py default.",
              "- No clips are silently dropped. A failure leaves the evaluation incomplete.",
              "- No claim is made about NVIDIA's proprietary pretraining overlap or closed-loop safety.",
              "- A8 is excluded: 112 gold clips actually received A8 training targets (147 in its pool).",
              "- Paired-bootstrap intervals, when complete, describe clip sampling, not training-seed uncertainty.",
              f"- Artifacts, commands, hashes, hardware and logs: `{output}`.", ""]
    if state.get("resume_from"):
        lines += ["## Recovery provenance", "",
                  f"- Preserved completed sweeps from `{state['resume_from']}`.",
                  "- Incomplete solver counts restart from clip zero with seed 42; partial RNG state is not guessed.",
                  "- HF range-read retries include HTTP 499; bounded to six attempts per request, with backoff.",
                  "- Raw historical results are unchanged; per-step source paths/hashes are in merged JSON files.",
                  "- Timings are shown individually; no cross-run latency speedup is calculated.", ""]
    return "\n".join(lines), table


def prepare(output, cache, gpus, extra_smoke_indices=()):
    import pandas as pd
    import torch
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetHFInterface
    from alpamayo.data.pai_trajectory import PAITrajectoryDataset
    from benchmark_hf_streaming import ValidatingTimedDataset

    protocol = read_json(PAPER_RUN / "training/protocol.json")
    manifest_path = Path(protocol["train_manifest"])
    if digest(manifest_path) != protocol["train_manifest_sha256"]:
        raise ValueError("A6 training manifest hash changed")
    ids = pd.read_parquet(GOLD)["key"].astype(str).tolist()
    audit = audit_training(set(ids), protocol, read_json(manifest_path),
                           read_json(PAPER_RUN / "training/raw_batch_plan.json"))
    if read_json(JOBS["A6"]["checkpoint"] / "COMPLETE.json")["global_step"] != 249:
        raise ValueError("A6 checkpoint incomplete")
    for model_id, job in JOBS.items():
        config = read_json(job["checkpoint"] / "config.json")
        if model_id == "R0" and any(k.startswith(("shortcut_", "paper_")) for k in config):
            raise ValueError("R0 must be the released checkpoint")
        index = read_json(job["checkpoint"] / "model.safetensors.index.json")
        for name in set(index["weight_map"].values()):
            shard = job["checkpoint"] / name
            if not shard.is_file() or shard.stat().st_size <= 0:
                raise FileNotFoundError(shard)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not ready")
    for gpu in gpus:
        x = torch.ones(8, device=f"cuda:{gpu}")
        assert x.sum().item() == 8
        del x
    torch.cuda.empty_cache()
    avdi = PhysicalAIAVDatasetHFInterface(revision=HF_REVISION, cache_dir=cache)
    rows = make_rows(ids, avdi.clip_index)
    features = [avdi.features.LABELS.EGOMOTION, avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
                avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV, avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
                avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV]
    missing = {feature: int((~avdi.feature_presence.loc[ids, feature]).sum())
               for feature in features if not avdi.feature_presence.loc[ids, feature].all()}
    if missing:
        raise ValueError(f"Missing required gold features: {missing}")
    write_manifest(output / "manifest", rows)
    write_manifest(output / "smoke_manifest", rows[:2])
    source = PAITrajectoryDataset(annotations_path=str(output / "manifest/val.json"),
                                 access_mode="hf_stream", hf_revision=HF_REVISION, hf_cache_dir=str(cache))
    stream = ValidatingTimedDataset(source, len(rows))
    indices = sorted({0, 1, *extra_smoke_indices})
    validated = [stream[i] for i in indices]
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,power.limit,enforced.power.limit,memory.used",
                                        "--format=csv"], text=True)
    write_json(output / "preflight.json", dict(
        gold_parquet=str(GOLD), gold_sha256=digest(GOLD), validation_clips=644,
        val_sha256=digest(output / "manifest/val.json"), training_audit=audit,
        train_manifest_sha256=digest(manifest_path), hf_revision=HF_REVISION,
        stream_smoke=validated, hardware=hardware, host=socket.gethostname(), gpus=gpus,
        checkpoints={key: dict(path=str(job["checkpoint"]), config_sha256=digest(job["checkpoint"] / "config.json"))
                     for key, job in JOBS.items()},
        source_hashes={name: digest(SCRIPT_DIR / name) for name in (
            "benchmark_inference_steps.py", "evaluate_gold_r0_a6.py", "hf_stream_retry.py")},
    ))


def launch(output, model_id, gpu, manifest, steps, cache, max_attempts=1):
    job = JOBS[model_id]
    output.mkdir()
    cmd = [sys.executable, str(SCRIPT_DIR / "benchmark_inference_steps.py"),
           "--checkpoint", str(job["checkpoint"]), "--config-name", job["recipe"],
           "--dataset", str(DATASET), "--manifest-dir", str(manifest), "--output-dir", str(output),
           "--steps", *map(str, steps), "--num-traj-samples", "6", "--seed", "42",
           "--attention-backend", "eager", "--warmup-samples", "1", "--access-mode", "hf_stream",
           "--hf-revision", HF_REVISION, "--hf-cache-dir", str(cache)]
    if max_attempts > 1:
        cmd += ["--hf-stream-max-attempts", str(max_attempts)]
    cmd += ["--zero-step-size-adapter"] if model_id == "R0" else [
        "--shortcut-inference-weights", "ema", "--expected-ema-updates", "249", "--verify-checkpoint-shortcut-config"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    rank_env = dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                    MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", DS_IGNORE_CUDA_DETECTION="1", **rank_env)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    write_json(output / "command.json", dict(argv=cmd, cwd=str(REPO), gpu=gpu, rank_env=rank_env,
                                            launcher="direct_single_rank", sighup="ignored"))
    with (output / "benchmark.log").open("x") as log:
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--hf-cache", type=Path, default=ASSETS / "hf_on_demand_smoke/cache")
    parser.add_argument("--resume-from", type=Path,
                        help="Prior gold run directory; preserve completed counts and rerun only missing whole sweeps.")
    parser.add_argument("--hf-stream-max-attempts", type=int, default=1)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        raise ValueError("Use two distinct GPUs")
    if args.hf_stream_max_attempts < 1:
        raise ValueError("hf-stream-max-attempts must be positive")
    if args.hf_stream_max_attempts > 1:
        from hf_stream_retry import configure_hf_stream_retries
        retry_policy = configure_hf_stream_retries(args.hf_stream_max_attempts)
    else:
        retry_policy = None
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    output = args.output_dir.resolve()
    if args.doc.exists():
        raise FileExistsError(args.doc)
    output.mkdir(parents=True, exist_ok=False)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status="preflight", pid=os.getpid(), host=socket.gethostname(), started_utc=utc())
    state.update(resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                 hf_stream_retry=retry_policy)
    results, children = {}, {}
    def save(**updates):
        state.update(updates, updated_utc=utc())
        write_json(output / "status.json", state)
        text, rows = report_text(results, state, output)
        atomic_text(output / "REPORT.md", text)
        atomic_text(args.doc, text)
        if rows:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            atomic_text(output / "comparison.csv", buffer.getvalue())
    try:
        save()
        extra_smoke_indices = set()
        if args.resume_from:
            for model in JOBS:
                progress_path = args.resume_from / f"{model}_evaluating/progress.json"
                if progress_path.exists():
                    count = int(read_json(progress_path)["completed_samples"])
                    extra_smoke_indices.update(i for i in (count, count + 1) if 0 <= i < 644)
        prepare(output, args.hf_cache, args.gpus, extra_smoke_indices)
        previous = load_previous(args.resume_from.resolve(), output / "manifest/val.json") if args.resume_from else {}
        results.update(copy.deepcopy(previous))
        for model, value in previous.items():
            write_json(output / f"{model}_imported_results.json", value)
            write_json(output / f"{model}_merged_benchmark_results.json", value)
        remaining = {model: missing_steps(previous.get(model, {})) for model in JOBS}
        save(remaining_solver_counts={model: list(steps) for model, steps in remaining.items()})
        for phase, dirname, steps in (("smoke", "smoke_manifest", (10, 2)),
                                      ("evaluating", "manifest", STEPS)):
            save(status=phase, jobs={})
            manifest = output / dirname
            children = {}
            requested = {model: steps if phase == "smoke" else remaining[model] for model in JOBS}
            for model, gpu in zip(JOBS, args.gpus, strict=True):
                if requested[model]:
                    children[model] = launch(output / f"{model}_{phase}", model, gpu, manifest,
                                             requested[model], args.hf_cache, args.hf_stream_max_attempts)
            while True:
                progress = {}
                for model, child in children.items():
                    directory = output / f"{model}_{phase}"
                    path = directory / "benchmark_results.json"
                    if path.exists():
                        result = read_json(path)
                        validate_result(result, model, manifest / "val.json", requested[model])
                        if phase == "evaluating":
                            results[model] = merge_completed(previous.get(model, {}), result, path)
                            write_json(output / f"{model}_merged_benchmark_results.json", results[model])
                    item = dict(pid=child.pid, returncode=child.poll())
                    if (directory / "progress.json").exists():
                        item.update(read_json(directory / "progress.json"))
                    progress[model] = item
                    if child.returncode is not None:
                        if child.returncode != 0:
                            raise RuntimeError(f"{model} {phase} failed: {child.returncode}; see {directory}/benchmark.log")
                        result = read_json(path)
                        validate_result(result, model, manifest / "val.json", requested[model])
                        if set(result["results"]) != set(map(str, requested[model])):
                            raise ValueError(f"{model}: incomplete {phase}")
                save(jobs=progress)
                if all(child.poll() is not None for child in children.values()):
                    break
                time.sleep(15)
            if phase == "smoke":
                write_json(output / "SMOKE_PASSED.json", dict(finished_utc=utc(), samples=2, steps=list(steps)))
        save(status="summarizing")
        for model, result in results.items():
            validate_result(result, model, output / "manifest/val.json", STEPS)
        paired = {}
        for step in map(str, STEPS):
            combined = dict(results={model: results[model]["results"][step] for model in JOBS})
            paired[step] = {}
            for metric in METRICS:
                old, new = paired_values(combined, "R0", "A6", metric)
                paired[step][metric] = bootstrap_metric(old, new, iterations=100_000, seed=20260929, threshold=0.1)
        write_json(output / "paired_bootstrap_a6_minus_r0.json", paired)
        within = {}
        for model, result in results.items():
            within[model] = {}
            for step in ("5", "4", "2"):
                within[model][step] = {}
                for metric in METRICS:
                    old, new = paired_values(result, "10", step, metric)
                    within[model][step][metric] = bootstrap_metric(old, new, iterations=100_000, seed=20260929, threshold=0.1)
        write_json(output / "paired_bootstrap_step_reduction.json", within)
        save(status="complete", finished_utc=utc())
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
