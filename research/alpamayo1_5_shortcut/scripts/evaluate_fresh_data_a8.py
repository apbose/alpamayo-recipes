#!/usr/bin/env python3
"""Detached A8 final EMA evaluation and matched A6 quality comparison.

One GPU, sequential solver counts, no training and no wall-clock timeout.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from bootstrap_paired_step_regression import bootstrap_metric, paired_values
from fill_missing_evaluations import (
    ASSETS, DATASET, JOBS, MANIFEST, METRICS, PAPER_RUN, REPO, SCRIPT_DIR,
    STEPS, VAL_SHA, WORKSPACE, digest, read_json, utc, validate_benchmark,
    write_json, atomic_text,
)

A8_RUN = ASSETS / "runs/alpamayo15_a8_fresh_data_3x5295_b64_20260925_r1"
MISSING_SUMMARY = WORKSPACE / "results/alpamayo15_missing_eval_128clips_20260924_r1/summary.json"


def missing_steps(result):
    completed = set(result.get("results", {}))
    if completed - set(map(str, STEPS)):
        raise ValueError("Unexpected solver count in saved A8 results")
    return [step for step in STEPS if str(step) not in completed]


def merge_results(previous, current, current_path):
    """Merge validated solver results, retaining independent timing provenance."""
    old_cfg, new_cfg = previous.get("configuration"), current.get("configuration")
    if old_cfg and new_cfg:
        for key in set(old_cfg) | set(new_cfg):
            if key != "inference_steps" and old_cfg.get(key) != new_cfg.get(key):
                raise ValueError(f"Resumed benchmark configuration changed: {key}")
    overlap = set(previous.get("results", {})) & set(current.get("results", {}))
    if overlap:
        raise ValueError(f"Refusing to overwrite completed solver results: {sorted(overlap)}")
    result = copy.deepcopy(previous if previous else current)
    result.setdefault("results", {}).update(copy.deepcopy(current.get("results", {})))
    sources = result.setdefault("step_sources", {})
    for step in current.get("results", {}):
        sources[step] = dict(path=str(current_path), sha256=digest(current_path))
    if "configuration" in result:
        result["configuration"]["inference_steps"] = [s for s in STEPS if str(s) in result["results"]]
    result["merged_across_runs"] = len({s["path"] for s in sources.values()}) > 1
    for step, value in result["results"].items():
        if sources.get(step, {}).get("path") != sources.get("10", {}).get("path"):
            value["comparison_to_10_step"] = None
    return result


def a6_reference():
    job = next(job for job in JOBS if job["id"] == "A6")
    rows = [row for row in read_json(MISSING_SUMMARY)["rows"] if row["experiment"] == "A6"]
    results, provenance = {}, {}
    loaded = {}
    for row in rows:
        source = row["source"]
        if source not in loaded:
            loaded[source] = read_json(source)
            validate_benchmark(loaded[source], job)
        step = str(row["steps"])
        results[step] = loaded[source]["results"][step]
        provenance[step] = dict(path=source, sha256=digest(source))
    if set(results) != set(map(str, STEPS)):
        raise ValueError("A6 matched baseline is incomplete")
    return dict(results=results), provenance


def report_text(result, reference, *, state, output, bootstrap=None):
    results = result.get("results", {})
    lines = ["# A8 fresh-data A6: matched 128-clip evaluation", "", f"Updated UTC: {utc()}", "",
             f"Status: **{state}**; solver settings complete: **{len(results)}/4**.", "",
             "EMA checkpoint-249; fixed 128 validation clips; six candidates per clip; seed 42; "
             "eager attention; one excluded warm-up per solver count. Same model-call path as A6.", "",
             "## Quality against A6", "",
             "All errors are metres, lower is better. Positive percentage change is worse. "
             "A8 versus A6 compares the same solver count; step regression compares A8 with its own ten-step result.", "",
             "| Steps | Metric | A6 | A8 | A8 vs A6 | A8 vs own 10-step |",
             "|---:|---|---:|---:|---:|---:|"]
    rows = []
    for step in map(str, STEPS):
        for metric in METRICS:
            old = reference["results"][step]["metrics"][metric]
            if step not in results:
                lines.append(f"| {step} | {metric} | {old:.4f} | Pending | — | — |")
                continue
            value = results[step]["metrics"][metric]
            relative = 100 * (value / old - 1)
            own = 100 * (value / results["10"]["metrics"][metric] - 1) if "10" in results else None
            own_text = "—" if own is None else f"{own:+.2f}%"
            lines.append(f"| {step} | {metric} | {old:.4f} | {value:.4f} | {relative:+.2f}% | {own_text} |")
            rows.append(dict(steps=int(step), metric=metric, a6=old, a8=value,
                             a8_vs_a6_percent=relative, a8_vs_own_10_percent=own))
    lines += ["", "## A8 timing (source runs kept separate)", "",
              "Speedup is shown only when both measurements came from the same benchmark run. "
              "The resumed two-step run must not be divided into historical ten-step latency.", "",
              "| Steps | Action Expert ms | Full model-call seconds | Expert speedup vs 10 | Source run |",
              "|---:|---:|---:|---:|---|"]
    sources = result.get("step_sources", {})
    for step in map(str, STEPS):
        if step not in results:
            continue
        timing = results[step]["latency_ms"]
        expert = timing["action_expert_diffusion"]["mean"]
        same_run = not sources or (step in sources and "10" in sources and
                                   sources[step]["path"] == sources["10"]["path"])
        speedup = results["10"]["latency_ms"]["action_expert_diffusion"]["mean"] / expert if "10" in results and same_run else None
        speedup_text = "—" if speedup is None else f"{speedup:.2f}x"
        source = sources.get(step, {}).get("path")
        source_text = f"[raw result]({source})" if source else "Current run"
        lines.append(f"| {step} | {expert:.2f} | {timing['end_to_end_model']['mean']/1000:.3f} | {speedup_text} | {source_text} |")
    if bootstrap:
        lines += ["", "## Paired clip-bootstrap: A8 minus A6", "",
                  "100,000 paired resamples; 95% intervals for the difference in mean error (metres). "
                  "These intervals cover clip sampling, not training-seed uncertainty.", "",
                  "| Steps | Metric | Difference | 95% interval |", "|---:|---|---:|---|"]
        for step, metrics in bootstrap.items():
            for metric, value in metrics.items():
                lo, hi = value["mean_delta_95_ci"]
                lines.append(f"| {step} | {metric} | {value['observed_mean_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines += ["", "## Limits and provenance", "",
              "- No cross-run A6/A8 latency speedup is claimed: historical GPU power caps differ.",
              "- Timing excludes loading/decoding and metric computation; each model call already includes six candidates.",
              "- Same A6 optimizer/loss/EMA budget; A8 changes data coverage and phase-boundary shuffle/padding.",
              "- Best-of-six open-loop quality is not a collision/off-road/closed-loop safety result.",
              "- No training, one-step run, checkpoint mutation, or GPU power-setting changes.",
              f"- Artifacts: `{output}`.",
              "- `preflight.json` records manifest, checkpoint, source hashes and baseline provenance.", ""]
    return "\n".join(lines), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume-from", type=Path,
                        help="Validated prior benchmark JSON; run only its missing solver counts")
    args = parser.parse_args()
    # A detached single-rank benchmark inherits this disposition. Avoid the
    # torchrun supervisor that converted the previous hang-up into worker exit.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    output, doc = args.output_dir.resolve(), args.doc.resolve()
    output.mkdir(parents=True, exist_ok=False)
    doc.parent.mkdir(parents=True, exist_ok=True)
    if doc.exists() and args.resume_from is None:
        raise FileExistsError(f"Refusing to overwrite report: {doc}")
    state = dict(status="preflight", pid=os.getpid(), gpu=args.gpu, host=socket.gethostname(), started_utc=utc())
    def status(**updates):
        state.update(updates, updated_utc=utc())
        write_json(output / "status.json", state)
    def report(result, reference, bootstrap=None):
        text, rows = report_text(result, reference, state=state["status"], output=output, bootstrap=bootstrap)
        atomic_text(output / "REPORT.md", text)
        atomic_text(doc, text)
        if rows:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            atomic_text(output / "a8_vs_a6.csv", buffer.getvalue())
    child = None
    result, reference = {}, None
    try:
        status()
        checkpoint = A8_RUN / "training/checkpoint-249"
        if read_json(checkpoint / "COMPLETE.json")["global_step"] != 249:
            raise ValueError("Final A8 checkpoint is incomplete")
        training = read_json(A8_RUN / "training/status.json")
        if training["status"] != "complete" or training["updates"] != 249:
            raise ValueError("A8 training is not complete")
        index = read_json(checkpoint / "model.safetensors.index.json")
        for shard in set(index["weight_map"].values()):
            if not (checkpoint / shard).is_file() or (checkpoint / shard).stat().st_size <= 0:
                raise FileNotFoundError(shard)
        if digest(MANIFEST / "val.json") != VAL_SHA:
            raise ValueError("Matched validation manifest changed")
        val = read_json(MANIFEST / "val.json")
        train = read_json(A8_RUN / "training/train_manifest.json")
        if len(val) != 128 or len({row['clip_id'] for row in val}) != 128:
            raise ValueError("Validation population changed")
        overlap = {row["clip_id"] for row in val} & {row["clip_id"] for row in train}
        if overlap:
            raise ValueError("A8 training/validation overlap")
        reference, provenance = a6_reference()
        job = dict(next(job for job in JOBS if job["id"] == "A6"), id="A8", checkpoint=checkpoint)
        previous = {}
        if args.resume_from:
            source = args.resume_from.resolve()
            previous = read_json(source)
            validate_benchmark(previous, job)
            source_preflight = read_json(source.parent.parent / "preflight.json")
            if source_preflight["source_hashes"]["benchmark_inference_steps.py"] != digest(SCRIPT_DIR / "benchmark_inference_steps.py"):
                raise ValueError("Benchmark implementation changed since original run")
            previous["step_sources"] = {step: dict(path=str(source), sha256=digest(source))
                                        for step in previous["results"]}
            write_json(output / "imported_results.json", previous)
        requested_steps = missing_steps(previous)
        if not requested_steps:
            raise ValueError("All solver counts are already complete; nothing to launch")
        if doc.exists():
            atomic_text(output / "previous_report.md", doc.read_text())
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,power.limit,enforced.power.limit,memory.used", "--format=csv"], text=True)
        write_json(output / "preflight.json", dict(checkpoint=str(checkpoint), checkpoint_config_sha256=digest(checkpoint / "config.json"),
            validation_manifest_sha256=VAL_SHA, validation_clips=128, train_val_overlap=0,
            a6_sources=provenance, hardware=hardware, gpu=args.gpu, host=socket.gethostname(),
            resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
            requested_steps=requested_steps, imported_step_sources=previous.get("step_sources", {}),
            training_protocol_sha256=digest(A8_RUN / "training/protocol.json"),
            source_hashes={name: digest(SCRIPT_DIR / name) for name in ("benchmark_inference_steps.py", "evaluate_fresh_data_a8.py")}))
        result = previous
        status(requested_steps=requested_steps, completed_steps=list(previous.get("results", {})))
        report(result, reference)
        benchmark_dir = output / "benchmark"
        cmd = [sys.executable, str(SCRIPT_DIR / "benchmark_inference_steps.py"), "--checkpoint", str(checkpoint),
               "--config-name", "sft_stage2_trajectory_shortcut_paper_ema", "--dataset", str(DATASET),
               "--manifest-dir", str(MANIFEST), "--output-dir", str(benchmark_dir), "--eval-split", "val",
               "--steps", *map(str, requested_steps), "--num-traj-samples", "6", "--seed", "42",
               "--attention-backend", "eager", "--warmup-samples", "1", "--shortcut-inference-weights", "ema",
               "--expected-ema-updates", "249", "--verify-checkpoint-shortcut-config"]
        # Identical one-rank NCCL execution, without an elastic supervisor.
        with socket.socket() as rendezvous:
            rendezvous.bind(("127.0.0.1", 0))
            port = rendezvous.getsockname()[1]
        rank_env = dict(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        write_json(output / "command.json", dict(argv=cmd, cwd=str(REPO), distributed_env=rank_env,
                                                 launcher="direct_single_rank", sighup="ignored"))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", DS_IGNORE_CUDA_DETECTION="1", **rank_env)
        with (output / "benchmark.log").open("w") as log:
            child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            status(status="evaluating", benchmark_pid=child.pid)
            result_path, last = benchmark_dir / "benchmark_results.json", None
            while child.poll() is None:
                if result_path.exists() and result_path.stat().st_mtime_ns != last:
                    current = read_json(result_path)
                    validate_benchmark(current, job)
                    result = merge_results(previous, current, result_path)
                    write_json(output / "merged_benchmark_results.json", result)
                    status(completed_steps=list(result["results"]))
                    report(result, reference)
                    last = result_path.stat().st_mtime_ns
                time.sleep(30)
            if child.returncode:
                raise RuntimeError(f"Benchmark failed with exit code {child.returncode}; see benchmark.log")
        current = read_json(result_path)
        validate_benchmark(current, job)
        result = merge_results(previous, current, result_path)
        write_json(output / "merged_benchmark_results.json", result)
        if set(result["results"]) != set(map(str, STEPS)):
            raise ValueError("Incomplete solver sweep")
        status(status="summarizing", completed_steps=list(map(str, STEPS)))
        bootstrap = {}
        for step in map(str, STEPS):
            combined = dict(results=dict(reference=reference["results"][step], candidate=result["results"][step]))
            bootstrap[step] = {}
            for metric in METRICS:
                old, new = paired_values(combined, "reference", "candidate", metric)
                bootstrap[step][metric] = bootstrap_metric(old, new, iterations=100_000, seed=20260925, threshold=0.10)
        write_json(output / "paired_bootstrap_a8_vs_a6.json", bootstrap)
        status(status="complete", finished_utc=utc())
        report(result, reference, bootstrap)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
        status(status="failed", error=f"{type(error).__name__}: {error}")
        if reference is not None and (output / "preflight.json").exists():
            report(result, reference)
        raise


if __name__ == "__main__":
    main()
