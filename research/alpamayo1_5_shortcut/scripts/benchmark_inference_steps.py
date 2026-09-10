#!/usr/bin/env python3
"""Compare Alpamayo 1.5 trajectory quality and latency across solver step counts.

The script deliberately loads the model once, resets the same random seed for
each step count, and times both the complete VLM+Action-Expert model call and
the inner flow-matching diffusion sampler.  It uses NVIDIA's released Stage-2
dataset, collator, model wrapper, and distance metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra.utils as hyu
import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from alpamayo.common import distributed
from alpamayo.metrics.metric_api import DistanceMetrics
from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Released A1-format wrapper or trained shortcut checkpoint.",
    )
    parser.add_argument(
        "--config-name",
        choices=(
            "sft_stage2_nav",
            "sft_stage2_nav_shortcut",
            "sft_stage2_trajectory_shortcut",
        ),
        default="sft_stage2_trajectory_shortcut",
        help="Use the shortcut config when loading a trained delta-t checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Local PhysicalAI-AV root containing the requested feature chunks.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "manifests/route_less_19chunks"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/benchmark",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="eager",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 4, 2, 1])
    parser.add_argument("--num-traj-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolved_attention_backend(config: Any) -> str | None:
    """Return the backend Transformers resolved for one model config."""
    internal = getattr(config, "_attn_implementation_internal", None)
    if internal is not None:
        return str(internal)
    requested = getattr(config, "_attn_implementation", None)
    return None if requested is None else str(requested)


def verify_attention_backends(
    model: torch.nn.Module, expected_backend: str
) -> dict[str, str | None]:
    """Prove that Cosmos-Reason2 and the expert selected the requested backend."""
    backends = {
        "cosmos_reason2_vlm": resolved_attention_backend(model.vlm.config),
        "cosmos_reason2_vision": resolved_attention_backend(
            model.vlm.config.vision_config
        ),
        "cosmos_reason2_text": resolved_attention_backend(model.vlm.config.text_config),
        "action_expert": resolved_attention_backend(model.expert.config),
    }
    required = (
        "cosmos_reason2_vision",
        "cosmos_reason2_text",
        "action_expert",
    )
    unexpected = {
        name: backends[name]
        for name in required
        if backends[name] != expected_backend
    }
    if unexpected:
        raise RuntimeError(
            f"Attention backend {expected_backend!r} was requested, but these model "
            f"components did not resolve it: {unexpected}. All backends: {backends}"
        )
    print(
        f"Verified model attention backends: {json.dumps(backends, sort_keys=True)}",
        flush=True,
    )
    return backends


def format_chunks(chunks: list[int]) -> str:
    return "[" + ",".join(str(chunk) for chunk in chunks) + "]"


def build_config(args: argparse.Namespace, split_summary: dict[str, Any]):
    train_chunks = format_chunks(split_summary["chunks"]["train"])
    val_chunks = format_chunks(split_summary["chunks"]["val"])
    manifest_prefix = (
        "" if args.config_name == "sft_stage2_trajectory_shortcut" else "nav_"
    )
    train_manifest = args.manifest_dir / f"{manifest_prefix}train.json"
    val_manifest = args.manifest_dir / f"{manifest_prefix}val.json"
    overrides = [
        f"model.pretrained_model_name_or_path={args.checkpoint}",
        f"+model.attn_implementation={args.attention_backend}",
        f"data.train_dataset.local_dir={args.dataset}",
        f"data.train_dataset.annotations_path={train_manifest}",
        f"data.train_dataset.chunk_ids={train_chunks}",
        f"data.val_dataset.local_dir={args.dataset}",
        f"data.val_dataset.annotations_path={val_manifest}",
        f"data.val_dataset.chunk_ids={val_chunks}",
        "trainer.per_device_eval_batch_size=1",
        "trainer.dataloader_num_workers=0",
        f"paths.output_dir={args.output_dir / 'trainer_output'}",
    ]
    with initialize_config_module(
        version_base=None, config_module="alpamayo1_5_sft.configs"
    ):
        return compose(config_name=args.config_name, overrides=overrides)


def sample_model(
    model: torch.nn.Module,
    data: dict[str, Any],
    inference_steps: int,
    num_traj_samples: int,
):
    return model.sample_trajectories_from_data(
        data=data,
        num_traj_samples=num_traj_samples,
        num_traj_sets=1,
        top_p=0.98,
        temperature=0.6,
        traj_only_generation=False,
        max_generation_length=256,
        return_extra=False,
        diffusion_kwargs={"inference_step": inference_steps},
    )


def benchmark() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_summary_path = args.manifest_dir / "summary.json"
    manifest_prefix = (
        "" if args.config_name == "sft_stage2_trajectory_shortcut" else "nav_"
    )
    val_manifest_path = args.manifest_dir / f"{manifest_prefix}val.json"
    split_summary = json.loads(split_summary_path.read_text())
    val_manifest = json.loads(val_manifest_path.read_text())

    if len(val_manifest) != split_summary["annotation_rows"]["val"]:
        raise ValueError("Validation manifest count disagrees with split summary")
    if len({row["clip_id"] for row in val_manifest}) != len(val_manifest):
        raise ValueError("Validation manifest is not unique by clip")

    cfg = build_config(args, split_summary)
    (args.output_dir / "composed_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    distributed.initialize_distributed_simple()
    if torch.distributed.get_world_size() != 1:
        raise ValueError("This benchmark intentionally requires exactly one process/GPU")

    model = hyu.instantiate(cfg.model, _convert_="partial")
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    model = trainer.accelerator.prepare_model(model, evaluation_mode=True)
    model = trainer.accelerator.unwrap_model(model)
    model.eval()
    attention_backends = verify_attention_backends(model, args.attention_backend)

    diffusion_timings_ms: list[float] = []
    original_diffusion_sample = model.diffusion.sample

    def timed_diffusion_sample(*sample_args, **sample_kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_diffusion_sample(*sample_args, **sample_kwargs)
        torch.cuda.synchronize()
        diffusion_timings_ms.append((time.perf_counter() - started) * 1000.0)
        return result

    model.diffusion.sample = timed_diffusion_sample

    results: dict[str, Any] = {
        "schema_version": 1,
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "recipe_config": args.config_name,
            "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
            "dataset": str(args.dataset),
            "validation_manifest": str(val_manifest_path),
            "validation_manifest_sha256": sha256(val_manifest_path),
            "validation_samples": len(val_manifest),
            "validation_unique_clips": len({row["clip_id"] for row in val_manifest}),
            "inference_steps": args.steps,
            "num_traj_samples": args.num_traj_samples,
            "seed_reset_for_each_step_count": args.seed,
            "warmup_samples_per_step_count": args.warmup_samples,
            "attention_backend": args.attention_backend,
            "resolved_attention_backends": attention_backends,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "results": {},
    }
    write_json(args.output_dir / "benchmark_results.json", results)

    distance_metric = DistanceMetrics()

    for inference_steps in args.steps:
        print(f"\n=== Benchmarking {inference_steps} diffusion steps ===", flush=True)

        if args.warmup_samples:
            seed_everything(args.seed)
            warmup_loader = iter(trainer.get_eval_dataloader())
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(args.warmup_samples):
                    warmup_data = next(warmup_loader)
                    sample_model(
                        model,
                        warmup_data,
                        inference_steps,
                        args.num_traj_samples,
                    )

        diffusion_timings_ms.clear()
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats()

        metric_sums: defaultdict[str, float] = defaultdict(float)
        metric_counts: defaultdict[str, int] = defaultdict(int)
        end_to_end_timings_ms: list[float] = []
        per_sample: list[dict[str, Any]] = []

        dataloader = trainer.get_eval_dataloader()
        for sample_index, data in enumerate(
            tqdm(dataloader, total=len(dataloader), desc=f"steps={inference_steps}")
        ):
            timing_count_before = len(diffusion_timings_ms)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot = sample_model(
                    model,
                    data,
                    inference_steps,
                    args.num_traj_samples,
                )
            torch.cuda.synchronize()
            end_to_end_ms = (time.perf_counter() - started) * 1000.0

            if len(diffusion_timings_ms) != timing_count_before + 1:
                raise RuntimeError("Expected exactly one diffusion sampler call per batch")
            if not torch.isfinite(pred_xyz).all() or not torch.isfinite(pred_rot).all():
                raise RuntimeError(f"Non-finite prediction at sample {sample_index}")

            output_batch = {"pred_xyz": pred_xyz, "pred_rot": pred_rot}
            sample_metrics = distance_metric.evaluate(model, data, output_batch)
            serialized_metrics: dict[str, list[float]] = {}
            for key, value in sample_metrics.items():
                value = value.detach().float()
                metric_sums[key] += value.sum().item()
                metric_counts[key] += value.numel()
                serialized_metrics[key] = value.cpu().tolist()

            end_to_end_timings_ms.append(end_to_end_ms)
            per_sample.append(
                {
                    "sample_index": sample_index,
                    "end_to_end_model_ms": end_to_end_ms,
                    "action_expert_diffusion_ms": diffusion_timings_ms[-1],
                    "metrics": serialized_metrics,
                }
            )

        averaged_metrics = {
            key: metric_sums[key] / metric_counts[key] for key in sorted(metric_sums)
        }
        step_result = {
            "metrics": averaged_metrics,
            "latency_ms": {
                "end_to_end_model": latency_stats(end_to_end_timings_ms),
                "action_expert_diffusion": latency_stats(diffusion_timings_ms),
                "fixed_vlm_and_other_mean": (
                    statistics.fmean(end_to_end_timings_ms)
                    - statistics.fmean(diffusion_timings_ms)
                ),
            },
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "per_sample": per_sample,
        }
        results["results"][str(inference_steps)] = step_result
        write_json(args.output_dir / "benchmark_results.json", results)
        print(json.dumps({str(inference_steps): step_result}, indent=2), flush=True)

    baseline = results["results"]["10"]
    baseline_e2e = baseline["latency_ms"]["end_to_end_model"]["mean"]
    baseline_expert = baseline["latency_ms"]["action_expert_diffusion"]["mean"]
    baseline_min_ade = baseline["metrics"]["min_ade"]
    for inference_steps in args.steps:
        result = results["results"][str(inference_steps)]
        result["comparison_to_10_step"] = {
            "end_to_end_speedup": (
                baseline_e2e / result["latency_ms"]["end_to_end_model"]["mean"]
            ),
            "action_expert_speedup": (
                baseline_expert
                / result["latency_ms"]["action_expert_diffusion"]["mean"]
            ),
            "min_ade_delta": result["metrics"]["min_ade"] - baseline_min_ade,
        }

    write_json(args.output_dir / "benchmark_results.json", results)

    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "steps",
                "min_ade",
                "ade",
                "corner_distance",
                "end_to_end_mean_ms",
                "action_expert_mean_ms",
                "end_to_end_speedup_vs_10",
                "action_expert_speedup_vs_10",
                "min_ade_delta_vs_10",
            ]
        )
        for inference_steps in args.steps:
            result = results["results"][str(inference_steps)]
            comparison = result["comparison_to_10_step"]
            writer.writerow(
                [
                    inference_steps,
                    result["metrics"]["min_ade"],
                    result["metrics"]["ade"],
                    result["metrics"]["corner_distance"],
                    result["latency_ms"]["end_to_end_model"]["mean"],
                    result["latency_ms"]["action_expert_diffusion"]["mean"],
                    comparison["end_to_end_speedup"],
                    comparison["action_expert_speedup"],
                    comparison["min_ade_delta"],
                ]
            )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    benchmark()
