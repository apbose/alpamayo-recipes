# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded DDP execution of one complete paper target batch per update."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import hydra.utils as hyu
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from alpamayo.common.misc import seed_everything
from alpamayo_r1.common.logging import setup_logging
from alpamayo1_5_sft.models.paper_shortcut_targets import paper_target_layout, require_full_hierarchy, raw_batch_plan

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "research/alpamayo1_5_shortcut"
ASSETS = Path("/home/scratch.abose_sw/alpamayo-assets")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class PlannedTargets(Dataset):
    def __init__(self, source, plan, layout, rank, world_size, seed):
        self.source, self.plan, self.layout = source, plan, layout
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.microbatches = len(layout.dt_base) // world_size

    def __len__(self):
        return len(self.plan) * self.microbatches

    def __getitem__(self, index):
        update, micro = divmod(index, self.microbatches)
        # Interleave slots across ranks to balance teacher work. All target
        # pairs and their weights are identical to the contiguous paper batch.
        slot = micro * self.world_size + self.rank
        source_index = self.plan[update][int(self.layout.source_indices[slot])]
        sample = self.source[source_index]
        sample["paper_slot"] = slot
        sample["paper_seed"] = self.seed + 100003 * update
        return sample


class PaperCollator:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError("Expected one local target per microbatch")
        sample = dict(samples[0])
        slot, seed = sample.pop("paper_slot"), sample.pop("paper_seed")
        result = self.collate([sample])
        result.update(paper_slot=slot, paper_seed=seed)
        return result


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ASSETS / "checkpoints/Alpamayo-1.5-10B-A1-format")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "manifests/hf_stream_300gb/train.json")
    parser.add_argument("--hf-cache", type=Path, default=ASSETS / "hf_on_demand_smoke/cache")
    parser.add_argument("--updates", type=int, default=249)
    parser.add_argument("--bootstrap-every", type=int, choices=(4, 8), default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=83)
    parser.add_argument("--no-save", action="store_true", help="Smoke only; do not save weights")
    parser.add_argument("--supervision", choices=("ema_bootstrap", "empirical_velocity"), default="ema_bootstrap")
    args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if args.updates < 1 or 64 % world or args.workers < 0:
        raise ValueError("Invalid updates/workers/world size")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    setup_logging()
    seed_everything(args.seed)
    layout = paper_target_layout(64, 128, args.bootstrap_every)
    require_full_hierarchy(layout)
    manifest = json.loads(args.manifest.read_text())
    val_path = PROJECT / "manifests/route_less_19chunks_128eval/val.json"
    val = json.loads(val_path.read_text())
    train_ids = {row["clip_id"] for row in manifest}
    if len(manifest) != 5295 or len(train_ids) != 5295 or train_ids & {row["clip_id"] for row in val}:
        raise ValueError("Expected the unchanged 5,295-clip train manifest, disjoint from validation")
    sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if sha != "21e94e04f441bceebb8c31159b660ff012669b7e2cbf2fdbb45243da00d1da5e":
        raise ValueError("Training manifest hash changed")
    plan = raw_batch_plan(len(manifest), 64, args.updates, args.seed)
    state = dict(status="initializing", started_utc=utc(), updates=0, target_updates=args.updates)

    def status(**updates):
        if rank == 0:
            state.update(updates, updated_utc=utc())
            write_json(args.output_dir / "status.json", state)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        status()
        used = [batch[int(i)] for batch in plan for i in layout.source_indices]
        protocol = dict(
            upstream_revision="601004348667094e1b71f30942199759412d4432",
            paper_preset="paper_table3_75_25" if args.bootstrap_every == 4 else "repository_default_87_5_12_5",
            raw_global_batch=64, global_target_pairs=64, world_size=world,
            local_batch=1, gradient_accumulation=64 // world,
            bootstrap_pairs=int(layout.bootstrap_mask.sum()),
            flow_pairs=int((~layout.bootstrap_mask).sum()),
            supervision=args.supervision,
            teacher_target_pairs=int(layout.bootstrap_mask.sum()) if args.supervision == "ema_bootstrap" else 0,
            empirical_target_pairs=int((~layout.bootstrap_mask).sum()) if args.supervision == "ema_bootstrap" else 64,
            dt_base=layout.dt_base.tolist(), source_indices=layout.source_indices.tolist(),
            train_manifest=str(args.manifest), train_manifest_sha256=sha,
            manifest_clips=5295, selected_unique_training_clips=len(set(used)),
            target_pairs=len(used), raw_rows_drawn=64 * args.updates,
            optimizer_updates=args.updates, seed=args.seed, base_checkpoint=str(args.checkpoint),
            learning_rate=1e-4, adam_betas=[0.9, 0.999], adam_epsilon=1e-8,
            weight_decay=0.1, weight_decay_all_trainable_parameters=True,
            schedule="constant", warmup=0, gradient_clipping=False,
            student_parameter_dtype="float32", ema_dtype="float32", autocast="bfloat16",
            ema_decay=0.999, ema_update="after each optimizer update",
            attention="eager", access_mode="hf_stream", hf_revision="33f9bf447ed3bcb7d545ce13f4226f824214fafb",
            architecture="Alpamayo 1.5 frozen VLM + Action Expert; not DiT",
            initialization="released checkpoint; new residual adapter starts at zero",
            not_a_bitwise_jax_reproduction=True,
        )
        write_json(args.output_dir / "protocol.json", protocol)
        write_json(args.output_dir / "raw_batch_plan.json", plan)
    dist.barrier()
    with initialize_config_module(version_base=None, config_module="alpamayo1_5_sft.configs"):
        cfg = compose(config_name="sft_stage2_trajectory_shortcut_hf_reference_ema", overrides=[
            f"model.pretrained_model_name_or_path={args.checkpoint}",
            "+model.attn_implementation=eager",
            f"data.train_dataset.hf_cache_dir={args.hf_cache}",
            f"data.train_dataset.annotations_path={args.manifest}",
        ])
        paper = compose(config_name="sft_stage2_trajectory_shortcut_paper_ema")
    cfg.model = paper.model
    cfg.model.pretrained_model_name_or_path = str(args.checkpoint)
    OmegaConf.update(cfg.model, "attn_implementation", "eager", force_add=True)
    cfg.model.shortcut_bootstrap_every = args.bootstrap_every
    cfg.model.shortcut_loss_weight = 1 / args.bootstrap_every
    OmegaConf.update(cfg.model, "paper_supervision", args.supervision, force_add=True)
    if rank == 0:
        (args.output_dir / "model_and_data_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    model = hyu.instantiate(cfg.model, _convert_="partial")
    # Match float32 master parameters/EMA, while keeping the frozen VLM BF16.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.to(device)
    if any(p.requires_grad for p in model.vlm.parameters()):
        raise RuntimeError("VLM must remain frozen")
    model.train()
    model.vlm.eval()
    model._freeze_shortcut_ema_teacher()
    source = hyu.instantiate(cfg.data.train_dataset, _convert_="partial", model_config=model.config)
    collate = hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
    planned = PlannedTargets(source, plan, layout, rank, world, args.seed)
    loader_kwargs = dict(batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=PaperCollator(collate))
    if args.workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="spawn")
    loader = iter(DataLoader(planned, **loader_kwargs))
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1, foreach=False)
    # Do not inherit HF Trainer's bias/norm decay exemptions or grad clipping.
    optimizer.zero_grad(set_to_none=True)
    micro_count = 64 // world
    allowed = {"tokenized_data", "ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot", "labels_mask", "paper_slot", "paper_seed"}
    status(status="training")
    start = time.monotonic()
    for step in range(args.updates):
        sums = torch.zeros(5, device=device, dtype=torch.float64)
        for micro in range(micro_count):
            batch = next(loader)
            data = move({k: v for k, v in batch.items() if k in allowed}, device)
            del batch
            sync = nullcontext() if micro == micro_count - 1 else ddp.no_sync()
            with sync, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp(**data)
                loss = output.loss / micro_count
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {step}, microbatch {micro}, rank {rank}")
            loss.backward()
            sums[0] += output.loss.detach().double()
            if output.flow_loss is not None:
                sums[1] += output.flow_loss.double()
                sums[3] += 1
            else:
                sums[2] += output.shortcut_loss.double()
                sums[4] += 1
            del data, output, loss
        grad_norm = torch.stack([p.grad.detach().float().norm().square() for p in parameters if p.grad is not None]).sum().sqrt()
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at update {step}")
        optimizer.step()
        model.update_shortcut_ema_teacher()
        optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(sums)
        if int(model.shortcut_ema_updates) != step + 1:
            raise RuntimeError("EMA did not advance exactly once per optimizer update")
        record = dict(update=step + 1, loss=float(sums[0] / 64),
                      flow_loss=float(sums[1] / sums[3]) if sums[3] else None,
                      shortcut_loss=float(sums[2] / sums[4]) if sums[4] else None,
                      flow_pairs=int(sums[3]), shortcut_pairs=int(sums[4]),
                      supervision=args.supervision,
                      grad_norm=float(grad_norm), ema_updates=int(model.shortcut_ema_updates),
                      elapsed_seconds=time.monotonic() - start)
        if rank == 0:
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        status(updates=step + 1, last_metrics=record)
        if not args.no_save and ((step + 1) % args.save_every == 0 or step + 1 == args.updates):
            status(status="saving")
            if rank == 0:
                checkpoint = args.output_dir / f"checkpoint-{step + 1}"
                model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="5GB")
                torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
                write_json(checkpoint / "trainer_state.json", {"global_step": step + 1, "ema_updates": step + 1, "raw_manifest_passes": (step + 1) * 64 / len(manifest)})
                write_json(checkpoint / "COMPLETE.json", {"completed_utc": utc(), "global_step": step + 1})
            dist.barrier()
            status(status="training", checkpoint=str(args.output_dir / f"checkpoint-{step + 1}"))
    status(status="complete", finished_utc=utc())
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
