# SPDX-License-Identifier: Apache-2.0
"""No-CFG target algorithm from kvfrans/shortcut-models (MIT), ported to torch.

Reference revision: 601004348667094e1b71f30942199759412d4432.
Target layout/formulas match; PyTorch and JAX RNG streams are not bitwise equal.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class PaperTargetLayout:
    source_indices: torch.Tensor
    dt_base: torch.Tensor
    bootstrap_mask: torch.Tensor
    base_steps: int

    @property
    def step_sizes(self):
        return torch.exp2(-self.dt_base.float())


def paper_target_layout(batch_size=64, base_steps=128, bootstrap_every=4):
    """Original repeat/fill hierarchy and overlapping source-image mapping."""
    if base_steps < 2 or base_steps & (base_steps - 1):
        raise ValueError("base_steps must be a power of two >= 2")
    if bootstrap_every <= 1 or batch_size <= 0 or batch_size % bootstrap_every:
        raise ValueError("batch_size must be divisible by bootstrap_every > 1")
    k, levels = batch_size // bootstrap_every, int(math.log2(base_steps))
    exponents = torch.arange(levels - 1, -1, -1).repeat_interleave(k // levels)
    exponents = torch.cat((exponents, torch.zeros(k - len(exponents), dtype=torch.long)))
    return PaperTargetLayout(
        torch.cat((torch.arange(k), torch.arange(batch_size - k))),
        torch.cat((exponents, torch.full((batch_size - k,), levels))),
        torch.arange(batch_size) < k, base_steps,
    )


def require_full_hierarchy(layout):
    expected = set(range(int(math.log2(layout.base_steps))))
    actual = set(layout.dt_base[layout.bootstrap_mask].tolist())
    if actual != expected:
        raise ValueError(f"Bootstrap slots do not cover all levels: {actual} != {expected}")


def paper_interpolation(clean, noise, t):
    return (1.0 - (1.0 - 1e-5) * t) * noise + t * clean


def paper_flow_target(clean, noise):
    return clean - (1.0 - 1e-5) * noise


def paper_bootstrap_target(teacher, noisy, t, d):
    with torch.no_grad():
        first = teacher(noisy, t, d / 2).float()
        midpoint = (noisy + (d / 2) * first).clamp(-4, 4)
        second = teacher(midpoint, t + d / 2, d / 2).float()
        return ((first + second) / 2).clamp(-4, 4).detach()


def draw_slot_randomness(layout, slot, action_shape, seed):
    """Branch-shaped draws; separate time/noise streams; deterministic replay."""
    k = int(layout.bootstrap_mask.sum())
    bootstrap = slot < k
    count = k if bootstrap else len(layout.dt_base)
    index = slot if bootstrap else slot - k
    noise = torch.randn((count, *action_shape), generator=torch.Generator().manual_seed(seed))
    time_gen = torch.Generator().manual_seed(seed + 1)
    if bootstrap:
        sections = (2 ** layout.dt_base[:k]).tolist()
        ticks = [int(torch.randint(n, (), generator=time_gen)) for n in sections]
        t = ticks[index] / sections[index]
    else:
        ticks = torch.randint(layout.base_steps, (count,), generator=time_gen)
        t = float(ticks[index]) / layout.base_steps
    return noise[index:index + 1], t


def raw_batch_plan(dataset_size, batch_size, updates, seed=10):
    """Finite-manifest counterpart of upstream's repeating shuffled TFDS stream."""
    if dataset_size <= 0:
        raise ValueError("dataset must be nonempty")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    cursor, batches = 0, []
    for _ in range(updates):
        batch = []
        while len(batch) < batch_size:
            take = min(batch_size - len(batch), dataset_size - cursor)
            batch.extend(permutation[cursor:cursor + take])
            cursor += take
            if cursor == dataset_size:
                permutation = torch.randperm(dataset_size, generator=generator).tolist()
                cursor = 0
        order = torch.randperm(batch_size, generator=generator).tolist()
        batches.append([batch[i] for i in order])
    return batches
