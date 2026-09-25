# SPDX-License-Identifier: Apache-2.0
"""Reference formula parity, batching, and accumulation regression tests."""
import importlib.util
import hashlib
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from alpamayo1_5_sft.models.paper_shortcut_targets import (
    paper_target_layout, require_full_hierarchy, paper_interpolation,
    paper_flow_target, paper_bootstrap_target, draw_slot_randomness, raw_batch_plan,
)
from alpamayo1_5_sft.models.paper_shortcut_alpamayo import PaperLogStepEncoder
from alpamayo1_5_sft.models.shortcut_modules import update_ema_module_


@pytest.mark.parametrize("every,expected", [(4, [6,6,5,5,4,4,3,3,2,2,1,1,0,0,0,0]), (8, [6,5,4,3,2,1,0,0])])
def test_exact_layout_and_input_reuse(every, expected):
    layout = paper_target_layout(64, 128, every)
    k = 64 // every
    assert layout.dt_base[:k].tolist() == expected
    assert layout.dt_base[k:].tolist() == [7] * (64 - k)
    assert layout.source_indices[:k].tolist() == list(range(k))
    assert layout.source_indices[k:].tolist() == list(range(64 - k))
    assert len(set(layout.source_indices.tolist())) == 64 - k
    require_full_hierarchy(layout)


def test_small_batch_reproduces_upstream_fallback_but_is_rejected_for_training():
    layout = paper_target_layout(8, 128, 4)
    assert layout.dt_base[:2].tolist() == [0, 0]
    with pytest.raises(ValueError, match="cover all levels"):
        require_full_hierarchy(layout)


@pytest.mark.parametrize("every", [4, 8])
def test_all_slot_times_are_on_the_correct_grid(every):
    layout = paper_target_layout(64, 128, every)
    for slot in range(64):
        noise, t = draw_slot_randomness(layout, slot, (64, 2), 42)
        assert noise.shape == (1, 64, 2)
        d = float(layout.step_sizes[slot])
        assert 0 <= t <= 1 - d
        assert t / d == round(t / d)
        repeat = draw_slot_randomness(layout, slot, (64, 2), 42)
        assert torch.equal(noise, repeat[0]) and t == repeat[1]


def test_target_clipping_half_steps_and_stop_gradient():
    param = torch.nn.Parameter(torch.tensor(9.0))
    calls = []
    def teacher(x, t, d):
        calls.append((x.clone(), t.clone(), d))
        return x + param
    x = torch.tensor([[[3.9, -9.0]]], requires_grad=True)
    t = torch.tensor([[[0.0]]])
    target = paper_bootstrap_target(teacher, x, t, 0.5)
    assert len(calls) == 2
    assert calls[0][2] == calls[1][2] == 0.25
    assert torch.equal(calls[1][1], t + 0.25)
    assert calls[1][0].abs().max() <= 4
    assert target.abs().max() <= 4 and not target.requires_grad
    assert param.grad is None


def test_microbatch_gradients_and_once_per_update_ema_match_full_batch():
    torch.manual_seed(12)
    x, target = torch.randn(64, 3), torch.randn(64, 2)
    full = torch.nn.Linear(3, 2)
    accumulated = torch.nn.Linear(3, 2)
    accumulated.load_state_dict(full.state_dict())
    torch.nn.functional.mse_loss(full(x), target).backward()
    for i in range(64):
        (torch.nn.functional.mse_loss(accumulated(x[i:i+1]), target[i:i+1]) / 64).backward()
    for a, b in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-6)
    teacher = torch.nn.Linear(3, 2)
    teacher.load_state_dict(full.state_dict())
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    torch.optim.AdamW(full.parameters(), lr=1e-4, weight_decay=0.1).step()
    update_ema_module_(teacher, full, decay=0.999)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, 0.999 * before[key] + 0.001 * full.state_dict()[key])


def test_log2_step_features_match_reference_embedding():
    d = torch.tensor([1/128, 1/64, 0.5, 1.0])
    half = 128
    freqs = np.exp(-math.log(10000) * np.arange(half, dtype=np.float32) / half)
    args = -np.log2(d.numpy())[:, None] * freqs[None]
    expected = np.concatenate((np.cos(args), np.sin(args)), axis=-1)
    np.testing.assert_allclose(PaperLogStepEncoder()(d).numpy(), expected, atol=1e-6)


def test_plan_and_rank_allocation_are_deterministic_and_complete():
    plan = raw_batch_plan(5295, 64, 249, 10)
    assert plan == raw_batch_plan(5295, 64, 249, 10)
    assert len(plan) == 249 and all(len(row) == 64 for row in plan)
    assert set(i for row in plan for i in row) == set(range(5295))
    assert sorted(micro * 8 + rank for micro in range(8) for rank in range(8)) == list(range(64))


def test_paper_hydra_config_is_explicit_and_separate():
    with initialize_config_module(version_base=None, config_module="alpamayo1_5_sft.configs"):
        cfg = compose(config_name="sft_stage2_trajectory_shortcut_paper_ema")
        old = compose(config_name="sft_stage2_trajectory_shortcut_10to5_reference_ema")
    assert cfg.model._target_.endswith("PaperEMAShortcutAlpamayo.from_pretrained")
    assert cfg.model.paper_target_batch_size == 64
    assert cfg.model.shortcut_flow_step_size == 1 / 128
    assert cfg.model.shortcut_bootstrap_every == 4
    assert cfg.model.shortcut_ema_decay == 0.999
    assert cfg.model.cotrain_vlm is False
    assert old.model.shortcut_flow_step_size == 0.1


@pytest.mark.parametrize("every", [4, 8])
def test_numerical_parity_with_original_get_targets(monkeypatch, every):
    """Execute the original function using NumPy-backed fixed random draws.

    This tests its actual source, not a second hand-written target function.
    JAX RNG is substituted so the same inputs are available to torch.
    """
    path = (Path(__file__).resolve().parents[3] / "research/alpamayo1_5_shortcut"
            / "third_party/shortcut_models/targets_shortcut.py")
    assert path.is_file(), "Pinned reference fixture must be present in the checkout"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "903543035d1aa641a16a5540d841c075a649310549e1795bf0eaabcc5784976f"
    ), "Reference fixture differs from the pinned source"
    draws = {"normal": [], "randint": []}
    def normal(key, shape):
        value = np.random.default_rng(int(key)).normal(size=shape).astype(np.float32)
        draws["normal"].append(value)
        return value
    def randint(key, shape, minval, maxval):
        value = np.random.default_rng(int(key)).integers(minval, maxval, size=shape, dtype=np.int32)
        draws["randint"].append(value)
        return value
    jax = types.ModuleType("jax")
    jax.numpy = np
    jax.random = types.SimpleNamespace(split=lambda key, n: np.arange(key, key+n), normal=normal,
        randint=randint, bernoulli=lambda key, p, shape: np.zeros(shape, dtype=bool))
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    spec = importlib.util.spec_from_file_location("upstream_shortcut_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flags = types.SimpleNamespace(batch_size=64, model=dict(bootstrap_every=every,
        denoise_timesteps=128, bootstrap_dt_bias=0, bootstrap_ema=1, bootstrap_cfg=0,
        class_dropout_prob=0.0, num_classes=1))
    def teacher_np(x, t, dt, labels, train=False):
        return 0.4 * x + t[:, None, None, None] + 0.05 * dt[:, None, None, None]
    state = types.SimpleNamespace(call_model_ema=teacher_np)
    clean = np.random.default_rng(99).normal(size=(64, 3, 2, 1)).astype(np.float32)
    upstream = module.get_targets(flags, 23, state, clean, np.zeros(64, dtype=np.int32))
    layout, k = paper_target_layout(64, 128, every), 64 // every
    noisy_rows, targets, times = [], [], []
    for slot in range(64):
        is_bootstrap = slot < k
        branch, index = (0, slot) if is_bootstrap else (1, slot-k)
        source = clean[int(layout.source_indices[slot]):int(layout.source_indices[slot])+1]
        noise = torch.from_numpy(draws["normal"][branch][index:index+1])
        n = 2 ** int(layout.dt_base[slot])
        t = float(draws["randint"][branch][index]) / n
        t_tensor = torch.tensor(t).reshape(1, 1, 1, 1)
        x = paper_interpolation(torch.from_numpy(source), noise, t_tensor)
        d = float(layout.step_sizes[slot])
        def teacher_torch(x, time, delta):
            return 0.4 * x + time + 0.05 * (-math.log2(delta))
        target = paper_bootstrap_target(teacher_torch, x, t_tensor, d) if is_bootstrap else paper_flow_target(torch.from_numpy(source), noise)
        noisy_rows.append(x.numpy()); targets.append(target.numpy()); times.append(t)
    np.testing.assert_allclose(np.concatenate(noisy_rows), upstream[0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(np.concatenate(targets), upstream[1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(times, upstream[2])
    np.testing.assert_array_equal(layout.dt_base.numpy(), upstream[3])
