# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shortcut-specific Action Expert modules."""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from alpamayo1_5_sft.models.shortcut_modules import (
    BalancedShortcutLevelSampler,
    StepSizeConditionedActionInProjV2,
    construct_shortcut_training_data,
    fork_kv_cache,
    shortcut_midpoint,
    shortcut_velocity_target,
)


def make_projection() -> StepSizeConditionedActionInProjV2:
    torch.manual_seed(7)
    return StepSizeConditionedActionInProjV2(
        in_dims=[8, 2],
        out_dim=16,
        num_enc_layers=2,
        hidden_size=12,
        max_freq=10.0,
        num_fourier_feats=8,
        step_size_fourier_feats=8,
        step_size_hidden_size=10,
        default_step_size=0.125,
    )


def test_zero_adapter_preserves_released_projection() -> None:
    projection = make_projection()
    x = torch.randn(3, 8, 2)
    timesteps = torch.rand(3, 1, 1)

    at_eighth = projection(x, timesteps, step_size=0.125)
    at_half = projection(x, timesteps, step_size=0.5)
    legacy_call = projection(x, timesteps)

    torch.testing.assert_close(at_eighth, at_half, rtol=0.0, atol=0.0)
    torch.testing.assert_close(at_eighth, legacy_call, rtol=0.0, atol=0.0)


def test_adapter_can_learn_distinct_step_sizes() -> None:
    projection = make_projection()
    torch.manual_seed(11)
    torch.nn.init.normal_(projection.step_size_adapter[-1].weight, std=0.05)

    x = torch.randn(2, 8, 2)
    timesteps = torch.rand(2, 1, 1)
    at_quarter = projection(x, timesteps, step_size=0.25)
    at_one = projection(x, timesteps, step_size=1.0)

    assert not torch.allclose(at_quarter, at_one)


def test_zero_output_layer_receives_gradient() -> None:
    projection = make_projection()
    x = torch.randn(2, 8, 2)
    timesteps = torch.rand(2, 1, 1)

    projection(x, timesteps, step_size=torch.tensor([0.25, 0.5])).square().mean().backward()

    output_weight = projection.step_size_adapter[-1].weight
    assert output_weight.grad is not None
    assert torch.count_nonzero(output_weight.grad) > 0


def test_released_projection_state_keys_remain_compatible() -> None:
    keys = set(make_projection().state_dict())
    assert "encoder.trunk.0.weight" in keys
    assert "timestep_fourier_encoder.freqs" in keys
    assert "step_size_adapter.2.weight" in keys
    assert not any(key.startswith("base_projection.") for key in keys)


@pytest.mark.parametrize("invalid_step_size", [0.0, -0.25, 1.1])
def test_rejects_invalid_scalar_step_size(invalid_step_size: float) -> None:
    projection = make_projection()
    x = torch.randn(1, 8, 2)
    timesteps = torch.rand(1, 1, 1)

    with pytest.raises(ValueError, match="step_size"):
        projection(x, timesteps, step_size=invalid_step_size)


def test_rejects_step_size_that_changes_within_trajectory() -> None:
    projection = make_projection()
    x = torch.randn(2, 8, 2)
    timesteps = torch.rand(2, 1, 1)
    step_size = torch.tensor([[0.25, 0.5], [0.5, 0.5]])

    with pytest.raises(ValueError, match="constant across each trajectory"):
        projection(x, timesteps, step_size=step_size)


def test_shortcut_samples_are_aligned_and_stay_inside_flow_path() -> None:
    x = torch.tensor(
        [
            [[2.0, -2.0]],
            [[4.0, 8.0]],
            [[1.0, 3.0]],
        ]
    )
    noise = torch.zeros_like(x)
    data = construct_shortcut_training_data(
        x,
        step_sizes=(0.25, 0.5, 1.0),
        noise=noise,
        level_indices=torch.tensor([0, 1, 2]),
        time_indices=torch.tensor([3, 1, 0]),
    )

    expected_t = torch.tensor([0.75, 0.5, 0.0]).reshape(3, 1, 1)
    expected_d = torch.tensor([0.25, 0.5, 1.0]).reshape(3, 1, 1)
    torch.testing.assert_close(data.timesteps, expected_t)
    torch.testing.assert_close(data.step_sizes, expected_d)
    torch.testing.assert_close(data.noisy_x, expected_t * x)
    torch.testing.assert_close(data.midpoint_timesteps, expected_t + expected_d / 2)
    assert bool((data.timesteps + data.step_sizes <= 1.0).all())


def test_shortcut_target_is_clamped_and_stopped_gradient() -> None:
    first = torch.tensor([[-10.0, 2.0]], requires_grad=True)
    second = torch.tensor([[-2.0, 4.0]], requires_grad=True)

    target = shortcut_velocity_target(first, second, clip_value=4.0)

    torch.testing.assert_close(target, torch.tensor([[-4.0, 3.0]]))
    assert not target.requires_grad


def test_shortcut_midpoint_uses_half_step_and_clamps() -> None:
    noisy_x = torch.tensor([[[3.5, -3.5]]])
    velocity = torch.tensor([[[4.0, -4.0]]])
    half_step = torch.tensor([[[0.25]]])

    midpoint = shortcut_midpoint(
        noisy_x,
        velocity,
        half_step,
        clip_value=4.0,
    )

    torch.testing.assert_close(midpoint, torch.tensor([[[4.0, -4.0]]]))


def test_cache_fork_does_not_mutate_prompt_cache() -> None:
    prompt = DynamicCache()
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)
    prompt.update(keys, values, layer_idx=0)
    original_keys = prompt.layers[0].keys

    fork = fork_kv_cache(prompt)
    assert fork.layers[0] is not prompt.layers[0]
    assert fork.layers[0].keys.data_ptr() == original_keys.data_ptr()

    fork.update(
        torch.randn(1, 2, 1, 4),
        torch.randn(1, 2, 1, 4),
        layer_idx=0,
    )

    assert prompt.get_seq_length() == 3
    assert fork.get_seq_length() == 4
    assert prompt.layers[0].keys.data_ptr() == original_keys.data_ptr()


def test_shortcut_sampling_rejects_non_dyadic_intervals() -> None:
    with pytest.raises(ValueError, match="dyadic"):
        construct_shortcut_training_data(
            torch.randn(1, 8, 2),
            step_sizes=(0.1,),
        )


def test_balanced_sampler_cycles_every_level_with_batch_size_one() -> None:
    sampler = BalancedShortcutLevelSampler(num_levels=3)

    observed = [
        int(sampler(1, device=torch.device("cpu"))[0])
        for _ in range(7)
    ]

    assert observed == [0, 1, 2, 0, 1, 2, 0]
    assert int(sampler.cursor) == 1


def test_balanced_sampler_assigns_consecutive_multi_sample_levels() -> None:
    sampler = BalancedShortcutLevelSampler(num_levels=3)

    first = sampler(5, device=torch.device("cpu"))
    second = sampler(2, device=torch.device("cpu"))

    assert first.tolist() == [0, 1, 2, 0, 1]
    assert second.tolist() == [2, 0]
    assert int(sampler.cursor) == 1


def test_balanced_sampler_partitions_distributed_global_batch() -> None:
    rank_zero = BalancedShortcutLevelSampler(num_levels=3)
    rank_one = BalancedShortcutLevelSampler(num_levels=3)

    first_rank = rank_zero(
        2,
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )
    second_rank = rank_one(
        2,
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert first_rank.tolist() == [0, 1]
    assert second_rank.tolist() == [2, 0]
    assert int(rank_zero.cursor) == int(rank_one.cursor) == 1


def test_balanced_sampler_resumes_from_state_dict_cursor() -> None:
    original = BalancedShortcutLevelSampler(num_levels=3)
    original(5, device=torch.device("cpu"))
    saved_state = {
        key: value.clone()
        for key, value in original.state_dict().items()
    }

    restored = BalancedShortcutLevelSampler(num_levels=3)
    restored.load_state_dict(saved_state)

    expected = original(3, device=torch.device("cpu"))
    actual = restored(3, device=torch.device("cpu"))
    assert actual.tolist() == expected.tolist() == [2, 0, 1]
    assert int(restored.cursor) == int(original.cursor) == 2


def test_balanced_sampler_rejects_partial_distributed_arguments() -> None:
    sampler = BalancedShortcutLevelSampler(num_levels=3)
    with pytest.raises(ValueError, match="provided together"):
        sampler(1, device=torch.device("cpu"), rank=0)
