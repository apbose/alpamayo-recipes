# SPDX-License-Identifier: Apache-2.0

"""Lightweight tests for the shortcut model wrapper and Hydra configuration."""

from __future__ import annotations

import json

from hydra import compose, initialize_config_module

from alpamayo1_5_sft.models.shortcut_alpamayo_r1 import (
    _SHORTCUT_PROJECTION_TARGET,
    checkpoint_has_step_size_adapter,
    shortcut_action_projection_config,
)


def test_shortcut_projection_config_preserves_base_fields() -> None:
    original = {
        "_target_": "alpamayo_r1.models.action_in_proj.PerWaypointActionInProjV2",
        "hidden_size": 512,
        "num_enc_layers": 2,
    }
    updated = shortcut_action_projection_config(
        original,
        flow_step_size=0.125,
        step_size_fourier_feats=20,
        step_size_hidden_size=256,
    )

    assert original["_target_"].endswith("PerWaypointActionInProjV2")
    assert updated["_target_"] == _SHORTCUT_PROJECTION_TARGET
    assert updated["hidden_size"] == 512
    assert updated["default_step_size"] == 0.125


def test_checkpoint_adapter_detection_from_index(tmp_path) -> None:
    index = {
        "weight_map": {
            "expert.layers.0.weight": "model-00001-of-00002.safetensors",
            "action_in_proj.step_size_adapter.2.weight": (
                "model-00002-of-00002.safetensors"
            ),
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert checkpoint_has_step_size_adapter(tmp_path)


def test_checkpoint_without_adapter_is_detected(tmp_path) -> None:
    index = {"weight_map": {"expert.layers.0.weight": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert not checkpoint_has_step_size_adapter(tmp_path)


def test_shortcut_stage2_hydra_config_is_separate_and_frozen() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        cfg = compose(config_name="sft_stage2_nav_shortcut")

    assert cfg.model._target_.endswith(
        "ShortcutTrainableAlpamayoR1.from_pretrained"
    )
    assert cfg.model.cotrain_vlm is False
    assert cfg.model.shortcut_flow_step_size == 0.125
    assert list(cfg.model.shortcut_step_sizes) == [0.25, 0.5, 1.0]
    assert cfg.model.shortcut_level_sampling == "balanced_cycle"
    assert cfg.model.shortcut_loss_weight == 0.125
    assert cfg.model.shortcut_target_clip == 4.0
    assert cfg.model.shortcut_teacher_mode == "online"
    assert cfg.task_name == "alpamayo_1_5_shortcut_stage2_nav"
