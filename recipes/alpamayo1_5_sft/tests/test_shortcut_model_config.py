# SPDX-License-Identifier: Apache-2.0

"""Lightweight tests for the shortcut model wrapper and Hydra configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import alpamayo1_5_sft.models.shortcut_alpamayo_r1 as shortcut_model_module
from alpamayo1_5_sft.models.shortcut_alpamayo_r1 import (
    _SHORTCUT_PROJECTION_TARGET,
    ShortcutTrainableAlpamayoR1,
    checkpoint_has_ema_teacher,
    checkpoint_has_step_size_adapter,
    shortcut_action_projection_config,
)
from hydra import compose, initialize_config_module
from safetensors.torch import save_file
from alpamayo1_5_sft.trainer import ShortcutEMATeacherCallback
from transformers import TrainerControl


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
            "action_in_proj.step_size_adapter.2.weight": ("model-00002-of-00002.safetensors"),
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

    assert cfg.model._target_.endswith("ShortcutTrainableAlpamayoR1.from_pretrained")
    assert cfg.model.cotrain_vlm is False
    assert cfg.model.shortcut_flow_step_size == 0.125
    assert list(cfg.model.shortcut_step_sizes) == [0.25, 0.5, 1.0]
    assert cfg.model.shortcut_level_sampling == "balanced_cycle"
    assert cfg.model.shortcut_loss_weight == 0.125
    assert cfg.model.shortcut_target_clip == 4.0
    assert cfg.model.shortcut_teacher_mode == "online"
    assert cfg.model.shortcut_loss_estimator == "per_sample_weighted"
    assert cfg.model.shortcut_bootstrap_every == 8
    assert cfg.task_name == "alpamayo_1_5_shortcut_stage2_nav"

def test_checkpoint_ema_detection_from_index(tmp_path) -> None:
    index = {
        "weight_map": {
            "expert.layers.0.weight": "model-00001-of-00002.safetensors",
            "ema_expert.layers.0.weight": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert checkpoint_has_ema_teacher(tmp_path)


def test_checkpoint_without_ema_is_detected(tmp_path) -> None:
    index = {"weight_map": {"expert.layers.0.weight": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    assert not checkpoint_has_ema_teacher(tmp_path)


def test_ema_hydra_config_is_matched_and_opt_in() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        cfg = compose(
            config_name="sft_stage2_trajectory_shortcut_hf_reference_ema"
        )

    assert cfg.model.shortcut_teacher_mode == "ema"
    assert cfg.model.shortcut_ema_decay == 0.999
    assert cfg.model.shortcut_ema_dtype == "float32"
    assert cfg.model.shortcut_inference_weights == "ema"
    assert cfg.model.shortcut_loss_estimator == "reference_partition"
    assert cfg.model.shortcut_bootstrap_every == 8
    assert cfg.model.shortcut_loss_weight == 0.125
    assert cfg.model.cotrain_vlm is False


def test_ten_to_five_ema_hydra_config_is_isolated_and_paper_ratio() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        cfg = compose(
            config_name=(
                "sft_stage2_trajectory_shortcut_hf_10to5_reference_ema"
            )
        )

    assert cfg.model.shortcut_flow_step_size == 0.1
    assert list(cfg.model.shortcut_step_sizes) == [0.2]
    assert cfg.model.shortcut_require_dyadic_steps is False
    assert cfg.model.shortcut_teacher_mode == "ema"
    assert cfg.model.shortcut_ema_decay == 0.999
    assert cfg.model.shortcut_inference_weights == "ema"
    assert cfg.model.shortcut_loss_estimator == "reference_partition"
    assert cfg.model.shortcut_bootstrap_every == 4
    assert cfg.model.shortcut_loss_weight == 0.25
    assert cfg.model.cotrain_vlm is False
    assert cfg.trainer.weight_decay == 0.1



def test_ema_callback_updates_once_per_optimizer_step() -> None:
    class StubModel:
        def __init__(self) -> None:
            self.ema_updates = 0

        def update_shortcut_ema_teacher(self) -> None:
            self.ema_updates += 1

    model = StubModel()
    control = TrainerControl()
    callback = ShortcutEMATeacherCallback()

    returned = callback.on_optimizer_step(None, None, control, model=model)

    assert returned is control
    assert model.ema_updates == 1


def test_ten_to_five_local_evaluation_preserves_training_model_config() -> None:
    with initialize_config_module(
        version_base=None, config_module="alpamayo1_5_sft.configs"
    ):
        train_cfg = compose(
            config_name="sft_stage2_trajectory_shortcut_hf_10to5_reference_ema"
        )
        eval_cfg = compose(
            config_name="sft_stage2_trajectory_shortcut_10to5_reference_ema"
        )
    assert eval_cfg.model == train_cfg.model
    assert eval_cfg.model.shortcut_flow_step_size == 0.1
    assert list(eval_cfg.model.shortcut_step_sizes) == [0.2]
    assert eval_cfg.model.shortcut_require_dyadic_steps is False
    assert eval_cfg.data.val_dataset._target_.endswith("PAITrajectoryDataset")
    assert eval_cfg.data.val_dataset.get("access_mode", "local") == "local"



def test_constructor_recovers_shortcut_values_consumed_into_saved_config(
    monkeypatch,
) -> None:
    """Model reload must not fall back to online/student constructor defaults."""

    class StubStepProjection(torch.nn.Linear):
        @staticmethod
        def _validate_scalar_step_size(value: float) -> None:
            assert value > 0.0

    def stub_base_init(self, **kwargs) -> None:
        del kwargs
        torch.nn.Module.__init__(self)
        self.action_in_proj = StubStepProjection(2, 2)
        self.expert = torch.nn.Linear(2, 2)
        self.action_out_proj = torch.nn.Linear(2, 2)

    monkeypatch.setattr(
        shortcut_model_module,
        "StepSizeConditionedActionInProjV2",
        StubStepProjection,
    )
    monkeypatch.setattr(
        shortcut_model_module.TrainableAlpamayoR1,
        "__init__",
        stub_base_init,
    )
    config = SimpleNamespace(
        action_in_proj_cfg={"_target_": "unused.in.stub"},
        shortcut_flow_step_size=0.25,
        shortcut_step_sizes=[0.5, 1.0],
        shortcut_level_sampling="balanced_cycle",
        shortcut_loss_weight=0.25,
        shortcut_target_clip=3.0,
        shortcut_teacher_mode="ema",
        shortcut_ema_decay=0.95,
        shortcut_ema_dtype="float32",
        shortcut_inference_weights="ema",
        shortcut_loss_estimator="reference_partition",
        shortcut_bootstrap_every=4,
        step_size_fourier_feats=12,
        step_size_hidden_size=32,
    )

    model = ShortcutTrainableAlpamayoR1(config)

    assert model.shortcut_flow_step_size == 0.25
    assert model.shortcut_step_sizes == (0.5, 1.0)
    assert model.shortcut_loss_weight == 0.25
    assert model.shortcut_target_clip == 3.0
    assert model.shortcut_teacher_mode == "ema"
    assert model.shortcut_ema_decay == 0.95
    assert model.shortcut_ema_dtype == "float32"
    assert model.shortcut_inference_weights == "ema"
    assert model.shortcut_loss_estimator == "reference_partition"
    assert model.shortcut_bootstrap_every == 4
    assert hasattr(model, "ema_expert")
    assert not any(
        parameter.requires_grad for parameter in model.ema_expert.parameters()
    )


def test_mixed_precision_ema_checkpoint_restore_includes_update_counter(
    tmp_path,
) -> None:
    model = ShortcutTrainableAlpamayoR1.__new__(ShortcutTrainableAlpamayoR1)
    torch.nn.Module.__init__(model)
    model.shortcut_teacher_mode = "ema"
    model.shortcut_ema_dtype = "float32"
    model.action_in_proj = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.expert = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.action_out_proj = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.ema_action_in_proj = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.ema_expert = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.ema_action_out_proj = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    model.ema_expert.register_buffer(
        "nonpersistent_probe",
        torch.ones((), dtype=torch.float32),
        persistent=False,
    )
    model.register_buffer("shortcut_ema_updates", torch.zeros((), dtype=torch.long))

    model._cast_shortcut_ema_teacher()
    assert {parameter.dtype for parameter in model.ema_expert.parameters()} == {
        torch.float32
    }

    checkpoint_state = {}
    ema_tensors = [
        (name, tensor)
        for name, tensor in model.state_dict(keep_vars=True).items()
        if name.startswith(
            ("ema_action_in_proj.", "ema_expert.", "ema_action_out_proj.")
        )
    ]
    assert "ema_expert.nonpersistent_probe" not in dict(ema_tensors)
    for index, (name, tensor) in enumerate(ema_tensors, start=1):
        checkpoint_state[name] = torch.full_like(tensor, float(index))
        tensor.data.zero_()
    checkpoint_state["shortcut_ema_updates"] = torch.tensor(17, dtype=torch.long)
    save_file(checkpoint_state, tmp_path / "model.safetensors")

    model._restore_shortcut_ema_teacher(tmp_path)

    assert model.shortcut_ema_updates.item() == 17
    for index, (_, tensor) in enumerate(ema_tensors, start=1):
        torch.testing.assert_close(tensor, torch.full_like(tensor, float(index)))
    torch.testing.assert_close(
        model.ema_expert.nonpersistent_probe,
        torch.ones_like(model.ema_expert.nonpersistent_probe),
    )
