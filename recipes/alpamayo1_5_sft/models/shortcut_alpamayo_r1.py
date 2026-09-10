# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 1.5 model variant with explicit shortcut step-size conditioning."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, Sequence

import einops
import torch
import torch.nn.functional as F
from safetensors import safe_open

from alpamayo1_5_sft.models.sft_base_model import ReasoningVLAOutput
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.shortcut_modules import (
    BalancedShortcutLevelSampler,
    StepSizeConditionedActionInProjV2,
    construct_shortcut_training_data,
    fork_kv_cache,
    shortcut_midpoint,
    shortcut_velocity_target,
    validate_shortcut_step_sizes,
)
from alpamayo_r1.common import logging
from alpamayo_r1.models.base_model import IGNORE_INDEX


logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


@dataclass
class ShortcutReasoningVLAOutput(ReasoningVLAOutput):
    """Training losses exposed separately for diagnostics."""

    flow_loss: torch.FloatTensor | None = None
    shortcut_loss: torch.FloatTensor | None = None
    shortcut_step_size: torch.FloatTensor | None = None


_SHORTCUT_PROJECTION_TARGET = (
    "alpamayo1_5_sft.models.shortcut_modules."
    "StepSizeConditionedActionInProjV2"
)
_ADAPTER_STATE_PREFIX = "action_in_proj.step_size_"


def checkpoint_has_step_size_adapter(checkpoint: str | Path) -> bool:
    """Check checkpoint metadata for trained step-size-adapter tensors."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        return False

    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        return any(key.startswith(_ADAPTER_STATE_PREFIX) for key in weight_map)

    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return any(key.startswith(_ADAPTER_STATE_PREFIX) for key in handle.keys())

    return False


def shortcut_action_projection_config(
    original_config: dict[str, Any],
    *,
    flow_step_size: float,
    step_size_fourier_feats: int,
    step_size_hidden_size: int,
) -> dict[str, Any]:
    """Return a checkpoint-compatible action projection configuration."""
    result = copy.deepcopy(original_config)
    result["_target_"] = _SHORTCUT_PROJECTION_TARGET
    result["default_step_size"] = flow_step_size
    result["step_size_fourier_feats"] = step_size_fourier_feats
    result["step_size_hidden_size"] = step_size_hidden_size
    return result


class ShortcutTrainableAlpamayoR1(TrainableAlpamayoR1):
    """Trainable Alpamayo expert conditioned on the requested solver interval."""

    def __init__(
        self,
        config,
        pretrained_modules=None,
        original_vocab_size: int | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        stage1_vlm_checkpoint_path: str | None = None,
        shortcut_flow_step_size: float = 0.125,
        shortcut_step_sizes: Sequence[float] = (0.25, 0.5, 1.0),
        shortcut_level_sampling: str = "balanced_cycle",
        shortcut_loss_weight: float = 0.125,
        shortcut_target_clip: float = 4.0,
        shortcut_teacher_mode: str = "online",
        step_size_fourier_feats: int = 20,
        step_size_hidden_size: int = 256,
    ) -> None:
        if cotrain_vlm or not stop_grad_from_vlm:
            raise ValueError(
                "Shortcut Stage-2 training requires a frozen, stop-gradient VLM"
            )
        StepSizeConditionedActionInProjV2._validate_scalar_step_size(
            shortcut_flow_step_size
        )
        shortcut_step_sizes = validate_shortcut_step_sizes(shortcut_step_sizes)
        expected_step_size = 2.0 * shortcut_flow_step_size
        for step_size in shortcut_step_sizes:
            if abs(step_size - expected_step_size) > 1e-6:
                raise ValueError(
                    "shortcut_step_sizes must form a doubling hierarchy above "
                    f"shortcut_flow_step_size; expected {expected_step_size}, "
                    f"got {step_size}"
                )
            expected_step_size *= 2.0
        if shortcut_level_sampling != "balanced_cycle":
            raise ValueError(
                "Only shortcut_level_sampling='balanced_cycle' is supported"
            )
        if not 0.0 <= shortcut_loss_weight <= 1.0:
            raise ValueError("shortcut_loss_weight must be in [0, 1]")
        if shortcut_target_clip <= 0.0:
            raise ValueError("shortcut_target_clip must be positive")
        if shortcut_teacher_mode != "online":
            raise ValueError(
                "Only the online stopped-gradient teacher is currently supported"
            )

        config.shortcut_flow_step_size = float(shortcut_flow_step_size)
        config.shortcut_step_sizes = list(shortcut_step_sizes)
        config.shortcut_level_sampling = shortcut_level_sampling
        config.shortcut_loss_weight = float(shortcut_loss_weight)
        config.shortcut_target_clip = float(shortcut_target_clip)
        config.shortcut_teacher_mode = shortcut_teacher_mode
        config.step_size_fourier_feats = int(step_size_fourier_feats)
        config.step_size_hidden_size = int(step_size_hidden_size)
        config.action_in_proj_cfg = shortcut_action_projection_config(
            config.action_in_proj_cfg,
            flow_step_size=shortcut_flow_step_size,
            step_size_fourier_feats=step_size_fourier_feats,
            step_size_hidden_size=step_size_hidden_size,
        )

        super().__init__(
            config=config,
            pretrained_modules=pretrained_modules,
            original_vocab_size=original_vocab_size,
            cotrain_vlm=cotrain_vlm,
            stop_grad_from_vlm=stop_grad_from_vlm,
            stage1_vlm_checkpoint_path=stage1_vlm_checkpoint_path,
        )
        if not isinstance(self.action_in_proj, StepSizeConditionedActionInProjV2):
            raise TypeError(
                "Shortcut model requires StepSizeConditionedActionInProjV2, got "
                f"{type(self.action_in_proj).__name__}"
            )
        self.shortcut_flow_step_size = float(shortcut_flow_step_size)
        self.shortcut_step_sizes = shortcut_step_sizes
        self.shortcut_level_sampling = shortcut_level_sampling
        self.shortcut_level_sampler = BalancedShortcutLevelSampler(
            num_levels=len(shortcut_step_sizes)
        )
        self.shortcut_loss_weight = float(shortcut_loss_weight)
        self.shortcut_target_clip = float(shortcut_target_clip)
        self.shortcut_teacher_mode = shortcut_teacher_mode

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        """Load base weights and preserve zero residual for a new adapter."""
        has_adapter_weights = checkpoint_has_step_size_adapter(
            pretrained_model_name_or_path
        )
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        if not has_adapter_weights:
            model.action_in_proj.reset_step_size_adapter()
        return model

    def _expert_velocity(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        step_sizes: torch.Tensor | float,
        *,
        prompt_cache: Any,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one vector field while preserving the reusable prompt cache."""
        action_embeds = self.action_in_proj(
            x,
            timesteps,
            step_size=step_sizes,
        )
        expert_cache = fork_kv_cache(prompt_cache)
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            past_key_values=expert_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        return pred.view(-1, *self.action_space.get_action_space_dims())

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ShortcutReasoningVLAOutput:
        """Train with a flow anchor and a two-half-steps shortcut target."""
        input_ids = tokenized_data.pop("input_ids")
        batch_size = input_ids.shape[0]
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # The mentor's Stage-2 plan freezes the entire VLM. Run it once and
        # reuse only its detached scene-context cache for every expert branch.
        with torch.no_grad():
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
                **tokenized_data,
            )

        future_start_token_id = self.config.traj_token_ids["future_start"]
        future_start_indices = (input_ids == future_start_token_id).nonzero(
            as_tuple=False
        )
        if future_start_indices.numel() == 0:
            raise ValueError("Training sample has no <traj_future_start> token")
        last_future_start_idx = future_start_indices[-1, 1] + 1

        flow_data = self._process_traj_future_training(traj_data)
        if flow_data["x"].shape[0] != batch_size:
            raise ValueError(
                "Shortcut Stage-2 currently requires one future trajectory per "
                f"input; got {flow_data['x'].shape[0]} actions for batch {batch_size}"
            )

        prompt_cache = vlm_outputs.past_key_values
        prompt_cache.crop(last_future_start_idx)
        for layer in prompt_cache.layers:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()

        num_expert_tokens = flow_data["x"].shape[1]
        position_ids = self._process_position_ids_qwen2_5_vl(
            vlm_outputs,
            batch_size,
            num_expert_tokens,
            flow_data["x"].device,
        )

        level_indices = self.shortcut_level_sampler(
            batch_size=batch_size,
            device=flow_data["x"].device,
        )
        shortcut_data = construct_shortcut_training_data(
            flow_data["x"],
            step_sizes=self.shortcut_step_sizes,
            noise=flow_data["noise"],
            level_indices=level_indices,
        )

        # The current teacher is the online Action Expert in eval mode. Its
        # outputs are stopped-gradient; an EMA teacher can be added later.
        expert_was_training = self.expert.training
        self.expert.eval()
        try:
            with torch.no_grad():
                first_velocity = self._expert_velocity(
                    shortcut_data.noisy_x,
                    shortcut_data.timesteps,
                    shortcut_data.half_step_sizes,
                    prompt_cache=prompt_cache,
                    position_ids=position_ids,
                )
                midpoint = shortcut_midpoint(
                    shortcut_data.noisy_x,
                    first_velocity,
                    shortcut_data.half_step_sizes,
                    clip_value=self.shortcut_target_clip,
                )
                second_velocity = self._expert_velocity(
                    midpoint,
                    shortcut_data.midpoint_timesteps,
                    shortcut_data.half_step_sizes,
                    prompt_cache=prompt_cache,
                    position_ids=position_ids,
                )
                target_velocity = shortcut_velocity_target(
                    first_velocity,
                    second_velocity,
                    clip_value=self.shortcut_target_clip,
                )
        finally:
            self.expert.train(expert_was_training)

        shortcut_pred = self._expert_velocity(
            shortcut_data.noisy_x,
            shortcut_data.timesteps,
            shortcut_data.step_sizes,
            prompt_cache=prompt_cache,
            position_ids=position_ids,
        )
        shortcut_loss = F.mse_loss(
            shortcut_pred,
            target_velocity.to(dtype=shortcut_pred.dtype),
        )

        flow_pred = self._expert_velocity(
            flow_data["noisy_x"],
            flow_data["timesteps"],
            self.shortcut_flow_step_size,
            prompt_cache=prompt_cache,
            position_ids=position_ids,
        )
        flow_loss = self.diffusion.compute_loss_from_pred(
            training_data=flow_data,
            pred=flow_pred,
        )

        loss = (
            (1.0 - self.shortcut_loss_weight) * flow_loss
            + self.shortcut_loss_weight * shortcut_loss
        )
        step_size_mean = shortcut_data.step_sizes.float().mean()
        logger.info(
            "Shortcut objective: "
            f"flow_loss={flow_loss.detach().float().item():.6f}, "
            f"shortcut_loss={shortcut_loss.detach().float().item():.6f}, "
            f"d_mean={step_size_mean.detach().item():.6f}, "
            f"level_indices={level_indices.detach().cpu().tolist()}, "
            f"cycle_cursor={int(self.shortcut_level_sampler.cursor.item())}"
        )
        return ShortcutReasoningVLAOutput(
            loss=loss,
            flow_loss=flow_loss.detach(),
            shortcut_loss=shortcut_loss.detach(),
            shortcut_step_size=step_size_mean.detach(),
        )

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        *args: Any,
        diffusion_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Condition legacy Alpamayo inference calls on 1 / solver_steps."""
        diffusion_kwargs = dict(diffusion_kwargs or {})
        inference_steps = int(
            diffusion_kwargs.get(
                "inference_step",
                self.diffusion.num_inference_steps,
            )
        )
        if inference_steps <= 0:
            raise ValueError(f"inference_step must be positive, got {inference_steps}")

        previous_step_size = self.action_in_proj.default_step_size
        self.action_in_proj.set_default_step_size(1.0 / inference_steps)
        try:
            return super().sample_trajectories_from_data_with_vlm_rollout(
                data,
                *args,
                diffusion_kwargs=diffusion_kwargs,
                **kwargs,
            )
        finally:
            self.action_in_proj.set_default_step_size(previous_step_size)
