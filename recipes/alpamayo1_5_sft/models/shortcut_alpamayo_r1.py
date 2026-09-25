# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 1.5 model variant with explicit shortcut step-size conditioning."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
import torch.nn.functional as F
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import ReasoningVLAOutput
from alpamayo1_5_sft.models.shortcut_modules import (
    BalancedShortcutLevelSampler,
    ReferenceShortcutBranchSampler,
    StepSizeConditionedActionInProjV2,
    construct_shortcut_training_data,
    fork_kv_cache,
    shortcut_midpoint,
    shortcut_velocity_target,
    update_ema_module_,
    validate_shortcut_step_sizes,
)
from alpamayo_r1.common import logging
from alpamayo_r1.models.base_model import IGNORE_INDEX
from safetensors import safe_open

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


@dataclass
class ShortcutReasoningVLAOutput(ReasoningVLAOutput):
    """Training losses exposed separately for diagnostics."""

    flow_loss: torch.FloatTensor | None = None
    shortcut_loss: torch.FloatTensor | None = None
    shortcut_step_size: torch.FloatTensor | None = None


_SHORTCUT_PROJECTION_TARGET = (
    "alpamayo1_5_sft.models.shortcut_modules.StepSizeConditionedActionInProjV2"
)
_ADAPTER_STATE_PREFIX = "action_in_proj.step_size_"
_EMA_STATE_PREFIX = "ema_expert."


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
            return any(key.startswith(_ADAPTER_STATE_PREFIX) for key in handle)

    return False

def checkpoint_has_ema_teacher(checkpoint: str | Path) -> bool:
    """Check whether a local checkpoint contains the EMA Action Expert."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir():
        return False

    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        return any(key.startswith(_EMA_STATE_PREFIX) for key in weight_map)

    safetensors_path = checkpoint_path / "model.safetensors"
    if safetensors_path.is_file():
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            return any(key.startswith(_EMA_STATE_PREFIX) for key in handle)

    return False


def _resolve_ema_dtype(name: str) -> torch.dtype:
    supported = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return supported[name]
    except KeyError as error:
        raise ValueError(
            f"shortcut_ema_dtype must be one of {sorted(supported)}, got {name!r}"
        ) from error




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
        shortcut_flow_step_size: float | None = None,
        shortcut_step_sizes: Sequence[float] | None = None,
        shortcut_require_dyadic_steps: bool | None = None,
        shortcut_level_sampling: str | None = None,
        shortcut_loss_weight: float | None = None,
        shortcut_target_clip: float | None = None,
        shortcut_teacher_mode: str | None = None,
        shortcut_ema_decay: float | None = None,
        shortcut_ema_dtype: str | None = None,
        shortcut_inference_weights: str | None = None,
        shortcut_loss_estimator: str | None = None,
        shortcut_bootstrap_every: int | None = None,
        step_size_fourier_feats: int | None = None,
        step_size_hidden_size: int | None = None,
    ) -> None:
        # ``PreTrainedModel.from_pretrained`` consumes keyword arguments that
        # already exist in a checkpoint config instead of forwarding them to
        # ``__init__``. Use ``None`` sentinels and recover those persisted
        # values here, otherwise an EMA checkpoint silently reloads with the
        # online/student Python defaults.
        shortcut_flow_step_size = float(
            getattr(config, "shortcut_flow_step_size", 0.125)
            if shortcut_flow_step_size is None
            else shortcut_flow_step_size
        )
        shortcut_step_sizes = (
            getattr(config, "shortcut_step_sizes", (0.25, 0.5, 1.0))
            if shortcut_step_sizes is None
            else shortcut_step_sizes
        )
        shortcut_require_dyadic_steps = bool(
            getattr(config, "shortcut_require_dyadic_steps", True)
            if shortcut_require_dyadic_steps is None
            else shortcut_require_dyadic_steps
        )
        shortcut_level_sampling = str(
            getattr(config, "shortcut_level_sampling", "balanced_cycle")
            if shortcut_level_sampling is None
            else shortcut_level_sampling
        )
        shortcut_loss_weight = float(
            getattr(config, "shortcut_loss_weight", 0.125)
            if shortcut_loss_weight is None
            else shortcut_loss_weight
        )
        shortcut_target_clip = float(
            getattr(config, "shortcut_target_clip", 4.0)
            if shortcut_target_clip is None
            else shortcut_target_clip
        )
        shortcut_teacher_mode = str(
            getattr(config, "shortcut_teacher_mode", "online")
            if shortcut_teacher_mode is None
            else shortcut_teacher_mode
        )
        shortcut_ema_decay = float(
            getattr(config, "shortcut_ema_decay", 0.999)
            if shortcut_ema_decay is None
            else shortcut_ema_decay
        )
        shortcut_ema_dtype = str(
            getattr(config, "shortcut_ema_dtype", "float32")
            if shortcut_ema_dtype is None
            else shortcut_ema_dtype
        )
        shortcut_inference_weights = str(
            getattr(config, "shortcut_inference_weights", "student")
            if shortcut_inference_weights is None
            else shortcut_inference_weights
        )
        shortcut_loss_estimator = str(
            getattr(config, "shortcut_loss_estimator", "per_sample_weighted")
            if shortcut_loss_estimator is None
            else shortcut_loss_estimator
        )
        shortcut_bootstrap_every = int(
            getattr(config, "shortcut_bootstrap_every", 8)
            if shortcut_bootstrap_every is None
            else shortcut_bootstrap_every
        )
        step_size_fourier_feats = int(
            getattr(config, "step_size_fourier_feats", 20)
            if step_size_fourier_feats is None
            else step_size_fourier_feats
        )
        step_size_hidden_size = int(
            getattr(config, "step_size_hidden_size", 256)
            if step_size_hidden_size is None
            else step_size_hidden_size
        )
        if cotrain_vlm or not stop_grad_from_vlm:
            raise ValueError("Shortcut Stage-2 training requires a frozen, stop-gradient VLM")
        StepSizeConditionedActionInProjV2._validate_scalar_step_size(shortcut_flow_step_size)
        shortcut_step_sizes = validate_shortcut_step_sizes(
            shortcut_step_sizes,
            require_dyadic=shortcut_require_dyadic_steps,
        )
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
            raise ValueError("Only shortcut_level_sampling='balanced_cycle' is supported")
        if not 0.0 <= shortcut_loss_weight <= 1.0:
            raise ValueError("shortcut_loss_weight must be in [0, 1]")
        if shortcut_target_clip <= 0.0:
            raise ValueError("shortcut_target_clip must be positive")
        if shortcut_teacher_mode not in {"online", "ema"}:
            raise ValueError("shortcut_teacher_mode must be 'online' or 'ema'")
        if not 0.0 <= shortcut_ema_decay < 1.0:
            raise ValueError("shortcut_ema_decay must be in [0, 1)")
        ema_dtype = _resolve_ema_dtype(shortcut_ema_dtype)
        if shortcut_inference_weights not in {"student", "ema"}:
            raise ValueError("shortcut_inference_weights must be 'student' or 'ema'")
        if shortcut_inference_weights == "ema" and shortcut_teacher_mode != "ema":
            raise ValueError(
                "shortcut_inference_weights='ema' requires "
                "shortcut_teacher_mode='ema'"
            )
        if shortcut_loss_estimator not in {
            "per_sample_weighted",
            "reference_partition",
        }:
            raise ValueError(
                "shortcut_loss_estimator must be 'per_sample_weighted' or 'reference_partition'"
            )
        if shortcut_bootstrap_every <= 1:
            raise ValueError("shortcut_bootstrap_every must be greater than one")
        if shortcut_loss_estimator == "reference_partition":
            expected_weight = 1.0 / shortcut_bootstrap_every
            if abs(shortcut_loss_weight - expected_weight) > 1e-9:
                raise ValueError(
                    "reference_partition requires shortcut_loss_weight == "
                    f"1 / shortcut_bootstrap_every ({expected_weight})"
                )

        config.shortcut_flow_step_size = float(shortcut_flow_step_size)
        config.shortcut_step_sizes = list(shortcut_step_sizes)
        config.shortcut_level_sampling = shortcut_level_sampling
        config.shortcut_require_dyadic_steps = shortcut_require_dyadic_steps
        config.shortcut_loss_weight = float(shortcut_loss_weight)
        config.shortcut_target_clip = float(shortcut_target_clip)
        config.shortcut_teacher_mode = shortcut_teacher_mode
        config.shortcut_loss_estimator = shortcut_loss_estimator
        config.shortcut_bootstrap_every = int(shortcut_bootstrap_every)
        config.shortcut_ema_decay = float(shortcut_ema_decay)
        config.shortcut_ema_dtype = shortcut_ema_dtype
        config.shortcut_inference_weights = shortcut_inference_weights
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
        self.shortcut_require_dyadic_steps = shortcut_require_dyadic_steps
        self.shortcut_level_sampling = shortcut_level_sampling
        self.shortcut_level_sampler = BalancedShortcutLevelSampler(
            num_levels=len(shortcut_step_sizes)
        )
        self.shortcut_loss_weight = float(shortcut_loss_weight)
        self.shortcut_target_clip = float(shortcut_target_clip)
        self.shortcut_teacher_mode = shortcut_teacher_mode
        self.shortcut_loss_estimator = shortcut_loss_estimator
        self.shortcut_bootstrap_every = int(shortcut_bootstrap_every)
        self.shortcut_ema_decay = float(shortcut_ema_decay)
        self.shortcut_ema_dtype = shortcut_ema_dtype
        self.shortcut_inference_weights = shortcut_inference_weights
        self.shortcut_branch_sampler = ReferenceShortcutBranchSampler(
            bootstrap_every=shortcut_bootstrap_every
        )
        if self.shortcut_teacher_mode == "ema":
            self.ema_action_in_proj = copy.deepcopy(self.action_in_proj).to(dtype=ema_dtype)
            self.ema_expert = copy.deepcopy(self.expert).to(dtype=ema_dtype)
            self.ema_action_out_proj = copy.deepcopy(self.action_out_proj).to(dtype=ema_dtype)
            self.register_buffer(
                "shortcut_ema_updates",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self._freeze_shortcut_ema_teacher()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        """Load base weights and preserve zero residual for a new adapter."""
        has_adapter_weights = checkpoint_has_step_size_adapter(pretrained_model_name_or_path)
        has_ema_weights = checkpoint_has_ema_teacher(pretrained_model_name_or_path)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        if not has_adapter_weights:
            model.action_in_proj.reset_step_size_adapter()
        if model.shortcut_teacher_mode == "ema":
            # ``from_pretrained(dtype="auto")`` applies the checkpoint's global
            # BF16 dtype after construction, including to our copied teacher.
            # Re-establish the requested mixed precision explicitly. For an EMA
            # checkpoint, reload those tensors directly so their saved FP32
            # precision is not lost through the global BF16 loading path.
            model._cast_shortcut_ema_teacher()
            if has_ema_weights:
                model._restore_shortcut_ema_teacher(pretrained_model_name_or_path)
                model._freeze_shortcut_ema_teacher()
            else:
                model.reset_shortcut_ema_teacher()
        return model

    def _shortcut_ema_module_pairs(
        self,
    ) -> tuple[tuple[torch.nn.Module, torch.nn.Module], ...]:
        if self.shortcut_teacher_mode != "ema":
            raise RuntimeError("EMA teacher modules are unavailable in online teacher mode")
        return (
            (self.ema_action_in_proj, self.action_in_proj),
            (self.ema_expert, self.expert),
            (self.ema_action_out_proj, self.action_out_proj),
        )

    def _cast_shortcut_ema_teacher(self) -> None:
        """Restore the teacher dtype after the global HF checkpoint cast."""
        ema_dtype = _resolve_ema_dtype(self.shortcut_ema_dtype)
        for ema_module, _ in self._shortcut_ema_module_pairs():
            ema_module.to(dtype=ema_dtype)

    @torch.no_grad()
    def _restore_shortcut_ema_teacher(self, checkpoint: str | Path) -> None:
        """Reload mixed-precision EMA tensors without a lossy global cast."""
        checkpoint_path = Path(checkpoint)
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            safetensors_path = checkpoint_path / "model.safetensors"
            if not safetensors_path.is_file():
                raise FileNotFoundError(
                    f"No safetensors checkpoint found in {checkpoint_path}"
                )
            with safe_open(
                safetensors_path,
                framework="pt",
                device="cpu",
            ) as handle:
                weight_map = {
                    key: safetensors_path.name for key in handle.keys()
                }

        ema_prefixes = (
            "ema_action_in_proj.",
            "ema_expert.",
            "ema_action_out_proj.",
        )
        # Use ``state_dict`` rather than ``named_buffers`` so dynamically
        # regenerated, non-persistent buffers (for example rotary ``inv_freq``)
        # are not incorrectly required in the checkpoint.
        target_tensors = {
            name: tensor
            for name, tensor in self.state_dict(keep_vars=True).items()
            if name.startswith(ema_prefixes) or name == "shortcut_ema_updates"
        }
        missing = sorted(set(target_tensors) - set(weight_map))
        if missing:
            raise ValueError(
                "EMA checkpoint is missing target tensors: "
                f"{missing[:5]} (total={len(missing)})"
            )

        keys_by_shard: dict[str, list[str]] = {}
        for key in target_tensors:
            keys_by_shard.setdefault(weight_map[key], []).append(key)
        loaded = 0
        for shard_name, keys in keys_by_shard.items():
            with safe_open(
                checkpoint_path / shard_name,
                framework="pt",
                device="cpu",
            ) as handle:
                for key in keys:
                    source = handle.get_tensor(key)
                    target = target_tensors[key]
                    if source.shape != target.shape:
                        raise ValueError(
                            f"EMA tensor shape mismatch for {key}: "
                            f"checkpoint={tuple(source.shape)}, "
                            f"model={tuple(target.shape)}"
                        )
                    target.copy_(
                        source.to(device=target.device, dtype=target.dtype)
                    )
                    loaded += 1
        logger.info(
            "Restored %d EMA tensors from %s in %s",
            loaded,
            checkpoint_path,
            self.shortcut_ema_dtype,
        )

    def _freeze_shortcut_ema_teacher(self) -> None:
        """Keep EMA modules out of the optimizer and in deterministic eval mode."""
        for ema_module, _ in self._shortcut_ema_module_pairs():
            ema_module.requires_grad_(False)
            ema_module.eval()

    @torch.no_grad()
    def reset_shortcut_ema_teacher(self) -> None:
        """Initialize EMA parameters exactly from the loaded online expert."""
        for ema_module, online_module in self._shortcut_ema_module_pairs():
            update_ema_module_(ema_module, online_module, decay=0.0)
        self.shortcut_ema_updates.zero_()
        self._freeze_shortcut_ema_teacher()

    @torch.no_grad()
    def update_shortcut_ema_teacher(self) -> None:
        """Advance the EMA teacher once after an optimizer update."""
        if self.shortcut_teacher_mode != "ema":
            raise RuntimeError("Cannot update an EMA teacher in online teacher mode")
        for ema_module, online_module in self._shortcut_ema_module_pairs():
            update_ema_module_(
                ema_module,
                online_module,
                decay=self.shortcut_ema_decay,
            )
        self.shortcut_ema_updates.add_(1)
        self._freeze_shortcut_ema_teacher()

    def _teacher_velocity(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        step_sizes: torch.Tensor | float,
        *,
        prompt_cache: Any,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a stopped-gradient target using the configured teacher."""
        if self.shortcut_teacher_mode == "online":
            return self._expert_velocity(
                x,
                timesteps,
                step_sizes,
                prompt_cache=prompt_cache,
                position_ids=position_ids,
            )
        return self._expert_velocity(
            x,
            timesteps,
            step_sizes,
            prompt_cache=prompt_cache,
            position_ids=position_ids,
            action_in_proj=self.ema_action_in_proj,
            expert=self.ema_expert,
            action_out_proj=self.ema_action_out_proj,
        )



    def _expert_velocity(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        step_sizes: torch.Tensor | float,
        *,
        prompt_cache: Any,
        position_ids: torch.Tensor,
        action_in_proj: torch.nn.Module | None = None,
        expert: torch.nn.Module | None = None,
        action_out_proj: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        """Predict one vector field while preserving the reusable prompt cache."""
        action_in_proj = self.action_in_proj if action_in_proj is None else action_in_proj
        expert = self.expert if expert is None else expert
        action_out_proj = self.action_out_proj if action_out_proj is None else action_out_proj
        action_embeds = action_in_proj(
            x,
            timesteps,
            step_size=step_sizes,
        )
        expert_cache = fork_kv_cache(prompt_cache)
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = expert(
            inputs_embeds=action_embeds,
            position_ids=position_ids,
            past_key_values=expert_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = action_out_proj(diffusion_out)
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
        future_start_indices = (input_ids == future_start_token_id).nonzero(as_tuple=False)
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

        run_shortcut_branch = True
        run_flow_branch = True
        branch_name = "both"
        if self.shortcut_loss_estimator == "reference_partition":
            shortcut_mask = self.shortcut_branch_sampler(
                batch_size=batch_size,
                device=flow_data["x"].device,
            )
            if bool(shortcut_mask.all()):
                run_flow_branch = False
                branch_name = "shortcut"
            elif bool((~shortcut_mask).all()):
                run_shortcut_branch = False
                branch_name = "flow"
            else:
                raise ValueError(
                    "reference_partition produced a mixed local batch. Use "
                    "per_device_train_batch_size=1 and form the effective batch "
                    "with DDP and/or gradient accumulation."
                )

        flow_loss = None
        shortcut_loss = None
        step_size_mean = None
        level_indices = None

        if run_shortcut_branch:
            level_indices = self.shortcut_level_sampler(
                batch_size=batch_size,
                device=flow_data["x"].device,
                rank=0,
                world_size=1,
            )
            shortcut_data = construct_shortcut_training_data(
                flow_data["x"],
                step_sizes=self.shortcut_step_sizes,
                noise=flow_data["noise"],
                level_indices=level_indices,
                require_dyadic=self.shortcut_require_dyadic_steps,
            )

            if self.shortcut_teacher_mode == "online":
                teacher_modules = (self.expert,)
            else:
                teacher_modules = tuple(
                    ema_module for ema_module, _ in self._shortcut_ema_module_pairs()
                )
            teacher_training_modes = tuple(module.training for module in teacher_modules)
            for module in teacher_modules:
                module.eval()
            try:
                with torch.no_grad():
                    first_velocity = self._teacher_velocity(
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
                    second_velocity = self._teacher_velocity(
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
                for module, was_training in zip(
                    teacher_modules,
                    teacher_training_modes,
                    strict=True,
                ):
                    module.train(was_training)

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
            step_size_mean = shortcut_data.step_sizes.float().mean()

        if run_flow_branch:
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

        if flow_loss is not None and shortcut_loss is not None:
            loss = (
                1.0 - self.shortcut_loss_weight
            ) * flow_loss + self.shortcut_loss_weight * shortcut_loss
        elif shortcut_loss is not None:
            loss = shortcut_loss
        elif flow_loss is not None:
            loss = flow_loss
        else:  # pragma: no cover - protected by branch allocation above
            raise RuntimeError("No shortcut objective branch was selected")

        flow_log = "none" if flow_loss is None else f"{flow_loss.detach().float().item():.6f}"
        shortcut_log = (
            "none" if shortcut_loss is None else f"{shortcut_loss.detach().float().item():.6f}"
        )
        step_log = "none" if step_size_mean is None else f"{step_size_mean.detach().item():.6f}"
        level_log = None if level_indices is None else level_indices.detach().cpu().tolist()
        logger.info(
            "Shortcut objective: "
            f"estimator={self.shortcut_loss_estimator}, "
            f"branch={branch_name}, "
            f"flow_loss={flow_log}, "
            f"shortcut_loss={shortcut_log}, "
            f"d_mean={step_log}, "
            f"level_indices={level_log}, "
            f"level_cursor={int(self.shortcut_level_sampler.cursor.item())}, "
            f"branch_cursor={int(self.shortcut_branch_sampler.cursor.item())}, "
            f"teacher={self.shortcut_teacher_mode}, "
            f"ema_updates={int(getattr(self, 'shortcut_ema_updates', torch.tensor(0)).item())}"
        )
        return ShortcutReasoningVLAOutput(
            loss=loss,
            flow_loss=None if flow_loss is None else flow_loss.detach(),
            shortcut_loss=None if shortcut_loss is None else shortcut_loss.detach(),
            shortcut_step_size=(None if step_size_mean is None else step_size_mean.detach()),
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

        with self._inference_action_modules():
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

    @contextmanager
    def _inference_action_modules(self):
        """Temporarily expose EMA action modules to the inherited sampler."""
        if self.shortcut_inference_weights == "student":
            yield
            return
        if self.shortcut_teacher_mode != "ema":  # pragma: no cover - constructor guard
            raise RuntimeError("EMA inference requested without an EMA teacher")

        student_modules = (
            self.action_in_proj,
            self.expert,
            self.action_out_proj,
        )
        self.action_in_proj = self.ema_action_in_proj
        self.expert = self.ema_expert
        self.action_out_proj = self.ema_action_out_proj
        try:
            yield
        finally:
            self.action_in_proj, self.expert, self.action_out_proj = student_modules
