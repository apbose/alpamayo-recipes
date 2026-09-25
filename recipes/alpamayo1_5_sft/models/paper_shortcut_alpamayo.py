# SPDX-License-Identifier: Apache-2.0
"""Paper-target EMA port; intentionally separate from all previous runs."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo1_5_sft.models.shortcut_alpamayo_r1 import (
    ShortcutTrainableAlpamayoR1, ShortcutReasoningVLAOutput, checkpoint_has_step_size_adapter,
)
from alpamayo1_5_sft.models.paper_shortcut_targets import (
    paper_target_layout, require_full_hierarchy, draw_slot_randomness,
    paper_interpolation, paper_flow_target, paper_bootstrap_target,
)


class PaperLogStepEncoder(nn.Module):
    """Upstream sinusoidal features of dt_base=-log2(d), injected into AE tokens.

    Token injection is an Alpamayo adaptation, not the original DiT adaLN.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(dim // 2).float() / (dim // 2))[None])

    def forward(self, step_sizes):
        args = -torch.log2(step_sizes.float())[:, None] * self.freqs.float()
        return torch.cat((args.cos(), args.sin()), dim=-1)


def select_paper_target(mode, bootstrap_slot, teacher, clean, noise, noisy, t, d):
    """Change only supervision; the caller retains the same x_t/t/d/source layout."""
    if mode not in {"ema_bootstrap", "empirical_velocity"}:
        raise ValueError(f"Unknown paper supervision: {mode}")
    uses_teacher = bootstrap_slot and mode == "ema_bootstrap"
    if uses_teacher:
        return paper_bootstrap_target(teacher, noisy, t, d), True
    return paper_flow_target(clean, noise).detach(), False


class PaperEMAShortcutAlpamayo(ShortcutTrainableAlpamayoR1):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        checkpoint = pretrained_model_name_or_path
        fresh_adapter = not checkpoint_has_step_size_adapter(checkpoint)
        model = super().from_pretrained(checkpoint, *args, **kwargs)
        # Keep analytic frequency buffers FP32 after global dtype='auto'.
        for projection in (model.action_in_proj, model.ema_action_in_proj):
            projection.step_size_fourier_encoder = PaperLogStepEncoder(256).to(
                device=projection.step_size_adapter[0].weight.device
            )
        if fresh_adapter:
            nn.init.normal_(model.action_in_proj.step_size_adapter[0].weight, std=0.02)
            nn.init.zeros_(model.action_in_proj.step_size_adapter[0].bias)
            # Preserve the released function: residual output starts at zero.
            # This differs from DiT's injection architecture and is documented.
            model.action_in_proj.reset_step_size_adapter()
            model.reset_shortcut_ema_teacher()
        return model

    def __init__(self, config, paper_target_batch_size=None, paper_supervision=None, **kwargs):
        self.paper_supervision = str(
            getattr(config, "paper_supervision", "ema_bootstrap")
            if paper_supervision is None else paper_supervision
        )
        if self.paper_supervision not in {"ema_bootstrap", "empirical_velocity"}:
            raise ValueError(f"Unknown paper supervision: {self.paper_supervision}")
        self.paper_target_batch_size = int(
            getattr(config, "paper_target_batch_size", 64)
            if paper_target_batch_size is None else paper_target_batch_size
        )
        super().__init__(config, **kwargs)
        self.paper_layout = paper_target_layout(self.paper_target_batch_size, 128, self.shortcut_bootstrap_every)
        require_full_hierarchy(self.paper_layout)
        if self.shortcut_teacher_mode != "ema" or self.shortcut_flow_step_size != 1 / 128:
            raise ValueError("Paper mode requires EMA and a 1/128 flow anchor")
        if tuple(self.shortcut_step_sizes) != tuple(2 ** -i for i in range(6, -1, -1)):
            raise ValueError("Paper mode requires the full 1/64-to-1 hierarchy")
        if config.step_size_fourier_feats != 256:
            raise ValueError("Paper embedding requires 256 sinusoidal features")
        for projection in (self.action_in_proj, self.ema_action_in_proj):
            projection.step_size_fourier_encoder = PaperLogStepEncoder(256)
        config.paper_target_batch_size = self.paper_target_batch_size
        config.paper_supervision = self.paper_supervision
        config.paper_target_algorithm = "kvfrans_no_cfg_repeat_fill_overlap_v1"
        config.paper_step_encoding = "negative_log2_sinusoidal_256_maxperiod10000"
        config.paper_upstream_revision = "601004348667094e1b71f30942199759412d4432"
        self._freeze_shortcut_ema_teacher()

    def forward(self, tokenized_data, ego_history_xyz=None, ego_history_rot=None,
                ego_future_xyz=None, ego_future_rot=None, labels_mask=None,
                paper_slot=None, paper_seed=None, **kwargs):
        if paper_slot is None or paper_seed is None:
            raise ValueError("Paper mode requires an explicit target slot and random seed")
        slot, seed = int(paper_slot), int(paper_seed)
        if not 0 <= slot < self.paper_target_batch_size:
            raise ValueError("Target slot out of range")
        tokenized_data = dict(tokenized_data)
        input_ids = tokenized_data.pop("input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError("Memory-bounded paper driver uses one target per microbatch")
        traj = dict(ego_history_xyz=ego_history_xyz, ego_history_rot=ego_history_rot,
                    ego_future_xyz=ego_future_xyz, ego_future_rot=ego_future_rot)
        input_ids = self.fuse_traj_tokens(input_ids, traj)
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)
        self.vlm.eval()
        with torch.no_grad():
            outputs = self.vlm(input_ids=input_ids, labels=labels, use_cache=True, **tokenized_data)
            clean = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz, traj_history_rot=ego_history_rot,
                traj_future_xyz=ego_future_xyz, traj_future_rot=ego_future_rot,
            ).reshape(1, *self.action_space.get_action_space_dims()).float()
        future_starts = (input_ids == self.config.traj_token_ids["future_start"]).nonzero()
        if len(future_starts) != 1:
            raise ValueError("Expected exactly one future-start token")
        cache = outputs.past_key_values
        cache.crop(int(future_starts[0, 1]) + 1)
        for layer in cache.layers:
            layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
        positions = self._process_position_ids_qwen2_5_vl(outputs, 1, clean.shape[1], clean.device)
        noise, t_scalar = draw_slot_randomness(self.paper_layout, slot, clean.shape[1:], seed)
        noise = noise.to(device=clean.device)
        t = torch.full((1, 1, 1), t_scalar, device=clean.device)
        d = float(self.paper_layout.step_sizes[slot])
        noisy = paper_interpolation(clean, noise, t)
        def teacher(x, time, delta):
            self._freeze_shortcut_ema_teacher()
            return self._teacher_velocity(x, time, delta, prompt_cache=cache, position_ids=positions)
        target, is_bootstrap = select_paper_target(
            self.paper_supervision, bool(self.paper_layout.bootstrap_mask[slot]),
            teacher, clean, noise, noisy, t, d,
        )
        prediction = self._expert_velocity(noisy, t, d, prompt_cache=cache, position_ids=positions)
        loss = F.mse_loss(prediction.float(), target.float())
        return ShortcutReasoningVLAOutput(
            loss=loss, flow_loss=None if is_bootstrap else loss.detach(),
            shortcut_loss=loss.detach() if is_bootstrap else None,
            shortcut_step_size=torch.tensor(d, device=clean.device),
        )
