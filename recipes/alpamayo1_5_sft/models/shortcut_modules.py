# SPDX-License-Identifier: Apache-2.0

"""Step-size-conditioned Action Expert input modules for shortcut training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Real
from typing import Any, Sequence

import torch
from torch import nn

from alpamayo_r1.models.action_in_proj import (
    FourierEncoderV2,
    PerWaypointActionInProjV2,
)


@dataclass(frozen=True)
class ShortcutTrainingData:
    """A dyadic interval sampled from one noise-to-data flow path."""

    x: torch.Tensor
    noise: torch.Tensor
    noisy_x: torch.Tensor
    timesteps: torch.Tensor
    step_sizes: torch.Tensor

    @property
    def half_step_sizes(self) -> torch.Tensor:
        return self.step_sizes / 2.0

    @property
    def midpoint_timesteps(self) -> torch.Tensor:
        return self.timesteps + self.half_step_sizes


def validate_shortcut_step_sizes(step_sizes: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(step_size) for step_size in step_sizes)
    if not result:
        raise ValueError("shortcut step_sizes must not be empty")

    for step_size in result:
        if not 0.0 < step_size <= 1.0:
            raise ValueError(
                f"every shortcut step_size must be in (0, 1], got {step_size}"
            )
        sections = round(1.0 / step_size)
        if sections <= 0 or abs(sections * step_size - 1.0) > 1e-6:
            raise ValueError(
                "shortcut step_sizes must partition [0, 1] exactly; "
                f"got {step_size}"
            )
        if sections & (sections - 1):
            raise ValueError(
                f"shortcut step_sizes must be dyadic, got {step_size}"
            )
    return result


class BalancedShortcutLevelSampler(nn.Module):
    """Cycle through shortcut levels with checkpoint-persistent state.

    Consecutive samples receive consecutive level indices. For distributed
    training, each rank receives a disjoint consecutive slice and every rank
    advances its replicated cursor by the same global batch size.
    """

    def __init__(self, num_levels: int, start_index: int = 0) -> None:
        super().__init__()
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if not 0 <= start_index < num_levels:
            raise ValueError("start_index must be in [0, num_levels)")
        self.num_levels = int(num_levels)
        self.register_buffer(
            "cursor",
            torch.tensor(start_index, dtype=torch.long),
            persistent=True,
        )

    def reset(self, start_index: int = 0) -> None:
        """Reset the next global level index."""
        if not 0 <= start_index < self.num_levels:
            raise ValueError("start_index must be in [0, num_levels)")
        self.cursor.fill_(start_index)

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        *,
        device: torch.device,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> torch.Tensor:
        """Return balanced level indices and advance the persistent cursor."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank = 0
                world_size = 1
        assert world_size is not None
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

        current = int(self.cursor.item())
        rank_start = current + rank * batch_size
        indices = (
            torch.arange(batch_size, device=device, dtype=torch.long) + rank_start
        ) % self.num_levels
        next_cursor = (current + world_size * batch_size) % self.num_levels
        self.cursor.fill_(next_cursor)
        return indices


def construct_shortcut_training_data(
    x: torch.Tensor,
    *,
    step_sizes: Sequence[float] = (0.25, 0.5, 1.0),
    noise: torch.Tensor | None = None,
    level_indices: torch.Tensor | None = None,
    time_indices: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> ShortcutTrainingData:
    """Sample aligned (x_t, t, d) values for shortcut consistency.

    For an interval d, t is selected from {0, d, ..., 1-d}. This guarantees
    that both half steps used to form the teacher target stay inside the
    training path from noise at time zero to data at time one.

    level_indices and time_indices are injectable for deterministic tests.
    Normal training leaves them unset.
    """
    if x.ndim < 2 or x.shape[0] <= 0:
        raise ValueError("x must have a non-empty batch dimension")
    levels = validate_shortcut_step_sizes(step_sizes)
    batch_size = x.shape[0]
    device = x.device

    if level_indices is None:
        level_indices = torch.randint(
            len(levels),
            (batch_size,),
            device=device,
            generator=generator,
        )
    else:
        level_indices = level_indices.to(device=device, dtype=torch.long).reshape(-1)
        if level_indices.shape[0] != batch_size:
            raise ValueError("level_indices batch dimension must match x")
        if not bool(((level_indices >= 0) & (level_indices < len(levels))).all()):
            raise ValueError("level_indices contains an out-of-range value")

    level_tensor = torch.tensor(levels, device=device, dtype=torch.float32)
    selected_step_sizes = level_tensor[level_indices]
    sections = torch.round(1.0 / selected_step_sizes).to(dtype=torch.long)

    if time_indices is None:
        random_values = torch.rand(
            (batch_size,),
            device=device,
            generator=generator,
        )
        time_indices = torch.floor(random_values * sections).to(dtype=torch.long)
    else:
        time_indices = time_indices.to(device=device, dtype=torch.long).reshape(-1)
        if time_indices.shape[0] != batch_size:
            raise ValueError("time_indices batch dimension must match x")
        if not bool(((time_indices >= 0) & (time_indices < sections)).all()):
            raise ValueError("time_indices must satisfy 0 <= index < 1 / step_size")

    timesteps = time_indices.to(dtype=torch.float32) / sections.to(dtype=torch.float32)
    broadcast_shape = (batch_size,) + (1,) * (x.ndim - 1)
    timesteps = timesteps.reshape(broadcast_shape)
    selected_step_sizes = selected_step_sizes.reshape(broadcast_shape)

    if noise is None:
        noise = torch.randn(
            x.shape,
            dtype=x.dtype,
            device=device,
            generator=generator,
        )
    elif noise.shape != x.shape:
        raise ValueError(f"noise shape must match x: {noise.shape} != {x.shape}")
    else:
        noise = noise.to(device=device, dtype=x.dtype)

    noisy_x = timesteps * x + (1.0 - timesteps) * noise
    return ShortcutTrainingData(
        x=x,
        noise=noise,
        noisy_x=noisy_x,
        timesteps=timesteps,
        step_sizes=selected_step_sizes,
    )


def shortcut_midpoint(
    noisy_x: torch.Tensor,
    first_velocity: torch.Tensor,
    half_step_sizes: torch.Tensor,
    *,
    clip_value: float = 4.0,
) -> torch.Tensor:
    """Euler-advance by half an interval and clamp like the reference code."""
    if clip_value <= 0.0:
        raise ValueError("clip_value must be positive")
    return torch.clamp(
        noisy_x + half_step_sizes * first_velocity.to(dtype=noisy_x.dtype),
        -clip_value,
        clip_value,
    )


def shortcut_velocity_target(
    first_velocity: torch.Tensor,
    second_velocity: torch.Tensor,
    *,
    clip_value: float = 4.0,
) -> torch.Tensor:
    """Average two half-step velocities into a stopped-gradient full-step target."""
    if first_velocity.shape != second_velocity.shape:
        raise ValueError("first_velocity and second_velocity shapes must match")
    if clip_value <= 0.0:
        raise ValueError("clip_value must be positive")
    target = (first_velocity + second_velocity) / 2.0
    return torch.clamp(target, -clip_value, clip_value).detach()


def fork_kv_cache(cache: Any) -> Any:
    """Fork a Transformers cache without copying its large prompt tensors.

    Cache layers are shallow-copied. Their prompt key/value tensors remain
    shared, but an expert update assigns concatenated tensors only to the
    copied layers, leaving the reusable VLM prompt cache unchanged.
    """
    if not hasattr(cache, "layers"):
        raise TypeError("expected a Transformers cache with a layers attribute")
    result = copy.copy(cache)
    result.layers = [copy.copy(layer) for layer in cache.layers]
    return result


class StepSizeConditionedActionInProjV2(PerWaypointActionInProjV2):
    """Add an explicit solver-step embedding to Alpamayo's action-token input.

    The inherited action/timestep projection retains exactly the same parameter
    names as the released checkpoint. A small residual adapter embeds the
    requested integration interval step_size (d in the shortcut paper) and
    adds it to every future waypoint token.

    The adapter's final layer is initialized to zero. Consequently, loading a
    released Alpamayo checkpoint produces exactly the released model function
    until shortcut training updates the adapter.
    """

    def __init__(
        self,
        *args,
        step_size_fourier_feats: int = 20,
        step_size_hidden_size: int = 256,
        default_step_size: float = 0.125,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if step_size_fourier_feats <= 0 or step_size_fourier_feats % 2:
            raise ValueError("step_size_fourier_feats must be a positive even integer")
        if step_size_hidden_size <= 0:
            raise ValueError("step_size_hidden_size must be positive")
        self._validate_scalar_step_size(default_step_size)

        self.default_step_size = float(default_step_size)
        self.step_size_fourier_encoder = FourierEncoderV2(
            dim=step_size_fourier_feats,
            max_freq=100.0,
        )
        self.step_size_adapter = nn.Sequential(
            nn.Linear(step_size_fourier_feats, step_size_hidden_size),
            nn.SiLU(),
            nn.Linear(step_size_hidden_size, self.out_dim),
        )
        self.reset_step_size_adapter()

    @staticmethod
    def _validate_scalar_step_size(step_size: float) -> None:
        if not 0.0 < step_size <= 1.0:
            raise ValueError(f"step_size must be in (0, 1], got {step_size}")

    def reset_step_size_adapter(self) -> None:
        """Restore the residual branch to an exact zero-output initialization."""
        output_layer = self.step_size_adapter[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("The final step-size adapter layer must be nn.Linear")
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def set_default_step_size(self, step_size: float) -> None:
        """Set the interval used when legacy callers omit step_size."""
        self._validate_scalar_step_size(step_size)
        self.default_step_size = float(step_size)

    def _batch_step_size(
        self,
        step_size: torch.Tensor | Real | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if step_size is None:
            result = torch.full(
                (batch_size,),
                self.default_step_size,
                dtype=torch.float32,
                device=device,
            )
        elif isinstance(step_size, Real):
            self._validate_scalar_step_size(float(step_size))
            result = torch.full(
                (batch_size,),
                float(step_size),
                dtype=torch.float32,
                device=device,
            )
        elif isinstance(step_size, torch.Tensor):
            result = step_size.detach().to(device=device, dtype=torch.float32)
            if result.numel() == 1:
                result = result.reshape(1).expand(batch_size)
            else:
                if result.shape[0] != batch_size:
                    raise ValueError(
                        "step_size batch dimension must match actions: "
                        f"{result.shape[0]} != {batch_size}"
                    )
                result = result.reshape(batch_size, -1)
                first = result[:, :1]
                if not torch.equal(result, first.expand_as(result)):
                    raise ValueError("step_size must be constant across each trajectory")
                result = first[:, 0]
        else:
            raise TypeError(
                "step_size must be a real scalar, tensor, or None; "
                f"got {type(step_size).__name__}"
            )

        if not bool(torch.isfinite(result).all()):
            raise ValueError("step_size contains non-finite values")
        if not bool(((result > 0.0) & (result <= 1.0)).all()):
            raise ValueError("every step_size value must be in (0, 1]")
        return result

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        step_size: torch.Tensor | Real | None = None,
    ) -> torch.Tensor:
        """Project actions while conditioning on the requested solver interval."""
        base_embedding = super().forward(x, timesteps)
        batch_step_size = self._batch_step_size(
            step_size,
            batch_size=x.shape[0],
            device=x.device,
        )

        step_features = self.step_size_fourier_encoder(batch_step_size)
        adapter_dtype = self.step_size_adapter[0].weight.dtype
        step_embedding = self.step_size_adapter(step_features.to(dtype=adapter_dtype))
        step_embedding = step_embedding[:, None, :].to(dtype=base_embedding.dtype)
        return base_embedding + step_embedding
