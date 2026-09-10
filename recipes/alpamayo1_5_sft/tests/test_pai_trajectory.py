# SPDX-License-Identifier: Apache-2.0

"""Tests for the explicit route-less PhysicalAI trajectory path."""

from __future__ import annotations

import json

import pytest
import torch
from hydra import compose, initialize_config_module

from alpamayo.data import pai_trajectory
from alpamayo.data.pai import PAIDataset
from alpamayo.data.pai_trajectory import PAITrajectoryDataset


class FakeAvdi:
    chunk_ids = [7]

    def get_all_clip_ids(self) -> list[str]:
        return ["clip-a"]


def fake_base_init(self, **kwargs) -> None:
    del kwargs
    self.avdi = FakeAvdi()
    self.num_history_steps = 16
    self.num_future_steps = 64
    self.time_step = 0.1
    self.vla_preprocess_func = lambda data: {"saw_nav_text": "nav_text" in data}


def test_route_less_dataset_loads_manifest_timestamp(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps([{"clip_id": "clip-a", "t0_relative": 7_000_000, "chunk": 7}])
    )
    monkeypatch.setattr(PAIDataset, "__init__", fake_base_init)

    seen: dict[str, object] = {}

    def fake_load(clip_id, *, t0_us, **kwargs):
        seen.update(clip_id=clip_id, t0_us=t0_us, kwargs=kwargs)
        return {"ego_future_xyz": torch.zeros(1, 1, 64, 3)}

    monkeypatch.setattr(pai_trajectory, "load_physical_aiavdataset", fake_load)
    dataset = PAITrajectoryDataset(str(manifest), local_dir="unused")
    sample = dataset[0]

    assert len(dataset) == 1
    assert seen["clip_id"] == "clip-a"
    assert seen["t0_us"] == 7_000_000
    assert sample["ego_future_xyz"].shape == (1, 64, 3)
    assert sample["tokenized_data"] == {"saw_nav_text": False}
    assert "nav_text" not in sample


@pytest.mark.parametrize(
    "rows,match",
    [
        (
            [
                {"clip_id": "clip-a", "t0_relative": 7_000_000},
                {"clip_id": "clip-a", "t0_relative": 7_000_000},
            ],
            "Duplicate",
        ),
        (
            [
                {
                    "clip_id": "clip-a",
                    "t0_relative": 7_000_000,
                    "nav_text": "Turn left in 5m",
                }
            ],
            "contains nav_text",
        ),
    ],
)
def test_route_less_dataset_rejects_ambiguous_rows(
    monkeypatch, tmp_path, rows, match
) -> None:
    manifest = tmp_path / "rows.json"
    manifest.write_text(json.dumps(rows))
    monkeypatch.setattr(PAIDataset, "__init__", fake_base_init)
    with pytest.raises(ValueError, match=match):
        PAITrajectoryDataset(str(manifest), local_dir="unused")


def test_route_less_shortcut_config_is_separate_and_frozen() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        cfg = compose(config_name="sft_stage2_trajectory_shortcut")

    assert cfg.model.cotrain_vlm is False
    assert cfg.model.shortcut_level_sampling == "balanced_cycle"
    assert cfg.data.train_dataset._target_.endswith("PAITrajectoryDataset")
    assert cfg.data.val_dataset._target_.endswith("PAITrajectoryDataset")
    assert "route" not in cfg.data.train_dataset.vla_preprocess_args.components_order
    assert cfg.task_name.endswith("trajectory_route_less")
