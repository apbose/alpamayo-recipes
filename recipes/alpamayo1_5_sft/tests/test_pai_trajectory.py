# SPDX-License-Identifier: Apache-2.0

"""Tests for the explicit route-less PhysicalAI trajectory path."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
import torch
from alpamayo.data import pai as pai_module
from alpamayo.data import pai_trajectory
from alpamayo.data.pai import PAIDataset
from alpamayo.data.pai_trajectory import PAITrajectoryDataset
from hydra import compose, initialize_config_module


class FakeAvdi:
    chunk_ids: ClassVar[list[int]] = [7]

    def get_all_clip_ids(self) -> list[str]:
        return ["clip-a"]


def fake_base_init(self, **kwargs) -> None:
    del kwargs
    self.avdi = FakeAvdi()
    self.maybe_stream = False
    self.num_history_steps = 16
    self.num_future_steps = 64
    self.time_step = 0.1
    self.vla_preprocess_func = lambda data: {"saw_nav_text": "nav_text" in data}


def test_route_less_dataset_loads_manifest_timestamp(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([{"clip_id": "clip-a", "t0_relative": 7_000_000, "chunk": 7}]))
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
    assert seen["kwargs"]["maybe_stream"] is False
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
def test_route_less_dataset_rejects_ambiguous_rows(monkeypatch, tmp_path, rows, match) -> None:
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


def test_hf_stream_access_uses_official_interface_shim(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeHFInterface:
        reasoning_db = None

        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def get_all_clip_ids(self) -> list[str]:
            return ["remote-clip"]

    monkeypatch.setattr(pai_module, "PhysicalAIAVDatasetHFInterface", FakeHFInterface)
    dataset = PAIDataset(
        access_mode="hf_stream",
        chunk_ids=[1, 2],
        hf_revision="immutable-revision",
        hf_cache_dir="/tmp/test-pai-cache",
    )

    assert dataset.access_mode == "hf_stream"
    assert dataset.maybe_stream is True
    assert dataset.clip_ids == ["remote-clip"]
    assert seen == {
        "chunk_ids": [1, 2],
        "revision": "immutable-revision",
        "token": None,
        "cache_dir": "/tmp/test-pai-cache",
    }


def test_local_access_still_requires_local_dir() -> None:
    with pytest.raises(ValueError, match="local_dir is required"):
        PAIDataset(access_mode="local")


def test_hf_stream_rejects_reasoning_metadata() -> None:
    with pytest.raises(ValueError, match="does not load reasoning annotations"):
        PAIDataset(access_mode="hf_stream", reasoning_metadata="reasoning.parquet")


def test_hf_reference_config_is_explicit_and_route_less() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        cfg = compose(config_name="sft_stage2_trajectory_shortcut_hf_reference")

    assert cfg.model.shortcut_loss_estimator == "reference_partition"
    assert cfg.model.shortcut_bootstrap_every == 8
    assert cfg.trainer.per_device_train_batch_size == 1
    assert cfg.data.train_dataset.access_mode == "hf_stream"
    assert cfg.data.train_dataset.hf_revision == ("33f9bf447ed3bcb7d545ce13f4226f824214fafb")
    assert "route" not in cfg.data.train_dataset.vla_preprocess_args.components_order
    assert cfg.performance.zip_cache is False


def test_hf_flow_control_matches_data_but_uses_base_expert() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="alpamayo1_5_sft.configs",
    ):
        flow = compose(config_name="sft_stage2_trajectory_hf_flow_control")
        shortcut = compose(config_name="sft_stage2_trajectory_shortcut_hf_reference")

    assert flow.model._target_.endswith("TrainableAlpamayoR1.from_pretrained")
    assert "Shortcut" not in flow.model._target_
    assert flow.model.cotrain_vlm is False
    assert flow.data.train_dataset.access_mode == "hf_stream"
    assert flow.data.train_dataset.hf_revision == shortcut.data.train_dataset.hf_revision
    assert flow.trainer.per_device_train_batch_size == 1
    assert flow.trainer.gradient_accumulation_steps == 1
    assert "route" not in flow.data.train_dataset.vla_preprocess_args.components_order
    assert flow.performance.zip_cache is False
