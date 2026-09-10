# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven PhysicalAI-AV trajectory dataset without route text."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alpamayo.data.pai import PAIDataset
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


logger = logging.getLogger(__name__)
REQUIRED_FIELDS = frozenset({"clip_id", "t0_relative"})


class PAITrajectoryDataset(PAIDataset):
    """Load explicit ``(clip_id, t0_relative)`` rows from a JSON manifest.

    Unlike :class:`alpamayo.data.pai_nav.PAIDatasetWithNav`, this dataset does
    not require or inject ``nav_text``.  It is intended for an explicitly
    route-less trajectory-prediction experiment using the default A1.5 chat
    template, which omits the route component.
    """

    def __init__(self, annotations_path: str, **kwargs: Any):
        super().__init__(**kwargs)

        manifest_path = Path(annotations_path)
        with manifest_path.open(encoding="utf-8") as handle:
            samples = json.load(handle)
        if not isinstance(samples, list) or not samples:
            raise ValueError(
                f"[PAITrajectoryDataset] {manifest_path} must contain a non-empty JSON list"
            )

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"[PAITrajectoryDataset] Row {index} in {manifest_path} is not an object"
                )
            missing = REQUIRED_FIELDS.difference(sample)
            if missing:
                raise ValueError(
                    f"[PAITrajectoryDataset] Row {index} in {manifest_path} is missing "
                    f"{sorted(missing)}"
                )
            if "nav_text" in sample:
                raise ValueError(
                    f"[PAITrajectoryDataset] Row {index} contains nav_text; use "
                    "PAIDatasetWithNav for a navigation-conditioned experiment"
                )
            clip_id = str(sample["clip_id"])
            t0_us = int(sample["t0_relative"])
            if not clip_id:
                raise ValueError(
                    f"[PAITrajectoryDataset] Row {index} has an empty clip_id"
                )
            if t0_us <= 0:
                raise ValueError(
                    f"[PAITrajectoryDataset] Row {index} has invalid t0_relative={t0_us}"
                )
            key = (clip_id, t0_us)
            if key in seen:
                raise ValueError(
                    f"[PAITrajectoryDataset] Duplicate (clip_id, t0_relative) row: {key}"
                )
            seen.add(key)
            normalized.append({"clip_id": clip_id, "t0_relative": t0_us})

        if self.avdi.chunk_ids is not None:
            allowed_clip_ids = set(self.avdi.get_all_clip_ids())
            kept = [row for row in normalized if row["clip_id"] in allowed_clip_ids]
            dropped = len(normalized) - len(kept)
            if not kept:
                raise ValueError(
                    f"[PAITrajectoryDataset] All {len(normalized)} manifest rows in "
                    f"{manifest_path} were filtered out by chunk_ids={self.avdi.chunk_ids}"
                )
            if dropped:
                logger.warning(
                    "[PAITrajectoryDataset] Filtered out %d/%d rows whose clips are "
                    "outside chunk_ids=%s; keeping %d.",
                    dropped,
                    len(normalized),
                    self.avdi.chunk_ids,
                    len(kept),
                )
            normalized = kept

        self._samples = normalized
        self.clip_ids = [row["clip_id"] for row in normalized]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._samples[idx]
        sample_data = load_physical_aiavdataset(
            entry["clip_id"],
            t0_us=entry["t0_relative"],
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        for key in list(sample_data):
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
