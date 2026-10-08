# SPDX-License-Identifier: Apache-2.0
"""Fresh-data selection, leakage rejection, DDP mapping and A6 compatibility."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from alpamayo1_5_sft.models.paper_data_plan import (
    HF_REVISION, audit_phases, file_sha256, fresh_batch_plan, heldout_clip_ids,
    heldout_manifest_paths, load_fresh_schedule, validate_training_rows,
)
from alpamayo1_5_sft.models.paper_shortcut_targets import paper_target_layout, raw_batch_plan
from alpamayo1_5_sft.train_paper_ema import PlannedTargets

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("fresh_selector", ROOT / "research/alpamayo1_5_shortcut/scripts/make_fresh_epoch_manifests.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
runner_spec = importlib.util.spec_from_file_location("fresh_runner", ROOT / "research/alpamayo1_5_shortcut/scripts/run_fresh_data_a6.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def rows(prefix, n, chunk=1):
    return [dict(clip_id=f"{prefix}-{i}", chunk=chunk, split="train", t0_relative=7_000_000) for i in range(n)]


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def test_clip_overlap_rejected_even_at_different_time():
    first, second = rows("a", 4), rows("b", 4)
    second[0].update(clip_id=first[0]["clip_id"], t0_relative=8_000_000)
    with pytest.raises(ValueError, match="earlier training"):
        audit_phases([first, second], {"heldout"}, 4)
    with pytest.raises(ValueError, match="held-out"):
        audit_phases([first], {first[0]["clip_id"]}, 4)


@pytest.mark.parametrize("change", [{"split": "val"}, {"nav_text": "turn"},
                                   {"t0_relative": 14_000_000}, {"t0_relative": True},
                                   {"chunk": -1}, {"clip_id": ""}])
def test_bad_training_rows_rejected(change):
    data = rows("a", 4)
    data[0].update(change)
    with pytest.raises(ValueError):
        validate_training_rows(data, 4)


def test_discover_all_historical_eval_sets(tmp_path):
    for name in ("val", "test", "nav_val", "nav_test", "val_all_timestamps", "test_one_per_clip"):
        save(tmp_path / f"{name}.json", rows(name, 1))
    save(tmp_path / "train.json", rows("train", 1))
    paths = heldout_manifest_paths(tmp_path)
    ids, evidence = heldout_clip_ids(paths)
    assert len(paths) == len(ids) == len(evidence) == 6
    with pytest.raises(ValueError, match="No held-out"):
        heldout_clip_ids([])


def test_249_updates_no_cross_phase_batches_and_first_phase_matches_a6():
    plan, phases = fresh_batch_plan([5295] * 3)
    assert len(plan) == 249 and phases == [1] * 83 + [2] * 83 + [3] * 83
    assert plan[:83] == raw_batch_plan(5295, 64, 83, 10)
    assert (plan, phases) == fresh_batch_plan([5295] * 3)
    assert (plan, phases) != fresh_batch_plan([5295] * 3, seed=11)
    for phase in (1, 2, 3):
        selected = [i for batch, p in zip(plan, phases) if p == phase for i in batch]
        assert len(selected) == 5312
        assert set(selected) == set(range((phase - 1) * 5295, phase * 5295))


def test_prefetched_ddp_slots_use_the_planned_phase_without_mutating_source():
    class Source:
        def __getitem__(self, index):
            return dict(index=index)
    plan, phases = fresh_batch_plan([65] * 3)
    layout = paper_target_layout()
    for rank in range(8):
        data = PlannedTargets(Source(), plan, layout, rank, 8, 10)
        for update in (0, 1, 2, 3, 4, 5):
            for micro in range(8):
                sample = data[update * 8 + micro]
                slot = micro * 8 + rank
                assert sample["index"] == plan[update][int(layout.source_indices[slot])]
                assert (phases[update] - 1) * 65 <= sample["index"] < phases[update] * 65
                assert sample["paper_slot"] == slot
                assert sample["paper_seed"] == 10 + 100003 * update


def test_selector_uses_only_eligible_disjoint_train_chunks():
    first = rows("original", 2, 1)
    catalog = first + [dict(clip_id=f"c-{chunk}-{i}", chunk=chunk, split="train", clip_is_valid=True)
                       for chunk in range(2, 15) for i in range(2)]
    index = pd.DataFrame(catalog).set_index("clip_id")
    index.loc[[row["clip_id"] for row in first], "clip_is_valid"] = True
    index.loc["c-2-0", "split"] = "val"
    index.loc["c-3-0", "clip_is_valid"] = False
    index["clip_is_valid"] = index["clip_is_valid"].astype(bool)
    presence = pd.DataFrame(True, index=index.index, columns=["camera", "ego"])
    presence.loc["c-4-0", "camera"] = False
    heldout = {"c-5-0"}
    phases = selector.select_phases(index, presence, ["camera", "ego"], first, heldout, {6}, 2)
    assert phases == selector.select_phases(index, presence, ["camera", "ego"], first, heldout, {6}, 2)
    audit_phases(phases, heldout, 2)
    chunks = [{row["chunk"] for row in phase} for phase in phases]
    assert not (chunks[0] & chunks[1] or chunks[0] & chunks[2] or chunks[1] & chunks[2])
    new_ids = {row["clip_id"] for phase in phases[1:] for row in phase}
    assert not new_ids & {"c-2-0", "c-3-0", "c-4-0", "c-5-0", "c-6-0", "c-6-1"}
    with pytest.raises(ValueError, match="Insufficient"):
        selector.select_phases(index, presence, ["camera", "ego"], first, heldout, set(range(2, 15)), 2)


def test_bundle_hash_and_live_holdout_checks(tmp_path):
    original = ROOT / "research/alpamayo1_5_shortcut/manifests/hf_stream_300gb/train.json"
    (tmp_path / "phase_1.json").write_bytes(original.read_bytes())
    phases = [json.loads(original.read_text()), rows("new2", 5295, 9001), rows("new3", 5295, 9002)]
    for i in (2, 3):
        save(tmp_path / f"phase_{i}.json", phases[i - 1])
    flat = [row for phase in phases for row in phase]
    train = save(tmp_path / "train.json", flat)
    schedule = dict(schema_version=1, revision=HF_REVISION, clips_per_phase=5295,
                    phases=[dict(manifest=f"phase_{i}.json", sha256=file_sha256(tmp_path / f"phase_{i}.json")) for i in (1, 2, 3)],
                    heldout_clip_ids=["heldout"], train_manifest_sha256=file_sha256(train))
    path = save(tmp_path / "schedule.json", schedule)
    holdout = save(tmp_path / "val.json", rows("heldout", 1))
    _, loaded, audit, sizes = load_fresh_schedule(path, [holdout])
    assert loaded == flat and sizes == [5295] * 3
    assert audit["total_unique_clips"] == 15885
    save(holdout, [dict(clip_id="new3-0", t0_relative=9_000_000)])
    with pytest.raises(ValueError, match="held-out"):
        load_fresh_schedule(path, [holdout])
    save(holdout, rows("heldout", 1))
    save(train, list(reversed(flat)))
    with pytest.raises(ValueError, match="Combined manifest"):
        load_fresh_schedule(path, [holdout])
    save(train, flat)
    save(tmp_path / "phase_2.json", list(reversed(phases[1])))
    with pytest.raises(ValueError, match="hash changed"):
        load_fresh_schedule(path, [holdout])


@pytest.mark.parametrize("train", [False, True])
def test_runner_requires_explicit_training_and_preserves_continuous_phases(tmp_path, monkeypatch, train):
    checkpoint = tmp_path / "released"
    checkpoint.mkdir()
    save(checkpoint / "config.json", {})
    output = tmp_path / "run"
    args = ["run_fresh_data_a6.py", "--run-dir", str(output), "--checkpoint", str(checkpoint),
            "--hf-cache", str(tmp_path / "cache")]
    if train:
        args.append("--train")
    monkeypatch.setattr(runner.sys, "argv", args)
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "torch.distributed.run" in argv:
            destination = Path(argv[argv.index("--output-dir") + 1])
            destination.mkdir()
            updates = int(argv[argv.index("--updates") + 1])
            metrics = [dict(flow_pairs=48, shortcut_pairs=16, ema_updates=i + 1, data_phase=i // 83 + 1)
                       for i in range(updates)]
            (destination / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics))
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.main()
    gpu_calls = [call for call in calls if "torch.distributed.run" in call]
    assert len(gpu_calls) == (2 if train else 0)
    if train:
        assert all(call[call.index("--checkpoint") + 1] == str(checkpoint) for call in gpu_calls)
        assert "--no-save" in gpu_calls[0] and "--no-save" not in gpu_calls[1]
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == ("training_complete_evaluation_pending" if train else "ready_not_training")


def test_runner_stops_on_preflight_failure(tmp_path, monkeypatch):
    checkpoint = tmp_path / "released"
    checkpoint.mkdir()
    save(checkpoint / "config.json", {})
    output = tmp_path / "run"
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--run-dir", str(output), "--checkpoint", str(checkpoint),
                                          "--hf-cache", str(tmp_path / "cache"), "--train"])
    calls = []
    def failed_run(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError("simulated failed preflight")
    monkeypatch.setattr(runner.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="preflight"):
        runner.main()
    assert len(calls) == 1 and "torch.distributed.run" not in calls[0]
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
