"""Ensure the empirical control changes targets, not the target inputs/layout."""
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from alpamayo1_5_sft.models.paper_shortcut_alpamayo import select_paper_target
from alpamayo1_5_sft.models.paper_shortcut_targets import (
    paper_target_layout, draw_slot_randomness, paper_interpolation,
    paper_bootstrap_target, paper_flow_target,
)


def test_default_bootstrap_targets_remain_identical_and_empirical_never_calls_teacher():
    layout = paper_target_layout()
    clean = torch.linspace(-2, 2, 128).reshape(1, 64, 2)
    teacher_parameter = torch.nn.Parameter(torch.tensor(0.2))
    def teacher(x, t, d):
        return x * teacher_parameter + t + d
    def forbidden(*args):
        raise AssertionError("Empirical target must not query the EMA teacher")
    for slot in range(64):
        noise, time = draw_slot_randomness(layout, slot, (64, 2), 42)
        t = torch.tensor(time).reshape(1, 1, 1)
        d = float(layout.step_sizes[slot])
        noisy = paper_interpolation(clean, noise, t)
        before = noisy.clone()
        bootstrap = bool(layout.bootstrap_mask[slot])
        old = paper_bootstrap_target(teacher, noisy, t, d) if bootstrap else paper_flow_target(clean, noise)
        selected, uses_teacher = select_paper_target("ema_bootstrap", bootstrap, teacher, clean, noise, noisy, t, d)
        torch.testing.assert_close(selected, old, rtol=0, atol=0)
        assert uses_teacher == bootstrap
        control, uses_teacher = select_paper_target("empirical_velocity", bootstrap, forbidden, clean, noise, noisy, t, d)
        torch.testing.assert_close(control, paper_flow_target(clean, noise), rtol=0, atol=0)
        assert not uses_teacher and not control.requires_grad
        assert not selected.requires_grad
        assert torch.equal(noisy, before)
    assert teacher_parameter.grad is None


def test_control_config_changes_only_supervision():
    with initialize_config_module(version_base=None, config_module="alpamayo1_5_sft.configs"):
        baseline = compose(config_name="sft_stage2_trajectory_shortcut_paper_ema")
        control = compose(config_name="sft_stage2_trajectory_paper_empirical_ema")
    a = OmegaConf.to_container(baseline.model)
    b = OmegaConf.to_container(control.model)
    assert b.pop("paper_supervision") == "empirical_velocity"
    assert a == b
    assert baseline.data == control.data


def test_reject_unknown_supervision():
    with pytest.raises(ValueError, match="Unknown paper supervision"):
        select_paper_target("bad", False, None, None, None, None, None, None)
