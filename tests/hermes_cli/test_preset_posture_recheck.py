"""Tests for preset generation — posture re-check after grown window and spill warning surfacing.

Covers:
- _recheck_posture_after_growth: stacked -> lean degradation when grown window exceeds VRAM
- _spill_warning_text: human-readable warning generation for CPU offload
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.local_runtime.context_policy import WindowDecision, ub_logits_bytes
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile, LayerKind, ctx_bytes
from hermes_cli.local_runtime.presets import (
    _recheck_posture_after_growth,
    PresetEntry,
)
from hermes_cli.web_routers.local_models import _spill_warning_text


# -- fixtures ---------------------------------------------------------------

def _make_profile(**overrides) -> ModelProfile:
    """A minimal Qwen-like MTP profile for testing."""
    defaults = dict(
        name="test-model",
        weights_bytes=16 * (1 << 30),           # 16 GiB
        embd_table_bytes=1 * (1 << 30),          # 1 GiB
        n_ctx_train=256 * 1024,
        layers=[(LayerKind.FULL, 256)] * 40,     # 40 attention layers
        swa_window=0,
        moe=False,
        architecture="qwen3",
        n_vocab=151_936,
        kv_scale=1.0,
    )
    defaults.update(overrides)
    return ModelProfile(**defaults)


def _make_budget(vram: int, ram: int = 64 * (1 << 30)) -> HardwareBudget:
    return HardwareBudget(
        usable_vram_bytes=vram,
        total_device_bytes=vram + (4 << 30),
        ram_available_bytes=ram,
        uma=False,
    )


# -- _recheck_posture_after_growth tests ------------------------------------

class TestRecheckPostureAfterGrowth:
    """When the initial MTP posture was stacked (big-ubatch) but the grown window pushes it past VRAM,
    the re-check should degrade to lean (smaller ubatch) to avoid CPU offload."""

    def test_stacked_degrades_to_lean_when_grown_spills(self):
        """Stacked posture at grown window exceeds VRAM -> must degrade to lean.

        Profile: 16 GiB weights, 152K vocab.  Budget: 18 GiB VRAM.
        At 96K grown window: stacked_need ~19.7 GiB > 18 GiB (spills).
        At 96K grown window: plain_need   ~18.6 GiB > 18 GiB (also spills, but lean minimizes).
        """
        profile = _make_profile()
        budget = _make_budget(18 * (1 << 30))
        fixed_overhead = int(1.5 * (1 << 30))

        stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)

        # Verify our assumption: stacked actually spills at 96K with this budget
        kv_96k = ctx_bytes(profile, 96 * 1024)
        stacked_need = profile.weights_bytes + kv_96k + fixed_overhead + stacked_logits
        assert stacked_need > budget.usable_vram_bytes, (
            f"test setup wrong: stacked_need {stacked_need / (1<<30):.2f} GiB "
            f"should exceed vram {budget.usable_vram_bytes / (1<<30):.2f} GiB")

        grown = WindowDecision(
            window=96 * 1024,
            spill_bytes=max(0, stacked_need - budget.usable_vram_bytes),
            kv_on_gpu=kv_96k <= budget.usable_vram_bytes,
            reasons=["grown window restored (96K)"],
        )

        result_prefill, result_logits = _recheck_posture_after_growth(
            profile, budget, fixed_overhead, True, stacked_logits, grown)

        # Should have degraded to lean (not stacked)
        assert result_prefill is False
        # Lean logits should be smaller than stacked logits
        assert result_logits < stacked_logits

    def test_stacked_kept_when_still_fits(self):
        """Stacked posture still fits VRAM at grown window -> keep stacked."""
        profile = _make_profile(weights_bytes=8 * (1 << 30))  # 8 GiB -- plenty of headroom
        budget = _make_budget(24 * (1 << 30))
        fixed_overhead = int(1.5 * (1 << 30))

        stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)

        # Verify: stacked fits at 96K
        kv_96k = ctx_bytes(profile, 96 * 1024)
        stacked_need = profile.weights_bytes + kv_96k + fixed_overhead + stacked_logits
        assert stacked_need <= budget.usable_vram_bytes

        grown = WindowDecision(window=96 * 1024, spill_bytes=0, kv_on_gpu=True)

        result_prefill, result_logits = _recheck_posture_after_growth(
            profile, budget, fixed_overhead, True, stacked_logits, grown)

        assert result_prefill is True
        assert result_logits == stacked_logits

    def test_already_lean_no_change(self):
        """If the initial posture is already lean, no re-check needed."""
        profile = _make_profile()
        budget = _make_budget(24 * (1 << 30))
        lean_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=False)

        grown = WindowDecision(window=96 * 1024, spill_bytes=2 * (1 << 30), kv_on_gpu=True)

        result_prefill, result_logits = _recheck_posture_after_growth(
            profile, budget, int(1.5 * (1 << 30)), False, lean_logits, grown)

        assert result_prefill is False
        assert result_logits == lean_logits

    def test_spilled_anyway_no_recheck(self):
        """If spill already happened regardless, no re-check -- already as bad as it gets."""
        profile = _make_profile()
        budget = _make_budget(24 * (1 << 30))
        stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)

        grown = WindowDecision(window=128 * 1024, spill_bytes=4 * (1 << 30), kv_on_gpu=False)

        result_prefill, result_logits = _recheck_posture_after_growth(
            profile, budget, int(1.5 * (1 << 30)), True, stacked_logits, grown)

        # Stays stacked -- spill already happened
        assert result_prefill is True
        assert result_logits == stacked_logits

    def test_both_postures_spill_uses_lean(self):
        """When both stacked and lean would spill at the grown window, use lean (minimizes spill).

        Profile: 16 GiB weights.  Budget: 10 GiB VRAM (very tight).
        At 96K: stacked ~19.7 GiB, plain ~18.6 GiB, both exceed 10 GiB.
        """
        profile = _make_profile()
        budget = _make_budget(10 * (1 << 30))
        fixed_overhead = int(1.5 * (1 << 30))

        stacked_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=True)
        plain_logits = ub_logits_bytes(profile.n_vocab, mtp_capable=True, mtp_prefill=False)

        # Verify: both postures spill at 96K
        kv_96k = ctx_bytes(profile, 96 * 1024)
        stacked_need = profile.weights_bytes + kv_96k + fixed_overhead + stacked_logits
        plain_need = profile.weights_bytes + kv_96k + fixed_overhead + plain_logits
        assert stacked_need > budget.usable_vram_bytes
        assert plain_need > budget.usable_vram_bytes

        grown = WindowDecision(window=96 * 1024, spill_bytes=stacked_need - budget.usable_vram_bytes, kv_on_gpu=False)

        result_prefill, result_logits = _recheck_posture_after_growth(
            profile, budget, fixed_overhead, True, stacked_logits, grown)

        # Falls back to lean (minimizes spill)
        assert result_prefill is False
        assert result_logits < stacked_logits


# -- _spill_warning_text tests -----------------------------------------------

class TestSpillWarningText:
    """Verify spill warnings are generated correctly for CPU offload."""

    def test_no_spill_returns_none(self):
        plan = PresetEntry(model_id="test", window=64 * 1024, spilled=False)
        assert _spill_warning_text(plan, 64 * 1024) is None

    def test_none_plan_returns_none(self):
        assert _spill_warning_text(None, 64 * 1024) is None

    def test_spill_with_granted_window(self):
        plan = PresetEntry(model_id="test", window=96 * 1024, spilled=True)
        warning = _spill_warning_text(plan, 96 * 1024)
        assert warning is not None
        assert "96K" in warning
        assert "CPU" in warning
        assert "reduces speed" in warning.lower()

    def test_spill_without_granted_window_uses_plan_window(self):
        plan = PresetEntry(model_id="test", window=128 * 1024, spilled=True)
        warning = _spill_warning_text(plan, None)
        assert warning is not None
        assert "128K" in warning

    def test_spill_warning_mentions_restart(self):
        plan = PresetEntry(model_id="test", window=96 * 1024, spilled=True)
        warning = _spill_warning_text(plan, 96 * 1024)
        assert "restart" in warning.lower() or "smaller" in warning.lower()
