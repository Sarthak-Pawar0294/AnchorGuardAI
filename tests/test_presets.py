"""
Unit tests for presets.py and UI result assessment logic.
"""

import pytest
import numpy as np
from presets import EXPERIMENT_PRESETS, ExperimentPreset
from metrics import PerformanceMetrics
from app import format_safe_pct, generate_plain_english_verdict


def test_presets_structure_and_counts():
    assert len(EXPERIMENT_PRESETS) == 4
    expected_keys = {"clean_room", "noisy_signals", "missing_readings", "blocked_signals"}
    assert set(EXPERIMENT_PRESETS.keys()) == expected_keys

    for k, p in EXPERIMENT_PRESETS.items():
        assert isinstance(p, ExperimentPreset)
        assert p.noise_std >= 0.0
        assert 0.0 <= p.drop_rate <= 1.0
        assert 0.0 <= p.nlos_probability <= 1.0
        assert p.nlos_bias_mean >= 0.0
        assert p.num_anchors >= 3

        # Verify simulation config creation
        cfg = p.to_simulation_config(seed=42)
        assert cfg.noise_std == p.noise_std
        assert cfg.drop_rate == p.drop_rate
        assert cfg.seed == 42


def test_format_safe_pct():
    # Regular improvement: from 2.0 to 1.0 -> +50.0%
    assert format_safe_pct(2.0, 1.0) == "+50.0%"
    # Regular degradation: from 1.0 to 2.0 -> -100.0%
    assert format_safe_pct(1.0, 2.0) == "-100.0%"
    # Edge cases: zero or NaN
    assert format_safe_pct(0.0, 1.0) == "N/A"
    assert format_safe_pct(np.nan, 1.0) == "N/A"
    assert format_safe_pct(1.0, np.nan) == "N/A"


def test_generate_plain_english_verdict():
    # Case 1: AnchorGuard wins
    m_base = PerformanceMetrics(
        rmse=2.50, mae=2.0, median_error=1.8, p95_error=4.0, max_error=5.0,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=1.2
    )
    m_ag = PerformanceMetrics(
        rmse=1.50, mae=1.2, median_error=1.0, p95_error=2.5, max_error=3.0,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=1.5
    )
    v1 = generate_plain_english_verdict(m_base, m_ag, "Blocked Signals")
    assert "AnchorGuard performed better" in v1
    assert "1.00 metres" in v1

    # Case 2: Baseline wins (e.g. pure Gaussian noise)
    m_base_win = PerformanceMetrics(
        rmse=0.40, mae=0.3, median_error=0.25, p95_error=0.8, max_error=1.0,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=1.0
    )
    m_ag_lose = PerformanceMetrics(
        rmse=0.60, mae=0.45, median_error=0.4, p95_error=1.1, max_error=1.3,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=1.4
    )
    v2 = generate_plain_english_verdict(m_base_win, m_ag_lose, "Noisy Signals")
    assert "Baseline performed better" in v2
    assert "0.40 m" in v2

    # Case 3: Screenshot case (Baseline 2.79m vs AG 2.90m - Baseline is strictly better)
    m_base_screen = PerformanceMetrics(
        rmse=2.788, mae=2.2, median_error=2.0, p95_error=4.5, max_error=6.0,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=0.5
    )
    m_ag_screen = PerformanceMetrics(
        rmse=2.913, mae=2.3, median_error=2.1, p95_error=4.8, max_error=6.2,
        success_rate_pct=100.0, total_steps=100, successful_steps=100, avg_runtime_ms=8.5
    )
    v_screen = generate_plain_english_verdict(m_base_screen, m_ag_screen, "Blocked Signals (NLOS)")
    assert "Baseline performed better" in v_screen
    assert "2.79 m" in v_screen
    assert "2.91 m" in v_screen

    # Case 4: Both methods fail
    m_fail = PerformanceMetrics(
        rmse=float("nan"), mae=float("nan"), median_error=float("nan"), p95_error=float("nan"),
        max_error=float("nan"), success_rate_pct=0.0, total_steps=100, successful_steps=0, avg_runtime_ms=0.5
    )
    v4 = generate_plain_english_verdict(m_fail, m_fail, "Hostile Test")
    assert "Neither estimator could calculate a valid position" in v4
