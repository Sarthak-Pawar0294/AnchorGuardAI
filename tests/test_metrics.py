"""
Unit tests for metrics.py - Positioning error calculations.
"""

import pytest
import numpy as np
from metrics import compute_positioning_metrics, metrics_to_dict


def test_metrics_perfect_accuracy():
    gt = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    est = gt.copy()

    m = compute_positioning_metrics(gt, est, avg_runtime_ms=1.5)
    assert m.rmse == 0.0
    assert m.mae == 0.0
    assert m.median_error == 0.0
    assert m.p95_error == 0.0
    assert m.success_rate_pct == 100.0
    assert m.successful_steps == 3
    assert m.total_steps == 3
    assert m.avg_runtime_ms == 1.5


def test_metrics_known_discrepancy():
    gt = np.array([[0.0, 0.0], [0.0, 0.0]])
    # Errors: point 1 error = 3m, point 2 error = 4m
    est = np.array([[3.0, 0.0], [0.0, 4.0]])

    m = compute_positioning_metrics(gt, est)
    # RMSE = sqrt((3^2 + 4^2)/2) = sqrt(25/2) = sqrt(12.5) ≈ 3.5355
    assert pytest.approx(m.rmse, rel=1e-3) == np.sqrt(12.5)
    assert pytest.approx(m.mae, rel=1e-3) == 3.5
    assert m.success_rate_pct == 100.0


def test_metrics_with_nan_failures():
    gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    est = np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 2.0], [np.nan, 3.0]])

    m = compute_positioning_metrics(gt, est)
    # Only steps 0 and 2 are fully valid
    assert m.successful_steps == 2
    assert m.total_steps == 4
    assert m.success_rate_pct == 50.0
    assert m.rmse == 0.0


def test_metrics_to_dict_formatting():
    gt = np.array([[0.0, 0.0], [1.0, 1.0]])
    est = np.array([[1.0, 0.0], [1.0, 2.0]])
    m = compute_positioning_metrics(gt, est, avg_runtime_ms=2.345)
    d = metrics_to_dict(m, prefix="Test")

    assert "Test_RMSE (m)" in d
    assert "Test_MAE (m)" in d
    assert "Test_Success Rate (%)" in d
    assert d["Test_Avg Runtime (ms)"] == 2.35
