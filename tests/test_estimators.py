"""
Unit tests for estimators.py - Positioning algorithms and geometry reliability.
"""

import pytest
import numpy as np
from estimators import (
    baseline_estimate_step,
    anchorguard_estimate_step,
    estimate_trajectory,
    check_geometry_conditioning,
    EstimationStatus,
    AnchorGuardConfig,
    PositionKalmanFilter,
)


@pytest.fixture
def standard_square_anchors():
    # 4 anchors in corners of 20x14 room
    return np.array([
        [1.0, 1.0],
        [19.0, 1.0],
        [19.0, 13.0],
        [1.0, 13.0]
    ])


def test_geometry_conditioning_valid(standard_square_anchors):
    is_valid, cond, msg = check_geometry_conditioning(standard_square_anchors)
    assert is_valid is True
    assert cond > 0.05


def test_geometry_conditioning_collinear():
    # 4 anchors along the same straight line
    collinear_anchors = np.array([
        [1.0, 5.0],
        [5.0, 5.0],
        [10.0, 5.0],
        [15.0, 5.0],
    ])
    is_valid, cond, msg = check_geometry_conditioning(collinear_anchors)
    assert is_valid is False
    assert "Degenerate geometry" in msg or "collinear" in msg.lower()


def test_geometry_conditioning_insufficient():
    two_anchors = np.array([[1.0, 1.0], [5.0, 5.0]])
    is_valid, cond, msg = check_geometry_conditioning(two_anchors)
    assert is_valid is False
    assert "Insufficient" in msg


def test_baseline_exact_noise_free_recovery(standard_square_anchors):
    true_pos = np.array([8.5, 6.2])
    # Exact Euclidean ranges
    ranges = np.linalg.norm(standard_square_anchors - true_pos, axis=1)

    est = baseline_estimate_step(ranges, standard_square_anchors)
    assert est.status == EstimationStatus.SUCCESS
    assert est.position is not None
    np.testing.assert_allclose(est.position, true_pos, atol=1e-4)


def test_baseline_insufficient_measurements_handling(standard_square_anchors):
    # Only 2 ranges available, 2 NaN
    ranges = np.array([5.0, 8.0, np.nan, np.nan])
    est = baseline_estimate_step(ranges, standard_square_anchors)
    assert est.status == EstimationStatus.INSUFFICIENT_ANCHORS
    assert est.position is None


def test_anchorguard_exact_recovery(standard_square_anchors):
    true_pos = np.array([12.0, 7.5])
    ranges = np.linalg.norm(standard_square_anchors - true_pos, axis=1)

    est = anchorguard_estimate_step(ranges, standard_square_anchors)
    assert est.status == EstimationStatus.SUCCESS
    assert est.position is not None
    np.testing.assert_allclose(est.position, true_pos, atol=1e-4)


def test_anchorguard_outlier_rejection():
    # 5 anchors to allow surplus for outlier rejection
    anchors = np.array([
        [1.0, 1.0],
        [19.0, 1.0],
        [19.0, 13.0],
        [1.0, 13.0],
        [10.0, 7.0],
    ])
    true_pos = np.array([10.0, 5.0])
    ranges = np.linalg.norm(anchors - true_pos, axis=1)

    # Inject severe NLOS positive bias (+6.0m) on Anchor 1
    corrupted_ranges = ranges.copy()
    corrupted_ranges[1] += 6.0

    # 1. Baseline NLS will be heavily pulled by the +6m outlier
    baseline_est = baseline_estimate_step(corrupted_ranges, anchors)
    assert baseline_est.status == EstimationStatus.SUCCESS
    baseline_err = np.linalg.norm(baseline_est.position - true_pos)

    # 2. AnchorGuard Robust will reject or down-weight the outlier
    ag_cfg = AnchorGuardConfig(enable_residual_pruning=True, nlos_residual_threshold=2.0)
    ag_est = anchorguard_estimate_step(corrupted_ranges, anchors, config=ag_cfg)
    assert ag_est.status == EstimationStatus.SUCCESS
    ag_err = np.linalg.norm(ag_est.position - true_pos)

    # AnchorGuard should be significantly more accurate than baseline
    assert ag_err < baseline_err
    assert ag_err < 0.5  # Sub-meter accuracy preserved
    assert 1 in ag_est.rejected_anchor_indices, "Anchor 1 should be isolated and pruned"


def test_kalman_filter_smoothing():
    kf = PositionKalmanFilter(dt=1.0, process_noise_std=0.2, default_meas_std=0.5)
    p1 = np.array([2.0, 3.0])
    out1 = kf.update(p1)
    np.testing.assert_allclose(out1, p1, atol=1e-5)

    # Prediction step
    pred = kf.predict()
    assert len(pred) == 2

    # Sequential updates
    p2 = np.array([3.0, 4.0])
    out2 = kf.update(p2)
    assert 2.0 <= out2[0] <= 3.5
    assert 3.0 <= out2[1] <= 4.5
