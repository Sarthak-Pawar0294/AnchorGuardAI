"""
tests/test_workbench_correctness.py - Rigorous correctness and evaluation tests for Workbench 2.0.
"""

import pytest
import numpy as np
from simulation import Room, generate_trajectory, simulate_measurements, SimulationConfig
from estimators import estimate_trajectory, AnchorGuardConfig
from metrics import compute_positioning_metrics, PerformanceMetrics
from config import WorkbenchConfig
from benchmark import run_workbench_trials
from visualization import create_residual_diagnostics_figure


def test_non_destructive_anchor_outage():
    """Verifies that simulate_measurements does NOT mutate room.anchors in-place."""
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    assert all(a.is_active for a in room.anchors)

    gt = generate_trajectory(room, pattern="smooth_loop", num_steps=20, seed=42)
    cfg = SimulationConfig(outage_anchor_ids=[0, 2])

    sim_res = simulate_measurements(room, gt, cfg)

    # Simulation result should only have 4 active anchors (ids 1, 3, 4, 5)
    assert len(sim_res.anchor_ids) == 4
    assert 0 not in sim_res.anchor_ids
    assert 2 not in sim_res.anchor_ids

    # But room.anchors must NOT have been mutated!
    assert all(a.is_active for a in room.anchors), "room.anchors was mutated in-place!"


def test_fair_measurement_reuse_across_estimators():
    """Verifies that both estimators receive strictly identical measurements."""
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    gt = generate_trajectory(room, pattern="smooth_loop", num_steps=50, seed=123)
    cfg = SimulationConfig(noise_std=0.2, drop_rate=0.1, nlos_probability=0.2, seed=123)
    sim_res = simulate_measurements(room, gt, cfg)

    # Identical measurements passed to baseline and anchorguard
    base_res = estimate_trajectory(
        measurements=sim_res.measurements,
        anchor_coords=sim_res.anchor_coords,
        method="baseline",
        noise_sigma=cfg.noise_std,
    )
    ag_res = estimate_trajectory(
        measurements=sim_res.measurements,
        anchor_coords=sim_res.anchor_coords,
        method="anchorguard",
        noise_sigma=cfg.noise_std,
    )

    assert len(base_res.estimated_trajectory) == len(sim_res.measurements)
    assert len(ag_res.estimated_trajectory) == len(sim_res.measurements)


def test_reproducibility_with_fixed_seed():
    """Two identical configs with the same seed must produce identical outputs."""
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    cfg = SimulationConfig(noise_std=0.3, drop_rate=0.15, nlos_probability=0.25, seed=777)

    gt1 = generate_trajectory(room, pattern="smooth_loop", num_steps=40, seed=777)
    sim1 = simulate_measurements(room, gt1, cfg)

    gt2 = generate_trajectory(room, pattern="smooth_loop", num_steps=40, seed=777)
    sim2 = simulate_measurements(room, gt2, cfg)

    np.testing.assert_array_equal(gt1, gt2)
    np.testing.assert_array_equal(sim1.true_ranges, sim2.true_ranges)
    # Check NaN and finite values match
    np.testing.assert_array_equal(np.isnan(sim1.measurements), np.isnan(sim2.measurements))
    valid = ~np.isnan(sim1.measurements)
    np.testing.assert_allclose(sim1.measurements[valid], sim2.measurements[valid], atol=1e-12)


def test_out_of_room_metric_and_denominators():
    """Verifies that out-of-room estimates are accurately counted with correct denominators."""
    room_dims = (20.0, 14.0)
    gt = np.array([
        [10.0, 7.0],
        [10.0, 7.0],
        [10.0, 7.0],
        [10.0, 7.0],
    ])
    # Step 0: Inside room
    # Step 1: Outside room (x = -2.0)
    # Step 2: Outside room (y = 18.0)
    # Step 3: Failed estimation (NaN)
    est = np.array([
        [10.5, 7.2],
        [-2.0, 7.0],
        [10.0, 18.0],
        [np.nan, np.nan],
    ])

    m = compute_positioning_metrics(
        ground_truth=gt,
        estimated=est,
        room_dims=room_dims,
    )

    assert m.total_steps == 4
    assert m.successful_steps == 3
    # Denominator for valid rate: 3 / 4 * 100 = 75.0%
    assert m.success_rate_pct == 75.0

    # Steps 1 and 2 are out of room => 2 steps
    assert m.out_of_room_steps == 2
    # Denominator for out of room rate: 2 / 3 * 100 = 66.666...%
    assert pytest.approx(m.out_of_room_rate_pct, rel=1e-3) == (2.0 / 3.0) * 100.0


def test_residual_diagnostics_figure():
    """Verifies that create_residual_diagnostics_figure creates a valid figure."""
    meas = np.array([
        [5.0, 6.0, 7.0],
        [5.2, 6.1, 7.3],
    ])
    anchors = np.array([
        [1.0, 1.0],
        [19.0, 1.0],
        [10.0, 13.0],
    ])
    est = np.array([
        [5.0, 5.0],
        [5.1, 5.2],
    ])
    fig = create_residual_diagnostics_figure(meas, anchors, est, anchor_ids=[0, 1, 2])
    assert len(fig.data) == 3
    assert fig.data[0].name == "Anchor 0"


def test_run_workbench_trials_statistics():
    """Verifies that run_workbench_trials executes K trials and produces mean and std."""
    cfg = WorkbenchConfig(
        room_width=20.0,
        room_height=14.0,
        trajectory_steps=30,
        noise_std=0.2,
        seed=100,
    )
    res = run_workbench_trials(cfg, num_trials=3, seed_offset=0)

    assert res["num_trials_completed"] == 3
    assert len(res["seeds_used"]) == 3
    assert res["seeds_used"] == [100, 101, 102]
    assert len(res["trial_records"]) == 3

    assert "rmse_mean" in res["baseline"]
    assert "rmse_std" in res["baseline"]
    assert res["baseline"]["rmse_mean"] > 0
    assert "rmse_mean" in res["anchorguard"]
    assert "rmse_std" in res["anchorguard"]
