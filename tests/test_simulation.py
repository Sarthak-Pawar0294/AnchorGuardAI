"""
Unit tests for simulation.py - Synthetic Indoor Positioning Environment.
"""

import pytest
import numpy as np
from simulation import (
    Room,
    Anchor,
    generate_trajectory,
    simulate_measurements,
    SimulationConfig,
)


def test_room_creation_and_defaults():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    assert room.width == 20.0
    assert room.height == 14.0
    assert len(room.anchors) == 6
    assert len(room.get_active_anchors()) == 6

    coords = room.get_anchor_coordinates(active_only=True)
    assert coords.shape == (6, 2)
    # Check bounds
    assert np.all(coords[:, 0] >= 0.0)
    assert np.all(coords[:, 0] <= 20.0)
    assert np.all(coords[:, 1] >= 0.0)
    assert np.all(coords[:, 1] <= 14.0)


def test_trajectory_generation_bounds():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=4)
    patterns = ["smooth_loop", "patrol_perimeter", "random_walk", "linear_crossing"]

    for pat in patterns:
        traj = generate_trajectory(room, pattern=pat, num_steps=50, seed=123)
        assert traj.shape == (50, 2)
        # Verify inside room with padding
        assert np.all(traj[:, 0] > 0.5)
        assert np.all(traj[:, 0] < 19.5)
        assert np.all(traj[:, 1] > 0.5)
        assert np.all(traj[:, 1] < 13.5)


def test_noise_free_ideal_simulation():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=4)
    traj = generate_trajectory(room, pattern="smooth_loop", num_steps=20, seed=42)

    cfg = SimulationConfig(
        noise_std=0.0,
        drop_rate=0.0,
        nlos_probability=0.0,
        nlos_bias_mean=0.0,
        outage_anchor_ids=[],
        seed=42,
    )

    sim_res = simulate_measurements(room, traj, cfg)
    # Measurements should be identical to true Euclidean ranges
    np.testing.assert_allclose(sim_res.measurements, sim_res.true_ranges, atol=1e-12)
    assert np.sum(np.isnan(sim_res.measurements)) == 0
    assert np.sum(sim_res.nlos_flags) == 0


def test_measurement_dropout_and_nlos_polarity():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    traj = generate_trajectory(room, pattern="smooth_loop", num_steps=100, seed=42)

    cfg = SimulationConfig(
        noise_std=0.0,  # Zero noise to isolate NLOS bias polarity
        drop_rate=0.30,
        nlos_probability=0.40,
        nlos_bias_mean=3.0,
        outage_anchor_ids=[],
        seed=99,
    )

    sim_res = simulate_measurements(room, traj, cfg)

    # Missing measurements must be present
    nan_count = np.sum(np.isnan(sim_res.measurements))
    assert nan_count > 0, "Expected missing measurements under 30% drop rate"

    # NLOS flags must produce positive distance differences
    for t in range(len(traj)):
        for i in range(6):
            if not np.isnan(sim_res.measurements[t, i]):
                if sim_res.nlos_flags[t, i]:
                    # Measured range must be strictly greater than true range
                    diff = sim_res.measurements[t, i] - sim_res.true_ranges[t, i]
                    assert diff > 0.0, "NLOS bias must be strictly positive"


def test_anchor_outage_simulation():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    traj = generate_trajectory(room, num_steps=10, seed=42)

    # Disable Anchor 0 and Anchor 3
    cfg = SimulationConfig(outage_anchor_ids=[0, 3], seed=42)
    sim_res = simulate_measurements(room, traj, cfg)

    # Only 4 anchors should remain in the result
    assert len(sim_res.anchor_ids) == 4
    assert 0 not in sim_res.anchor_ids
    assert 3 not in sim_res.anchor_ids
    assert sim_res.measurements.shape[1] == 4
