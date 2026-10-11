"""
Unit tests for visualization.py - Plotly charts creation and schema verification.
"""

import pytest
import numpy as np
import plotly.graph_objects as go

from simulation import Room
from visualization import (
    create_floorplan_figure,
    create_error_comparison_figure,
    create_cdf_figure,
)


@pytest.fixture
def sample_data():
    room = Room.create_default(width=20.0, height=14.0, num_anchors=6)
    # 10 steps of mock positions
    gt = np.linspace([2.0, 2.0], [18.0, 12.0], 10)
    baseline = gt + np.random.normal(0, 0.2, gt.shape)
    anchorguard = gt + np.random.normal(0, 0.1, gt.shape)
    return room, gt, baseline, anchorguard


def test_floorplan_figure_generation(sample_data):
    room, gt, baseline, anchorguard = sample_data
    fig = create_floorplan_figure(
        room=room,
        ground_truth=gt,
        baseline_traj=baseline,
        anchorguard_traj=anchorguard,
        current_step=5,
        show_anchors=True,
    )
    assert isinstance(fig, go.Figure)
    # Ensure layout and axes do not raise exceptions when serialized
    json_rep = fig.to_json()
    assert len(json_rep) > 0
    assert fig.layout.xaxis.title.text == "X Coordinate (Metres)"
    assert fig.layout.yaxis.title.text == "Y Coordinate (Metres)"


def test_error_comparison_figure_generation(sample_data):
    _, gt, baseline, anchorguard = sample_data
    fig = create_error_comparison_figure(
        ground_truth=gt,
        baseline_traj=baseline,
        anchorguard_traj=anchorguard,
    )
    assert isinstance(fig, go.Figure)
    json_rep = fig.to_json()
    assert len(json_rep) > 0
    assert fig.layout.xaxis.title.text == "Trajectory Time Step (t)"
    assert fig.layout.yaxis.title.text == "Euclidean Error (Metres)"


def test_cdf_figure_generation(sample_data):
    _, gt, baseline, anchorguard = sample_data
    fig = create_cdf_figure(
        ground_truth=gt,
        baseline_traj=baseline,
        anchorguard_traj=anchorguard,
    )
    assert isinstance(fig, go.Figure)
    json_rep = fig.to_json()
    assert len(json_rep) > 0
    assert fig.layout.xaxis.title.text == "Position Error (Metres)"
    assert fig.layout.yaxis.title.text == "Cumulative Probability"


def test_floorplan_with_out_of_room_estimates(sample_data):
    room, gt, baseline, anchorguard = sample_data
    # Add large out-of-room estimate (-5m X, 25m Y)
    baseline_out = baseline.copy()
    baseline_out[0] = [-5.0, 25.0]

    fig = create_floorplan_figure(
        room=room,
        ground_truth=gt,
        baseline_traj=baseline_out,
        anchorguard_traj=anchorguard,
    )
    # Axes must dynamically expand to encompass out-of-room points
    assert fig.layout.xaxis.range[0] <= -5.0
    assert fig.layout.yaxis.range[1] >= 25.0


def test_floorplan_with_failed_estimates(sample_data):
    room, gt, baseline, anchorguard = sample_data
    # Introduce NaN failed estimates at steps 2 and 4
    baseline_failed = baseline.copy()
    baseline_failed[2] = [np.nan, np.nan]
    baseline_failed[4] = [np.nan, np.nan]

    fig = create_floorplan_figure(
        room=room,
        ground_truth=gt,
        baseline_traj=baseline_failed,
        anchorguard_traj=anchorguard,
    )
    # Check that 'Failed Estimation Steps' trace exists
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert "Failed Estimation Steps" in trace_names
