"""
tests/test_config.py - Tests for WorkbenchConfig and AnchorConfig.
"""

import pytest
import json
from config import (
    WorkbenchConfig,
    AnchorConfig,
    generate_corner_anchors,
    generate_perimeter_anchors,
    generate_grid_anchors,
)


def test_default_config_valid():
    cfg = WorkbenchConfig()
    is_valid, errors = cfg.validate()
    assert is_valid is True
    assert len(errors) == 0


def test_invalid_room_dimensions():
    cfg = WorkbenchConfig(room_width=-10.0, room_height=0.0)
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("width" in e for e in errors)
    assert any("height" in e for e in errors)


def test_invalid_probabilities():
    cfg = WorkbenchConfig(drop_rate=1.5, nlos_probability=-0.2)
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("drop rate" in e for e in errors)
    assert any("NLOS probability" in e for e in errors)


def test_insufficient_anchors_detection():
    # Only 2 active anchors
    cfg = WorkbenchConfig(
        anchors=[
            AnchorConfig(id=0, name="A0", x=1.0, y=1.0, is_active=True),
            AnchorConfig(id=1, name="A1", x=5.0, y=1.0, is_active=True),
            AnchorConfig(id=2, name="A2", x=5.0, y=5.0, is_active=False),
        ]
    )
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("At least 3 active anchors" in e for e in errors)


def test_outage_causes_insufficient_anchors():
    # 3 active anchors, but one is in outage_anchor_ids
    cfg = WorkbenchConfig(
        anchors=[
            AnchorConfig(id=0, name="A0", x=1.0, y=1.0, is_active=True),
            AnchorConfig(id=1, name="A1", x=5.0, y=1.0, is_active=True),
            AnchorConfig(id=2, name="A2", x=5.0, y=5.0, is_active=True),
        ],
        outage_anchor_ids=[2]
    )
    is_valid, errors = cfg.validate()
    assert is_valid is False
    assert any("At least 3 active anchors" in e for e in errors)


def test_layout_generators():
    corners = generate_corner_anchors(20.0, 14.0)
    assert len(corners) == 4
    assert corners[0].x == 1.0 and corners[0].y == 1.0
    assert corners[2].x == 19.0 and corners[2].y == 13.0

    perimeter = generate_perimeter_anchors(20.0, 14.0, count=6)
    assert len(perimeter) == 6

    perimeter_8 = generate_perimeter_anchors(20.0, 14.0, count=8)
    assert len(perimeter_8) == 8

    grid = generate_grid_anchors(20.0, 14.0, rows=3, cols=3)
    assert len(grid) == 9


def test_json_roundtrip():
    cfg = WorkbenchConfig(
        room_width=25.0,
        room_height=15.0,
        noise_std=0.4,
        drop_rate=0.2,
        seed=999,
        apply_kalman=True,
    )
    json_str = cfg.to_json()
    cfg_loaded = WorkbenchConfig.from_json(json_str)

    assert cfg_loaded.room_width == 25.0
    assert cfg_loaded.room_height == 15.0
    assert cfg_loaded.noise_std == 0.4
    assert cfg_loaded.drop_rate == 0.2
    assert cfg_loaded.seed == 999
    assert cfg_loaded.apply_kalman is True
    assert len(cfg_loaded.anchors) == len(cfg.anchors)


def test_fingerprint_stale_detection():
    cfg1 = WorkbenchConfig(noise_std=0.25)
    cfg2 = WorkbenchConfig(noise_std=0.25)
    cfg3 = WorkbenchConfig(noise_std=0.50)

    fp1 = cfg1.compute_fingerprint()
    fp2 = cfg2.compute_fingerprint()
    fp3 = cfg3.compute_fingerprint()

    assert fp1 == fp2
    assert fp1 != fp3


def test_model_bridges():
    cfg = WorkbenchConfig(noise_std=0.35, huber_delta=1.5)
    room = cfg.to_room()
    sim_cfg = cfg.to_simulation_config()
    ag_cfg = cfg.to_anchorguard_config()

    assert room.width == cfg.room_width
    assert len(room.anchors) == len(cfg.anchors)
    assert sim_cfg.noise_std == 0.35
    assert ag_cfg.huber_delta == 1.5
