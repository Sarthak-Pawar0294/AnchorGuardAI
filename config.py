"""
config.py - Validated Experiment Configuration Model for AnchorGuard AI 2.0 Workbench.

Provides:
- AnchorConfig and WorkbenchConfig dataclasses with strict validation.
- Presets for Anchor Layouts (Corners, Perimeter, Grid).
- Fingerprinting / Hashing for Stale-Result Detection.
- JSON Serialization and Deserialization with schema versioning.
- Bridges to simulation.Room, simulation.SimulationConfig, and estimators.AnchorGuardConfig.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import numpy as np

from simulation import Room, Anchor, SimulationConfig
from estimators import AnchorGuardConfig


CONFIG_SCHEMA_VERSION = "2.0"


@dataclass
class AnchorConfig:
    id: int
    name: str
    x: float
    y: float
    is_active: bool = True

    def to_anchor(self) -> Anchor:
        return Anchor(
            id=int(self.id),
            name=str(self.name),
            x=float(self.x),
            y=float(self.y),
            is_active=bool(self.is_active),
        )

    @classmethod
    def from_anchor(cls, a: Anchor) -> "AnchorConfig":
        return cls(
            id=a.id,
            name=a.name,
            x=float(a.x),
            y=float(a.y),
            is_active=bool(a.is_active),
        )


def generate_corner_anchors(width: float = 20.0, height: float = 14.0, pad: float = 1.0) -> List[AnchorConfig]:
    """Generates 4 anchors at the 4 room corners with a safety padding."""
    return [
        AnchorConfig(id=0, name="Anchor 0 (SW Corner)", x=pad, y=pad, is_active=True),
        AnchorConfig(id=1, name="Anchor 1 (SE Corner)", x=width - pad, y=pad, is_active=True),
        AnchorConfig(id=2, name="Anchor 2 (NE Corner)", x=width - pad, y=height - pad, is_active=True),
        AnchorConfig(id=3, name="Anchor 3 (NW Corner)", x=pad, y=height - pad, is_active=True),
    ]


def generate_perimeter_anchors(width: float = 20.0, height: float = 14.0, count: int = 6) -> List[AnchorConfig]:
    """Generates 6 to 8 anchors distributed around the room perimeter."""
    pad = 1.0
    anchors = [
        AnchorConfig(id=0, name="Anchor 0 (SW Corner)", x=pad, y=pad, is_active=True),
        AnchorConfig(id=1, name="Anchor 1 (SE Corner)", x=width - pad, y=pad, is_active=True),
        AnchorConfig(id=2, name="Anchor 2 (NE Corner)", x=width - pad, y=height - pad, is_active=True),
        AnchorConfig(id=3, name="Anchor 3 (NW Corner)", x=pad, y=height - pad, is_active=True),
        AnchorConfig(id=4, name="Anchor 4 (South Center)", x=width / 2.0, y=pad * 0.8, is_active=True),
        AnchorConfig(id=5, name="Anchor 5 (North Center)", x=width / 2.0, y=height - (pad * 0.8), is_active=True),
    ]
    if count >= 8:
        anchors.extend([
            AnchorConfig(id=6, name="Anchor 6 (West Center)", x=pad * 0.8, y=height / 2.0, is_active=True),
            AnchorConfig(id=7, name="Anchor 7 (East Center)", x=width - (pad * 0.8), y=height / 2.0, is_active=True),
        ])
    return anchors


def generate_grid_anchors(width: float = 20.0, height: float = 14.0, rows: int = 3, cols: int = 3) -> List[AnchorConfig]:
    """Generates a uniform grid layout of anchors across the room."""
    pad = 1.2
    xs = np.linspace(pad, width - pad, cols)
    ys = np.linspace(pad, height - pad, rows)
    anchors = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            name = f"Anchor {idx} (Grid R{r+1}C{c+1})"
            anchors.append(AnchorConfig(id=idx, name=name, x=round(float(xs[c]), 2), y=round(float(ys[r]), 2), is_active=True))
            idx += 1
    return anchors


@dataclass
class WorkbenchConfig:
    # 1. Environment & Geometry
    room_width: float = 20.0
    room_height: float = 14.0
    layout_preset: str = "perimeter"  # "corners", "perimeter", "grid", "custom"
    anchors: List[AnchorConfig] = field(default_factory=lambda: generate_perimeter_anchors(20.0, 14.0, 6))

    # 2. Target Trajectory
    trajectory_pattern: str = "smooth_loop"  # "smooth_loop", "patrol_perimeter", "linear_crossing", "random_walk"
    trajectory_steps: int = 100
    trajectory_speed: float = 1.0

    # 3. Measurement Fault Settings
    noise_std: float = 0.25           # Metres Gaussian sigma
    drop_rate: float = 0.10           # Packet drop probability [0.0, 1.0]
    nlos_probability: float = 0.20    # Probability of NLOS condition [0.0, 1.0]
    nlos_bias_mean: float = 2.5       # Exponential scale in metres
    nlos_bias_max: float = 6.0        # Max NLOS bias cap
    outage_anchor_ids: List[int] = field(default_factory=list)

    # 4. Reproducibility
    seed: int = 42

    # 5. Estimator Selection & Parameters
    selected_estimators: List[str] = field(default_factory=lambda: ["baseline", "anchorguard"])
    huber_delta: float = 1.345
    enable_residual_pruning: bool = True
    nlos_residual_threshold: float = 2.0
    max_pruned_anchors: int = 2
    min_anchors_after_pruning: int = 3
    apply_kalman: bool = False
    kalman_process_std: float = 0.5

    # 6. Metadata
    config_version: str = CONFIG_SCHEMA_VERSION

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validates all configuration parameters against physical and mathematical limits.
        Returns: (is_valid, list_of_error_messages)
        """
        errors: List[str] = []

        if self.room_width <= 0:
            errors.append(f"Room width must be positive (got {self.room_width} m).")
        if self.room_height <= 0:
            errors.append(f"Room height must be positive (got {self.room_height} m).")

        if self.trajectory_steps < 5:
            errors.append(f"Trajectory steps must be at least 5 (got {self.trajectory_steps}).")
        if self.trajectory_speed <= 0:
            errors.append(f"Trajectory speed factor must be positive (got {self.trajectory_speed}).")

        if self.noise_std < 0:
            errors.append(f"Noise standard deviation cannot be negative (got {self.noise_std} m).")

        if not (0.0 <= self.drop_rate <= 1.0):
            errors.append(f"Missing-reading drop rate must be in [0.0, 1.0] (got {self.drop_rate}).")

        if not (0.0 <= self.nlos_probability <= 1.0):
            errors.append(f"NLOS probability must be in [0.0, 1.0] (got {self.nlos_probability}).")

        if self.nlos_bias_mean < 0:
            errors.append(f"NLOS mean bias cannot be negative (got {self.nlos_bias_mean} m).")

        if self.nlos_bias_max < self.nlos_bias_mean and self.nlos_probability > 0:
            errors.append(f"NLOS max cap ({self.nlos_bias_max} m) should be >= mean bias ({self.nlos_bias_mean} m).")

        if self.seed < 0:
            errors.append(f"Random seed must be non-negative (got {self.seed}).")

        if self.huber_delta <= 0:
            errors.append(f"Huber tuning delta must be positive (got {self.huber_delta}).")

        if self.kalman_process_std <= 0:
            errors.append(f"Kalman process noise std must be positive (got {self.kalman_process_std}).")

        # Active anchor count check
        active_count = sum(1 for a in self.anchors if a.is_active and a.id not in self.outage_anchor_ids)
        if active_count < 3:
            errors.append(f"At least 3 active anchors are mathematically required for 2D positioning (found {active_count}).")

        return len(errors) == 0, errors

    def to_room(self) -> Room:
        """Constructs a simulation.Room instance."""
        room = Room(width=float(self.room_width), height=float(self.room_height))
        room.anchors = [a.to_anchor() for a in self.anchors]
        return room

    def to_simulation_config(self) -> SimulationConfig:
        """Constructs a simulation.SimulationConfig instance."""
        return SimulationConfig(
            noise_std=float(self.noise_std),
            drop_rate=float(self.drop_rate),
            nlos_probability=float(self.nlos_probability),
            nlos_bias_mean=float(self.nlos_bias_mean),
            nlos_bias_max=float(self.nlos_bias_max),
            outage_anchor_ids=list(self.outage_anchor_ids),
            seed=int(self.seed),
        )

    def to_anchorguard_config(self) -> AnchorGuardConfig:
        """Constructs an estimators.AnchorGuardConfig instance."""
        return AnchorGuardConfig(
            loss="huber",
            huber_delta=float(self.huber_delta),
            enable_residual_pruning=bool(self.enable_residual_pruning),
            nlos_residual_threshold=float(self.nlos_residual_threshold),
            max_pruned_anchors=int(self.max_pruned_anchors),
            min_anchors_after_pruning=int(self.min_anchors_after_pruning),
            noise_sigma=float(self.noise_std),
        )

    def compute_fingerprint(self) -> str:
        """
        Computes a deterministic SHA256 hex digest of the configuration fields.
        Used to detect when any setting has been modified since the last simulation run.
        """
        payload = self.to_dict()
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to dictionary."""
        d = asdict(self)
        # Ensure anchors are dicts
        d["anchors"] = [asdict(a) for a in self.anchors]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkbenchConfig":
        """Reconstructs WorkbenchConfig from dictionary with schema tolerance."""
        anchors_raw = data.get("anchors", [])
        anchors = [
            AnchorConfig(
                id=int(a["id"]),
                name=str(a.get("name", f"Anchor {a['id']}")),
                x=float(a["x"]),
                y=float(a["y"]),
                is_active=bool(a.get("is_active", True)),
            )
            for a in anchors_raw
        ]

        return cls(
            room_width=float(data.get("room_width", 20.0)),
            room_height=float(data.get("room_height", 14.0)),
            layout_preset=str(data.get("layout_preset", "perimeter")),
            anchors=anchors if anchors else generate_perimeter_anchors(20.0, 14.0, 6),
            trajectory_pattern=str(data.get("trajectory_pattern", "smooth_loop")),
            trajectory_steps=int(data.get("trajectory_steps", 100)),
            trajectory_speed=float(data.get("trajectory_speed", 1.0)),
            noise_std=float(data.get("noise_std", 0.25)),
            drop_rate=float(data.get("drop_rate", 0.10)),
            nlos_probability=float(data.get("nlos_probability", 0.20)),
            nlos_bias_mean=float(data.get("nlos_bias_mean", 2.5)),
            nlos_bias_max=float(data.get("nlos_bias_max", 6.0)),
            outage_anchor_ids=[int(i) for i in data.get("outage_anchor_ids", [])],
            seed=int(data.get("seed", 42)),
            selected_estimators=[str(e) for e in data.get("selected_estimators", ["baseline", "anchorguard"])],
            huber_delta=float(data.get("huber_delta", 1.345)),
            enable_residual_pruning=bool(data.get("enable_residual_pruning", True)),
            nlos_residual_threshold=float(data.get("nlos_residual_threshold", 2.0)),
            max_pruned_anchors=int(data.get("max_pruned_anchors", 2)),
            min_anchors_after_pruning=int(data.get("min_anchors_after_pruning", 3)),
            apply_kalman=bool(data.get("apply_kalman", False)),
            kalman_process_std=float(data.get("kalman_process_std", 0.5)),
            config_version=str(data.get("config_version", CONFIG_SCHEMA_VERSION)),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serializes configuration to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkbenchConfig":
        """Deserializes configuration from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def apply_preset(self, preset_id: str) -> None:
        """Applies one of the standard measurement fault presets."""
        from presets import EXPERIMENT_PRESETS
        if preset_id in EXPERIMENT_PRESETS:
            p = EXPERIMENT_PRESETS[preset_id]
            self.noise_std = p.noise_std
            self.drop_rate = p.drop_rate
            self.nlos_probability = p.nlos_probability
            self.nlos_bias_mean = p.nlos_bias_mean
            self.outage_anchor_ids = list(p.outage_anchor_ids)
            self.trajectory_pattern = p.trajectory_pattern
            self.trajectory_steps = p.trajectory_steps
