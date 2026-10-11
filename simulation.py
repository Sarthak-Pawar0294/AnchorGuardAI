"""
simulation.py - Synthetic Indoor Positioning Environment & Measurement Simulator.

Provides:
- Configurable 2D Room and Anchor Geometry
- Realistic Target Trajectory Generators (Smooth Lissajous/8-Loop, Waypoint Patrol, Random Walk)
- Ground Truth Separation and Distance Calculation
- Fault Simulation:
  * Additive Gaussian noise
  * Packet drops (Missing measurements)
  * Positive Non-Line-of-Sight (NLOS) bias
  * Specific anchor outages / sensor blackouts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np


@dataclass
class Anchor:
    id: int
    name: str
    x: float
    y: float
    is_active: bool = True


@dataclass
class Room:
    width: float = 20.0  # in metres
    height: float = 14.0  # in metres
    anchors: List[Anchor] = field(default_factory=list)

    @classmethod
    def create_default(cls, width: float = 20.0, height: float = 14.0, num_anchors: int = 6) -> "Room":
        """
        Creates a room with well-distributed anchors along perimeter and interior corners.
        Supports 3 to 12 anchors.
        """
        room = cls(width=width, height=height)
        # Canonical anchor candidate locations for 20x14 room (or scaled)
        candidates = [
            (1.0, 1.0, "Anchor 1 (SW Corner)"),
            (width - 1.0, 1.0, "Anchor 2 (SE Corner)"),
            (width - 1.0, height - 1.0, "Anchor 3 (NE Corner)"),
            (1.0, height - 1.0, "Anchor 4 (NW Corner)"),
            (width / 2.0, 0.8, "Anchor 5 (South Center)"),
            (width / 2.0, height - 0.8, "Anchor 6 (North Center)"),
            (0.8, height / 2.0, "Anchor 7 (West Center)"),
            (width - 0.8, height / 2.0, "Anchor 8 (East Center)"),
            (width * 0.3, height * 0.4, "Anchor 9 (Interior West)"),
            (width * 0.7, height * 0.6, "Anchor 10 (Interior East)"),
            (width * 0.3, height * 0.7, "Anchor 11 (Interior North)"),
            (width * 0.7, height * 0.3, "Anchor 12 (Interior South)"),
        ]

        num_anchors = max(3, min(num_anchors, len(candidates)))
        room.anchors = [
            Anchor(id=i, name=candidates[i][2], x=candidates[i][0], y=candidates[i][1], is_active=True)
            for i in range(num_anchors)
        ]
        return room

    def get_active_anchors(self) -> List[Anchor]:
        return [a for a in self.anchors if a.is_active]

    def get_anchor_coordinates(self, active_only: bool = True) -> np.ndarray:
        anchors = self.get_active_anchors() if active_only else self.anchors
        if not anchors:
            return np.empty((0, 2))
        return np.array([[a.x, a.y] for a in anchors], dtype=float)


TrajectoryPattern = Literal["smooth_loop", "patrol_perimeter", "random_walk", "linear_crossing"]


def generate_trajectory(
    room: Room,
    pattern: TrajectoryPattern = "smooth_loop",
    num_steps: int = 100,
    speed: float = 1.0,  # nominal step speed factor
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Generates a realistic 2D ground truth trajectory inside the room boundaries.
    Returns:
        np.ndarray of shape (num_steps, 2), where each row is [x, y].
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    pad = 1.5
    w_min, w_max = pad, room.width - pad
    h_min, h_max = pad, room.height - pad
    cx, cy = room.width / 2.0, room.height / 2.0
    rx, ry = (room.width / 2.0) - pad, (room.height / 2.0) - pad

    if pattern == "smooth_loop":
        # Smooth Lissajous / Figure-8 trajectory
        t = np.linspace(0, 2 * np.pi, num_steps, endpoint=False)
        x = cx + rx * 0.85 * np.sin(t)
        y = cy + ry * 0.75 * np.sin(2 * t)
        return np.column_stack([x, y])

    elif pattern == "patrol_perimeter":
        # Smooth rectangular corridor path with rounded corners
        waypoints = np.array([
            [w_min, h_min],
            [w_max, h_min],
            [w_max, h_max],
            [w_min, h_max],
            [w_min, h_min]
        ])
        # Interpolate continuously along closed perimeter waypoints
        dists = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        step_dists = np.linspace(0, total_len, num_steps, endpoint=False)
        xs = np.interp(step_dists, cum_dist, waypoints[:, 0])
        ys = np.interp(step_dists, cum_dist, waypoints[:, 1])
        return np.column_stack([xs, ys])

    elif pattern == "linear_crossing":
        # Diagonal and cross sweeps inside room
        p_start = np.array([w_min, h_min])
        p_end = np.array([w_max, h_max])
        alphas = np.linspace(0, 1, num_steps)[:, np.newaxis]
        return p_start + alphas * (p_end - p_start)

    elif pattern == "random_walk":
        # Constrained smooth random walk with inertial heading
        points = np.zeros((num_steps, 2))
        pos = np.array([cx, cy])
        heading = rng.uniform(0, 2 * np.pi)
        step_len = 0.4 * speed

        for i in range(num_steps):
            points[i] = pos
            # Smooth heading variation
            heading += rng.normal(0.0, 0.25)
            next_pos = pos + np.array([np.cos(heading), np.sin(heading)]) * step_len
            
            # Boundary reflection
            if next_pos[0] < w_min or next_pos[0] > w_max:
                heading = np.pi - heading
                next_pos[0] = np.clip(next_pos[0], w_min, w_max)
            if next_pos[1] < h_min or next_pos[1] > h_max:
                heading = -heading
                next_pos[1] = np.clip(next_pos[1], w_min, w_max)
            pos = next_pos
        return points

    else:
        raise ValueError(f"Unknown trajectory pattern: {pattern}")


@dataclass
class SimulationConfig:
    noise_std: float = 0.25           # Metres, Gaussian standard deviation
    drop_rate: float = 0.10           # Missing measurement probability [0.0, 1.0]
    nlos_probability: float = 0.20    # Probability of an NLOS condition [0.0, 1.0]
    nlos_bias_mean: float = 2.5       # Exponential or positive bias mean in metres
    nlos_bias_max: float = 6.0        # Cap for positive NLOS bias
    outage_anchor_ids: List[int] = field(default_factory=list)  # Permanently or temporarily disabled
    seed: Optional[int] = 42


@dataclass
class SimulationResult:
    trajectory: np.ndarray            # Shape (T, 2) Ground truth coordinates (NEVER exposed to estimators)
    measurements: np.ndarray          # Shape (T, N) Distances with NaN for missing
    true_ranges: np.ndarray           # Shape (T, N) Ideal Euclidean distances (for inspection/analysis only)
    nlos_flags: np.ndarray            # Shape (T, N) Boolean mask of actual NLOS occurrences
    anchor_coords: np.ndarray         # Shape (N, 2) Coordinates of anchors corresponding to columns
    anchor_ids: List[int]             # List of anchor IDs


def simulate_measurements(
    room: Room,
    trajectory: np.ndarray,
    config: SimulationConfig
) -> SimulationResult:
    """
    Computes true Euclidean distances and injects noise, missing measurements, and NLOS bias.
    Strictly decouples ground truth from output measurements.
    """
    if config.seed is not None:
        rng = np.random.default_rng(config.seed)
    else:
        rng = np.random.default_rng()

    # Determine active anchors without mutating room's persistent state
    active_anchors = [
        a for a in room.anchors
        if a.is_active and (a.id not in config.outage_anchor_ids)
    ]
    num_anchors = len(active_anchors)
    num_steps = len(trajectory)

    if num_anchors == 0:
        raise ValueError("Cannot simulate measurements: No active anchors in room!")

    anchor_coords = np.array([[a.x, a.y] for a in active_anchors], dtype=float)
    anchor_ids = [a.id for a in active_anchors]

    # Compute Ideal Euclidean Distances: shape (T, N)
    # trajectory: (T, 1, 2), anchor_coords: (1, N, 2)
    diff = trajectory[:, np.newaxis, :] - anchor_coords[np.newaxis, :, :]
    true_ranges = np.linalg.norm(diff, axis=-1)

    # 1. Additive Gaussian Measurement Noise
    gaussian_noise = rng.normal(0.0, max(0.0, config.noise_std), size=(num_steps, num_anchors))

    # 2. Positive NLOS Distance Bias
    nlos_mask = rng.uniform(0.0, 1.0, size=(num_steps, num_anchors)) < config.nlos_probability
    # Exponential distribution yields realistic asymmetric positive multipath tail
    nlos_biases = rng.exponential(scale=max(0.1, config.nlos_bias_mean), size=(num_steps, num_anchors))
    nlos_biases = np.clip(nlos_biases, 0.5, config.nlos_bias_max)
    # Apply bias only where NLOS mask is True
    applied_nlos = np.where(nlos_mask, nlos_biases, 0.0)

    # Raw measurements before dropout
    noisy_ranges = true_ranges + gaussian_noise + applied_nlos
    # Ensure physical distances cannot be negative
    noisy_ranges = np.clip(noisy_ranges, 0.01, None)

    # 3. Missing Measurements (Packet Dropout)
    drop_mask = rng.uniform(0.0, 1.0, size=(num_steps, num_anchors)) < config.drop_rate
    measurements = np.where(drop_mask, np.nan, noisy_ranges)

    return SimulationResult(
        trajectory=trajectory.copy(),
        measurements=measurements,
        true_ranges=true_ranges,
        nlos_flags=nlos_mask,
        anchor_coords=anchor_coords,
        anchor_ids=anchor_ids,
    )
