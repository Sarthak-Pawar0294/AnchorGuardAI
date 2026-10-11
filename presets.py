"""
presets.py - Standard Experiment Presets for AnchorGuard AI.
Provides documented configurations for the 4 interactive scenarios:
1. Clean Room (Ideal baseline)
2. Noisy Signals (Thermal AWGN noise)
3. Missing Readings (Packet dropouts)
4. Blocked Signals (NLOS multipath positive bias)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from simulation import SimulationConfig


@dataclass(frozen=True)
class ExperimentPreset:
    id: str
    name: str
    short_description: str
    detailed_explanation: str
    noise_std: float              # Metres
    drop_rate: float              # [0.0 - 1.0]
    nlos_probability: float       # [0.0 - 1.0]
    nlos_bias_mean: float         # Metres
    num_anchors: int
    outage_anchor_ids: List[int]
    trajectory_pattern: str
    trajectory_steps: int

    def to_simulation_config(self, seed: Optional[int] = 42) -> SimulationConfig:
        return SimulationConfig(
            noise_std=self.noise_std,
            drop_rate=self.drop_rate,
            nlos_probability=self.nlos_probability,
            nlos_bias_mean=self.nlos_bias_mean,
            outage_anchor_ids=list(self.outage_anchor_ids),
            seed=seed,
        )


EXPERIMENT_PRESETS: Dict[str, ExperimentPreset] = {
    "clean_room": ExperimentPreset(
        id="clean_room",
        name="1. Clean Room",
        short_description="Near-ideal indoor conditions with line-of-sight signals.",
        detailed_explanation=(
            "Simulates an open room with clear Line-of-Sight (LOS) to all 6 anchors. "
            "Measurement noise is minimal (σ = 0.08 m), with 0% missing readings and zero obstacle reflections. "
            "Under pure Gaussian conditions, standard Least Squares is mathematically near-optimal."
        ),
        noise_std=0.08,
        drop_rate=0.0,
        nlos_probability=0.0,
        nlos_bias_mean=0.0,
        num_anchors=6,
        outage_anchor_ids=[],
        trajectory_pattern="smooth_loop",
        trajectory_steps=100,
    ),
    "noisy_signals": ExperimentPreset(
        id="noisy_signals",
        name="2. Noisy Signals",
        short_description="Heavy random measurement noise from radio interference.",
        detailed_explanation=(
            "Simulates high radio-frequency interference and receiver thermal noise (σ = 0.70 m, 5% packet drops). "
            "The error is purely zero-mean Gaussian (AWGN). Standard Least Squares is theoretically well-suited for "
            "uncorrupted Gaussian distributions, while AnchorGuard's Huber loss provides stable performance."
        ),
        noise_std=0.70,
        drop_rate=0.05,
        nlos_probability=0.0,
        nlos_bias_mean=0.0,
        num_anchors=6,
        outage_anchor_ids=[],
        trajectory_pattern="smooth_loop",
        trajectory_steps=100,
    ),
    "missing_readings": ExperimentPreset(
        id="missing_readings",
        name="3. Missing Readings",
        short_description="Severe packet drops and intermittent sensor outages.",
        detailed_explanation=(
            "Simulates high wireless packet collisions and temporary sensor dropouts (35% missing readings). "
            "Estimators must dynamically determine positions from surviving subsets of anchors. "
            "AnchorGuard gracefully evaluates geometric conditioning and prevents erratic position spikes."
        ),
        noise_std=0.20,
        drop_rate=0.35,
        nlos_probability=0.05,
        nlos_bias_mean=1.5,
        num_anchors=6,
        outage_anchor_ids=[],
        trajectory_pattern="smooth_loop",
        trajectory_steps=100,
    ),
    "blocked_signals": ExperimentPreset(
        id="blocked_signals",
        name="4. Blocked Signals (NLOS)",
        short_description="Severe non-line-of-sight multipath wall/furniture reflections.",
        detailed_explanation=(
            "Simulates non-line-of-sight (NLOS) conditions where radio signals penetrate thick concrete walls "
            "or bounce off obstacles, adding large positive distance errors (35% probability, +3.5 m mean bias). "
            "Standard Least Squares is heavily corrupted by these positive outliers, whereas AnchorGuard isolates "
            "and down-weights or rejects the corrupted anchors using geometric consensus."
        ),
        noise_std=0.25,
        drop_rate=0.10,
        nlos_probability=0.35,
        nlos_bias_mean=3.5,
        num_anchors=6,
        outage_anchor_ids=[],
        trajectory_pattern="smooth_loop",
        trajectory_steps=100,
    ),
}
