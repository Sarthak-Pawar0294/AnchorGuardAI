"""
benchmark.py - Automated Evaluation Benchmark & Stress Testing for AnchorGuard AI.

Executes reproducible comparative evaluation across varied stress environments:
1. Low-Noise Scenario (ideal/minimal conditions)
2. High-Noise Scenario (severe AWGN)
3. High Dropout Scenario (packet drops / missing measurements)
4. Heavy NLOS Scenario (multipath positive range bias)
5. Sparse Infrastructure Scenario (reduced anchor density)

Generates empirical comparison tables and exports results to CSV.
"""

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from simulation import Room, generate_trajectory, simulate_measurements, SimulationConfig
from estimators import estimate_trajectory, AnchorGuardConfig
from metrics import compute_positioning_metrics


@dataclass
class BenchmarkScenario:
    name: str
    description: str
    num_anchors: int
    noise_std: float
    drop_rate: float
    nlos_probability: float
    nlos_bias_mean: float
    outage_anchor_ids: List[int]


BENCHMARK_SCENARIOS = [
    BenchmarkScenario(
        name="1_Baseline_Ideal",
        description="Low noise, zero NLOS, zero packet loss, 8 anchors",
        num_anchors=8,
        noise_std=0.10,
        drop_rate=0.0,
        nlos_probability=0.0,
        nlos_bias_mean=0.0,
        outage_anchor_ids=[],
    ),
    BenchmarkScenario(
        name="2_High_AWGN_Noise",
        description="High Gaussian noise (sigma=0.75m), zero NLOS",
        num_anchors=8,
        noise_std=0.75,
        drop_rate=0.05,
        nlos_probability=0.0,
        nlos_bias_mean=0.0,
        outage_anchor_ids=[],
    ),
    BenchmarkScenario(
        name="3_High_Packet_Loss",
        description="Sparse packet loss (35% drop rate), nominal noise",
        num_anchors=6,
        noise_std=0.20,
        drop_rate=0.35,
        nlos_probability=0.05,
        nlos_bias_mean=1.5,
        outage_anchor_ids=[],
    ),
    BenchmarkScenario(
        name="4_Severe_NLOS_Biases",
        description="Severe NLOS conditions (40% probability, +3.5m bias)",
        num_anchors=8,
        noise_std=0.25,
        drop_rate=0.10,
        nlos_probability=0.40,
        nlos_bias_mean=3.5,
        outage_anchor_ids=[],
    ),
    BenchmarkScenario(
        name="5_Sparse_Outage_Hostile",
        description="Reduced anchors (4 total with 1 outage = 3 remaining), NLOS + drops",
        num_anchors=4,
        noise_std=0.30,
        drop_rate=0.15,
        nlos_probability=0.25,
        nlos_bias_mean=3.0,
        outage_anchor_ids=[0],  # Permanent outage of 1 anchor -> only 3 active
    ),
]


def run_benchmark(
    scenarios: List[BenchmarkScenario] = BENCHMARK_SCENARIOS,
    num_steps: int = 120,
    num_trials: int = 3,
    output_csv: str = "benchmark_results.csv",
    seed_base: int = 100,
) -> pd.DataFrame:
    """
    Runs multi-trial benchmark across all scenarios and returns consolidated summary DataFrame.
    Guarantees that both algorithms are provided IDENTICAL trajectories and measurements.
    """
    records: List[Dict[str, Any]] = []

    for sc in scenarios:
        baseline_rmses = []
        baseline_maes = []
        baseline_p95s = []
        baseline_success = []

        ag_rmses = []
        ag_maes = []
        ag_p95s = []
        ag_success = []

        ag_kf_rmses = []
        ag_kf_maes = []
        ag_kf_p95s = []

        for trial in range(num_trials):
            current_seed = seed_base + (trial * 37)
            room = Room.create_default(width=20.0, height=14.0, num_anchors=sc.num_anchors)
            gt_traj = generate_trajectory(room, pattern="smooth_loop", num_steps=num_steps, seed=current_seed)

            sim_cfg = SimulationConfig(
                noise_std=sc.noise_std,
                drop_rate=sc.drop_rate,
                nlos_probability=sc.nlos_probability,
                nlos_bias_mean=sc.nlos_bias_mean,
                outage_anchor_ids=sc.outage_anchor_ids,
                seed=current_seed,
            )

            sim_res = simulate_measurements(room, gt_traj, sim_cfg)

            # 1. Baseline NLS
            base_res = estimate_trajectory(
                measurements=sim_res.measurements,
                anchor_coords=sim_res.anchor_coords,
                method="baseline",
                noise_sigma=sc.noise_std,
            )
            base_m = compute_positioning_metrics(gt_traj, base_res.estimated_trajectory, base_res.success_flags)
            baseline_rmses.append(base_m.rmse)
            baseline_maes.append(base_m.mae)
            baseline_p95s.append(base_m.p95_error)
            baseline_success.append(base_m.success_rate_pct)

            # 2. AnchorGuard Robust (Huber + Residual Rejection)
            ag_cfg = AnchorGuardConfig(noise_sigma=sc.noise_std)
            ag_res = estimate_trajectory(
                measurements=sim_res.measurements,
                anchor_coords=sim_res.anchor_coords,
                method="anchorguard",
                noise_sigma=sc.noise_std,
                anchorguard_config=ag_cfg,
                apply_kalman=False,
            )
            ag_m = compute_positioning_metrics(gt_traj, ag_res.estimated_trajectory, ag_res.success_flags)
            ag_rmses.append(ag_m.rmse)
            ag_maes.append(ag_m.mae)
            ag_p95s.append(ag_m.p95_error)
            ag_success.append(ag_m.success_rate_pct)

            # 3. AnchorGuard + Kalman Smoothing
            ag_kf_res = estimate_trajectory(
                measurements=sim_res.measurements,
                anchor_coords=sim_res.anchor_coords,
                method="anchorguard",
                noise_sigma=sc.noise_std,
                anchorguard_config=ag_cfg,
                apply_kalman=True,
            )
            ag_kf_m = compute_positioning_metrics(gt_traj, ag_kf_res.estimated_trajectory, None)
            ag_kf_rmses.append(ag_kf_m.rmse)
            ag_kf_maes.append(ag_kf_m.mae)
            ag_kf_p95s.append(ag_kf_m.p95_error)

        def safe_stats(vals):
            valid = [v for v in vals if not np.isnan(v)]
            if not valid:
                return np.nan, np.nan
            mean_v = float(np.mean(valid))
            std_v = float(np.std(valid)) if len(valid) > 1 else 0.0
            return mean_v, std_v

        b_rmse_avg, b_rmse_std = safe_stats(baseline_rmses)
        ag_rmse_avg, ag_rmse_std = safe_stats(ag_rmses)
        ag_kf_rmse_avg, ag_kf_rmse_std = safe_stats(ag_kf_rmses)

        rmse_improvement_pct = ((b_rmse_avg - ag_rmse_avg) / b_rmse_avg * 100.0) if (not np.isnan(b_rmse_avg) and b_rmse_avg > 0) else 0.0

        records.append({
            "Scenario": sc.name,
            "Description": sc.description,
            "Trials": num_trials,
            "Baseline RMSE (m)": round(b_rmse_avg, 3),
            "Baseline RMSE Std (m)": round(b_rmse_std, 3),
            "Baseline P95 (m)": round(safe_stats(baseline_p95s)[0], 3),
            "Baseline Success (%)": round(safe_stats(baseline_success)[0], 1),
            "AnchorGuard RMSE (m)": round(ag_rmse_avg, 3),
            "AnchorGuard RMSE Std (m)": round(ag_rmse_std, 3),
            "AnchorGuard P95 (m)": round(safe_stats(ag_p95s)[0], 3),
            "AnchorGuard Success (%)": round(safe_stats(ag_success)[0], 1),
            "AG+Kalman RMSE (m)": round(ag_kf_rmse_avg, 3),
            "AG Error Reduction (%)": round(rmse_improvement_pct, 1),
        })

    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
    return df


def run_workbench_trials(
    config: Any,  # WorkbenchConfig
    num_trials: int = 5,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """
    Executes repeated Monte Carlo trials using a WorkbenchConfig.
    Guarantees:
    - Identical ground truth and measurements fed to every enabled estimator within each trial.
    - Explicit seed stepping (seed, seed+1, ...).
    - Returns summary statistics (mean ± std) and complete per-trial records.
    """
    room = config.to_room()
    room_dims = (config.room_width, config.room_height)
    base_seed = config.seed + seed_offset

    trial_records = []
    base_rmses, base_maes, base_p95s, base_succs, base_oors = [], [], [], [], []
    ag_rmses, ag_maes, ag_p95s, ag_succs, ag_oors = [], [], [], [], []
    ag_kf_rmses, ag_kf_maes, ag_kf_p95s, ag_kf_succs, ag_kf_oors = [], [], [], [], []

    for trial_idx in range(num_trials):
        current_seed = base_seed + trial_idx
        # Copy config for trial with current seed
        sim_cfg = config.to_simulation_config()
        sim_cfg.seed = current_seed

        gt_traj = generate_trajectory(
            room=room,
            pattern=config.trajectory_pattern,
            num_steps=config.trajectory_steps,
            speed=config.trajectory_speed,
            seed=current_seed,
        )
        sim_res = simulate_measurements(room, gt_traj, sim_cfg)

        row = {"trial": trial_idx + 1, "seed": current_seed}

        # 1. Baseline
        base_res = estimate_trajectory(
            measurements=sim_res.measurements,
            anchor_coords=sim_res.anchor_coords,
            method="baseline",
            noise_sigma=config.noise_std,
        )
        base_m = compute_positioning_metrics(
            ground_truth=gt_traj,
            estimated=base_res.estimated_trajectory,
            success_flags=base_res.success_flags,
            avg_runtime_ms=base_res.avg_runtime_ms,
            total_runtime_ms=base_res.total_runtime_ms,
            room_dims=room_dims,
        )
        base_rmses.append(base_m.rmse)
        base_maes.append(base_m.mae)
        base_p95s.append(base_m.p95_error)
        base_succs.append(base_m.success_rate_pct)
        base_oors.append(base_m.out_of_room_rate_pct)
        row["baseline_rmse"] = base_m.rmse
        row["baseline_success_pct"] = base_m.success_rate_pct

        # 2. AnchorGuard Robust
        ag_cfg = config.to_anchorguard_config()
        ag_res = estimate_trajectory(
            measurements=sim_res.measurements,
            anchor_coords=sim_res.anchor_coords,
            method="anchorguard",
            noise_sigma=config.noise_std,
            anchorguard_config=ag_cfg,
            apply_kalman=False,
        )
        ag_m = compute_positioning_metrics(
            ground_truth=gt_traj,
            estimated=ag_res.estimated_trajectory,
            success_flags=ag_res.success_flags,
            avg_runtime_ms=ag_res.avg_runtime_ms,
            total_runtime_ms=ag_res.total_runtime_ms,
            room_dims=room_dims,
        )
        ag_rmses.append(ag_m.rmse)
        ag_maes.append(ag_m.mae)
        ag_p95s.append(ag_m.p95_error)
        ag_succs.append(ag_m.success_rate_pct)
        ag_oors.append(ag_m.out_of_room_rate_pct)
        row["anchorguard_rmse"] = ag_m.rmse
        row["anchorguard_success_pct"] = ag_m.success_rate_pct

        # 3. AnchorGuard + Kalman
        ag_kf_res = estimate_trajectory(
            measurements=sim_res.measurements,
            anchor_coords=sim_res.anchor_coords,
            method="anchorguard",
            noise_sigma=config.noise_std,
            anchorguard_config=ag_cfg,
            apply_kalman=True,
            kalman_process_std=config.kalman_process_std,
        )
        ag_kf_m = compute_positioning_metrics(
            ground_truth=gt_traj,
            estimated=ag_kf_res.estimated_trajectory,
            success_flags=None,
            avg_runtime_ms=ag_kf_res.avg_runtime_ms,
            total_runtime_ms=ag_kf_res.total_runtime_ms,
            room_dims=room_dims,
        )
        ag_kf_rmses.append(ag_kf_m.rmse)
        ag_kf_maes.append(ag_kf_m.mae)
        ag_kf_p95s.append(ag_kf_m.p95_error)
        ag_kf_succs.append(ag_kf_m.success_rate_pct)
        ag_kf_oors.append(ag_kf_m.out_of_room_rate_pct)
        row["ag_kf_rmse"] = ag_kf_m.rmse
        row["ag_kf_success_pct"] = ag_kf_m.success_rate_pct

        trial_records.append(row)

    def calc_mean_std(vals):
        valid = [v for v in vals if not np.isnan(v)]
        if not valid:
            return np.nan, np.nan
        m = float(np.mean(valid))
        s = float(np.std(valid)) if len(valid) > 1 else 0.0
        return m, s

    b_rmse_m, b_rmse_s = calc_mean_std(base_rmses)
    ag_rmse_m, ag_rmse_s = calc_mean_std(ag_rmses)
    ag_kf_rmse_m, ag_kf_rmse_s = calc_mean_std(ag_kf_rmses)

    return {
        "num_trials_completed": num_trials,
        "base_seed": base_seed,
        "seeds_used": [base_seed + i for i in range(num_trials)],
        "baseline": {
            "rmse_mean": b_rmse_m, "rmse_std": b_rmse_s,
            "mae_mean": calc_mean_std(base_maes)[0],
            "p95_mean": calc_mean_std(base_p95s)[0],
            "success_mean": calc_mean_std(base_succs)[0],
            "out_of_room_mean": calc_mean_std(base_oors)[0],
        },
        "anchorguard": {
            "rmse_mean": ag_rmse_m, "rmse_std": ag_rmse_s,
            "mae_mean": calc_mean_std(ag_maes)[0],
            "p95_mean": calc_mean_std(ag_p95s)[0],
            "success_mean": calc_mean_std(ag_succs)[0],
            "out_of_room_mean": calc_mean_std(ag_oors)[0],
        },
        "anchorguard_kalman": {
            "rmse_mean": ag_kf_rmse_m, "rmse_std": ag_kf_rmse_s,
            "mae_mean": calc_mean_std(ag_kf_maes)[0],
            "p95_mean": calc_mean_std(ag_kf_p95s)[0],
            "success_mean": calc_mean_std(ag_kf_succs)[0],
            "out_of_room_mean": calc_mean_std(ag_kf_oors)[0],
        },
        "trial_records": trial_records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnchorGuard AI Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run 1 trial per scenario for quick validation")
    parser.add_argument("--trials", type=int, default=3, help="Number of Monte Carlo trials per scenario")
    parser.add_argument("--out", type=str, default="benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    trials = 1 if args.quick else args.trials
    print(f"Running AnchorGuard AI Benchmark ({len(BENCHMARK_SCENARIOS)} scenarios, {trials} trials each)...")
    res_df = run_benchmark(num_trials=trials, output_csv=args.out)
    print("\n=== BENCHMARK SUMMARY RESULTS ===")
    print(res_df[["Scenario", "Baseline RMSE (m)", "AnchorGuard RMSE (m)", "AG+Kalman RMSE (m)", "AG Error Reduction (%)"]].to_string())
    print(f"\nFull results saved to: {args.out}")
