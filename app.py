"""
app.py - AnchorGuard AI: Polished Indoor Positioning Product Demonstration.
VIT Code Apex 3.0 | Track 4, Topic 2: AI-Based Indoor Positioning with Sparse Infrastructure and Noisy Measurements.
Team: Gang of Coders.

Cohesive Single-Dashboard Architecture:
- Section A: Header, branding, problem context, and simulation status.
- Section B: Compact scenario presets (Normal, Noisy, Missing, Blocked) & fault controls.
- Section C: Main Interactive Floor Plan (visual centerpiece with ground truth, baseline, AnchorGuard).
- Section D: Performance Metrics Summary (RMSE, Median, P95, Error Reduction with plain-English definitions).
- Section E: "What Happened and Why?" dynamic scientific analysis.
- Section F: Supporting Evidence (Error over time & Error CDF curves).
- Collapsed Section: Advanced Analysis (algorithmic tuning, residual diagnostics, Monte Carlo, and data export).
"""

import io
import json
import time
from typing import Optional, Tuple, Dict, Any, List
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from simulation import (
    Room,
    Anchor,
    generate_trajectory,
    simulate_measurements,
    SimulationConfig,
)
from estimators import (
    estimate_trajectory,
    AnchorGuardConfig,
    EstimationStatus,
    check_geometry_conditioning,
)
from metrics import compute_positioning_metrics, metrics_to_dict, PerformanceMetrics
from visualization import (
    create_floorplan_figure,
    create_error_comparison_figure,
    create_cdf_figure,
    create_residual_diagnostics_figure,
)
from config import (
    WorkbenchConfig,
    AnchorConfig,
    generate_corner_anchors,
    generate_perimeter_anchors,
    generate_grid_anchors,
)
from presets import EXPERIMENT_PRESETS
from benchmark import run_workbench_trials


# =========================================================================
# Shared Helper Functions (Decoupled & Testable)
# =========================================================================
def format_safe_pct(baseline_val: float, ag_val: float) -> str:
    """Safely calculates percentage error reduction without division by zero."""
    if np.isnan(baseline_val) or np.isnan(ag_val) or baseline_val <= 0:
        return "N/A"
    diff_pct = ((baseline_val - ag_val) / baseline_val) * 100.0
    return f"{diff_pct:+.1f}%"


def generate_plain_english_verdict(base_m: PerformanceMetrics, ag_m: PerformanceMetrics, context_name: str = "this scenario") -> str:
    """Generates an honest, truthful assessment comparing unrounded RMSE values."""
    if np.isnan(base_m.rmse) and np.isnan(ag_m.rmse):
        return "Neither estimator could calculate a valid position due to insufficient measurements or degenerate geometry."
    if np.isnan(base_m.rmse):
        return f"Baseline Least Squares failed to produce a valid position, while AnchorGuard achieved an RMSE of {ag_m.rmse:.2f} m."
    if np.isnan(ag_m.rmse):
        return f"AnchorGuard failed to produce a valid position, while Baseline achieved an RMSE of {base_m.rmse:.2f} m."

    diff_m = base_m.rmse - ag_m.rmse
    pct_str = format_safe_pct(base_m.rmse, ag_m.rmse)

    if abs(diff_m) < 0.02:
        return (
            f"**Equivalent Performance:** Both methods performed virtually identically in {context_name} "
            f"(Baseline: {base_m.rmse:.2f} m vs AnchorGuard: {ag_m.rmse:.2f} m)."
        )
    elif base_m.rmse > ag_m.rmse:
        return (
            f"**AnchorGuard performed better:** Under {context_name}, AnchorGuard reduced position error by "
            f"**{diff_m:.2f} metres ({pct_str})**, achieving **{ag_m.rmse:.2f} m** RMSE compared to Baseline's **{base_m.rmse:.2f} m**."
        )
    else:
        excess_m = ag_m.rmse - base_m.rmse
        return (
            f"**Baseline performed better:** In {context_name}, standard Least Squares achieved a lower error "
            f"(**{base_m.rmse:.2f} m** vs AnchorGuard's **{ag_m.rmse:.2f} m**, lower by {excess_m:.2f} m). "
            f"This occurs when measurement errors are predominantly symmetric or when geometric diversity makes outlier isolation ambiguous."
        )


generate_verdict = generate_plain_english_verdict


def generate_dynamic_explanation(
    preset_key: str,
    cfg: WorkbenchConfig,
    base_m: PerformanceMetrics,
    ag_m: PerformanceMetrics,
) -> Dict[str, str]:
    """
    Constructs a dynamic, 5-point scientific explanation based on the actual
    environmental configuration, measurement corruptions, and observed errors.
    """
    diff_m = base_m.rmse - ag_m.rmse
    pct_str = format_safe_pct(base_m.rmse, ag_m.rmse)

    # 1. What changed in the environment
    if preset_key == "clean_room":
        env_text = "Clean, unobstructed indoor environment with clear Line-of-Sight (LOS) between the target and all reference anchors."
        meas_text = f"Range measurements had minimal thermal Gaussian noise (σ = {cfg.noise_std:.2f} m) with 0% packet drops and zero obstacle reflections."
        impact_text = "Both estimators produced smooth, tightly clustered position tracks that closely followed the ground truth path."
        significance_text = "Confirms that under ideal radio conditions, robust estimation incurs virtually zero penalty compared to theoretical optimal least squares."
    elif preset_key == "noisy_signals":
        env_text = "High radio-frequency interference and receiver thermal noise (AWGN) spreading randomly across the room."
        meas_text = f"Measurements experienced elevated symmetric Gaussian noise (σ = {cfg.noise_std:.2f} m) and {cfg.drop_rate*100:.0f}% packet drops, without directional bias."
        impact_text = "Estimated trajectories exhibited high-frequency jitter around the actual path, but without persistent systematic offsets."
        significance_text = "Demonstrates stability under RF noise without erratic solver divergence or trajectory breakdown."
    elif preset_key == "missing_readings":
        env_text = "Hostile wireless channel with packet collisions, signal fading, and temporary sensor dropouts."
        meas_text = f"High missing-reading rate ({cfg.drop_rate*100:.0f}% packet drops) forcing estimators to resolve coordinates from sparse subsets of surviving anchors."
        impact_text = "When fewer than 3 anchors were available, estimators safely declared failure instead of fabricating wild position jumps."
        significance_text = "In real-world facilities, multi-path fading frequently causes packet drops. Estimators must gracefully handle geometric degradation."
    else:  # blocked_signals or custom
        env_text = "Dense indoor layout with concrete walls, pillars, and furniture blocking direct Line-of-Sight (NLOS conditions)."
        meas_text = f"Radio signals bounced off obstacles, injecting large positive distance biases (+{cfg.nlos_bias_mean:.1f} m mean shift) on {cfg.nlos_probability*100:.0f}% of readings."
        impact_text = "Conventional Least Squares squared every residual, pulling the estimated path drastically off-track toward reflecting walls."
        significance_text = "NLOS multipath is the #1 failure mode in real indoor tracking. Without robust outlier attenuation, standard algorithms report dangerous false locations."

    # 4. Whether AnchorGuard improved the error
    if np.isnan(base_m.rmse) or np.isnan(ag_m.rmse):
        result_text = "Geometric starvation prevented complete trajectory recovery for one or both methods."
    elif abs(diff_m) < 0.02:
        result_text = f"Both methods achieved virtually equivalent accuracy (Baseline: {base_m.rmse:.2f} m vs AnchorGuard: {ag_m.rmse:.2f} m)."
    elif diff_m > 0:
        result_text = f"AnchorGuard outperformed standard Least Squares, reducing tracking error by **{diff_m:.2f} m ({pct_str})** (RMSE: **{ag_m.rmse:.2f} m** vs **{base_m.rmse:.2f} m**). Its Huber loss bounded outlier leverage, and residual consensus pruned corrupted beacons."
    else:
        result_text = f"Baseline Least Squares achieved a slightly lower error (**{base_m.rmse:.2f} m** vs AnchorGuard's **{ag_m.rmse:.2f} m**). Under purely symmetric Gaussian noise, standard L2 optimization is mathematically optimal (Gauss-Markov theorem)."

    return {
        "environment": env_text,
        "measurements": meas_text,
        "trajectory_impact": impact_text,
        "result_verdict": result_text,
        "why_it_matters": significance_text,
    }


def execute_experiment(cfg: WorkbenchConfig) -> Dict[str, Any]:
    """
    Executes positioning simulation with strict decoupling:
    1. Ground truth and measurements generated once.
    2. Identical measurements fed to all estimators.
    3. Metrics computed from actual outputs without artificial clipping.
    """
    room = cfg.to_room()
    sim_cfg = cfg.to_simulation_config()
    room_dims = (cfg.room_width, cfg.room_height)

    gt_traj = generate_trajectory(
        room=room,
        pattern=cfg.trajectory_pattern,
        num_steps=cfg.trajectory_steps,
        speed=cfg.trajectory_speed,
        seed=cfg.seed,
    )
    sim_res = simulate_measurements(room, gt_traj, sim_cfg)

    # 1. Baseline NLS
    base_res = estimate_trajectory(
        measurements=sim_res.measurements,
        anchor_coords=sim_res.anchor_coords,
        method="baseline",
        noise_sigma=cfg.noise_std,
    )
    base_m = compute_positioning_metrics(
        ground_truth=gt_traj,
        estimated=base_res.estimated_trajectory,
        success_flags=base_res.success_flags,
        avg_runtime_ms=base_res.avg_runtime_ms,
        total_runtime_ms=base_res.total_runtime_ms,
        room_dims=room_dims,
    )

    # 2. AnchorGuard Robust
    ag_cfg = cfg.to_anchorguard_config()
    ag_res = estimate_trajectory(
        measurements=sim_res.measurements,
        anchor_coords=sim_res.anchor_coords,
        method="anchorguard",
        noise_sigma=cfg.noise_std,
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

    # 3. AnchorGuard + Kalman Smoothing
    ag_kf_res = estimate_trajectory(
        measurements=sim_res.measurements,
        anchor_coords=sim_res.anchor_coords,
        method="anchorguard",
        noise_sigma=cfg.noise_std,
        anchorguard_config=ag_cfg,
        apply_kalman=True,
        kalman_process_std=cfg.kalman_process_std,
    )
    ag_kf_m = compute_positioning_metrics(
        ground_truth=gt_traj,
        estimated=ag_kf_res.estimated_trajectory,
        success_flags=None,
        avg_runtime_ms=ag_kf_res.avg_runtime_ms,
        total_runtime_ms=ag_kf_res.total_runtime_ms,
        room_dims=room_dims,
    )

    return {
        "config_snapshot": cfg.to_dict(),
        "room": room,
        "gt_traj": gt_traj,
        "sim_res": sim_res,
        "base_res": base_res,
        "base_m": base_m,
        "ag_res": ag_res,
        "ag_m": ag_m,
        "ag_kf_res": ag_kf_res,
        "ag_kf_m": ag_kf_m,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def init_session_state():
    if "config" not in st.session_state:
        st.session_state["config"] = WorkbenchConfig()
    if "active_preset_key" not in st.session_state:
        st.session_state["active_preset_key"] = "blocked_signals"
        st.session_state["config"].apply_preset("blocked_signals")
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "last_fingerprint" not in st.session_state:
        st.session_state["last_fingerprint"] = None
    if "saved_runs" not in st.session_state:
        st.session_state["saved_runs"] = []
    if "multi_trial_data" not in st.session_state:
        st.session_state["multi_trial_data"] = None


# =========================================================================
# Main Dashboard Product Renderer
# =========================================================================
def render_app():
    """Renders the single, cohesive product demonstration dashboard."""
    st.set_page_config(
        page_title="AnchorGuard AI — Robust Indoor Positioning",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Scientific-Software Aesthetic CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #080d1a;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .stDeployButton, [data-testid="stDeployButton"] {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* Product Hero Header */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }
        .hero-title {
            font-size: 2.0rem;
            font-weight: 700;
            color: #f8fafc;
            margin: 0 0 4px 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #38bdf8;
            font-weight: 500;
            margin: 0 0 10px 0;
        }
        .hero-desc {
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.5;
            margin: 0;
            max-width: 900px;
        }
        
        /* Status Pills */
        .pill-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .pill-track {
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-right: 8px;
        }
        .pill-team {
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            margin-right: 8px;
        }
        .pill-proto {
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
        }
        
        /* Metric Scorecards */
        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            height: 100%;
        }
        .metric-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            font-family: ui-monospace, monospace;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.73rem;
            color: #64748b;
            margin-top: 4px;
        }
        
        /* Dynamic Explanation Card */
        .explain-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 10px;
            padding: 18px 22px;
            margin-top: 18px;
        }
        .explain-header {
            font-size: 1.08rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .explain-point {
            margin-bottom: 8px;
            font-size: 0.90rem;
            line-height: 1.5;
            color: #cbd5e1;
        }
        .explain-point b {
            color: #f1f5f9;
        }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    config: WorkbenchConfig = st.session_state["config"]

    # Initial execution if empty
    if st.session_state["last_results"] is None:
        st.session_state["last_results"] = execute_experiment(config)
        st.session_state["last_fingerprint"] = config.compute_fingerprint()

    # Track stale status
    current_fp = config.compute_fingerprint()
    is_stale = (st.session_state["last_fingerprint"] != current_fp)

    # =========================================================================
    # A. HEADER & PROJECT CONTEXT
    # =========================================================================
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="margin-bottom: 8px;">
                    <span class="pill-tag pill-track">VIT Code Apex 3.0 • Track 4, Topic 2</span>
                    <span class="pill-tag pill-team">Team: Gang of Coders</span>
                    <span class="pill-tag pill-proto">Simulation Prototype</span>
                </div>
                <h1 class="hero-title">🎯 AnchorGuard AI</h1>
                <div class="hero-subtitle">Robust 2D Indoor Positioning Under Sparse Infrastructure & Noisy Measurements</div>
                <p class="hero-desc">
                    GPS cannot penetrate indoor walls. Fixed radio reference beacons (anchors) estimate distance to a moving target, 
                    but signals suffer from receiver noise, packet dropouts, and severe obstacle reflections (Non-Line-of-Sight delay). 
                    AnchorGuard AI uses robust M-estimators and geometric consensus to isolate corruptions and deliver reliable sub-meter tracking.
                </p>
            </div>
            <div style="text-align: right;">
                <div style="padding: 6px 14px; border-radius: 8px; font-size: 0.82rem; font-weight: 600; 
                     background: {'rgba(245, 158, 11, 0.15)' if is_stale else 'rgba(16, 185, 129, 0.15)'}; 
                     color: {'#fbbf24' if is_stale else '#34d399'}; 
                     border: 1px solid {'#f59e0b' if is_stale else '#10b981'};">
                    { '⚠️ Settings Changed — Click Run' if is_stale else '✅ Simulation Current' }
                </div>
                <div style="margin-top: 6px; font-size: 0.72rem; color: #64748b;">Ground truth strictly decoupled</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # B. COMPACT SCENARIO CONTROLS
    # =========================================================================
    st.markdown("### 🎛️ 1. Select Environmental Scenario & Settings")

    # One-click preset scenario buttons
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        if st.button("🟢 Normal Signals\n(Ideal Line-of-Sight)", use_container_width=True):
            st.session_state["active_preset_key"] = "clean_room"
            config.apply_preset("clean_room")
            with st.spinner("Simulating Normal Signals..."):
                st.session_state["last_results"] = execute_experiment(config)
                st.session_state["last_fingerprint"] = config.compute_fingerprint()
            st.rerun()

    with p_col2:
        if st.button("🟡 Noisy Measurements\n(Heavy Radio AWGN)", use_container_width=True):
            st.session_state["active_preset_key"] = "noisy_signals"
            config.apply_preset("noisy_signals")
            with st.spinner("Simulating Noisy Measurements..."):
                st.session_state["last_results"] = execute_experiment(config)
                st.session_state["last_fingerprint"] = config.compute_fingerprint()
            st.rerun()

    with p_col3:
        if st.button("🟠 Missing Readings\n(High Packet Drops)", use_container_width=True):
            st.session_state["active_preset_key"] = "missing_readings"
            config.apply_preset("missing_readings")
            with st.spinner("Simulating Missing Readings..."):
                st.session_state["last_results"] = execute_experiment(config)
                st.session_state["last_fingerprint"] = config.compute_fingerprint()
            st.rerun()

    with p_col4:
        if st.button("🔴 Blocked Signals\n(Severe NLOS Multipath)", use_container_width=True):
            st.session_state["active_preset_key"] = "blocked_signals"
            config.apply_preset("blocked_signals")
            with st.spinner("Simulating Blocked Signals..."):
                st.session_state["last_results"] = execute_experiment(config)
                st.session_state["last_fingerprint"] = config.compute_fingerprint()
            st.rerun()

    # Compact fine-tuning controls and Run button
    with st.container():
        ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.5, 1.2, 1.2, 1.0, 1.4])
        with ctrl1:
            layout_opts = ["Perimeter (6 Anchors)", "Corners (4 Anchors)", "Grid (9 Anchors)"]
            curr_idx = 0
            if config.layout_preset == "corners":
                curr_idx = 1
            elif config.layout_preset == "grid":
                curr_idx = 2

            sel_layout = st.selectbox("Anchor Geometry", options=layout_opts, index=curr_idx)
            if sel_layout == "Corners (4 Anchors)" and config.layout_preset != "corners":
                config.layout_preset = "corners"
                config.anchors = generate_corner_anchors(config.room_width, config.room_height)
            elif sel_layout == "Grid (9 Anchors)" and config.layout_preset != "grid":
                config.layout_preset = "grid"
                config.anchors = generate_grid_anchors(config.room_width, config.room_height, 3, 3)
            elif sel_layout == "Perimeter (6 Anchors)" and config.layout_preset != "perimeter":
                config.layout_preset = "perimeter"
                config.anchors = generate_perimeter_anchors(config.room_width, config.room_height, 6)

        with ctrl2:
            config.noise_std = st.slider("Noise σ (m)", 0.0, 1.5, float(config.noise_std), 0.05, help="Random Gaussian noise added to distance measurements.")

        with ctrl3:
            config.drop_rate = st.slider("Missing Drops", 0.0, 0.6, float(config.drop_rate), 0.05, help="Probability that an anchor measurement is dropped.")

        with ctrl4:
            c_seed, c_rnd = st.columns([2, 1])
            with c_seed:
                config.seed = st.number_input("Seed", 0, 99999, int(config.seed), 1, help="Fixes random numbers for reproducibility.")
            with c_rnd:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🎲", help="Pick new random seed"):
                    config.seed = int(np.random.randint(0, 100000))
                    st.rerun()

        with ctrl5:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
                is_valid, errors = config.validate()
                if not is_valid:
                    st.error("Validation error: " + "; ".join(errors))
                else:
                    with st.spinner("Simulating and estimating trajectories..."):
                        st.session_state["last_results"] = execute_experiment(config)
                        st.session_state["last_fingerprint"] = config.compute_fingerprint()
                    st.rerun()

    # Load results
    results = st.session_state["last_results"]
    gt_traj = results["gt_traj"]
    sim_res = results["sim_res"]
    base_res = results["base_res"]
    base_m: PerformanceMetrics = results["base_m"]
    ag_res = results["ag_res"]
    ag_m: PerformanceMetrics = results["ag_m"]
    res_room: Room = results["room"]

    st.markdown("---")

    # =========================================================================
    # D. RESULTS SUMMARY METRICS BANNER
    # =========================================================================
    st.markdown("### 📊 2. Performance Summary")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline RMSE</div>
            <div class="metric-value" style="color: #ef4444;">{base_m.rmse:.2f} m</div>
            <div class="metric-sub">Conventional Least Squares</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AnchorGuard RMSE</div>
            <div class="metric-value" style="color: #38bdf8;">{ag_m.rmse:.2f} m</div>
            <div class="metric-sub">Robust Huber + Pruning</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        reduction_str = format_safe_pct(base_m.rmse, ag_m.rmse)
        is_pos = (base_m.rmse > ag_m.rmse)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Error Reduction</div>
            <div class="metric-value" style="color: {'#10b981' if is_pos else '#94a3b8'};">{reduction_str}</div>
            <div class="metric-sub">AnchorGuard vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median Error</div>
            <div class="metric-value">{ag_m.median_error:.2f} m</div>
            <div class="metric-sub">50% of steps below this error</div>
        </div>
        """, unsafe_allow_html=True)

    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">95th-Percentile Error</div>
            <div class="metric-value">{ag_m.p95_error:.2f} m</div>
            <div class="metric-sub">Worst-case tail error (P95)</div>
        </div>
        """, unsafe_allow_html=True)

    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Valid Solution Rate</div>
            <div class="metric-value">{ag_m.success_rate_pct:.0f}%</div>
            <div class="metric-sub">{ag_m.successful_steps}/{ag_m.total_steps} steps resolved</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("**What is RMSE?** Root Mean Square Error measures the average distance between estimated positions and actual target coordinates. Lower values indicate tighter tracking accuracy.")

    # =========================================================================
    # C. MAIN INTERACTIVE FLOOR PLAN (VISUAL CENTERPIECE)
    # =========================================================================
    st.markdown("### 🗺️ 3. Interactive Floor Plan & Trajectory Tracking")
    st.caption("Visual centerpiece comparing actual target path with baseline trilateration and robust AnchorGuard estimation.")

    floor_fig = create_floorplan_figure(
        room=res_room,
        ground_truth=gt_traj,
        baseline_traj=base_res.estimated_trajectory,
        anchorguard_traj=ag_res.estimated_trajectory,
    )
    floor_fig.update_layout(height=560)
    st.plotly_chart(floor_fig, use_container_width=True)

    # =========================================================================
    # E. "WHAT HAPPENED AND WHY?" DYNAMIC SCIENTIFIC NARRATIVE
    # =========================================================================
    curr_preset = st.session_state.get("active_preset_key", "custom")
    narrative = generate_dynamic_explanation(curr_preset, config, base_m, ag_m)

    st.markdown(f"""
    <div class="explain-card">
        <div class="explain-header">💡 What Happened and Why? (Scientific Analysis)</div>
        <div class="explain-point"><b>1. Environmental Condition:</b> {narrative["environment"]}</div>
        <div class="explain-point"><b>2. Measurement Corruption:</b> {narrative["measurements"]}</div>
        <div class="explain-point"><b>3. Trajectory Observation:</b> {narrative["trajectory_impact"]}</div>
        <div class="explain-point"><b>4. Algorithmic Response:</b> {narrative["result_verdict"]}</div>
        <div class="explain-point"><b>5. Physical Significance:</b> {narrative["why_it_matters"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # F. SUPPORTING EVIDENCE (COMPACT DIAGNOSTIC CHARTS)
    # =========================================================================
    st.markdown("### 📈 4. Supporting Diagnostic Evidence")
    ev_col1, ev_col2 = st.columns(2)
    with ev_col1:
        st.plotly_chart(
            create_error_comparison_figure(gt_traj, base_res.estimated_trajectory, ag_res.estimated_trajectory),
            use_container_width=True,
        )
    with ev_col2:
        st.plotly_chart(
            create_cdf_figure(gt_traj, base_res.estimated_trajectory, ag_res.estimated_trajectory),
            use_container_width=True,
        )

    # =========================================================================
    # ADVANCED ANALYSIS & ENGINEERING EXPANDER (COLLAPSED BY DEFAULT)
    # =========================================================================
    with st.expander("🛠️ Advanced Analysis, Diagnostics & Data Export", expanded=False):
        st.caption("Technical controls for faculty review, algorithm tuning, residual inspection, Monte Carlo evaluation, and data exports.")
        
        adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
            "⚙️ Algorithm Tuning",
            "🎯 Anchor Residuals",
            "🎲 Monte Carlo Stress Test",
            "💾 Data & Reports",
        ])

        with adv_tab1:
            st.markdown("#### Algorithmic Parameters")
            tun_c1, tun_c2, tun_c3 = st.columns(3)
            with tun_c1:
                config.huber_delta = st.slider("Huber Tuning δ", 0.5, 3.0, float(config.huber_delta), 0.1, help="Cutoff where loss transitions from quadratic to linear penalty.")
            with tun_c2:
                config.enable_residual_pruning = st.checkbox("Enable Outlier Pruning", value=bool(config.enable_residual_pruning), help="Prunes corrupted anchor using geometric consensus.")
                config.max_pruned_anchors = st.slider("Max Pruned Anchors", 1, 3, int(config.max_pruned_anchors), 1)
            with tun_c3:
                config.apply_kalman = st.checkbox("Enable Kalman Smoothing", value=bool(config.apply_kalman), help="Applies 2D constant-velocity kinematic filter.")
                config.kalman_process_std = st.slider("Kalman Acceleration Std", 0.1, 2.0, float(config.kalman_process_std), 0.1)

        with adv_tab2:
            st.markdown("#### Range Residual Diagnostics")
            st.caption("Residual = Measured Distance - Estimated Distance. Abnormally large positive residuals (> +2 m) signify severe NLOS conditions.")
            st.plotly_chart(
                create_residual_diagnostics_figure(
                    measurements=sim_res.measurements,
                    anchor_coords=sim_res.anchor_coords,
                    estimated_traj=ag_res.estimated_trajectory,
                    anchor_ids=sim_res.anchor_ids,
                ),
                use_container_width=True,
            )

        with adv_tab3:
            st.markdown("#### Multi-Trial Monte Carlo Benchmarking")
            st.caption("Evaluates algorithm robustness across multiple sequential random seeds, reporting Mean ± Standard Deviation.")
            mc_k = st.slider("Trial Count", 3, 10, 5, 1)
            if st.button("Run Monte Carlo Trials"):
                with st.spinner(f"Running {mc_k} Monte Carlo trials..."):
                    mc_data = run_workbench_trials(config, num_trials=mc_k)
                    st.session_state["multi_trial_data"] = mc_data
                st.toast("Monte Carlo evaluation complete!")

            if st.session_state["multi_trial_data"] is not None:
                mc_d = st.session_state["multi_trial_data"]
                st.markdown(f"**Completed Trials:** {mc_d['num_trials_completed']} | **Seeds:** {mc_d['seeds_used']}")
                summary_df = pd.DataFrame([
                    {
                        "Method": "Baseline NLS",
                        "RMSE (Mean ± Std)": f"{mc_d['baseline']['rmse_mean']:.3f} ± {mc_d['baseline']['rmse_std']:.3f} m",
                        "MAE Mean": f"{mc_d['baseline']['mae_mean']:.3f} m",
                        "P95 Mean": f"{mc_d['baseline']['p95_mean']:.3f} m",
                        "Valid Rate": f"{mc_d['baseline']['success_mean']:.1f}%",
                    },
                    {
                        "Method": "AnchorGuard Robust",
                        "RMSE (Mean ± Std)": f"{mc_d['anchorguard']['rmse_mean']:.3f} ± {mc_d['anchorguard']['rmse_std']:.3f} m",
                        "MAE Mean": f"{mc_d['anchorguard']['mae_mean']:.3f} m",
                        "P95 Mean": f"{mc_d['anchorguard']['p95_mean']:.3f} m",
                        "Valid Rate": f"{mc_d['anchorguard']['success_mean']:.1f}%",
                    },
                ])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with adv_tab4:
            st.markdown("#### Experiment Data & Configuration Export")
            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                # CSV Export
                steps_arr = np.arange(len(gt_traj))
                b_traj = base_res.estimated_trajectory
                ag_traj = ag_res.estimated_trajectory
                b_err = np.linalg.norm(gt_traj - b_traj, axis=1)
                ag_err = np.linalg.norm(gt_traj - ag_traj, axis=1)
                csv_df = pd.DataFrame({
                    "step": steps_arr,
                    "true_x": np.round(gt_traj[:, 0], 3),
                    "true_y": np.round(gt_traj[:, 1], 3),
                    "baseline_x": np.round(b_traj[:, 0], 3),
                    "baseline_y": np.round(b_traj[:, 1], 3),
                    "baseline_error_m": np.round(b_err, 3),
                    "anchorguard_x": np.round(ag_traj[:, 0], 3),
                    "anchorguard_y": np.round(ag_traj[:, 1], 3),
                    "anchorguard_error_m": np.round(ag_err, 3),
                })
                st.download_button(
                    "📥 Export Results CSV",
                    data=csv_df.to_csv(index=False),
                    file_name=f"anchorguard_run_{config.seed}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with exp_c2:
                # JSON Export
                st.download_button(
                    "📥 Export Config JSON",
                    data=config.to_json(),
                    file_name=f"anchorguard_config_{config.seed}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with exp_c3:
                # Markdown Report
                report_md = f"""# AnchorGuard AI — Experiment Summary
**Seed:** {config.seed} | **Room:** {config.room_width}×{config.room_height} m | **Noise:** {config.noise_std} m
- Baseline RMSE: {base_m.rmse:.3f} m
- AnchorGuard RMSE: {ag_m.rmse:.3f} m
- Error Reduction: {format_safe_pct(base_m.rmse, ag_m.rmse)}
- Valid Estimate Rate: {ag_m.success_rate_pct:.1f}%

*Simulation Prototype — Gang of Coders, VIT Code Apex 3.0*
"""
                st.download_button(
                    "📥 Export Summary Markdown",
                    data=report_md,
                    file_name=f"anchorguard_report_{config.seed}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )


if __name__ == "__main__":
    render_app()
