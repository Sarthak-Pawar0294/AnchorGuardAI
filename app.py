"""
app.py - AnchorGuard AI 2.0: Industry-Style Indoor Positioning Engineering Workbench.
VIT Code Apex 3.0, Track 4, Topic 2: AI-Based Indoor Positioning with Sparse Infrastructure and Noisy Measurements.
Team: Gang of Coders.

Six Navigation Sections:
1. Overview - Live room preview, primary metrics, simulation status (current vs stale), prominent Run button.
2. Environment Studio - Room dimensions, layout presets (Perimeter, Corners, Grid), coordinate editor, SVD geometry diagnostics.
3. Measurement Lab - Noise, packet drop rate, NLOS bias, anchor outages, reproducibility seed, presets, fault model disclosure.
4. Algorithm Lab - Baseline NLS, AnchorGuard Robust Huber, AnchorGuard + Kalman, parameter tuning, reset-to-recommended.
5. Evaluation Lab - Comprehensive metrics, out-of-room rate, error vs time, CDF, anchor residuals, repeated Monte Carlo trials.
6. Data & Reports - JSON configuration export/import, CSV results export, saved run history, structured engineering report.
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

    if abs(diff_m) < 0.01:
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


def init_session_state():
    if "config" not in st.session_state:
        st.session_state["config"] = WorkbenchConfig()
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "last_fingerprint" not in st.session_state:
        st.session_state["last_fingerprint"] = None
    if "saved_runs" not in st.session_state:
        st.session_state["saved_runs"] = []
    if "multi_trial_data" not in st.session_state:
        st.session_state["multi_trial_data"] = None


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


def render_app():
    """Main Streamlit Workbench UI renderer."""
    st.set_page_config(
        page_title="AnchorGuard AI 2.0 — Engineering Workbench",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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
        h1, h2, h3, h4 {
            color: #f8fafc !important;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        .brand-header {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 18px 24px;
            margin-bottom: 20px;
        }
        .badge-sub {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 8px;
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-track {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 8px;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .sim-tag {
            display: inline-block;
            padding: 3px 10px;
            background: #1e293b;
            color: #94a3b8;
            border-radius: 6px;
            font-size: 0.75rem;
            font-family: monospace;
            border: 1px solid #334155;
        }
        .status-current {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid #10b981;
            font-weight: 600;
            font-size: 0.82rem;
        }
        .status-stale {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 6px;
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid #f59e0b;
            font-weight: 600;
            font-size: 0.82rem;
        }
        .stale-banner {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 12px 18px;
            color: #fbbf24;
            margin-bottom: 16px;
            font-size: 0.9rem;
        }
        .metric-card {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
            height: 100%;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        .metric-num {
            font-size: 1.7rem;
            font-weight: 700;
            font-family: ui-monospace, monospace;
            color: #f8fafc;
        }
        .metric-caption {
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 4px;
        }
        .takeaway-box {
            background: #0f172a;
            border-left: 4px solid #10b981;
            border-top: 1px solid #1e293b;
            border-right: 1px solid #1e293b;
            border-bottom: 1px solid #1e293b;
            border-radius: 6px;
            padding: 14px 18px;
            margin: 14px 0;
            font-size: 0.92rem;
            line-height: 1.5;
        }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    config: WorkbenchConfig = st.session_state["config"]

    # Initial simulation run if none exists
    if st.session_state["last_results"] is None:
        st.session_state["last_results"] = execute_experiment(config)
        st.session_state["last_fingerprint"] = config.compute_fingerprint()

    # Determine if current results match the active config
    current_fp = config.compute_fingerprint()
    is_stale = (st.session_state["last_fingerprint"] != current_fp)

    # Sidebar Navigation & Master Run Button
    with st.sidebar:
        st.markdown("### 🎯 Workbench Navigation")
        section = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Environment Studio",
                "Measurement Lab",
                "Algorithm Lab",
                "Evaluation Lab",
                "Data & Reports",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### ⚡ Experiment Execution")
        
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            is_valid, errors = config.validate()
            if not is_valid:
                st.error("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))
            else:
                with st.spinner("Simulating and estimating trajectories..."):
                    st.session_state["last_results"] = execute_experiment(config)
                    st.session_state["last_fingerprint"] = config.compute_fingerprint()
                    is_stale = False
                st.rerun()

        if is_stale:
            st.markdown('<div class="status-stale">⚠️ Status: Outdated (Config Changed)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-current">✅ Status: Current (Up to Date)</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"**Room:** {config.room_width:.0f}×{config.room_height:.0f} m | **Anchors:** {len(config.anchors)} | **Seed:** {config.seed}")
        st.caption("AnchorGuard AI 2.0 • Simulation Prototype")

    # Header Banner
    st.markdown("""
    <div class="brand-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="margin-bottom: 6px;">
                    <span class="badge-track">VIT Code Apex 3.0 • Track 4, Topic 2</span>
                    <span class="badge-sub">Team: Gang of Coders</span>
                </div>
                <h1 style="margin: 0; font-size: 1.85rem; color: #f8fafc;">AnchorGuard AI 2.0</h1>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                    Industry-Style Indoor Positioning Engineering Workbench for Sparse Infrastructure & Noisy Measurements.
                </p>
            </div>
            <div style="text-align: right;">
                <span class="sim-tag">SIMULATION-BASED PROTOTYPE</span>
                <div style="margin-top: 6px; font-size: 0.75rem; color: #64748b;">Ground truth strictly decoupled</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state["last_results"]
    gt_traj = results["gt_traj"]
    sim_res = results["sim_res"]
    base_res = results["base_res"]
    base_m: PerformanceMetrics = results["base_m"]
    ag_res = results["ag_res"]
    ag_m: PerformanceMetrics = results["ag_m"]
    ag_kf_res = results["ag_kf_res"]
    ag_kf_m: PerformanceMetrics = results["ag_kf_m"]
    res_room: Room = results["room"]

    # =========================================================================
    # SECTION 1: OVERVIEW
    # =========================================================================
    if section == "Overview":
        st.markdown("## 📊 Overview")

        if is_stale:
            st.markdown("""
            <div class="stale-banner">
                ⚠️ <b>Results Outdated:</b> Configuration parameters have been modified since the last simulation run.
                The metrics and plots below reflect previous run settings. Click <b>Run Simulation</b> in the sidebar to update.
            </div>
            """, unsafe_allow_html=True)

        active_anchors_count = sum(1 for a in config.anchors if a.is_active and a.id not in config.outage_anchor_ids)
        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
        col_c1.metric("Room Dimensions", f"{config.room_width:.0f} × {config.room_height:.0f} m")
        col_c2.metric("Active Anchors", f"{active_anchors_count} of {len(config.anchors)}")
        col_c3.metric("Noise Std (σ)", f"{config.noise_std:.2f} m")
        col_c4.metric("NLOS Probability", f"{config.nlos_probability * 100:.0f}%")
        col_c5.metric("Random Seed", f"{config.seed}")

        st.markdown("---")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Baseline RMSE</div>
                <div class="metric-num" style="color: #ef4444;">{base_m.rmse:.2f} m</div>
                <div class="metric-caption">Nonlinear Least Squares</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AnchorGuard RMSE</div>
                <div class="metric-num" style="color: #38bdf8;">{ag_m.rmse:.2f} m</div>
                <div class="metric-caption">Robust Huber + Pruning</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Error Reduction</div>
                <div class="metric-num" style="color: #10b981;">{format_safe_pct(base_m.rmse, ag_m.rmse)}</div>
                <div class="metric-caption">AnchorGuard vs Baseline</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Valid Estimate Rate</div>
                <div class="metric-num">{ag_m.success_rate_pct:.1f}%</div>
                <div class="metric-caption">{ag_m.successful_steps} / {ag_m.total_steps} steps</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Out-of-Room Rate</div>
                <div class="metric-num" style="color: {'#ef4444' if ag_m.out_of_room_rate_pct > 0 else '#34d399'};">{ag_m.out_of_room_rate_pct:.1f}%</div>
                <div class="metric-caption">{ag_m.out_of_room_steps} estimates beyond room</div>
            </div>
            """, unsafe_allow_html=True)

        verdict = generate_plain_english_verdict(base_m, ag_m, "this configuration")
        st.markdown(f"""
        <div class="takeaway-box">
            {verdict}
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            create_floorplan_figure(
                room=res_room,
                ground_truth=gt_traj,
                baseline_traj=base_res.estimated_trajectory,
                anchorguard_traj=ag_res.estimated_trajectory,
            ),
            use_container_width=True,
        )

    # =========================================================================
    # SECTION 2: ENVIRONMENT STUDIO
    # =========================================================================
    elif section == "Environment Studio":
        st.markdown("## 🏛️ Environment Studio")
        st.caption("Configure room dimensions, anchor layouts, individual coordinate placement, and target trajectories.")

        t1, t2 = st.tabs(["📐 Room & Anchors", "🚶 Target Trajectory"])

        with t1:
            c_dim1, c_dim2, c_preset = st.columns([1, 1, 1.5])
            with c_dim1:
                new_w = st.slider("Room Width (m)", 10.0, 40.0, float(config.room_width), 1.0)
            with c_dim2:
                new_h = st.slider("Room Height (m)", 8.0, 30.0, float(config.room_height), 1.0)
            with c_preset:
                layout_choice = st.selectbox(
                    "Anchor Layout Preset",
                    options=["perimeter", "corners", "grid", "custom"],
                    index=["perimeter", "corners", "grid", "custom"].index(config.layout_preset) if config.layout_preset in ["perimeter", "corners", "grid", "custom"] else 0,
                    help="Applies a standard geometric anchor distribution.",
                )

            if new_w != config.room_width or new_h != config.room_height:
                config.room_width = new_w
                config.room_height = new_h

            if layout_choice != config.layout_preset:
                config.layout_preset = layout_choice
                if layout_choice == "corners":
                    config.anchors = generate_corner_anchors(config.room_width, config.room_height)
                elif layout_choice == "perimeter":
                    config.anchors = generate_perimeter_anchors(config.room_width, config.room_height, count=6)
                elif layout_choice == "grid":
                    config.anchors = generate_grid_anchors(config.room_width, config.room_height, rows=3, cols=3)

            st.markdown("#### 📍 Anchor Coordinate Management")
            st.caption("Edit coordinates directly. Changes update actual simulation geometry upon next run.")

            anchor_data = [
                {"id": a.id, "name": a.name, "x": a.x, "y": a.y, "is_active": a.is_active}
                for a in config.anchors
            ]
            df_anchors = pd.DataFrame(anchor_data)

            edited_df = st.data_editor(
                df_anchors,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "id": st.column_config.NumberColumn("Anchor ID", disabled=True),
                    "name": st.column_config.TextColumn("Name"),
                    "x": st.column_config.NumberColumn("X (m)", step=0.1, format="%.2f"),
                    "y": st.column_config.NumberColumn("Y (m)", step=0.1, format="%.2f"),
                    "is_active": st.column_config.CheckboxColumn("Active"),
                },
                key="anchor_editor",
            )

            new_anchors = []
            for idx, row in edited_df.iterrows():
                new_anchors.append(AnchorConfig(
                    id=int(row["id"]) if not pd.isna(row["id"]) else int(idx),
                    name=str(row["name"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    is_active=bool(row["is_active"]),
                ))
            config.anchors = new_anchors

            st.markdown("#### 🔍 Spatial Geometry Diagnostics")
            active_coords = np.array([[a.x, a.y] for a in config.anchors if a.is_active and a.id not in config.outage_anchor_ids])
            is_geom_valid, aspect_ratio, geom_msg = check_geometry_conditioning(active_coords)

            g_col1, g_col2, g_col3 = st.columns(3)
            g_col1.metric("Active Anchors", f"{len(active_coords)}")
            g_col2.metric("SVD Aspect Ratio (σ₂/σ₁)", f"{aspect_ratio:.3f}" if is_geom_valid else "Degenerate")
            g_col3.metric("Geometry Condition", "✅ Good" if is_geom_valid else "❌ Ill-Conditioned")
            st.caption(f"**Diagnostic note:** {geom_msg}")

        with t2:
            st.markdown("#### 🎯 Target Trajectory Settings")
            c_patt, c_steps, c_spd = st.columns(3)
            with c_patt:
                config.trajectory_pattern = st.selectbox(
                    "Trajectory Pattern",
                    options=["smooth_loop", "patrol_perimeter", "linear_crossing", "random_walk"],
                    index=["smooth_loop", "patrol_perimeter", "linear_crossing", "random_walk"].index(config.trajectory_pattern),
                    help="Mathematical path used for target motion.",
                )
            with c_steps:
                config.trajectory_steps = st.slider("Duration (Time Steps)", 20, 300, config.trajectory_steps, 10)
            with c_spd:
                config.trajectory_speed = st.slider("Nominal Speed Factor", 0.5, 3.0, float(config.trajectory_speed), 0.1)

        temp_room = config.to_room()
        preview_gt = generate_trajectory(temp_room, pattern=config.trajectory_pattern, num_steps=config.trajectory_steps, seed=config.seed)
        st.plotly_chart(
            create_floorplan_figure(room=temp_room, ground_truth=preview_gt, baseline_traj=None, anchorguard_traj=None),
            use_container_width=True,
        )

    # =========================================================================
    # SECTION 3: MEASUREMENT LAB
    # =========================================================================
    elif section == "Measurement Lab":
        st.markdown("## 🔬 Measurement Lab")
        st.caption("Configure range noise, packet loss dropout, NLOS multipath bias, and anchor outages.")

        st.markdown("#### ⚡ Experiment Presets")
        p_cols = st.columns(4)
        for i, (p_id, p_obj) in enumerate(EXPERIMENT_PRESETS.items()):
            with p_cols[i]:
                if st.button(p_obj.name, key=f"btn_{p_id}", use_container_width=True):
                    config.apply_preset(p_id)
                    st.toast(f"Applied preset: {p_obj.name}")
                    st.rerun()

        st.markdown("---")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 📡 Radio Channel Fault Models")
            config.noise_std = st.slider(
                "Measurement Noise Std σ (m)",
                0.0, 2.0, float(config.noise_std), 0.05,
                help="Additive Gaussian noise simulating thermal AWGN noise in distance readings."
            )
            config.drop_rate = st.slider(
                "Missing-Reading Rate (Packet Drops)",
                0.0, 0.8, float(config.drop_rate), 0.05,
                help="Probability that a sensor reading is dropped completely (missing measurement)."
            )
            config.nlos_probability = st.slider(
                "NLOS Contamination Probability",
                0.0, 0.8, float(config.nlos_probability), 0.05,
                help="Probability that a line-of-sight signal is obstructed, introducing a positive distance bias."
            )
            config.nlos_bias_mean = st.slider(
                "NLOS Exponential Mean Bias (m)",
                0.5, 8.0, float(config.nlos_bias_mean), 0.5,
                help="Mean positive shift added to ranging measurements under NLOS conditions."
            )

        with col_m2:
            st.markdown("#### 🔌 Anchor Sensor Outages & Reproducibility")
            active_ids = [a.id for a in config.anchors]
            config.outage_anchor_ids = st.multiselect(
                "Disabled / Blackout Anchor IDs",
                options=active_ids,
                default=[i for i in config.outage_anchor_ids if i in active_ids],
                help="Specific anchors turned off completely during the experiment.",
            )

            st.markdown("#### 🎲 Reproducibility Management")
            c_seed, c_rnd = st.columns([2, 1])
            with c_seed:
                config.seed = st.number_input("Random Seed", 0, 999999, int(config.seed), 1, help="Fixed seed guarantees identical noise across runs.")
            with c_rnd:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🎲 Randomize"):
                    config.seed = int(np.random.randint(0, 100000))
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 📋 Fault Model Disclosure & Physical Realism")
        st.info(r"""
        **Implemented Fault Models in Simulator:**
        - ✅ **Additive White Gaussian Noise (AWGN):** Modeled as $\eta \sim \mathcal{N}(0, \sigma^2)$ on all ranges.
        - ✅ **Independent Packet Dropout:** Modeled as Bernoulli drop mask with probability $p_{\text{drop}}$.
        - ✅ **Positive NLOS Distance Shift:** Modeled as exponential positive multipath bias $b \ge 0$.
        - ✅ **Anchor Outages:** Specific radio beacons permanently disabled.

        **Physical Radio Limitations (Not Modeled):**
        - ⚠️ *Ray-tracing diffraction & reflection geometry is not modeled; NLOS biases are simulated statistically.*
        - ⚠️ *Antenna polarization, radiation patterns, and human-body shadow fading are not modeled.*
        - ⚠️ *Clock synchronization drift and phase noise are not modeled.*
        """)

    # =========================================================================
    # SECTION 4: ALGORITHM LAB
    # =========================================================================
    elif section == "Algorithm Lab":
        st.markdown("## ⚙️ Algorithm Lab")
        st.caption("Select and configure positioning algorithms and robustness parameters.")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### 🧮 Estimator Selection")
            eval_baseline = st.checkbox("Baseline Nonlinear Least Squares (Linear Loss)", value=True, disabled=True, help="Standard L2 solver without outlier downweighting.")
            eval_ag = st.checkbox("AnchorGuard Robust Estimator (Huber Loss + Outlier Pruning)", value=True, disabled=True, help="M-estimator attenuating large residuals with iterative pruning.")
            config.apply_kalman = st.checkbox("Optional Kalman Temporal Smoothing", value=bool(config.apply_kalman), help="2D constant-velocity kinematic filter applied across steps.")

            st.markdown("#### 🛡️ Robust Loss Parameters")
            config.huber_delta = st.slider(
                "Huber Tuning Parameter δ",
                0.5, 3.5, float(config.huber_delta), 0.1,
                help="Residual cutoff where loss transitions from quadratic (Gaussian-optimal) to linear (L1 robust). Standard 95% asymptotic efficiency is δ = 1.345."
            )

        with col_a2:
            st.markdown("#### 🔍 NLOS Residual Pruning")
            config.enable_residual_pruning = st.checkbox(
                "Enable Iterative Outlier Pruning",
                value=bool(config.enable_residual_pruning),
                help="Performs leave-one-out geometric consensus: prunes the anchor that drops the fit cost by > 35%."
            )
            config.max_pruned_anchors = st.slider(
                "Max Pruned Anchors Per Step",
                1, 3, int(config.max_pruned_anchors), 1,
                help="Upper bound on anchors pruned per step to prevent geometric starvation."
            )

            st.markdown("#### 📈 Kalman Filter Parameters")
            config.kalman_process_std = st.slider(
                "Kalman Process Acceleration Std σ_w (m/s²)",
                0.1, 2.0, float(config.kalman_process_std), 0.1,
                help="Kinematic process noise allowance for target acceleration and direction shifts."
            )

        if st.button("🔄 Reset to Recommended Settings"):
            config.huber_delta = 1.345
            config.enable_residual_pruning = True
            config.max_pruned_anchors = 2
            config.apply_kalman = False
            config.kalman_process_std = 0.5
            st.toast("Algorithm settings reset to recommended defaults.")
            st.rerun()

    # =========================================================================
    # SECTION 5: EVALUATION LAB
    # =========================================================================
    elif section == "Evaluation Lab":
        st.markdown("## 📈 Evaluation Lab")
        st.caption("Quantitative positioning diagnostics, residual analysis, error CDF, and repeated-trial Monte Carlo evaluations.")

        tab_single, tab_multi = st.tabs(["🔬 Single-Run Diagnostics", "📊 Repeated-Trial Comparisons (Monte Carlo)"])

        with tab_single:
            st.markdown("#### 📋 Comprehensive Performance Metrics")
            m_data = {
                "Metric": [
                    "RMSE (Root Mean Square Error)",
                    "MAE (Mean Absolute Error)",
                    "Median Error (50th percentile)",
                    "P95 Error (95th percentile)",
                    "Max Error",
                    "Valid Position Estimate Rate",
                    "Out-of-Room Estimate Rate",
                    "Avg Runtime Per Step",
                    "Total Runtime (Complete Run)",
                ],
                "Baseline NLS": [
                    f"{base_m.rmse:.3f} m",
                    f"{base_m.mae:.3f} m",
                    f"{base_m.median_error:.3f} m",
                    f"{base_m.p95_error:.3f} m",
                    f"{base_m.max_error:.3f} m",
                    f"{base_m.success_rate_pct:.1f}% ({base_m.successful_steps}/{base_m.total_steps})",
                    f"{base_m.out_of_room_rate_pct:.1f}% ({base_m.out_of_room_steps} steps)",
                    f"{base_m.avg_runtime_ms:.2f} ms",
                    f"{base_m.total_runtime_ms:.1f} ms",
                ],
                "AnchorGuard Robust": [
                    f"{ag_m.rmse:.3f} m",
                    f"{ag_m.mae:.3f} m",
                    f"{ag_m.median_error:.3f} m",
                    f"{ag_m.p95_error:.3f} m",
                    f"{ag_m.max_error:.3f} m",
                    f"{ag_m.success_rate_pct:.1f}% ({ag_m.successful_steps}/{ag_m.total_steps})",
                    f"{ag_m.out_of_room_rate_pct:.1f}% ({ag_m.out_of_room_steps} steps)",
                    f"{ag_m.avg_runtime_ms:.2f} ms",
                    f"{ag_m.total_runtime_ms:.1f} ms",
                ],
                "AnchorGuard + Kalman": [
                    f"{ag_kf_m.rmse:.3f} m",
                    f"{ag_kf_m.mae:.3f} m",
                    f"{ag_kf_m.median_error:.3f} m",
                    f"{ag_kf_m.p95_error:.3f} m",
                    f"{ag_kf_m.max_error:.3f} m",
                    f"{ag_kf_m.success_rate_pct:.1f}%",
                    f"{ag_kf_m.out_of_room_rate_pct:.1f}%",
                    f"{ag_kf_m.avg_runtime_ms:.2f} ms",
                    f"{ag_kf_m.total_runtime_ms:.1f} ms",
                ],
            }
            st.dataframe(pd.DataFrame(m_data), use_container_width=True, hide_index=True)
            st.caption(
                "**Metric Denominator Definitions:** "
                "Valid Position Estimate Rate = (N_valid / N_total) × 100%. "
                "Out-of-Room Estimate Rate = (N_out_of_room / N_valid) × 100%."
            )

            st.markdown("---")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(
                    create_error_comparison_figure(gt_traj, base_res.estimated_trajectory, ag_res.estimated_trajectory),
                    use_container_width=True,
                )
            with col_v2:
                st.plotly_chart(
                    create_cdf_figure(gt_traj, base_res.estimated_trajectory, ag_res.estimated_trajectory),
                    use_container_width=True,
                )

            st.markdown("#### 🎯 Anchor Range Residual Diagnostics")
            st.caption("Residual = Observed Distance - Estimated Distance. Positive residuals indicate severe NLOS bias.")
            st.plotly_chart(
                create_residual_diagnostics_figure(
                    measurements=sim_res.measurements,
                    anchor_coords=sim_res.anchor_coords,
                    estimated_traj=ag_res.estimated_trajectory,
                    anchor_ids=sim_res.anchor_ids,
                ),
                use_container_width=True,
            )

        with tab_multi:
            st.markdown("#### 🎲 Repeated Monte Carlo Trials")
            st.caption("Executes multiple runs using sequential seeds (Seed, Seed+1, ...) to evaluate mean and standard deviation.")

            mc_col1, mc_col2 = st.columns([1, 2])
            with mc_col1:
                num_trials = st.slider("Number of Trials", 2, 15, 5, 1)
                run_mc = st.button("🚀 Run Monte Carlo Trials", type="primary", use_container_width=True)

            if run_mc:
                with st.spinner(f"Running {num_trials} Monte Carlo trials..."):
                    mc_data = run_workbench_trials(config, num_trials=num_trials)
                    st.session_state["multi_trial_data"] = mc_data
                st.toast("Monte Carlo trials completed!")

            mc_res = st.session_state["multi_trial_data"]
            if mc_res is not None:
                st.markdown(f"**Completed Trials:** {mc_res['num_trials_completed']} | **Seeds:** {mc_res['seeds_used']}")

                mc_summary = [
                    {
                        "Method": "Baseline NLS",
                        "RMSE (Mean ± Std)": f"{mc_res['baseline']['rmse_mean']:.3f} ± {mc_res['baseline']['rmse_std']:.3f} m",
                        "MAE Mean": f"{mc_res['baseline']['mae_mean']:.3f} m",
                        "P95 Mean": f"{mc_res['baseline']['p95_mean']:.3f} m",
                        "Success Rate Mean": f"{mc_res['baseline']['success_mean']:.1f}%",
                        "Out-of-Room Mean": f"{mc_res['baseline']['out_of_room_mean']:.1f}%",
                    },
                    {
                        "Method": "AnchorGuard Robust",
                        "RMSE (Mean ± Std)": f"{mc_res['anchorguard']['rmse_mean']:.3f} ± {mc_res['anchorguard']['rmse_std']:.3f} m",
                        "MAE Mean": f"{mc_res['anchorguard']['mae_mean']:.3f} m",
                        "P95 Mean": f"{mc_res['anchorguard']['p95_mean']:.3f} m",
                        "Success Rate Mean": f"{mc_res['anchorguard']['success_mean']:.1f}%",
                        "Out-of-Room Mean": f"{mc_res['anchorguard']['out_of_room_mean']:.1f}%",
                    },
                    {
                        "Method": "AnchorGuard + Kalman",
                        "RMSE (Mean ± Std)": f"{mc_res['anchorguard_kalman']['rmse_mean']:.3f} ± {mc_res['anchorguard_kalman']['rmse_std']:.3f} m",
                        "MAE Mean": f"{mc_res['anchorguard_kalman']['mae_mean']:.3f} m",
                        "P95 Mean": f"{mc_res['anchorguard_kalman']['p95_mean']:.3f} m",
                        "Success Rate Mean": f"{mc_res['anchorguard_kalman']['success_mean']:.1f}%",
                        "Out-of-Room Mean": f"{mc_res['anchorguard_kalman']['out_of_room_mean']:.1f}%",
                    },
                ]
                st.dataframe(pd.DataFrame(mc_summary), use_container_width=True, hide_index=True)

                with st.expander("🔍 View Per-Trial Breakdown Records"):
                    st.dataframe(pd.DataFrame(mc_res["trial_records"]), use_container_width=True, hide_index=True)

    # =========================================================================
    # SECTION 6: DATA & REPORTS
    # =========================================================================
    elif section == "Data & Reports":
        st.markdown("## 💾 Data & Reports")
        st.caption("Export configurations, time-series data, run comparisons, and generate technical reports.")

        t_cfg, t_data, t_runs, t_rep = st.tabs(["⚙️ Config JSON", "📊 Trajectory CSV", "🗂️ Run History", "📑 Engineering Report"])

        with t_cfg:
            st.markdown("#### 📥 Export Active Configuration")
            cfg_json = config.to_json()
            st.download_button(
                label="💾 Download Configuration (JSON)",
                data=cfg_json,
                file_name=f"anchorguard_config_seed{config.seed}.json",
                mime="application/json",
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("#### 📤 Import Configuration")
            uploaded_cfg = st.file_uploader("Upload Configuration JSON", type=["json"])
            if uploaded_cfg is not None:
                try:
                    raw_text = uploaded_cfg.read().decode("utf-8")
                    loaded_cfg = WorkbenchConfig.from_json(raw_text)
                    is_val, val_errs = loaded_cfg.validate()
                    if not is_val:
                        st.error("Uploaded configuration failed validation:\n" + "\n".join(f"- {e}" for e in val_errs))
                    else:
                        st.session_state["config"] = loaded_cfg
                        st.success("Configuration loaded successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse configuration file: {e}")

        with t_data:
            st.markdown("#### 📥 Export Simulation Time-Series Data (CSV)")
            st.caption("Exports step-by-step Ground Truth, Measurements, Baseline Estimates, AnchorGuard Estimates, and Errors.")

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
                "baseline_status": [s.value for s in base_res.statuses],
                "anchorguard_status": [s.value for s in ag_res.statuses],
            })

            st.dataframe(csv_df.head(10), use_container_width=True)
            st.download_button(
                label="💾 Download Results Data (CSV)",
                data=csv_df.to_csv(index=False),
                file_name=f"anchorguard_data_seed{config.seed}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with t_runs:
            st.markdown("#### 🗂️ Session Run History & Comparisons")
            c_name, c_save = st.columns([3, 1])
            with c_name:
                run_name = st.text_input("Run Label", value=f"Run {len(st.session_state['saved_runs']) + 1}: Noise={config.noise_std} NLOS={config.nlos_probability}")
            with c_save:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("Save Run to History"):
                    st.session_state["saved_runs"].append({
                        "name": run_name,
                        "timestamp": results["timestamp"],
                        "noise_std": config.noise_std,
                        "nlos_prob": config.nlos_probability,
                        "baseline_rmse": base_m.rmse,
                        "anchorguard_rmse": ag_m.rmse,
                        "improvement": format_safe_pct(base_m.rmse, ag_m.rmse),
                    })
                    st.toast("Run saved to history!")

            if st.session_state["saved_runs"]:
                st.dataframe(pd.DataFrame(st.session_state["saved_runs"]), use_container_width=True)
            else:
                st.caption("No runs saved in session history yet.")

        with t_rep:
            st.markdown("#### 📑 Technical Engineering Report")
            report_text = f"""# AnchorGuard AI 2.0 — Engineering Workbench Technical Report
**Generated:** {results["timestamp"]}
**System Version:** AnchorGuard AI 2.0 (Prototype)
**Evaluation Track:** VIT Code Apex 3.0, Track 4, Topic 2
**Team:** Gang of Coders

---

## 1. Environment & Geometry Configuration
- **Room Width:** {config.room_width:.1f} m
- **Room Height:** {config.room_height:.1f} m
- **Anchor Count:** {len(config.anchors)} total ({sum(1 for a in config.anchors if a.is_active)} active)
- **Trajectory Pattern:** {config.trajectory_pattern}
- **Trajectory Steps:** {config.trajectory_steps}

## 2. Measurement Channel Fault Settings
- **Measurement Noise Std (σ):** {config.noise_std:.2f} m
- **Packet Dropout Rate:** {config.drop_rate * 100:.1f}%
- **NLOS Contamination Probability:** {config.nlos_probability * 100:.1f}%
- **NLOS Mean Exponential Bias:** {config.nlos_bias_mean:.2f} m
- **Random Seed:** {config.seed}

## 3. Algorithmic Parameters
- **Huber Loss Tuning (δ):** {config.huber_delta:.3f}
- **Iterative Outlier Pruning:** {config.enable_residual_pruning}
- **Max Pruned Anchors:** {config.max_pruned_anchors}
- **Kalman Smoothing Enabled:** {config.apply_kalman}

## 4. Empirical Performance Evaluation
| Metric | Baseline NLS | AnchorGuard Robust | AG + Kalman |
| :--- | :--- | :--- | :--- |
| **RMSE (m)** | {base_m.rmse:.3f} | {ag_m.rmse:.3f} | {ag_kf_m.rmse:.3f} |
| **MAE (m)** | {base_m.mae:.3f} | {ag_m.mae:.3f} | {ag_kf_m.mae:.3f} |
| **Median Error (m)** | {base_m.median_error:.3f} | {ag_m.median_error:.3f} | {ag_kf_m.median_error:.3f} |
| **P95 Error (m)** | {base_m.p95_error:.3f} | {ag_m.p95_error:.3f} | {ag_kf_m.p95_error:.3f} |
| **Valid Estimate Rate (%)** | {base_m.success_rate_pct:.1f}% | {ag_m.success_rate_pct:.1f}% | {ag_kf_m.success_rate_pct:.1f}% |
| **Out-of-Room Rate (%)** | {base_m.out_of_room_rate_pct:.1f}% | {ag_m.out_of_room_rate_pct:.1f}% | {ag_kf_m.out_of_room_rate_pct:.1f}% |
| **Avg Runtime (ms)** | {base_m.avg_runtime_ms:.2f} | {ag_m.avg_runtime_ms:.2f} | {ag_kf_m.avg_runtime_ms:.2f} |

**Verdict:** {generate_plain_english_verdict(base_m, ag_m, "this configuration")}

---

## 5. Explicit Limitations & Disclaimers
1. **Simulation-Based Prototype:** Results are derived from synthetic 2D Euclidean range simulations and statistical fault injections. Real-world wireless multipath, antenna cross-polarization, and human-body shadow fading require physical RF measurement validation.
2. **Ground Truth Decoupling:** Ground truth trajectories are strictly decoupled from estimators. No artificial position clipping or fictitious benchmark data has been introduced.
"""
            st.text_area("Report Preview", value=report_text, height=300)
            st.download_button(
                label="💾 Download Engineering Report (Markdown)",
                data=report_text,
                file_name=f"anchorguard_report_seed{config.seed}.md",
                mime="text/markdown",
                use_container_width=True,
            )


if __name__ == "__main__":
    render_app()
