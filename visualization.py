"""
visualization.py - Interactive Plotly Floor Plan & Diagnostics for AnchorGuard AI.

Renders:
1. Floor plan with perimeter room boundaries, anchor locations (active vs disabled/outage),
   ground-truth trajectory, baseline estimates, and AnchorGuard robust estimates.
2. Step-by-step error distribution, residual analysis charts, and CDF plots.
"""

from typing import List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from simulation import Room, Anchor


def create_floorplan_figure(
    room: Room,
    ground_truth: np.ndarray,
    baseline_traj: Optional[np.ndarray] = None,
    anchorguard_traj: Optional[np.ndarray] = None,
    current_step: Optional[int] = None,
    show_anchors: bool = True,
    show_legend: bool = True,
) -> go.Figure:
    """
    Creates an interactive 2D floor plan visualization with dark technical theme.
    """
    fig = go.Figure()

    # Room Boundary Polygon
    room_x = [0, room.width, room.width, 0, 0]
    room_y = [0, 0, room.height, room.height, 0]

    fig.add_trace(go.Scatter(
        x=room_x,
        y=room_y,
        mode="lines",
        line=dict(color="#3b82f6", width=2, dash="solid"),
        name="Room Boundary",
        hoverinfo="skip"
    ))

    # Room Grid / Zones
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=room.width, y1=room.height,
        fillcolor="rgba(15, 23, 42, 0.6)",
        line=dict(width=0),
        layer="below",
    )

    # 1. Ground Truth Path
    fig.add_trace(go.Scatter(
        x=ground_truth[:, 0],
        y=ground_truth[:, 1],
        mode="lines+markers",
        name="Ground Truth (Actual Path)",
        line=dict(color="#10b981", width=3),
        marker=dict(size=4, color="#10b981"),
        hovertemplate="<b>Ground Truth</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
    ))

    # 2. Baseline Estimated Path
    if baseline_traj is not None:
        fig.add_trace(go.Scatter(
            x=baseline_traj[:, 0],
            y=baseline_traj[:, 1],
            mode="lines+markers",
            name="Baseline NLS Trilateration",
            line=dict(color="#ef4444", width=2, dash="dot"),
            marker=dict(size=5, symbol="x", color="#ef4444"),
            connectgaps=False,  # Breaks line over missing/failed estimates
            hovertemplate="<b>Baseline</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
        ))

    # 3. AnchorGuard Robust Path
    if anchorguard_traj is not None:
        fig.add_trace(go.Scatter(
            x=anchorguard_traj[:, 0],
            y=anchorguard_traj[:, 1],
            mode="lines+markers",
            name="AnchorGuard Robust AI",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5, symbol="circle-open", color="#38bdf8"),
            connectgaps=False,
            hovertemplate="<b>AnchorGuard</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
        ))

    # 4. Failed Estimation Steps (explicit markers on ground-truth path where estimation failed)
    if baseline_traj is not None:
        failed_mask = np.isnan(baseline_traj[:, 0])
        if np.any(failed_mask):
            failed_gt = ground_truth[failed_mask]
            fig.add_trace(go.Scatter(
                x=failed_gt[:, 0],
                y=failed_gt[:, 1],
                mode="markers",
                name="Failed Estimation Steps",
                marker=dict(size=8, symbol="cross", color="#f59e0b", line=dict(color="#000000", width=1)),
                hovertemplate="<b>Estimation Failed Step</b><br>True X: %{x:.2f} m<br>True Y: %{y:.2f} m<br>Status: INSUFFICIENT / DEGENERATE<extra></extra>"
            ))

    # 5. Current Target Position (if step specified)
    if current_step is not None and 0 <= current_step < len(ground_truth):
        curr_gt = ground_truth[current_step]
        fig.add_trace(go.Scatter(
            x=[curr_gt[0]],
            y=[curr_gt[1]],
            mode="markers",
            name="Current True Target",
            marker=dict(size=14, color="#10b981", symbol="star", line=dict(color="#ffffff", width=2)),
            hovertemplate="<b>Current Position (t=%{customdata})</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>",
            customdata=[current_step]
        ))

        if baseline_traj is not None and not np.isnan(baseline_traj[current_step, 0]):
            curr_b = baseline_traj[current_step]
            fig.add_trace(go.Scatter(
                x=[curr_b[0]],
                y=[curr_b[1]],
                mode="markers",
                name="Current Baseline Pos",
                marker=dict(size=11, color="#ef4444", symbol="x", line=dict(width=2)),
                hovertemplate="<b>Current Baseline</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
            ))

        if anchorguard_traj is not None and not np.isnan(anchorguard_traj[current_step, 0]):
            curr_ag = anchorguard_traj[current_step]
            fig.add_trace(go.Scatter(
                x=[curr_ag[0]],
                y=[curr_ag[1]],
                mode="markers",
                name="Current AnchorGuard Pos",
                marker=dict(size=12, color="#38bdf8", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>Current AnchorGuard</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<extra></extra>"
            ))

    # 6. Fixed Anchors
    if show_anchors:
        active_a = [a for a in room.anchors if a.is_active]
        inactive_a = [a for a in room.anchors if not a.is_active]

        if active_a:
            fig.add_trace(go.Scatter(
                x=[a.x for a in active_a],
                y=[a.y for a in active_a],
                mode="markers+text",
                name="Active Anchors",
                text=[f"A{a.id}" for a in active_a],
                textposition="top center",
                textfont=dict(color="#e2e8f0", size=10, family="monospace"),
                marker=dict(size=13, color="#fbbf24", symbol="triangle-up", line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>%{customdata}</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Status: ACTIVE<extra></extra>",
                customdata=[a.name for a in active_a]
            ))

        if inactive_a:
            fig.add_trace(go.Scatter(
                x=[a.x for a in inactive_a],
                y=[a.y for a in inactive_a],
                mode="markers+text",
                name="Offline / Outage Anchors",
                text=[f"A{a.id} (OFF)" for a in inactive_a],
                textposition="top center",
                textfont=dict(color="#94a3b8", size=9, family="monospace"),
                marker=dict(size=12, color="#64748b", symbol="x", line=dict(color="#ef4444", width=2)),
                hovertemplate="<b>%{customdata}</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Status: OUTAGE<extra></extra>",
                customdata=[a.name for a in inactive_a]
            ))

    # Dynamically compute axis ranges to make out-of-room estimates and anchors visible
    all_x = [0.0, room.width] + [a.x for a in room.anchors] + list(ground_truth[:, 0])
    all_y = [0.0, room.height] + [a.y for a in room.anchors] + list(ground_truth[:, 1])

    if baseline_traj is not None:
        valid_b = baseline_traj[~np.isnan(baseline_traj[:, 0])]
        if len(valid_b) > 0:
            all_x.extend(list(valid_b[:, 0]))
            all_y.extend(list(valid_b[:, 1]))

    if anchorguard_traj is not None:
        valid_ag = anchorguard_traj[~np.isnan(anchorguard_traj[:, 0])]
        if len(valid_ag) > 0:
            all_x.extend(list(valid_ag[:, 0]))
            all_y.extend(list(valid_ag[:, 1]))

    x_min = min(-1.0, float(np.min(all_x)) - 1.0)
    x_max = max(room.width + 1.0, float(np.max(all_x)) + 1.0)
    y_min = min(-1.0, float(np.min(all_y)) - 1.0)
    y_max = max(room.height + 1.0, float(np.max(all_y)) + 1.0)

    # Layout styling with Dark Engineering Aesthetic
    fig.update_layout(
        title=dict(
            text="Indoor Floor Plan & Positioning Trajectory",
            font=dict(size=16, color="#f8fafc", family="Inter, system-ui")
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title=dict(
                text="X Coordinate (Metres)",
                font=dict(color="#cbd5e1"),
            ),
            range=[x_min, x_max],
            zeroline=False,
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(
                text="Y Coordinate (Metres)",
                font=dict(color="#cbd5e1"),
            ),
            range=[y_min, y_max],
            scaleanchor="x",
            scaleratio=1,
            zeroline=False,
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color="#e2e8f0", size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ) if show_legend else dict(visible=False),
        margin=dict(l=40, r=40, t=50, b=80),
        height=550,
    )

    return fig


def create_error_comparison_figure(
    ground_truth: np.ndarray,
    baseline_traj: np.ndarray,
    anchorguard_traj: np.ndarray,
) -> go.Figure:
    """
    Renders time-series tracking error (Euclidean distance error vs time step).
    """
    t_steps = np.arange(len(ground_truth))

    # Compute Euclidean distance errors
    b_diff = ground_truth - baseline_traj
    b_err = np.linalg.norm(b_diff, axis=1)

    ag_diff = ground_truth - anchorguard_traj
    ag_err = np.linalg.norm(ag_diff, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_steps,
        y=b_err,
        mode="lines",
        name="Baseline NLS Error",
        line=dict(color="#ef4444", width=2),
        connectgaps=False,
        hovertemplate="Step %{x}<br>Baseline Error: %{y:.3f} m<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=t_steps,
        y=ag_err,
        mode="lines",
        name="AnchorGuard Robust Error",
        line=dict(color="#38bdf8", width=2.5),
        connectgaps=False,
        hovertemplate="Step %{x}<br>AnchorGuard Error: %{y:.3f} m<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Positioning Error over Trajectory Steps",
            font=dict(size=14, color="#f8fafc")
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title=dict(
                text="Trajectory Time Step (t)",
                font=dict(color="#cbd5e1"),
            ),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(
                text="Euclidean Error (Metres)",
                font=dict(color="#cbd5e1"),
            ),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(color="#e2e8f0", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=300,
    )
    return fig


def create_cdf_figure(
    ground_truth: np.ndarray,
    baseline_traj: np.ndarray,
    anchorguard_traj: np.ndarray,
) -> go.Figure:
    """
    Renders Cumulative Distribution Function (CDF) of Positioning Error.
    Standard metric used in indoor localization research (e.g. IEEE IPIN).
    """
    b_diff = ground_truth - baseline_traj
    b_err = np.linalg.norm(b_diff, axis=1)
    b_valid = b_err[~np.isnan(b_err)]

    ag_diff = ground_truth - anchorguard_traj
    ag_err = np.linalg.norm(ag_diff, axis=1)
    ag_valid = ag_err[~np.isnan(ag_err)]

    fig = go.Figure()

    if len(b_valid) > 0:
        b_sorted = np.sort(b_valid)
        b_cdf = np.arange(1, len(b_sorted) + 1) / len(b_sorted)
        fig.add_trace(go.Scatter(
            x=b_sorted,
            y=b_cdf,
            mode="lines",
            name="Baseline NLS",
            line=dict(color="#ef4444", width=2),
            hovertemplate="Error ≤ %{x:.2f} m: %{y:.1%}<extra></extra>"
        ))

    if len(ag_valid) > 0:
        ag_sorted = np.sort(ag_valid)
        ag_cdf = np.arange(1, len(ag_sorted) + 1) / len(ag_sorted)
        fig.add_trace(go.Scatter(
            x=ag_sorted,
            y=ag_cdf,
            mode="lines",
            name="AnchorGuard Robust",
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate="Error ≤ %{x:.2f} m: %{y:.1%}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="Cumulative Error Distribution (CDF)",
            font=dict(size=14, color="#f8fafc")
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title=dict(
                text="Position Error (Metres)",
                font=dict(color="#cbd5e1"),
            ),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(
                text="Cumulative Probability",
                font=dict(color="#cbd5e1"),
            ),
            range=[0, 1.05],
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(color="#e2e8f0", size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=300,
    )
    return fig


def create_residual_diagnostics_figure(
    measurements: np.ndarray,
    anchor_coords: np.ndarray,
    estimated_traj: np.ndarray,
    anchor_ids: Optional[List[int]] = None,
    title: str = "Anchor Range Residual Diagnostics (NLOS Identification)",
) -> go.Figure:
    """
    Renders range residuals (Measured Range - Estimated Euclidean Distance) per anchor.
    Large positive residuals (> 1.5 - 2.0 m) indicate severe NLOS bias or corrupted measurements.
    """
    num_steps, num_anchors = measurements.shape
    t_steps = np.arange(num_steps)

    # Compute Euclidean distance to each anchor for each step: (T, N)
    diff = estimated_traj[:, np.newaxis, :] - anchor_coords[np.newaxis, :, :]
    est_dists = np.linalg.norm(diff, axis=-1)

    residuals = measurements - est_dists

    fig = go.Figure()
    palette = [
        "#38bdf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#f472b6",
        "#60a5fa", "#4ade80", "#facc15", "#fb923c", "#c084fc", "#e879f9"
    ]

    for i in range(num_anchors):
        a_id = anchor_ids[i] if anchor_ids is not None and i < len(anchor_ids) else i
        r_i = residuals[:, i]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=t_steps,
            y=r_i,
            mode="lines+markers",
            name=f"Anchor {a_id}",
            marker=dict(size=4),
            line=dict(width=1.5, color=color),
            connectgaps=False,
            hovertemplate=f"Step %{{x}}<br>Anchor {a_id} Residual: %{{y:.2f}} m<extra></extra>"
        ))

    # Reference zero residual
    fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b")

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color="#f8fafc")
        ),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            title=dict(
                text="Trajectory Time Step (t)",
                font=dict(color="#cbd5e1"),
            ),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(
                text="Residual (m): Range - Est Dist",
                font=dict(color="#cbd5e1"),
            ),
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(color="#e2e8f0", size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig
