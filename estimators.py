"""
estimators.py - Position Estimation Algorithms for AnchorGuard AI.

Implements:
1. Geometry conditioning & degeneracy checks (Condition Number, Rank, Collinearity)
2. Method A: Baseline Nonlinear Least Squares (NLS) Trilateration
3. Method B: AnchorGuard Robust Estimator (Huber loss + iterative residual analysis + NLOS mitigation)
4. Uncertainty Covariance & Geometric Dilution of Precision (GDOP) calculation
5. Optional Configurable 2D Constant-Velocity Kalman Filter for temporal trajectory smoothing
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import least_squares


class EstimationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INSUFFICIENT_ANCHORS = "INSUFFICIENT_ANCHORS"
    DEGENERATE_GEOMETRY = "DEGENERATE_GEOMETRY"
    CONVERGENCE_FAILURE = "CONVERGENCE_FAILURE"
    MISSING_ALL = "MISSING_ALL"


@dataclass
class SingleStepEstimate:
    position: Optional[np.ndarray]           # [x, y] or None if failed
    status: EstimationStatus
    num_used_anchors: int
    used_anchor_indices: List[int]
    residuals: Optional[np.ndarray] = None   # Residual for each used anchor
    covariance: Optional[np.ndarray] = None  # 2x2 estimated position error covariance
    gdop: Optional[float] = None             # Geometric Dilution of Precision
    rejected_anchor_indices: List[int] = field(default_factory=list)
    computation_time_ms: float = 0.0


@dataclass
class TrajectoryEstimateResult:
    estimated_trajectory: np.ndarray         # Shape (T, 2), NaN for failed steps
    statuses: List[EstimationStatus]
    success_flags: np.ndarray               # Boolean array of shape (T,)
    step_estimates: List[SingleStepEstimate]
    avg_runtime_ms: float
    total_runtime_ms: float


def check_geometry_conditioning(anchor_coords: np.ndarray) -> Tuple[bool, float, str]:
    """
    Evaluates whether the anchor spatial configuration allows non-degenerate 2D position estimation.
    Trilateration requires at least 3 non-collinear anchors.
    
    Returns:
        (is_valid, condition_metric, message)
    """
    k = len(anchor_coords)
    if k < 3:
        return False, 0.0, f"Insufficient anchors: only {k} active, minimum required is 3"

    # Centered coordinate covariance / SVD to check collinearity
    mean_coord = np.mean(anchor_coords, axis=0)
    centered = anchor_coords - mean_coord
    
    # Singular values of spatial spread
    _, s, _ = np.linalg.svd(centered)
    
    # If smaller singular value is very small compared to the larger one, anchors are collinear
    if len(s) < 2 or s[1] < 1e-4:
        return False, 0.0, "Degenerate geometry: Anchors are collinear or co-located"

    aspect_ratio = s[1] / (s[0] + 1e-9)
    if aspect_ratio < 0.05:
        return False, aspect_ratio, f"Ill-conditioned anchor geometry (aspect ratio={aspect_ratio:.3f})"

    return True, aspect_ratio, "Good geometric diversity"


def compute_gdop_and_covariance(
    pos: np.ndarray,
    anchors: np.ndarray,
    sigma: float = 0.25
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Calculates the 2D Geometric Dilution of Precision (GDOP) and 2x2 error covariance matrix.
    Jacobian H of range model: H_i = (pos - anchor_i) / ||pos - anchor_i||
    """
    diff = pos - anchors
    dists = np.linalg.norm(diff, axis=1, keepdims=True)
    dists = np.maximum(dists, 1e-6)
    H = diff / dists  # shape (k, 2)
    
    try:
        HtH = H.T @ H
        cov = np.linalg.inv(HtH) * (sigma ** 2)
        gdop = float(np.sqrt(np.trace(np.linalg.inv(HtH))))
        return gdop, cov
    except np.linalg.LinAlgError:
        return None, None


# =========================================================================
# Method A: Baseline Nonlinear Least Squares Trilateration
# =========================================================================

def baseline_estimate_step(
    measurements: np.ndarray,
    anchor_coords: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    noise_sigma: float = 0.25,
) -> SingleStepEstimate:
    """
    Estimates 2D coordinates for a single time step using standard Nonlinear Least Squares (Linear Loss).
    Fails safely if measurements are inadequate or anchors are collinear.
    """
    t_start = time.perf_counter()

    # Filter out missing measurements (NaN)
    valid_mask = ~np.isnan(measurements)
    valid_indices = np.where(valid_mask)[0].tolist()
    k = len(valid_indices)

    if k < 3:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.INSUFFICIENT_ANCHORS,
            num_used_anchors=k,
            used_anchor_indices=valid_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    valid_anchors = anchor_coords[valid_mask]
    valid_ranges = measurements[valid_mask]

    # Check geometric conditioning
    is_valid_geom, cond, _ = check_geometry_conditioning(valid_anchors)
    if not is_valid_geom:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.DEGENERATE_GEOMETRY,
            num_used_anchors=k,
            used_anchor_indices=valid_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    # Initial guess: centroid of valid anchors if not provided
    if initial_guess is None or np.any(np.isnan(initial_guess)):
        x0 = np.mean(valid_anchors, axis=0)
    else:
        x0 = initial_guess

    def residual_fun(p):
        dists = np.linalg.norm(valid_anchors - p, axis=1)
        return dists - valid_ranges

    try:
        opt_res = least_squares(
            residual_fun,
            x0=x0,
            loss="linear",  # Standard L2 squared loss
            method="lm",
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=150,
        )

        if not opt_res.success:
            return SingleStepEstimate(
                position=None,
                status=EstimationStatus.CONVERGENCE_FAILURE,
                num_used_anchors=k,
                used_anchor_indices=valid_indices,
                computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        pos = opt_res.x
        residuals = residual_fun(pos)
        gdop, cov = compute_gdop_and_covariance(pos, valid_anchors, sigma=noise_sigma)

        return SingleStepEstimate(
            position=pos,
            status=EstimationStatus.SUCCESS,
            num_used_anchors=k,
            used_anchor_indices=valid_indices,
            residuals=residuals,
            covariance=cov,
            gdop=gdop,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    except Exception:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.CONVERGENCE_FAILURE,
            num_used_anchors=k,
            used_anchor_indices=valid_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )


# =========================================================================
# Method B: AnchorGuard Robust Estimator (Huber Loss + Residual NLOS Pruning)
# =========================================================================

@dataclass
class AnchorGuardConfig:
    loss: str = "huber"              # "huber", "cauchy", "soft_l1"
    huber_delta: float = 1.345       # Huber tuning parameter (standard 95% asymptotic efficiency)
    enable_residual_pruning: bool = True
    nlos_residual_threshold: float = 2.0  # Normalized residual threshold to trigger NLOS rejection
    max_pruned_anchors: int = 2     # Maximum anchors to reject per step (preserves geometry)
    min_anchors_after_pruning: int = 3
    noise_sigma: float = 0.25


def anchorguard_estimate_step(
    measurements: np.ndarray,
    anchor_coords: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    config: Optional[AnchorGuardConfig] = None,
) -> SingleStepEstimate:
    """
    AnchorGuard Robust Estimator:
    1. Uses Huber M-estimator loss function to attenuate the leverage of positive NLOS range errors.
    2. Performs normalized residual inspection: if a single or small subset of anchors exhibits
       abnormally large positive residuals (indicating severe NLOS bias) and surplus anchors exist,
       the estimator prunes the worst outlier and re-fits.
    3. Gracefully rejects degenerate geometry or insufficient measurements.
    """
    if config is None:
        config = AnchorGuardConfig()

    t_start = time.perf_counter()

    valid_mask = ~np.isnan(measurements)
    current_indices = np.where(valid_mask)[0].tolist()
    rejected_indices: List[int] = []

    if len(current_indices) < 3:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.INSUFFICIENT_ANCHORS,
            num_used_anchors=len(current_indices),
            used_anchor_indices=current_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    is_valid_geom, _, _ = check_geometry_conditioning(anchor_coords[current_indices])
    if not is_valid_geom:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.DEGENERATE_GEOMETRY,
            num_used_anchors=len(current_indices),
            used_anchor_indices=current_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    # Initial guess
    if initial_guess is None or np.any(np.isnan(initial_guess)):
        x0 = np.mean(anchor_coords[current_indices], axis=0)
    else:
        x0 = initial_guess

    def solve_robust(idx_list: List[int], start_p: np.ndarray):
        a_sub = anchor_coords[idx_list]
        r_sub = measurements[idx_list]

        def residual_fun(p):
            dists = np.linalg.norm(a_sub - p, axis=1)
            # Normalized residual: (observed - distance)
            return (r_sub - dists) / max(0.01, config.noise_sigma)

        res = least_squares(
            residual_fun,
            x0=start_p,
            loss="huber",
            f_scale=config.huber_delta,
            method="trf",
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=150,
        )
        return res, a_sub, r_sub

    # Initial robust fit
    try:
        opt_res, active_anchors, active_ranges = solve_robust(current_indices, x0)
        if not opt_res.success:
            return SingleStepEstimate(
                position=None,
                status=EstimationStatus.CONVERGENCE_FAILURE,
                num_used_anchors=len(current_indices),
                used_anchor_indices=current_indices,
                computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        pos = opt_res.x

        # Robust NLOS Outlier Rejection via Geometric Consensus & Residual Analysis
        if config.enable_residual_pruning and len(current_indices) > config.min_anchors_after_pruning:
            # If surplus anchors exist (>= 4), test if dropping any single anchor dramatically reduces
            # total consensus residual, characteristic of an NLOS outlier.
            best_indices = current_indices
            best_cost = opt_res.cost
            best_pos = pos
            pruned_candidate = None

            # 1. First evaluate single-anchor leave-one-out candidates
            for cand in current_indices:
                sub_idx = [i for i in current_indices if i != cand]
                valid_geom, _, _ = check_geometry_conditioning(anchor_coords[sub_idx])
                if not valid_geom:
                    continue
                sub_res, _, _ = solve_robust(sub_idx, pos)
                if sub_res.success and sub_res.cost < best_cost:
                    # Significant drop in residual cost indicates the excluded anchor was corrupting the fit
                    if sub_res.cost < best_cost * 0.65 or (best_cost - sub_res.cost > 2.0):
                        best_cost = sub_res.cost
                        best_indices = sub_idx
                        best_pos = sub_res.x
                        pruned_candidate = cand

            if pruned_candidate is not None:
                current_indices = best_indices
                pos = best_pos
                rejected_indices.append(pruned_candidate)

                # 2. If still >= 4 anchors remain and allowed, check second outlier
                if config.max_pruned_anchors > 1 and len(current_indices) > config.min_anchors_after_pruning:
                    second_cand = None
                    for cand2 in current_indices:
                        sub2_idx = [i for i in current_indices if i != cand2]
                        valid_geom, _, _ = check_geometry_conditioning(anchor_coords[sub2_idx])
                        if not valid_geom:
                            continue
                        sub2_res, _, _ = solve_robust(sub2_idx, pos)
                        if sub2_res.success and sub2_res.cost < best_cost * 0.65:
                            best_cost = sub2_res.cost
                            best_indices = sub2_idx
                            best_pos = sub2_res.x
                            second_cand = cand2
                    if second_cand is not None:
                        current_indices = best_indices
                        pos = best_pos
                        rejected_indices.append(second_cand)

        final_anchors = anchor_coords[current_indices]
        final_dists = np.linalg.norm(final_anchors - pos, axis=1)
        final_residuals = measurements[current_indices] - final_dists
        gdop, cov = compute_gdop_and_covariance(pos, final_anchors, sigma=config.noise_sigma)

        return SingleStepEstimate(
            position=pos,
            status=EstimationStatus.SUCCESS,
            num_used_anchors=len(current_indices),
            used_anchor_indices=current_indices,
            residuals=final_residuals,
            covariance=cov,
            gdop=gdop,
            rejected_anchor_indices=rejected_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    except Exception:
        return SingleStepEstimate(
            position=None,
            status=EstimationStatus.CONVERGENCE_FAILURE,
            num_used_anchors=len(current_indices),
            used_anchor_indices=current_indices,
            computation_time_ms=(time.perf_counter() - t_start) * 1000.0,
        )


# =========================================================================
# Temporal Filtering: 2D Constant Velocity Kalman Filter
# =========================================================================

class PositionKalmanFilter:
    """
    Discrete 2D Constant-Velocity Kalman Filter:
    State: x = [x, y, vx, vy]^T
    Integrates position estimates when available and performs prediction when measurements fail.
    """
    def __init__(self, dt: float = 1.0, process_noise_std: float = 0.5, default_meas_std: float = 0.3):
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.default_meas_std = default_meas_std

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # Discrete process noise covariance Q (continuous white noise acceleration model)
        q = process_noise_std ** 2
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        self.Q = q * np.array([
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
            [dt2, 0.0, dt,  0.0],
            [0.0, dt2, 0.0, dt ]
        ])

        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0
        self.is_initialized = False

    def predict(self) -> np.ndarray:
        if not self.is_initialized:
            return np.zeros(2)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]

    def update(self, pos_meas: np.ndarray, meas_cov: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_initialized:
            self.x = np.array([pos_meas[0], pos_meas[1], 0.0, 0.0])
            self.P = np.eye(4) * (self.default_meas_std ** 2)
            self.is_initialized = True
            return self.x[:2]

        R = meas_cov if meas_cov is not None else np.eye(2) * (self.default_meas_std ** 2)

        # Innovation
        y = pos_meas - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        # Joseph form for numerical stability
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x[:2]


# =========================================================================
# Trajectory-Level Batch Runner
# =========================================================================

def estimate_trajectory(
    measurements: np.ndarray,
    anchor_coords: np.ndarray,
    method: str = "anchorguard",  # "baseline" or "anchorguard"
    noise_sigma: float = 0.25,
    anchorguard_config: Optional[AnchorGuardConfig] = None,
    apply_kalman: bool = False,
    kalman_process_std: float = 0.5,
) -> TrajectoryEstimateResult:
    """
    Processes an entire sequence of range measurements across all time steps.
    Never silently fabricates positions when measurement geometry is inadequate.
    If apply_kalman=True, a Kalman filter smoothly fuses point estimates.
    """
    num_steps = len(measurements)
    estimated_traj = np.full((num_steps, 2), np.nan)
    statuses: List[EstimationStatus] = []
    step_estimates: List[SingleStepEstimate] = []
    success_flags = np.zeros(num_steps, dtype=bool)

    kf = PositionKalmanFilter(dt=1.0, process_noise_std=kalman_process_std, default_meas_std=noise_sigma) if apply_kalman else None
    prev_pos = None

    t_total_start = time.perf_counter()

    for t in range(num_steps):
        m_t = measurements[t]

        if method == "baseline":
            step_res = baseline_estimate_step(
                measurements=m_t,
                anchor_coords=anchor_coords,
                initial_guess=prev_pos,
                noise_sigma=noise_sigma,
            )
        elif method == "anchorguard":
            step_res = anchorguard_estimate_step(
                measurements=m_t,
                anchor_coords=anchor_coords,
                initial_guess=prev_pos,
                config=anchorguard_config or AnchorGuardConfig(noise_sigma=noise_sigma),
            )
        else:
            raise ValueError(f"Unknown estimator method: {method}")

        statuses.append(step_res.status)
        step_estimates.append(step_res)

        if step_res.status == EstimationStatus.SUCCESS and step_res.position is not None:
            raw_pos = step_res.position
            success_flags[t] = True
            prev_pos = raw_pos.copy()

            if kf is not None:
                kf.predict()
                filtered_pos = kf.update(raw_pos, step_res.covariance)
                estimated_traj[t] = filtered_pos
            else:
                estimated_traj[t] = raw_pos
        else:
            # If estimation fails:
            if kf is not None and kf.is_initialized:
                # Predict only without updating measurement (temporal extrapolation)
                pred_pos = kf.predict()
                # Do NOT silently claim it was a measured success
                estimated_traj[t] = pred_pos
            else:
                estimated_traj[t] = [np.nan, np.nan]

    total_time_ms = (time.perf_counter() - t_total_start) * 1000.0
    avg_runtime_ms = total_time_ms / max(1, num_steps)

    return TrajectoryEstimateResult(
        estimated_trajectory=estimated_traj,
        statuses=statuses,
        success_flags=success_flags,
        step_estimates=step_estimates,
        avg_runtime_ms=avg_runtime_ms,
        total_runtime_ms=total_time_ms,
    )
