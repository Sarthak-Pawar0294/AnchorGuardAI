"""
metrics.py - Evaluation Metrics for Indoor Positioning.

Computes:
- Position RMSE (Root Mean Square Error in metres)
- MAE (Mean Absolute Positioning Error in metres)
- Median Positioning Error (50th percentile)
- 95th Percentile Positioning Error
- Percentage of Successfully Estimated Positions (%)
- Runtime per Position Estimate (ms)
- Average GDOP (Geometric Dilution of Precision)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class PerformanceMetrics:
    rmse: float                         # Metres
    mae: float                          # Metres
    median_error: float                 # Metres
    p95_error: float                    # Metres
    max_error: float                    # Metres
    success_rate_pct: float             # Percentage [0.0 - 100.0] = (successful_steps / total_steps) * 100
    total_steps: int
    successful_steps: int
    avg_runtime_ms: float               # Per-step runtime in ms
    total_runtime_ms: float = 0.0       # Complete run runtime in ms
    out_of_room_steps: int = 0          # Number of valid estimates located outside room boundaries
    out_of_room_rate_pct: float = 0.0   # Percentage = (out_of_room_steps / successful_steps) * 100
    avg_gdop: Optional[float] = None


def compute_positioning_metrics(
    ground_truth: np.ndarray,
    estimated: np.ndarray,
    success_flags: Optional[np.ndarray] = None,
    avg_runtime_ms: float = 0.0,
    total_runtime_ms: float = 0.0,
    gdop_values: Optional[list] = None,
    room_dims: Optional[Tuple[float, float]] = None,
) -> PerformanceMetrics:
    """
    Computes rigorous empirical positioning metrics comparing ground truth to estimates.
    Only evaluates points where the estimate is valid (not NaN).

    Definitions:
    - success_rate_pct: (successful_steps / total_steps) * 100%
    - out_of_room_rate_pct: (out_of_room_steps / successful_steps) * 100% if successful_steps > 0 else 0%
    """
    assert len(ground_truth) == len(estimated), "Ground truth and estimates must have same length"
    total_steps = len(ground_truth)

    # Valid mask: both coordinates must be finite
    valid_mask = ~np.isnan(estimated[:, 0]) & ~np.isnan(estimated[:, 1])
    if success_flags is not None:
        valid_mask = valid_mask & success_flags

    successful_steps = int(np.sum(valid_mask))
    success_rate_pct = (successful_steps / total_steps) * 100.0 if total_steps > 0 else 0.0

    if total_runtime_ms == 0.0 and avg_runtime_ms > 0.0:
        total_runtime_ms = avg_runtime_ms * total_steps

    if successful_steps == 0:
        return PerformanceMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            median_error=float("nan"),
            p95_error=float("nan"),
            max_error=float("nan"),
            success_rate_pct=0.0,
            total_steps=total_steps,
            successful_steps=0,
            avg_runtime_ms=avg_runtime_ms,
            total_runtime_ms=total_runtime_ms,
            out_of_room_steps=0,
            out_of_room_rate_pct=0.0,
            avg_gdop=None,
        )

    # Euclidean distance errors: sqrt((x_true - x_est)^2 + (y_true - y_est)^2)
    diff = ground_truth[valid_mask] - estimated[valid_mask]
    errors = np.linalg.norm(diff, axis=1)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(errors))
    med = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    max_err = float(np.max(errors))

    # Out of room detection on valid estimates
    out_of_room_steps = 0
    out_of_room_rate_pct = 0.0
    if room_dims is not None:
        rw, rh = room_dims
        valid_est = estimated[valid_mask]
        oor_mask = (valid_est[:, 0] < 0.0) | (valid_est[:, 0] > rw) | (valid_est[:, 1] < 0.0) | (valid_est[:, 1] > rh)
        out_of_room_steps = int(np.sum(oor_mask))
        out_of_room_rate_pct = (out_of_room_steps / successful_steps) * 100.0

    avg_gdop = None
    if gdop_values is not None:
        valid_gdop = [g for g in gdop_values if g is not None and not np.isnan(g)]
        if valid_gdop:
            avg_gdop = float(np.mean(valid_gdop))

    return PerformanceMetrics(
        rmse=rmse,
        mae=mae,
        median_error=med,
        p95_error=p95,
        max_error=max_err,
        success_rate_pct=success_rate_pct,
        total_steps=total_steps,
        successful_steps=successful_steps,
        avg_runtime_ms=avg_runtime_ms,
        total_runtime_ms=total_runtime_ms,
        out_of_room_steps=out_of_room_steps,
        out_of_room_rate_pct=out_of_room_rate_pct,
        avg_gdop=avg_gdop,
    )


def metrics_to_dict(m: PerformanceMetrics, prefix: str = "") -> Dict[str, Any]:
    """Serializes metrics to flat dictionary for tabular display and CSV export."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}RMSE (m)": round(m.rmse, 3) if not np.isnan(m.rmse) else None,
        f"{p}MAE (m)": round(m.mae, 3) if not np.isnan(m.mae) else None,
        f"{p}Median Error (m)": round(m.median_error, 3) if not np.isnan(m.median_error) else None,
        f"{p}P95 Error (m)": round(m.p95_error, 3) if not np.isnan(m.p95_error) else None,
        f"{p}Success Rate (%)": round(m.success_rate_pct, 1),
        f"{p}Out-of-Room Rate (%)": round(m.out_of_room_rate_pct, 1),
        f"{p}Avg Runtime (ms)": round(m.avg_runtime_ms, 2),
        f"{p}Total Runtime (ms)": round(m.total_runtime_ms, 2),
        f"{p}Avg GDOP": round(m.avg_gdop, 2) if m.avg_gdop is not None else None,
    }
