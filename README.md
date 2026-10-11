# AnchorGuard AI 2.0 — Indoor Positioning Engineering Workbench

**VIT Code Apex 3.0** | **Track 4, Topic 2**: *AI-Based Indoor Positioning with Sparse Infrastructure and Noisy Measurements*  
**Team**: **Gang of Coders**  
**Version**: **2.0 (Engineering Workbench)**

---

## 1. Project Overview & Scope

AnchorGuard AI 2.0 is an industry-style indoor positioning engineering workbench designed to evaluate, stress-test, and benchmark 2D target tracking algorithms under sparse reference infrastructure and hostile radio channel conditions:
- **Additive White Gaussian Noise (AWGN):** Thermal noise and receiver jitter.
- **Packet Dropout:** Channel fading, packet collisions, and sensor dropouts ($p_{\text{drop}} \in [0.0, 0.8]$).
- **Severe Positive Non-Line-of-Sight (NLOS) Biases:** Multipath delays and through-wall penetration ($b \ge 0$).
- **Anchor Outages & Degenerate Geometries:** Permanent or intermittent beacon outages, collinear placements, or $<3$ active reference points.

### Statement of Physical Realism & Limitations
> [!NOTE]
> AnchorGuard AI 2.0 is a **simulation-based engineering workbench**. Ground-truth trajectories and synthetic range measurements are generated using calibrated mathematical fault models. Physical radio propagation phenomena (such as 3D ray-tracing diffraction, wall material dielectric losses, human-body shadow fading, and clock drift) require integration with real RF hardware (UWB, Wi-Fi RTT, or BLE Channel Sounding) for operational deployment.

---

## 2. System Architecture & Modular Decoupling

AnchorGuard AI 2.0 strictly separates **Configuration**, **Simulation**, **Estimation**, **Evaluation**, and **Presentation**:

```
AnchorGuard/
├── app.py               # Streamlit Workbench with 6 dedicated engineering sections
├── config.py            # Validated experiment configuration model (schema v2.0)
├── presets.py           # Standard measurement fault presets
├── simulation.py        # 2D Room, trajectory synthesis, and decoupled measurement simulator
├── estimators.py        # Baseline NLS, AnchorGuard Robust Huber + Outlier Pruning, Kalman Filter
├── metrics.py           # Empirical positioning metrics & strict denominator accounting
├── benchmark.py         # Multi-trial Monte Carlo benchmark engine & CSV exporter
├── visualization.py     # Interactive Plotly floor plans, error CDFs, and residual diagnostics
├── requirements.txt     # Locked project dependencies
├── README.md            # System documentation and instructions
└── tests/
    ├── test_config.py               # 9 tests: validation, layout generators, JSON roundtrip, hashing
    ├── test_estimators.py           # 8 tests: conditioning, NLS, AnchorGuard, Kalman smoothing
    ├── test_metrics.py              # 4 tests: metric computation, NaN handling, formatting
    ├── test_presets.py              # 3 tests: presets schema, safe formatting, plain-English verdicts
    ├── test_simulation.py           # 5 tests: geometry bounds, noise-free recovery, outage handling
    ├── test_visualization.py        # 5 tests: floorplans, CDF, error comparisons, canvas limits
    ├── test_workbench_correctness.py# 6 tests: fair reuse, reproducibility, out-of-room metrics, std stats
    └── test_streamlit_app.py        # 3 tests: AppTest clean startup, 6-tab navigation, button execution
```

### Strict Decoupling Contract
1. **Ground Truth Decoupling:** Ground truth target trajectories $\mathbf{p}_t$ are never accessible to estimators. Estimators receive only corrupted distance vectors $\tilde{\mathbf{d}}_t$ and known active anchor coordinates $\mathbf{a}_i$.
2. **Fair Measurement Reuse:** During single-run or multi-trial comparisons, all estimators receive **identical** simulated measurements and geometry.
3. **Preservation of Real Errors:** Estimates are never artificially clipped to room boundaries to inflate benchmark scores. Failed steps remain `NaN` and are plotted with broken segments (`connectgaps=False`).

---

## 3. Validated Experiment Configuration Model (`config.py`)

Centralized configuration model with explicit domain validation:

| Component | Parameters | Validation Rules |
| :--- | :--- | :--- |
| **Room & Geometry** | `room_width`, `room_height`, `layout_preset` | Width & height $> 0$; minimum 3 active non-collinear anchors required. |
| **Anchors** | List of `AnchorConfig(id, name, x, y, is_active)` | Presets for Perimeter, Corners, and Grid; coordinates editable in Environment Studio. |
| **Target Trajectory** | `trajectory_pattern`, `trajectory_steps`, `trajectory_speed` | Patterns: `smooth_loop`, `patrol_perimeter`, `linear_crossing`, `random_walk`; steps $\ge 5$. |
| **Channel Faults** | `noise_std`, `drop_rate`, `nlos_probability`, `nlos_bias_mean` | Noise $\ge 0$; probabilities $\in [0.0, 1.0]$; NLOS bias mean $\ge 0$. |
| **Outages** | `outage_anchor_ids` | Specific anchor IDs excluded non-destructively without mutating persistent room state. |
| **Reproducibility** | `seed` | Seed $\ge 0$; deterministic noise and trajectory across runs. |
| **Algorithm Tuning** | `huber_delta`, `enable_residual_pruning`, `apply_kalman`, `kalman_process_std` | Huber $\delta > 0$; Kalman process std $> 0$. |

### Hashing & Stale Result Tracking
Every configuration instance computes an SHA-256 fingerprint:
$$\text{fingerprint} = \text{SHA256}(\text{JSON}(\mathbf{C}))_{[:16]}$$
If a user modifies any setting in the Workbench, the application marks previous results as **Outdated (Stale)** until **Run Simulation** is executed.

---

## 4. Product Demonstration Dashboard Architecture

AnchorGuard AI is presented as a single cohesive, presentation-ready product demonstration:

1. **Header & Context:** AnchorGuard AI branding, subtitle, 1-sentence indoor positioning problem definition, and live simulation status badge.
2. **Compact Scenario Controls:**
   - One-click Scenario Presets: **Normal Signals** (Ideal LOS), **Noisy Measurements** (AWGN interference), **Missing Readings** (Packet drops), and **Blocked Signals** (Severe NLOS multipath).
   - Quick settings: Anchor layout (Perimeter 6, Corners 4, Grid 9), noise level $\sigma$, packet drop rate, and random seed.
   - Prominent **Run Simulation** button.
3. **Interactive Floor Plan (Visual Centerpiece):**
   - High-contrast 2D floor plan with dynamic axis scaling.
   - Displays Ground Truth path (emerald green), Conventional Baseline NLS (red dashed), AnchorGuard Robust (sky blue), active anchors (amber triangles), and failed estimation steps (crosses).
4. **Performance Summary Metrics:**
   - Baseline RMSE, AnchorGuard RMSE, Percentage Error Reduction, Median Error, 95th-Percentile Error, and Valid Solution Rate.
   - Plain-English explanation of RMSE and fair improvement metrics.
5. **What Happened and Why? (Dynamic Scientific Narrative):**
   - Concise, dynamic 5-point explanation covering environmental conditions, measurement impact, trajectory observations, AnchorGuard algorithmic response, and physical indoor positioning significance.
6. **Supporting Diagnostic Evidence:**
   - Tracking Error over Time ($e_t$ vs $t$) and Cumulative Distribution Function (Error CDF).
7. **Advanced Analysis & Diagnostics (Collapsed Expander):**
   - Technical controls housed under a clean collapsed expander: algorithm tuning (Huber $\delta$, outlier pruning, Kalman filtering), raw anchor range residual diagnostics, multi-trial Monte Carlo evaluations (Mean $\pm$ Std), and full data exports (CSV time-series, JSON configuration, Markdown summary report).

---

## 5. Metric Definitions & Mathematical Denominators

| Metric | Mathematical Definition | Denominator |
| :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{N_{\text{valid}}} \sum_{t \in \mathcal{V}} \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2^2}$ | Valid steps $N_{\text{valid}}$ |
| **MAE** | $\frac{1}{N_{\text{valid}}} \sum_{t \in \mathcal{V}} \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2$ | Valid steps $N_{\text{valid}}$ |
| **Median Error** | $\text{Median}(\{ \|\hat{\mathbf{p}}_t - \mathbf{p}_t\|_2 \}_{t \in \mathcal{V}})$ | Valid steps $N_{\text{valid}}$ |
| **P95 Error** | 95th percentile of Euclidean errors | Valid steps $N_{\text{valid}}$ |
| **Valid Estimate Rate** | $(N_{\text{valid}} / N_{\text{total}}) \times 100\%$ | Total trajectory steps $N_{\text{total}}$ |
| **Out-of-Room Rate** | $(N_{\text{out\_of\_room}} / N_{\text{valid}}) \times 100\%$ | Valid estimated steps $N_{\text{valid}}$ |

*Out-of-room condition:* $\hat{x}_t < 0 \lor \hat{x}_t > W \lor \hat{y}_t < 0 \lor \hat{y}_t > H$.

---

## 6. Installation & Execution

### Prerequisites
- Python 3.12 (or 3.10+)

```bash
git clone https://github.com/GangOfCoders/AnchorGuard.git
cd AnchorGuard
pip install -r requirements.txt
```

### Launch Interactive Workbench
```bash
python -m streamlit run app.py
```
Open browser at `http://localhost:8501`.

### Run Automated Test Suite (43 Tests)
```bash
python -m pytest -v
```

### Run Multi-Trial Benchmark
```bash
python benchmark.py --trials 3 --out benchmark_results.csv
```
For quick validation (1 trial):
```bash
python benchmark.py --quick
```

---

## 7. Streamlit Community Cloud Deployment

### Step 1: Push Project to GitHub
Initialize the git repository locally and push to your GitHub account:

```bash
git init
git add .
git commit -m "feat: AnchorGuard AI 2.0 Workbench ready for deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **Create app** (or **Deploy an app**).
3. Fill in the deployment details:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

Streamlit Community Cloud will automatically detect `requirements.txt`, install dependencies, and launch `app.py`.

---

## 8. Quality Assurance Summary

- **Total Automated Tests:** 43 passed (100% pass rate).
- **Streamlit AppTest Coverage:** Verified clean startup, navigation across all 6 sections, and execution button behavior.
- **Execution Semantics:** Zero mutations on source configuration or room geometry; fair measurement reuse across estimators; reproducible seeds.
