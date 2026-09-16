# Zero-Shot FFB Mass Estimation via RGB-D

Automated mass estimation of oil palm Fresh Fruit Bunches (FFBs) from a single Intel RealSense D455 recording. Segmentation and geometry need no fine-tuning or labelled training data; one calibration gain is fitted to weighed bunches.

Full analysis: [report.md](report.md).

---

## Results

Evaluated on 10 FFBs (FFB18 excluded: person in frame).

**Held-out (leave-one-out: each bunch predicted by a gain fitted on the other nine)**

| Approach A calibration | MAE (95% CI) | MAPE | Pearson r² | R² |
|---|---|---|---|---|
| Per-session gain (as in v2) | 1.65 kg (1.24–2.10) | 12.3% | 0.81 | 0.73 |
| One gain for both sessions | 1.60 kg (1.16–2.01) | 11.8% | 0.89 | 0.74 |

| Baseline | MAE | MAPE | Pearson r² |
|---|---|---|---|
| Caliper ellipsoid (π/6·W·L·T), fitted on the other 40 bunches | 1.67 kg | 10.7% | 0.74 |

- **Matches a tape measure.** The pipeline equals a caliper measurement on absolute error and ranks bunches by size better, without manual measurement.
- **Prediction intervals.** 90% intervals are about ±3 kg wide.
- **Sample size.** With 10 bunches, MAE differences under about 0.6 kg cannot be detected.

**In-sample (gain fitted on the evaluated bunches; measures fit, not prediction)**

| Approach | Segmentation | MAE | MAPE | Pearson r² |
|---|---|---|---|---|
| v1 — baseline | Depth mask, 15 frames | 1.95 kg | 14.2% | 0.691 |
| A — best | Temporal depth mask | 1.27 kg | 9.8% | 0.857 |
| C — comparison | Grounding DINO → SAM2 | 1.99 kg | 15.8% | 0.753 |

**Repeated splits, Approach A** (all 330 splits; FFB18 is used for training but never scored).

| Calibration | 4 train / 7 test | 7 train / 4 test |
|---|---|---|
| One gain | 1.66 kg, per-FFB Pearson r² 0.886 | 1.59 kg, 0.89 |
| Per-session gains (the in-sample model) | 1.97 kg, 0.770 | 1.65 kg, 0.79 |

**Attempted fixes.** None of the v3 changes (steady frames, filter order, grid-free volume, tarp plane, unaligned depth) beats v2 under leave-one-out, 4/7 or 7/4 evaluation (report §9).

**Open problem.** Errors differ systematically between the two recording sessions (report §9). More bunches recorded in one fixed setup are needed before any change can be confirmed.

---

## Overview

The v2 notebook takes a short `.bag` recording of an FFB on a tarp and outputs an estimated mass. It fuses depth over time (per-pixel median) instead of aggregating per-frame volumes, which removes sensor noise before segmentation.

```
.bag recording
    │
    ├─ Temporal accumulation: per-pixel median over the frames read → fused depth map
    │
    ├─ Approach A (best): depth foreground mask → 2.5D grid volume
    │
    └─ Approach C (comparison): Grounding DINO → SAM2 mask → 2.5D grid volume
```

The `ffb` package re-implements v2 exactly for CPU and adds optional changes, evaluated in `notebooks/ffb_pipeline_v3.ipynb` (none improved held-out accuracy):

- **Steady frames:** fuse only frames recorded before the scene is disturbed.
- **Frame reading:** frames are copied (no 32-frame cap), and invalid pixels are kept out of the median filter.
- **Volume:** grid-free volume above a fitted tarp plane.
- **Depth grid:** optionally leave session 1 depth unaligned to colour.
- **Segmentation:** SAM 3 in place of Grounding DINO → SAM 2 → YOLO-World.

---

## Method (v2)

### 1. Temporal depth accumulation

- Reads up to 120 frames, applies a 3 × 3 median filter to each depth frame, marks zero depth invalid, and takes the per-pixel median over time.
- **As run:**
  - Split-bag bunches (FFB31–35) used only 32 frames, because the reader exhausts the RealSense frame pool.
  - Combined-bag bunches (FFB10–19) used every frame, including frames after the scene was disturbed.
  - The median filter runs before invalid pixels are masked.
- Fused maps are cached per FFB; delete `fused_cache/` or set `CLEAR_CACHE = True` to recompute.

### 2. Segmentation

**Depth foreground mask (Approaches A, v1)**

1. `z_front` = 5th-percentile depth of valid pixels (closest surface).
2. `_auto_margin` locates the tarp peak in a depth histogram 5–45 cm behind `z_front`. Protrusion height = tarp depth − `z_front` − 5 cm, clipped to [8, 22] cm.
3. Adaptive HSV thresholds come from the 25-pixel image border (`s_min` = 85th-percentile border saturation, clamped 15–70).
4. Split-bag bunches (848 × 480 depth grid): the seed contour is expanded with `_expand_mask_bbox`, padding 8% of the bounding-box diagonal (clamped 8–40 px).
5. Fallback to Grounding DINO → YOLO-World if no contour is found. It never triggered on this dataset.

**Depth grid.** One camera records all depth at 848 × 480. Combined bags align depth to the 1280 × 720 colour frame, so the pipeline's "848 vs 1280" branch separates the two recording sessions, not two cameras. Combined-bag colour frames reach the pipeline in BGR order. The depth mask is unaffected, but Approach C's detectors see swapped channels for session 1.

**Grounding DINO + SAM2 (Approach C)**

- `IDEA-Research/grounding-dino-base` in fp32 under `torch.autocast(dtype=float16)`.
- Three prompts are tried; the first detection above 0.50 confidence is kept.
- SAM2 (`sam2_hiera_small`) segments the detected box into a pixel mask.

### 3. Volume integration (2.5D grid)

Projects masked depth pixels into 3D, clips to `z_front + z_window`, then integrates:

```
V_raw = Σ (z_ref − z_min_per_cell) × (0.002 m)²
z_ref = 95th-percentile depth of the point cloud
```

The 2 mm cells are finer than the pixel footprint on the 848 × 480 grid, so many cells stay empty. With tight masks, `z_ref` sits above the tarp. A per-session gain fitted to ground truth absorbs both effects: **2.25** for session 1 and **2.53** for session 2 (refitted from starting values 2.02 / 2.53).

### 4. Mass prediction

```
mass = gain × V_raw × DENSITY_CONSTANT
DENSITY_CONSTANT = 956.28 kg/m³   (mean of all 50 rows in ground_truth.csv)
```

Density cancels out of the fitted prediction (mass = V_raw · Σ(m·r)/Σ(r²)), so only one gain is actually fitted.

---

## Repository Structure

```
RGBD-Mass/
├── notebooks/
│   ├── kaggle_ffb_pipeline_v2.ipynb   # v2 pipeline (Kaggle, GPU optional)
│   ├── kaggle_ffb_pipeline.ipynb      # v1 baseline (legacy)
│   └── ffb_pipeline_v3.ipynb          # CPU re-evaluation: parity, fixes, geometry checks
├── ffb/                               # Package: bag IO, fusion, segmentation, volume, evaluation, SAM 3
├── scripts/
│   ├── cpu_eval.py                    # Reproduces v2 on CPU and evaluates each fix
│   └── moge_check.py                  # MoGe-2 monocular-depth cross-check
├── tests/                             # Synthetic geometry, fusion, evaluation and SAM 3 tests
├── results/                           # CPU evaluation outputs (CSV, JSON, plots)
├── perception_pipeline.py             # Base pipeline class (patched by the v2 notebook)
├── bag_reader.py                      # RealSense .bag reading used by the v2 notebook
├── ground_truth.csv                   # 50 FFBs: mass, caliper W/L/T, displaced volume, derived density
├── validation_density.csv             # 10 FFBs for density validation
├── requirements.txt                   # Pinned CPU dependencies for ffb/, scripts and tests
├── report.md                          # Full results and analysis
└── data/
    └── FFB{N}/                        # .bag recording(s) per FFB
```

### Notebook cell map (v2)

| Cell | Purpose |
|---|---|
| `preflight` | Locate data, detect GPU count |
| `install` | Install ultralytics, sam2, pyrealsense2, transformers |
| `weights` | Download SAM2 + YOLO-World weights |
| `patches` | All pipeline patches: segmentation, volume integration, frame processing |
| `bundlepatch` | `process_bundle_multi_frame` patch (v1 ranked selection + IQR) |
| `factory` | Pipeline factory (`make_pipeline`) |
| `a9cc3e00` | Grounding DINO loader + SAM2 fp32 guard |
| `approach_a` | Temporal accumulation + depth mask (Approach A) |
| `approach_c` | Grounding DINO → SAM2 segmentation (Approach C) |
| `run_all` | Process all bundles; fused depth shared between A and C |
| `01c74738` | Per-session gain refit (least squares) |
| `compare` | In-sample results table + MAE / MAPE / r² + scatter plots |
| `fd5e63ad` | 4/7 exhaustive cross-validation with one gain (330 splits, 2310 predictions) |

---

## Setup

### v2 notebook on Kaggle

Requirements: Python 3.10+, torch ≥ 2.0, ultralytics ≥ 8.2, sam2, pyrealsense2, transformers ≥ 4.38, opencv-python, scipy, pandas.

Upload the dataset as a private Kaggle dataset, set `PROJECT_DIR` in the `preflight` cell to its path, and run cells top-to-bottom. The notebook detects GPUs and falls back to CPU.

### CPU re-evaluation (local)

```bash
pip install -r requirements.txt
python -m pytest                    # 32 tests, a few seconds
python scripts/cpu_eval.py          # v2 parity + fix ladder, ~6 min cold, seconds with fused_cache/
jupyter nbconvert --to notebook --execute notebooks/ffb_pipeline_v3.ipynb
```

Bags are read from `data/`; fused depth is cached in `fused_cache/` under a key that includes the fusion settings.

### Optional models (MoGe-2, SAM 3)

Use a separate environment so the system transformers install is untouched:

```bash
python -m venv --system-site-packages .venv-models
.venv-models\Scripts\python -m pip install -U pip "transformers>=5" git+https://github.com/microsoft/MoGe.git
.venv-models\Scripts\python scripts\moge_check.py        # ~30 s per bunch on CPU
```

SAM 3 additionally needs approved access to `facebook/sam3` and `.venv-models\Scripts\hf auth login`. Then run `.venv-models\Scripts\python scripts\cpu_eval.py --sam3`.

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `DENSITY_CONSTANT` | 956.28 kg/m³ | Mean of all 50 ground-truth rows (mass ÷ rounded volume); cancels after the gain refit |
| `SCALE_BY_WIDTH` | `{1280: 2.02, 848: 2.53}` | Starting gains, keyed by depth-grid width (i.e. recording session); refitted to 2.25 / 2.53 |
| `MAX_SCAN` | 120 | Frame limit for temporal accumulation; split bags stop at 32 in v2 |
| `N_FRAMES` | 15 | Frames used by the v1 baseline |
| `_GDINO_BOX_THRESH` | 0.15 | Minimum Grounding DINO score to accept a detection |
| `CLEAR_CACHE` | False | Set True to force recomputation of fused depth maps |

### For new data

1. Record in one fixed setup (background, recording layout, camera height). Keep the scene still for the first ~5 s, and set depth units to 100 µm.
2. Fit one gain per recording setup from at least ~10 bunches; density does not need re-deriving because it cancels.
3. Measure displaced volume to 0.1 L, and document whether caliper measurements include spikes.
4. Check the tarp saturation range (`colour_s_min`) and the [8, 22] cm protrusion clip for the new background and bunch population.
5. Report held-out metrics (leave-one-out or a separate test set) with confidence intervals.

---

## Known Limitations

| Item | Issue |
|---|---|
| FFB18 | Person in frame; excluded from all metrics (v1 overestimated its volume by 10 L, Approach A by 3.2 L) |
| FFB32 | Largest error; underestimated by every approach; not caused by missing depth |
| Session effect | The two sessions differ in scene, background, recording layout and bunch batch; with 5–6 bunches each, their effects cannot be separated |
| Ground truth | Displaced volume rounded to whole litres (≈0.24 kg MAE floor); density column derived from it |
| Recordings | Most scenes are disturbed partway through; v2 fuses those frames for session 1 |
| Sample size | 10 evaluable bunches; MAE differences under ~0.6 kg cannot be detected |

The pipeline is a **controlled-protocol instrument**: flat background, top-down camera, camera-to-tarp distance of about 1.55 m (as measured in both sessions). Performance at other distances or backgrounds is untested.

---

## Comparison

| System | n | MAE | 1 − MAPE | r² | Notes |
|---|---|---|---|---|---|
| Aqil, manual CloudCompare | 10 | **1.45 kg** | 90.1% | 0.797 (Pearson) | Manual segmentation; no gain fitted to vision output |
| This pipeline (A), leave-one-out, one gain | 10 | 1.60 kg | 88.2% | **0.886** (Pearson) | Automated; gain fitted on the other 9 |
| This pipeline (A), leave-one-out, per-session gains | 10 | 1.65 kg | 87.7% | 0.810 (Pearson) | Automated; gains fitted on the other 9 |
| Caliper ellipsoid, fitted on 40 other bunches | 10 | 1.67 kg | 89.3% | 0.740 (Pearson) | Manual measurement, same bunches |
| Group 2 (YOLOv8 + PCA ellipsoid) | 5 | 4.4 kg | 74.4% | — | Scale factor tuned; session 1 only |

All rows use the same ground truth and the same bunches. Against Aqil the paired difference is **+0.15 kg (95% CI −0.36 to +0.68)** — indistinguishable on 10 bunches. Aqil's own headline over all 50 bunches is better (MAE 1.15 kg, Pearson r² 0.902) and would need roughly 0.45 kg lower MAE to beat. Group 2's published 73% is the mean of (1 − |error| ÷ actual) over 6 bunches including the excluded FFB18, and their report calls it a projected goal; on the 5 comparable bunches this pipeline scores 1.57 kg and 90.4%. See Section 7 of [report.md](report.md).
