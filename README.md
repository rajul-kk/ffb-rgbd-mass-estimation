# Zero-Shot FFB Mass Estimation via RGB-D

Automated mass estimation of oil palm Fresh Fruit Bunches (FFBs) from a single Intel RealSense D455 depth camera, with no fine-tuning or labelled training data required.

---

## Results

Evaluated on 11 FFBs (excluding FFB18 — person in frame):

| Approach | Segmentation | MAE | MAPE | r² |
|---|---|---|---|---|
| v1 — baseline | Depth mask, 15 frames | 1.95 kg | 14.2% | 0.691 |
| **A — best** | Temporal depth mask | **1.27 kg** | **9.8%** | **0.857** |
| C — comparison | Grounding DINO → SAM2 | 1.99 kg | 15.8% | 0.753 |

**4/7 exhaustive CV — Approach A (excl. FFB18):** MAE = 1.66 kg · r² = 0.886 per-FFB mean  
**Aqil thesis benchmark** (n = 50, manual CloudCompare): r² = 0.900

---

## Overview

The pipeline takes a short `.bag` recording of an FFB on a tarp and outputs an estimated mass. The key innovation in v2 is computing a single temporally-fused depth map (nanmedian over 120 frames) instead of aggregating per-frame volumes, which removes sensor noise before segmentation.

```
.bag recording
    │
    ├─ Temporal accumulation: nanmedian(120 frames) → fused depth map
    │     (cached to fused_cache/{FFB}_fused.npz)
    │
    ├─ Approach A (best): depth foreground mask → 2.5D grid volume
    │
    └─ Approach C (comparison): Grounding DINO → SAM2 mask → 2.5D grid volume
```

Detection cascade (fallback when depth mask fails):

```
Grounding DINO (fp32 + autocast)  →  YOLO-World  →  default centre-crop box
```

---

## Method

### 1. Temporal depth accumulation (Approach A / C)

- Streams up to 120 frames from the `.bag` file; applies median blur to each depth frame
- Stacks frames into a 3D array and takes `nanmedian` along the time axis → single dense depth map
- Fills remaining NaN pixels (no valid depth in any frame) with 0
- Result cached per FFB; delete `fused_cache/` or set `CLEAR_CACHE = True` to recompute

### 2. Segmentation

**Depth foreground mask (primary — Approaches A, v1)**

1. `z_front` = 5th-percentile depth of valid pixels (closest surface)
2. `_auto_margin`: fits a depth histogram over 5–45 cm behind `z_front` to locate the tarp peak; protrusion height = tarp depth − `z_front` − 5 cm, clipped to [8, 22] cm
3. Adaptive HSV thresholds from 25-pixel image border (`s_min` = 85th-percentile border saturation, clamped 15–70)
4. For 848 × 480 cameras: seed contour → `_expand_mask_bbox` with adaptive padding (8% of bbox diagonal, clamped 8–40 px)
5. Falls through to GDino → YOLO-World if no contour found

**Grounding DINO + SAM2 (primary — Approach C; fallback — Approach A)**

- `IDEA-Research/grounding-dino-base` kept in fp32; `torch.autocast(dtype=float16)` handles mixed precision for the forward pass (avoids BERT text encoder dtype conflicts)
- Three prompts tried in order; stops at first detection above 0.50 confidence
- SAM2 (`sam2_hiera_small`) segments the detected box into a pixel mask

### 3. Volume integration (2.5D grid)

Projects masked depth pixels into 3D, clips to `z_front + z_window`, then integrates:

```
V_raw = Σ (z_ref − z_min_per_cell) × (0.002 m)²
z_ref = 95th-percentile depth of point cloud
```

Per-camera hemisphere correction (least-squares fit to ground truth):

| Resolution | Scale |
|---|---|
| 848 × 480 | 2.53 |
| 1280 × 720 | 2.02 |

### 4. Mass prediction

```
mass = volume × DENSITY_CONSTANT
DENSITY_CONSTANT = 956.28 kg/m³   (mean of all 50 rows in ground_truth.csv)
```

Density cancels out of the fitted prediction (mass = V_raw · Σ(m·r)/Σ(r²)), so only one gain is actually fitted.

### What each adaptation contributed (v1 → A)

| Adaptation | MAE |
|---|---|
| v1 baseline | 1.95 kg |
| + Temporal nanmedian (120 frames) | ~1.58 kg |
| + Adaptive HSV threshold + z_window | ~1.42 kg |
| + Per-camera scale recalibration | **1.27 kg** |

---

## Repository Structure

```
RGBD-Mass/
├── notebooks/
│   ├── kaggle_ffb_pipeline_v2.ipynb   # Primary notebook (run top-to-bottom)
│   └── kaggle_ffb_pipeline.ipynb      # v1 baseline (legacy)
├── perception_pipeline.py             # Base pipeline class (patched by notebook)
├── bag_reader.py                      # RealSense .bag reading utilities
├── ground_truth.csv                   # 11 FFBs: mass, volume, density
├── validation_density.csv             # 10 independent FFBs for density validation
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
| `01c74738` | Per-camera scale recalibration (least-squares refit) |
| `compare` | Results table + MAE / MAPE / r² + scatter plots |
| `fd5e63ad` | 4/7 exhaustive cross-validation (330 splits, 2310 predictions) |

---

## Setup

### Requirements

```
Python 3.10+
torch >= 2.0
ultralytics >= 8.2
sam2
pyrealsense2
transformers >= 4.38
opencv-python
scipy
pandas
```

### Running on Kaggle

Upload the dataset as a private Kaggle dataset. Set `PROJECT_DIR` in the `preflight` cell to the dataset path. Run cells top-to-bottom.

The notebook auto-detects GPU availability and falls back to CPU. Pipeline initialisation is serialised via a threading lock to avoid a TorchScript redefinition crash on multi-GPU environments.

### Running locally

```bash
pip install ultralytics sam2 pyrealsense2 transformers scipy

# Run notebook
jupyter notebook notebooks/kaggle_ffb_pipeline_v2.ipynb
```

Set `PROJECT_DIR` in `preflight` to the directory containing `perception_pipeline.py`, `bag_reader.py`, and the CSV files.

---

## Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| `DENSITY_CONSTANT` | 956.28 kg/m³ | Mean of all 50 GT rows (mass ÷ rounded volume); cancels after scale refit |
| `SCALE_BY_WIDTH` | `{1280: 2.02, 848: 2.53}` | Hemisphere correction — re-derive if shooting distance changes |
| `MAX_SCAN` | 120 | Frames scanned for temporal accumulation |
| `N_FRAMES` | 15 | Frames used by v1 baseline |
| `_GDINO_BOX_THRESH` | 0.15 | Minimum GDino score to accept a detection |
| `CLEAR_CACHE` | False | Set True to force recomputation of fused depth maps |

### Recalibration checklist for new data

1. Re-derive `DENSITY_CONSTANT` from ≥ 10 FFBs (weigh + water displacement)
2. Re-run the scale recalibration cell after `run_all` to update `SCALE_BY_WIDTH`
3. For a new camera resolution: add it to `SCALE_BY_WIDTH`; add a bbox-expansion branch if depth is noisy
4. Check tarp colour matches `colour_s_min` range (green tarp default)
5. Verify `np.clip(margin, 0.08, 0.22)` bounds cover your FFB population's protrusion height

---

## Known Limitations

| FFB | Issue |
|---|---|
| FFB18 | Person in frame — +10 kg systematic error, excluded from all metrics |
| FFB32 | Sparse depth, desaturated fronds — underestimated by all approaches |
| FFB31, FFB17 | Consistently underestimated; possible non-standard camera distance |

The pipeline is a **controlled-protocol instrument**: it requires a flat tarp background, top-down camera placement, and consistent shooting distance (≈ 1.2–1.5 m). Performance will degrade outside these bounds without recalibration.

---

## Comparison

| System | n | Accuracy (1−MAPE) | r² | Notes |
|---|---|---|---|---|
| **This pipeline (A, CV)** | **11** | **87.7%** | **0.886** | Zero-shot, automated |
| Aqil thesis | 50 | — | 0.900 | Manual segmentation, CloudCompare |
| Group 2 (YOLOv8 + PCA ellipsoid) | 6 | ~73% | — | Manual setup, 6 samples |
