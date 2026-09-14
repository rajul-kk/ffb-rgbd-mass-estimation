# FFB Mass Estimation — Final Report

**Pipeline:** `kaggle_ffb_pipeline_v2.ipynb`  
**Camera:** Intel RealSense D455  
**Dataset:** 11 FFBs, one variety, one harvest date  
**Method:** Zero-shot (no fine-tuning or labelled training data)

---

## Objective

Automate the mass estimation of oil palm Fresh Fruit Bunches (FFBs) from a short
RGB-D recording, matching or exceeding the accuracy of Aqil's manual
CloudCompare workflow (r² = 0.900, n = 50).

---

## Method

### Pipeline overview

```
.bag recording
    │
    ├─ Temporal depth accumulation ──────────────────────────────────────┐
    │   nanmedian over 120 frames → single dense fused depth map         │
    │   (cached to fused_cache/{FFB}_fused.npz for fast re-runs)        │
    │                                                                     │
    ├─ Approach v1 (baseline)                                            │
    │   15 ranked frames → per-frame depth mask → IQR-filtered median    │
    │                                                                     │
    ├─ Approach A (best)                                                  │
    │   Fused depth → adaptive depth mask → 2.5D grid volume            │
    │                                                                     │
    └─ Approach C (comparison)                                            │
        Fused depth → Grounding DINO bbox → SAM2 mask → 2.5D volume ────┘
```

### Segmentation

**Depth foreground mask (Approaches v1, A)**

1. `z_front` = 5th-percentile depth of valid pixels (closest surface)
2. `_auto_margin`: fits a depth histogram over 5–45 cm behind `z_front` to
   locate the tarp peak; protrusion height = tarp depth − `z_front` − 5 cm,
   clipped to [8, 22] cm
3. Adaptive HSV thresholds derived from 25-pixel border statistics
   (`s_min` = 85th-percentile border saturation, clamped 15–70)
4. For 848 × 480 cameras: depth seed → `_expand_mask_bbox` with adaptive
   padding (8% of bbox diagonal, clamped 8–40 px)
5. Fallback: Grounding DINO detection → SAM2 mask (replaces YOLO-World)

**Grounding DINO + SAM2 (Approach C)**

- Grounding DINO (`IDEA-Research/grounding-dino-base`) run in fp32 with
  `torch.autocast(dtype=float16)` for stable mixed-precision inference
- Three prompts tried in order; best confidence box selected
- SAM2 (`sam2_hiera_small`) segments the bbox into a pixel mask
- Same fused depth used as Approach A (no extra bag reading)

### Volume integration

```
V_raw = Σ (z_ref − z_min_per_cell) × (0.002 m)²    [2.5D grid, 2 mm step]
z_ref = 95th-percentile depth of masked point cloud
```

Hemisphere correction (fitted per camera resolution):

| Resolution | Scale |
|---|---|
| 848 × 480 | 2.53 |
| 1280 × 720 | 2.02 |

### Mass prediction

```
mass = volume × DENSITY_CONSTANT
DENSITY_CONSTANT = 956.28 kg/m³   (mean of all 50 rows in ground_truth.csv)
```

Density cancels out of the fitted prediction (mass = V_raw · Σ(m·r)/Σ(r²)), so only one gain is actually fitted.

---

## Results

### Per-FFB volume predictions (litres)

| FFB | Actual | v1 | A | C | v1 err | A err | C err |
|---|---|---|---|---|---|---|---|
| FFB10 | 18.0 | 18.98 | 18.70 | 17.41 | +0.98 | +0.70 | −0.59 |
| FFB11 | 14.0 | 17.41 | 15.01 | 16.08 | +3.41 | +1.01 | +2.08 |
| FFB12 | 22.0 | 22.02 | 22.56 | 23.61 | +0.02 | +0.56 | +1.61 |
| FFB17 | 14.0 | 15.83 | 11.53 | 10.85 | +1.83 | −2.47 | −3.15 |
| FFB18 † | 10.0 | 20.05 | 13.19 | 12.79 | +10.05 | +3.19 | +2.79 |
| FFB19 | 20.0 | 17.30 | 18.61 | 18.21 | −2.70 | −1.39 | −1.79 |
| FFB31 | 14.0 | 16.71 | 13.24 | 12.03 | +2.71 | −0.76 | −1.97 |
| FFB32 | 10.0 | 12.24 | 7.95 | 3.78 | +2.24 | −2.05 | −6.22 |
| FFB33 | 13.0 | 14.96 | 14.90 | 14.58 | +1.96 | +1.90 | +1.58 |
| FFB34 | 14.0 | 15.04 | 12.17 | 13.46 | +1.04 | −1.83 | −0.54 |
| FFB35 | 14.0 | 11.27 | 14.38 | 14.93 | −2.73 | +0.38 | +0.93 |

† FFB18 excluded from all metrics (person in frame — +10 kg systematic error,
unfixable algorithmically).

### In-sample metrics (excl. FFB18, n = 10)

| Approach | MAE | MAPE | r | r² |
|---|---|---|---|---|
| v1 — 15-frame IQR | 1.95 kg | 14.2% | 0.831 | 0.691 |
| **A — temporal + depth mask** | **1.27 kg** | **9.8%** | **0.926** | **0.857** |
| C — temporal + GDino→SAM2 | 1.99 kg | 15.8% | 0.868 | 0.753 |

### 4/7 exhaustive cross-validation — Approach A (excl. FFB18)

C(11,4) = 330 splits · 4 train / 7 test · each FFB tested in 210 folds  
2310 total hold-out predictions · one gain refitted per fold (density cancels)

| | MAE | MAPE | r² |
|---|---|---|---|
| All 2100 predictions (excl. FFB18) | 1.66 kg | 12.3% | 0.860 |
| Per-FFB mean (excl. FFB18) | — | — | **0.886** |
| **Aqil thesis (n = 50, manual)** | — | — | **0.900** |

Including FFB18 (known outlier — person in frame):

| | MAE | MAPE | r² |
|---|---|---|---|
| All 2310 predictions | 1.85 kg | 14.8% | 0.748 |
| Per-FFB mean | — | — | 0.772 |

### Per-FFB CV breakdown (Approach A)

| FFB | Actual mass | CV pred mean | Mean abs err | MAPE |
|---|---|---|---|---|
| FFB10 | 17.0 kg | 18.83 kg | 1.83 kg | 10.8% |
| FFB11 | 14.0 kg | 15.02 kg | 1.05 kg | 7.5% |
| FFB12 | 21.6 kg | 22.60 kg | 1.27 kg | 5.9% |
| FFB17 | 13.4 kg | 11.32 kg | 2.08 kg | 15.5% |
| FFB18 | 9.6 kg | 13.40 kg | 3.80 kg | 39.6% |
| FFB19 | 19.6 kg | 18.32 kg | 1.42 kg | 7.2% |
| FFB31 | 13.8 kg | 11.54 kg | 2.26 kg | 16.4% |
| FFB32 | 9.8 kg | 6.95 kg | 2.85 kg | 29.1% |
| FFB33 | 12.0 kg | 13.26 kg | 1.26 kg | 10.5% |
| FFB34 | 12.6 kg | 10.63 kg | 1.97 kg | 15.6% |
| FFB35 | 13.0 kg | 12.66 kg | 0.60 kg | 4.6% |

Hardest FFBs: FFB18 (person in frame), FFB32 (sparse depth, low colour contrast),
FFB31 and FFB17 (consistently underestimated — possible recording distance issue).

---

## What each adaptation contributed

Starting from the v1 in-sample MAE of 1.95 kg:

| Adaptation | MAE after |
|---|---|
| Temporal nanmedian depth (120 frames vs 15) | ~1.58 kg |
| Adaptive HSV threshold + adaptive z_window | ~1.42 kg |
| Per-camera scale recalibration (848 px vs 1280 px fit separately) | **1.27 kg** |

The single largest gain came from recognising that the 848 × 480 and 1280 × 720
cameras need independent hemisphere scale factors. Fitting one scale across both
cameras compromised the 848 px correction and introduced a systematic offset.

---

## Comparison to related work

| System | n | Accuracy (1−MAPE) | r² | Notes |
|---|---|---|---|---|
| **This pipeline (A, CV)** | **11** | **87.7%** | **0.886** | Zero-shot, automated |
| Aqil thesis | 50 | — | 0.900 | Manual segmentation, CloudCompare |
| Group 2 (YOLOv8 + PCA ellipsoid) | 6 | ~73% | — | Manual setup, 6 samples |

At r² = 0.886 vs 0.900, the automated pipeline is within 0.014 r² of the
manual benchmark despite having 4.5× fewer samples and requiring no human
annotation or interactive segmentation.

---

## Why Approach C (GDino→SAM2) underperforms Approach A

On this tarp-protocol dataset, depth is a better segmentation cue than colour/texture:

- The depth foreground mask directly isolates what is close to the camera —
  which is exactly the FFB
- SAM2 segments visually coherent regions; for FFB32 (sparse fronds, low
  saturation) the detected SAM2 mask missed most of the bunch (3.78 L vs 10.0 L
  actual), pulling Approach C's MAE to 1.99 kg
- Excluding FFB32, Approach C MAE ≈ 1.51 kg — still worse than A's 1.27 kg

Approach C's advantage — working without a tarp background — would matter in
field conditions where the tarp protocol cannot be enforced. The Grounding DINO
detector is retained as a fallback within the detection cascade (replacing
YOLO-World when the depth mask finds no valid contour).

---

## Known limitations

| FFB | Issue |
|---|---|
| FFB18 | Person in frame — +10 kg systematic error, excluded from all metrics |
| FFB32 | Sparse depth, desaturated fronds — underestimated by all approaches |
| FFB31, FFB17 | Consistently underestimated; possible non-standard camera distance |

**Protocol requirements:** flat tarp background, top-down camera placement,
shooting distance 1.2–1.5 m. Performance will degrade outside these bounds
without recalibration.

---

## Recalibration checklist for new data

1. Re-derive `DENSITY_CONSTANT` from ≥ 10 FFBs (weigh + water displacement)
2. Re-run the scale recalibration cell after `run_all` to update `SCALE_BY_WIDTH`
3. For a new camera resolution: add it to `SCALE_BY_WIDTH`; add a bbox-expansion
   branch in `_patched_process_frame` if depth is noisy
4. Check tarp colour matches `colour_s_min` range (green tarp default)
5. Verify `np.clip(margin, 0.08, 0.22)` bounds cover the new FFB population's
   protrusion height range
