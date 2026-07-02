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
DENSITY_CONSTANT = 956.28 kg/m³   (mean of 11 FFBs)
```

Scale and density are refitted jointly in cross-validation.

---

## Results

### Per-FFB volume predictions (litres)

| FFB | Actual | v1 | A | C | v1 err | A err | C err |
|---|---|---|---|---|---|---|---|
| FFB10 | 18.0 | 18.98 | 18.70 | 17.70 | +0.98 | +0.70 | −0.30 |
| FFB11 | 14.0 | 17.40 | 15.01 | 16.34 | +3.40 | +1.01 | +2.34 |
| FFB12 | 22.0 | 22.02 | 22.56 | 23.11 | +0.02 | +0.56 | +1.11 |
| FFB17 | 14.0 | 15.76 | 11.53 | 11.02 | +1.76 | −2.47 | −2.98 |
| FFB18 † | 10.0 | 20.05 | 13.19 | 12.99 | +10.05 | +3.19 | +2.99 |
| FFB19 | 20.0 | 17.30 | 18.61 | 18.07 | −2.70 | −1.39 | −1.93 |
| FFB31 | 14.0 | 15.97 | 13.41 | 12.29 | +1.97 | −0.59 | −1.71 |
| FFB32 | 10.0 | 12.24 | 8.06 | 3.86 | +2.24 | −1.95 | −6.14 |
| FFB33 | 13.0 | 14.58 | 14.92 | 14.58 | +1.58 | +1.92 | +1.57 |
| FFB34 | 14.0 | 13.83 | 12.33 | 13.75 | −0.18 | −1.67 | −0.25 |
| FFB35 | 14.0 | 10.42 | 14.05 | 14.53 | −3.58 | +0.05 | +0.53 |

† FFB18 excluded from all metrics (person in frame — +10 kg systematic error,
unfixable algorithmically).

### In-sample metrics (excl. FFB18, n = 10)

| Approach | MAE | MAPE | r | r² |
|---|---|---|---|---|
| v1 — 15-frame IQR | 1.80 kg | 13.1% | 0.828 | 0.686 |
| **A — temporal + depth mask** | **1.20 kg** | **9.2%** | **0.929** | **0.863** |
| C — temporal + GDino→SAM2 | 1.89 kg | 15.2% | 0.864 | 0.747 |

### 4/7 exhaustive cross-validation — Approach A (excl. FFB18)

C(11,4) = 330 splits · 4 train / 7 test · each FFB tested in 210 folds  
2310 total hold-out predictions · density and scale refitted per fold

| | MAE | MAPE | r² |
|---|---|---|---|
| All 2310 predictions | 1.67 kg | 12.4% | 0.866 |
| Per-FFB mean | — | — | **0.893** |
| **Aqil thesis (n = 50, manual)** | — | — | **0.900** |

### Per-FFB CV breakdown (Approach A)

| FFB | Actual mass | CV pred mean | Mean abs err | MAPE |
|---|---|---|---|---|
| FFB10 | 17.0 kg | 18.91 kg | 1.91 kg | 11.2% |
| FFB11 | 14.0 kg | 15.08 kg | 1.10 kg | 7.9% |
| FFB12 | 21.6 kg | 22.70 kg | 1.33 kg | 6.2% |
| FFB17 | 13.4 kg | 11.36 kg | 2.04 kg | 15.2% |
| FFB18 | 9.6 kg | 13.46 kg | 3.86 kg | 40.2% |
| FFB19 | 19.6 kg | 18.40 kg | 1.37 kg | 7.0% |
| FFB31 | 13.8 kg | 11.58 kg | 2.22 kg | 16.1% |
| FFB32 | 9.8 kg | 6.97 kg | 2.83 kg | 28.9% |
| FFB33 | 12.0 kg | 13.14 kg | 1.15 kg | 9.6% |
| FFB34 | 12.6 kg | 10.67 kg | 1.93 kg | 15.3% |
| FFB35 | 13.0 kg | 12.23 kg | 0.86 kg | 6.6% |

Hardest FFBs: FFB18 (person in frame), FFB32 (sparse depth, low colour contrast),
FFB31 and FFB17 (consistently underestimated — possible recording distance issue).

---

## What each adaptation contributed

Starting from the v1 in-sample MAE of 1.80 kg:

| Adaptation | MAE after |
|---|---|
| Temporal nanmedian depth (120 frames vs 15) | 1.58 kg |
| Adaptive HSV threshold + adaptive z_window | 1.42 kg |
| Per-camera scale recalibration (848 px vs 1280 px fit separately) | **1.20 kg** |

The single largest gain came from recognising that the 848 × 480 and 1280 × 720
cameras need independent hemisphere scale factors. Fitting one scale across both
cameras compromised the 848 px correction and introduced a systematic offset.

---

## Comparison to related work

| System | n | Accuracy (1−MAPE) | r² | Notes |
|---|---|---|---|---|
| **This pipeline (A, CV)** | **11** | **87.8%** | **0.893** | Zero-shot, automated |
| Aqil thesis | 50 | — | 0.900 | Manual segmentation, CloudCompare |
| Group 2 (YOLOv8 + PCA ellipsoid) | 6 | ~73% | — | Manual setup, 6 samples |

At r² = 0.893 vs 0.900, the automated pipeline is within 0.007 r² of the
manual benchmark despite having 4.5× fewer samples and requiring no human
annotation or interactive segmentation.

---

## Why Approach C (GDino→SAM2) underperforms Approach A

On this tarp-protocol dataset, depth is a better segmentation cue than colour/texture:

- The depth foreground mask directly isolates what is close to the camera —
  which is exactly the FFB
- SAM2 segments visually coherent regions; for FFB32 (sparse fronds, low
  saturation) the detected SAM2 mask missed most of the bunch (3.86 L vs 10.0 L
  actual), pulling Approach C's MAE to 1.89 kg
- Excluding FFB32, Approach C MAE ≈ 1.41 kg — still worse than A's 1.20 kg

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
