# FFB Mass Estimation — Final Report

**Pipeline:** `notebooks/kaggle_ffb_pipeline_v2.ipynb` (results); `notebooks/ffb_pipeline_v3.ipynb` and `scripts/cpu_eval.py` (re-evaluation)  
**Camera:** Intel RealSense D455 (depth 848 × 480, colour 1280 × 720)  
**Dataset:** 11 FFBs of one variety, recorded in two sessions; ground truth for 50 FFBs  
**Method:** zero-shot segmentation and geometry, with one calibration gain fitted to ground truth

> **Revision note.** This version corrects the first report after a full CPU re-run of the pipeline. The re-run reproduces every v2 number exactly. It also shows that the headline figures were measured on the bunches used for calibration, that the cross-validation used a different calibration from the one described, and that several method details and four input bugs were misreported. Section 5 gives the corrected evaluation; Section 12 lists every change.

---

## 1. Summary

- **Held-out accuracy.** Approach A predicts mass with a leave-one-out MAE of **1.60–1.65 kg** (MAPE 11.8–12.3%, 95% CI about 1.2–2.1 kg) on 10 bunches. The previously reported 1.27 kg was in-sample.
- **Baseline.** A caliper ellipsoid fitted on 40 other bunches reaches **1.67 kg** on the same 10. The pipeline matches it on absolute error and ranks bunches by size better (Pearson r² 0.89 vs 0.74), with no manual measurement.
- **Uncertainty.** 90% prediction intervals are about ±3 kg wide.
- **Sample size.** With 10 bunches, only MAE differences of about 0.6 kg or more can be detected.
- **Main open problem.** Errors differ systematically between the two recording sessions. More bunches recorded in one fixed setup are needed to resolve this.

---

## 2. Objective

Automate the mass estimation of oil palm Fresh Fruit Bunches (FFBs) from a short RGB-D recording, matching or exceeding the accuracy of Aqil's manual CloudCompare workflow (r² = 0.900, n = 50).

---

## 3. Data

| | Session 1 | Session 2 |
|---|---|---|
| Bunches | FFB10, 11, 12, 17, 18, 19 | FFB31, 32, 33, 34, 35 |
| Recording | One bag with depth and colour | Separate depth and colour bags |
| Scene (sample frames) | Indoors, cloth tarp | Outdoors, flat board |
| Camera-to-tarp distance (measured from depth) | 1.55 m | 1.55–1.57 m |

- **Depth.** Recorded at 848 × 480 with 1 mm depth units in both sessions.
- **Ground truth.** `ground_truth.csv` lists 50 bunches with mass, caliper width, length and thickness, and displaced volume rounded to whole litres.
  - `True_Density_kg_L` is mass ÷ rounded volume. It is derived, not measured separately.
  - The rounding alone puts a floor of about 0.24 kg under any method's MAE.
  - With exact volumes, a linear fit of mass on displaced volume gives R² 0.982 (MAE 0.44 kg, n = 50). All remaining error is in volume estimation.
- **Exclusion.** FFB18 is excluded from all metrics because a person is in frame. v1 overestimated its volume by 10 L and Approach A by 3.2 L.

---

## 4. Method (as run in v2)

### 4.1 Pipeline overview

```
.bag recording
    │
    ├─ Temporal depth accumulation
    │   per-pixel median over the frames read (up to 120; see 4.2)
    │
    ├─ Approach v1 (baseline)
    │   15 ranked frames → per-frame depth mask → IQR-filtered median
    │
    ├─ Approach A (best)
    │   fused depth → adaptive depth mask → 2.5D grid volume
    │
    └─ Approach C (comparison)
        fused depth → Grounding DINO box → SAM2 mask → 2.5D grid volume
```

### 4.2 Temporal depth accumulation

Each depth frame gets a 3 × 3 median filter, zero depth is marked invalid, and the per-pixel median is taken over time. Three problems affect the inputs as run:

- **Frame count.** Session 2 bunches used only **32 frames**, not 120: the bag reader keeps references to frames and exhausts the RealSense frame pool. Session 1 bunches used all 59–102 frames.
- **Disturbed frames.** In most recordings the scene is disturbed partway through (from about frame 21–35 in session 1). Session 1's fused depth includes those frames.
- **Filter order.** The median filter runs before invalid pixels are masked. Zero-depth neighbours shift edge pixels toward nearer depths or remove them.

### 4.3 Segmentation

**Depth foreground mask (Approaches v1, A)**

1. `z_front` = 5th-percentile depth of valid pixels (closest surface).
2. `_auto_margin` finds the tarp peak in a depth histogram 5–45 cm behind `z_front`. Protrusion height = tarp depth − `z_front` − 5 cm, clipped to [8, 22] cm.
3. Adaptive HSV thresholds come from 25-pixel border statistics (`s_min` = 85th-percentile border saturation, clamped 15–70).
4. Session 2 bunches (depth grid 848 × 480): the depth seed is expanded with `_expand_mask_bbox`, padding 8% of the bounding-box diagonal (clamped 8–40 px).
5. Fallback: Grounding DINO detection → SAM2 mask. It was never used: the depth mask found a contour for all 11 bunches.

**Depth grid.** The pipeline branches on image width (848 vs 1280), but there is one camera. Session 1 depth is aligned to the 1280 × 720 colour frame; session 2 depth stays at 848 × 480. The "848 vs 1280" split therefore separates the two sessions, not two cameras.

**Colour order.** Session 1 colour frames reach the pipeline in BGR order. The depth mask uses only saturation and value, which the swap does not change. Grounding DINO and SAM2 (Approach C) received channel-swapped images for session 1.

**Grounding DINO + SAM2 (Approach C)**

- Grounding DINO (`IDEA-Research/grounding-dino-base`) runs in fp32 under `torch.autocast(dtype=float16)`.
- Three prompts are tried; the highest-confidence box is kept.
- SAM2 (`sam2_hiera_small`) segments the box into a pixel mask, using the same fused depth as Approach A.

### 4.4 Volume integration

```
V_raw = Σ (z_ref − z_min_per_cell) × (0.002 m)²    [2.5D grid, 2 mm cells]
z_ref = 95th-percentile depth of the masked point cloud
```

Two properties of this integral matter for interpretation:

- **Empty cells.** At 1.55 m one 848 × 480 depth pixel covers 3.6 mm, larger than a 2 mm cell, so many cells get no point. On synthetic scenes the grid recovers 33–36% of the volume under the visible surface at 848 × 480, and 74–81% at 1280 × 720.
- **Reference depth.** Session 1 masks are tight, so `z_ref` lies 11–20 cm above the tarp and only the upper part of the bunch is integrated. Session 2 masks are expanded, so `z_ref` is close to the tarp.

**Calibration.** One gain per session, fitted by least squares on all 11 bunches (including FFB18): **2.25** (session 1) and **2.53** (session 2). The values 2.02 and 2.53 in earlier versions were starting values before this refit.

### 4.5 Mass prediction

```
mass = gain × V_raw × DENSITY_CONSTANT
DENSITY_CONSTANT = 956.28 kg/m³   (mean of all 50 rows in ground_truth.csv)
```

Density cancels out of the fitted prediction (mass = V_raw · Σ(m·r)/Σ(r²)), so only one gain is actually fitted.

---

## 5. Results

### 5.1 In-sample results (v2 output)

Gains fitted on the same 11 bunches that are evaluated.

| FFB | Actual (L) | v1 | A | C | v1 err | A err | C err |
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

† Excluded from all metrics (person in frame).

**In-sample mass metrics (excluding FFB18, n = 10)**

| Approach | MAE | MAPE | Pearson r | Pearson r² |
|---|---|---|---|---|
| v1 — 15-frame IQR | 1.95 kg | 14.2% | 0.831 | 0.691 |
| A — temporal + depth mask | 1.27 kg | 9.8% | 0.926 | 0.857 |
| C — temporal + GDino→SAM2 | 1.99 kg | 15.8% | 0.868 | 0.753 |

These figures measure fit, not prediction. The re-run reproduces Approach A exactly (MAE 1.273 kg, Pearson r² 0.857; R² = 1 − SS_res/SS_tot = 0.810). v1 and C were not re-evaluated.

### 5.2 Held-out evaluation (Approach A)

Leave-one-out, n = 10: each bunch is predicted with a gain fitted on the other nine.

| Calibration | MAE (95% CI) | MAPE | Pearson r² | R² |
|---|---|---|---|---|
| Per-session gain (as in v2) | 1.65 kg (1.24–2.10) | 12.3% | 0.81 | 0.73 |
| One gain for both sessions | 1.60 kg (1.16–2.01) | 11.8% | 0.89 | 0.74 |

- **CIs** are percentile bootstrap intervals. Pearson r² ignores bias and scale errors; R² does not.
- **Prediction intervals.** 90% jackknife+ intervals have a median width of 6.2 kg (per-session gain) and 5.6 kg (one gain). Nine of 10 bunches fell inside their interval.

### 5.3 4-train/7-test cross-validation

All C(11, 4) = 330 splits, with FFB18 in the training pool but excluded from scoring. The previously reported cross-validation fitted **one gain for both sessions**, not the per-session gains behind the in-sample results. Both are shown:

| Calibration | Predictions scored | MAE | MAPE | Pearson r² (all) | R² (all) | Pearson r² (per-FFB mean) |
|---|---|---|---|---|---|---|
| One gain (previously reported) | 2,100 | 1.66 kg | 12.3% | 0.860 | 0.696 | 0.886 |
| Per-session gain | 2,000* | 1.97 kg | 14.4% | 0.676 | 0.541 | 0.770 |

\*100 predictions are impossible: in those folds, one session has no training bunch.

Including FFB18 in scoring (one gain, from the v2 run): MAE 1.85 kg, MAPE 14.8%, Pearson r² 0.748 (all predictions) and 0.772 (per-FFB mean).

**Per-FFB breakdown (one gain)**

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

Besides FFB18, the largest errors are FFB32, FFB31 and FFB17. Neither missing depth nor camera distance explains them: the fused depth has no gaps inside any bunch's mask, and the camera-to-tarp distance is 1.55–1.57 m for every bunch.

### 5.4 Baseline: caliper ellipsoid

Mass ≈ a · (π/6 · W · L · T) + b, using the ground-truth caliper measurements.

| Fitted on | Evaluated on | MAE | MAPE | Pearson r² | R² |
|---|---|---|---|---|---|
| The other 40 bunches | The same 10 bunches | 1.67 kg | 10.7% | 0.74 | 0.54 |
| Leave-one-out within the 10 | The same 10 bunches | 2.02 kg | 13.5% | 0.58 | 0.57 |

### 5.5 Statistical power

- **Detectable difference.** Per-bunch error differences between methods have a standard deviation of about 0.9 kg, so with 10 bunches only MAE differences of about 0.6 kg can be detected reliably.
- **v1 vs A.** In-sample, Approach A beats v1 by 0.68 kg (95% CI 0.18–1.21).
- **C vs A.** Approach A beats C by 0.72 kg (0.04–1.57).
- **What more data buys.** About 30 bunches would detect differences of about 0.33 kg.

---

## 6. What each adaptation contributed

Starting from the v1 in-sample MAE of 1.95 kg (in-sample figures from development; not reproduced):

| Adaptation | In-sample MAE after |
|---|---|
| Temporal median depth (instead of 15 ranked frames) | ~1.58 kg |
| Adaptive HSV threshold + adaptive z_window | ~1.42 kg |
| Separate gain for each session | 1.27 kg |

The last step was previously described as the largest gain. Held out, it is no better than a single gain (1.65 vs 1.60 kg, leave-one-out), and it fits a separate gain to 5–6 bunches per session. It should not be counted as an improvement. What the two gains absorb is a difference between recording sessions (Section 9), not a difference between cameras.

---

## 7. Comparison to related work

| System | n | MAE | 1 − MAPE | r² | Notes |
|---|---|---|---|---|---|
| This pipeline (A), leave-one-out | 10 | 1.60–1.65 kg | 87.7–88.2% | 0.81–0.89 (Pearson) | Automated; one calibration gain |
| Caliper ellipsoid, fitted on 40 other bunches | 10 | 1.67 kg | 89.3% | 0.74 (Pearson) | Manual measurement, same bunches |
| Aqil thesis | 50 | — | — | 0.900 | Manual segmentation, CloudCompare |
| Group 2 (YOLOv8 + PCA ellipsoid) | 6 | — | ~73% | — | Manual setup |

The previous version stated that the pipeline was within 0.014 r² of Aqil's benchmark. The two figures are not comparable. The 0.886 was a per-bunch mean over 2,100 cross-validation predictions, on 10 bunches, using a calibration that differed from the in-sample model; Aqil's 0.900 comes from 50 bunches and a possibly different r² definition. On the same bunches, the automated pipeline matches a caliper ellipsoid on absolute error and tracks relative size better.

---

## 8. Why Approach C (GDino→SAM2) underperforms Approach A

On this tarp-protocol dataset, depth is a better segmentation cue than colour and texture:

- **Depth isolates the bunch.** The depth foreground mask selects what sits above the tarp, which is the bunch.
- **SAM2 missed most of FFB32.** SAM2 segments visually coherent regions. For FFB32 (session 2; sparse fronds, low saturation) its mask missed most of the bunch (3.78 L vs 10.0 L actual), pulling Approach C's in-sample MAE to 1.99 kg.
- **Without FFB32.** Approach C's in-sample MAE is about 1.51 kg, still worse than A.
- **Session 1 caveat.** Both detectors received BGR-ordered images for these bunches (Section 4.3), so Approach C's session 1 results understate what it can do.

Approach C's advantage — working without a tarp background — would matter in field conditions where the tarp protocol cannot be enforced. The detection fallback did not affect Approach A in this dataset.

---

## 9. Follow-up re-evaluation (v3, CPU)

The `ffb` package reproduces v2 exactly and adds changes one at a time (`notebooks/ffb_pipeline_v3.ipynb`). **Steady frames** means frames from the start of the recording up to the first frame where more than 8% of pixels moved over 2 cm or lost depth, relative to frames 0–7.

**Leave-one-out MAE (kg), n = 10**

| Step | Per-session gain | One gain |
|---|---|---|
| v2 (reproduced) | 1.65 | 1.60 |
| + steady frames only | 1.58 | 1.43 |
| + median filter after masking | 2.29 | 2.28 |
| + grid-free volume | 2.44 | 5.84 |
| + tarp-plane reference | 1.69 | 3.26 |
| + unaligned session 1 depth | 1.78 | 2.69 |

Findings:

- **Steady frames.** The only change that helped both calibrations. Paired difference from v2: −0.07 kg (95% CI −0.48 to +0.20) and −0.17 kg (−0.48 to +0.07). The direction is favourable but not statistically confirmed.
- **Geometry fixes are correct but raise error.** The grid-free volume is within 0.1% on synthetic scenes, and object height above the fitted tarp plane correlates at r = 0.87 with caliper thickness. Error still rises, because v2's empty grid cells and high reference depth were partly cancelling each other between sessions.
- **Session difference.** With the tarp-plane volume, volume above the tarp ÷ displaced volume is **1.58 in session 1 and 1.07 in session 2**. This 1.5× difference is the dominant remaining error. What was tested:
  - **Depth-to-colour alignment:** accounts for part of it (1.78 → 1.58).
  - **Disturbed frames:** no effect.
  - **Tarp folds:** ruled out; session 1's tarp is flatter near the bunch.
  - **Colour-only depth:** a monocular depth model (MoGe-2) gives the same direction (2.00 vs 1.20), so the effect is unlikely to be specific to the depth camera.
  - **Bunch shape or measurement:** session 1 bunches measure about 2 cm taller than their caliper thickness, against 0.3 cm in session 2. This fits spikier bunches or a different caliper protocol, but is not confirmed.
- **Status.** These comparisons were chosen after inspecting the data and should be treated as exploratory.

---

## 10. Known limitations

| Item | Issue |
|---|---|
| FFB18 | Person in frame; excluded from all metrics |
| FFB32 | Largest error; underestimated by every approach; not caused by missing depth |
| Session effect | Sessions differ in scene, background, bag layout and bunch batch; with 5–6 bunches each, their effects cannot be separated |
| Ground truth | Displaced volume rounded to whole litres; the density column is derived from it |
| Recordings | Most scenes are disturbed partway through; v2 fuses those frames for session 1 and only 32 frames for session 2 |
| Sample size | 10 evaluable bunches; MAE differences under about 0.6 kg cannot be detected |

**Protocol requirements.** Flat background, top-down camera, camera-to-tarp distance of about 1.55 m (as measured). Performance at other distances or backgrounds is untested.

---

## 11. Recommendations for new data

1. Record 25–30 bunches in one fixed setup: same background, recording layout and camera height. Record a few bunches in both previous setups to measure the setup effect directly.
2. Keep the scene still for the first ~5 s of each recording, and set RealSense depth units to 100 µm.
3. Include an object of known volume in some recordings, to check the geometry independently of bunch ground truth.
4. Measure displaced volume to 0.1 L, and document whether caliper measurements include spikes.
5. Fix the model before looking at results — steady-frame fusion, tarp-plane volume, object height, one gain — and report held-out metrics with confidence intervals.
6. If the v2 calibration is reused, fit one gain per recording setup from at least ~10 bunches per setup. Density does not need re-deriving for this model because it cancels.
7. For a new background or bunch population, check the tarp saturation range (`colour_s_min`) and the [8, 22] cm protrusion clip.

---

## 12. Corrections from the previous version

| Previous statement | Correction |
|---|---|
| Approach A MAE 1.27 kg, r² 0.857 (headline) | In-sample. Held-out MAE is 1.60–1.65 kg (Section 5.2) |
| CV r² 0.886 presented with per-camera scaling | The CV fitted one gain; with per-session gains, per-FFB r² is 0.770 |
| Scales 2.02 / 2.53 for 1280 / 848 cameras | One camera; the split is by recording session; gains used were 2.25 / 2.53 |
| Per-camera scaling gave the largest gain | Not supported under held-out evaluation |
| Density: mean of 11 FFBs, refitted per fold | Mean of all 50 rows; cancels out of predictions |
| Temporal median over 120 frames | 32 frames in session 2; session 1 includes disturbed frames |
| FFB18: +10 kg systematic error | v1 overestimated its volume by 10 L; Approach A by 3.2 L |
| FFB31/FFB17: possible camera-distance issue; FFB32: sparse depth | Distance is 1.55–1.57 m for every bunch; no depth gaps inside any mask |
| Within 0.014 r² of Aqil's benchmark | Not comparable (Section 7); compare with the caliper baseline instead |
| Shooting distance 1.2–1.5 m | Measured 1.55–1.57 m |
| GDino/SAM2 fallback used when the depth mask fails | Never triggered in this dataset; session 1 colour was BGR-ordered |
