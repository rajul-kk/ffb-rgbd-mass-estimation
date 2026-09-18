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
- **Attempted fixes.** None of the v3 changes beats v2 under leave-one-out, 4/7 or 7/4 evaluation: 0 of 60 paired comparisons are significantly better and 25 are significantly worse (Section 9).
- **Main open problem.** Errors differ systematically between the two recording sessions. More bunches recorded in one fixed setup are needed to resolve this.

---

## 2. Objective

Automate the mass estimation of oil palm Fresh Fruit Bunches (FFBs) from a short RGB-D recording, matching or exceeding the accuracy of Aqil's manual CloudCompare workflow. Scored on the same 10 bunches, the two are indistinguishable (Section 7); Aqil's published result over all 50 bunches remains better.

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
- **Exclusion.** FFB18 is excluded from all metrics. A person is in frame throughout the recording (a documentation/protocol issue, not a measured cause; the automated mask itself is clean, checked visually against the RGB frame). The real driver is that it is the least undercounted bunch of all 11 by v2's grid volume method (Section 9), so the session gain — fit to correct the typical shortfall — overcorrects it: v1 overestimated its volume by 10 L and Approach A by 3.2 kg.
- **Camera identity.** The bag metadata reports a RealSense D455 (serial 215122256082), while Aqil's thesis describes a D435i. These may therefore not be his recordings, although the bunch IDs and ground truth match.
- **Capture protocol.** Aqil's thesis rotates each bunch through four orientations of about 7 s, which fits the 24–32 s session 1 bags. The "disturbances" partway through most recordings are probably these planned rotations rather than noise, so v2's all-frame median blends several poses.

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

With 7 training bunches per split (all 330 splits of 7 train / 4 test), v2 scores MAE 1.59 kg with one gain and 1.65 kg with per-session gains (per-bunch Pearson r² 0.89 and 0.79). Every split then has training bunches from both sessions.

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

Both comparison systems used the same ground truth, so they can be scored on the same bunches. Aqil's per-bunch estimates come from his Appendix Tables C and D; Group 2's from their Figure A6, read off the chart.

**Same 10 bunches (FFB18 excluded).**

| System | MAE | MAPE | R² | Pearson r² | Calibration |
|---|---|---|---|---|---|
| Aqil, manual CloudCompare | **1.45 kg** | **9.9%** | **0.761** | 0.797 | None fitted to vision output; density from the same 50 bunches |
| This pipeline (A), leave-one-out, one gain | 1.60 kg | 11.8% | 0.743 | **0.886** | Gain fitted on the other 9 |
| This pipeline (A), leave-one-out, per-session gains | 1.65 kg | 12.3% | 0.730 | 0.810 | Gains fitted on the other 9 |
| Caliper ellipsoid, fitted on 40 other bunches | 1.67 kg | 10.7% | 0.540 | 0.740 | Manual measurement |

Paired difference, this pipeline minus Aqil: **+0.15 kg (95% CI −0.36 to +0.68)** with one gain, +0.20 kg (−0.58 to +0.99) with per-session gains. The two are indistinguishable on this sample. Aqil's own headline is better than either (MAE 1.15 kg, MAPE 8.8%, Pearson r² 0.902 over all 50 bunches; 1.17 kg and 6.6% on his 10 validation bunches), but those sets include easier bunches. Beating it would need roughly 0.45 kg lower MAE, which 10 bunches cannot demonstrate (Section 5.5).

**Group 2 (YOLOv8 + PCA ellipsoid), same 5 session-1 bunches.** Their headline 73% is the mean of (1 − |error| ÷ actual) over 6 bunches, including the excluded FFB18, with an empirical scale factor whose fitting set is not stated; their own report also describes the 73% as a projected goal. On FFB10, 11, 12, 17 and 19 their medians give MAE 4.4 kg and MAPE 25.6%, against 1.57 kg and 9.6% for this pipeline (leave-one-out, one gain) and about 1.6 kg and 10% for Aqil. Their errors change sign between bunches (+3.5 to +4.6 kg on the three largest, −8.2 kg on FFB19), so a single scale factor cannot correct them.

**Calibration, compared across all three.** All three methods correct for the same underlying problem — a single top-down view understates a bunch's true volume — but differ in how openly that correction is made:

| | Density | Volume correction | Fitted to held-out ground truth? |
|---|---|---|---|
| Aqil | One constant (956.28 kg/m³) over all 50 bunches | None; his 2.5D volume is not corrected by a fitted factor | No fitted parameter at all |
| This pipeline | One or two gains (2.25 / 2.53), fit by least squares | The gain itself, folded into one number (Section 9) | Yes — leave-one-out, 4/7, 7/4 (Section 5) |
| Group 2 | Same constant as Aqil, applied uniformly | An unstated "empirical scaling factor" (their Section VI) | Not stated; no held-out check reported |

Group 2's own limitations section attributes their error to "reliance on a constant density assumption" and recommends, as future work, "moving beyond a constant density assumption" toward "a data-driven regression model trained on a larger dataset" — a fitted, per-setup correction similar in spirit to the gain used here, but they did not build or evaluate one. So this pipeline's calibration is more transparent and held-out validated than either alternative, but it is also the only one of the three that needs two different numbers for what is nominally the same measurement — itself a symptom of the two uncorrected mechanisms in Section 9, not a genuine physical difference between the sessions.

The previous version stated that the pipeline was within 0.014 r² of Aqil's benchmark. The two figures are not comparable: the 0.886 was a per-bunch mean over 2,100 cross-validation predictions on 10 bunches, while Aqil's 0.900 comes from 50 bunches and a possibly different r² definition. The like-for-like comparison above replaces it.

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

**Held-out MAE (kg).** Leave-one-out (n = 10), and all 330 splits of 4 train / 7 test and of 7 train / 4 test (FFB18 used for training, never scored).

| Step | One gain: LOO | 4/7 | 7/4 | Per-session: LOO | 4/7 | 7/4 |
|---|---|---|---|---|---|---|
| v2 (reproduced) | **1.60** | **1.66** | **1.59** | **1.65** | **1.97** | **1.65** |
| + steady frames only | 1.43 | 1.79 | 1.58 | 1.58 | 2.21 | 1.85 |
| + median filter after masking | 2.28 | 2.72 | 2.50 | 2.29 | 3.35 | 2.69 |
| + grid-free volume | 5.84 | 5.80 | 5.78 | 2.44 | 3.76 | 2.98 |
| + tarp-plane reference | 3.26 | 3.46 | 3.29 | 1.69 | 2.06 | 1.81 |
| + unaligned session 1 depth | 2.69 | 2.85 | 2.70 | 1.78 | 2.11 | 1.86 |

Across these three schemes and four calibration models (the two above, plus volume with an intercept and volume plus object height), none of 60 paired comparisons with v2 is significantly better, and 25 are significantly worse (paired bootstrap over bunches, 95% CI).

Findings:

- **v2 is the most stable.** With one gain it scores 1.59–1.66 kg under every scheme, level with the caliper baseline. Its per-session gains degrade to 1.97 kg when each split has only 4 training bunches.
- **Steady frames do not improve accuracy.** Paired difference from v2 with one gain: −0.17 kg (leave-one-out), +0.13 kg (4/7) and −0.01 kg (7/4); with per-session gains: −0.07, +0.24 and +0.20 kg. Every CI spans zero. They remain a reasonable data-quality step for disturbed recordings.
- **Tarp-plane volume.** With per-session gains it ties v2 under all three schemes (+0.04, +0.09 and +0.15 kg); with one gain it is clearly worse (+1.66 to +1.80 kg). It ranks bunches within a session about as well as v2, but exposes the difference between sessions.
- **Geometry fixes are correct but raise error.** The grid-free volume is within 0.1% on synthetic scenes, and object height above the fitted tarp plane correlates at r = 0.87 with caliper thickness. Error still rises, because v2's empty grid cells and high reference depth were partly cancelling each other between sessions.
- **The grid undercount has two separate causes, confirmed per bunch.** Comparing each bunch's grid `z_ref` (95th percentile of the *masked* points) against the true tarp plane depth:

  | Session | `z_ref` vs tarp gap | Raw grid undercount |
  |---|---|---|
  | 1 (FFB10/11/12/17/18/19) | 11–20 cm | 41–63%, tracking the gap |
  | 2 (FFB31–35) | ≈0 cm | 55–69% despite no gap |

  In session 1, `z_ref` never reaches the tarp — it is the percentile of bunch-only points, so it sits partway up the bunch instead of at the ground, and height is undercounted by however far short it falls. FFB18 has the smallest gap of the six (11.4 cm), which is why it is the least undercounted bunch overall (Section 3) and why the session gain, fit to the typical 50–55% shortfall, overcorrects it. In session 2, `z_ref` already sits at the tarp, yet the undercount persists: the native 848-wide depth has a pixel pitch of ≈3.6 mm at this distance, coarser than the fixed 2 mm grid cell, so a sweep of grid cell size (1.5–10 mm) confirms many cells are empty and silently dropped regardless of bunch size. Neither is a small-footprint effect — the largest-footprint bunch (FFB10) has one of the largest shortfalls.
- **Two targeted fixes, tested and rejected.** Each diagnosis above suggests an obvious fix — a grid cell size tied to the pixel pitch (`pixel_pitch`), and a `z_ref` taken from a ring of tarp points around the mask instead of the mask's own percentile (`ring_z_ref`) — and both were validated on synthetic data first, recovering >90% and the true tarp depth (±1 mm) respectively. Neither beats v2 held-out on the real bunches. The pitch-matched grid cell is clearly worse (LOO 3.25–6.47 kg): real depth noise means a bigger cell more often catches a spurious near-outlier as the cell's surface height, and this dominates over the aliasing it fixes. The ring-based `z_ref` is closer — LOO 1.86 kg with per-session gains refit to it, against v2's 1.65 kg, not statistically distinguishable (paired diff +0.21 kg, 95% CI −0.97 to +1.43) — but it overshoots session 1 by 10–64% instead of landing near the true volume, because a real mask's boundary isn't the clean dome the synthetic check used. This is the fourth independently validated geometry fix (after the grid-free and tarp-plane volumes above) to fail this bar. The consistent pattern across all four is evidence that, with 11 bunches split 6/5 across two sessions, no volume-method fix can be shown to help — not that v2's calibration is secretly correct.
- **How much of the session gain is real physics versus curve-fitting?** To test this without leaking our own mass values into the answer, mass was predicted as density (956.28 kg/m³, Aqil's external constant, not fit to our 11 bunches) times volume, with zero parameters fit to our data at any stage:

  | Volume used | Correction applied | MAE, n = 10, zero fit to our masses |
  |---|---|---|
  | Plain v2 grid | none | 8.98 kg |
  | + pitch-matched cell | session-2 mechanism only | 6.54 kg |
  | + ring-based `z_ref` | both mechanisms | 4.52 kg |
  | v2's fitted per-session gain | — | **1.65 kg** |

  Each confirmed mechanism moves the zero-fit model in the right direction, which is independent evidence the two diagnoses above are real. But even with both applied and an external density, the zero-fit model is still 2.7× worse than the fitted gain. So roughly half of what the gain corrects for is explained, physically, by the two mechanisms found here; the remaining half is either genuine per-bunch density variation (the 50-bunch ground truth spans about 0.85–1.05 kg/L) or a further systematic error not yet identified — this dataset cannot distinguish the two. The gain is therefore partly principled and partly an unexplained fit to 11 bunches; closing that remainder needs more data, not more geometry fixes.
- **Session difference.** With the tarp-plane volume, volume above the tarp ÷ displaced volume is **1.58 in session 1 and 1.07 in session 2**. This 1.5× difference is the dominant remaining error. What was tested:
  - **Depth-to-colour alignment:** accounts for part of it (1.78 → 1.58).
  - **Disturbed frames:** no effect.
  - **Tarp folds:** ruled out; session 1's tarp is flatter near the bunch.
  - **Colour-only depth:** a monocular depth model (MoGe-2) gives the same direction (2.00 vs 1.20). This is not independent evidence: it used this pipeline's masks and tarp distance, so a mask or reference error would carry through.
  - **Bunch shape or ground truth:** ruled out. Aqil's manual estimates, which need no fitted gain, show almost no session difference (estimate ÷ truth 1.036 in session 1 and 0.983 in session 2, a ratio of 1.05) on the same bunches and the same displaced volumes, against 1.48 for the tarp-plane volume and 1.15 for the v2 grid. The gap is therefore in these recordings or in the automatic segmentation, not in the bunches or the ground truth. An earlier version attributed it to spikier bunches or a different caliper protocol; that is not supported.
- **Exploratory lead.** v2 volume plus object height (mass = a·V + b·H) scores 1.04 kg (leave-one-out), 1.17 kg (7/4) and 1.65 kg (4/7). It needs about 7 training bunches and was chosen after seeing the data.
- **Status.** These comparisons were chosen after inspecting the data and should be treated as exploratory.

---

## 10. Known limitations

| Item | Issue |
|---|---|
| FFB18 | Excluded from all metrics: person in frame (protocol issue) and grid volume undercounts this small, compact bunch by ~41% (Section 3) |
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
5. Fix the candidate models before looking at results — for example v2 volume + object height, and tarp-plane volume + object height, each with one gain — and report held-out metrics (leave-one-out and repeated splits) with confidence intervals.
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
| Within 0.014 r² of Aqil's benchmark | Not comparable. Scored on the same 10 bunches, this pipeline is +0.15 kg MAE against Aqil, CI −0.36 to +0.68 (Section 7) |
| Session gap fits spikier bunches or a different caliper protocol | Not supported. Aqil's manual estimates show no session gap on the same bunches, so it lies in these recordings or in the segmentation (Section 9) |
| MoGe-2 shows the effect is not specific to the depth camera | It used this pipeline's masks and tarp distance, so it is not independent evidence (Section 9) |
| Shooting distance 1.2–1.5 m | Measured 1.55–1.57 m |
| GDino/SAM2 fallback used when the depth mask fails | Never triggered in this dataset; session 1 colour was BGR-ordered |
