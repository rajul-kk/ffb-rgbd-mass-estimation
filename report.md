# FFB Mass Estimation — Report

**Camera:** Intel RealSense D455 (depth 848 × 480, colour 1280 × 720)  
**Data:** 11 fresh fruit bunches (FFBs) recorded in two sessions; ground truth for 50  
**Code:** `ffb/` package; v2 results in `notebooks/kaggle_ffb_pipeline_v2.ipynb`, re-evaluation in `notebooks/ffb_pipeline_v3.ipynb`, depth-only mask in `notebooks/ffb_pipeline_v4.ipynb`

> **Revision note.** A full CPU re-run reproduces every v2 number exactly. It showed that v2's headline figures were in-sample, that the cross-validation used a different calibration from the one described, and that the inputs had five bugs, one of which (unregistered session 2 colour) explains most of the difference between sessions. Section 10 lists every correction.

---

## 1. Summary

- **v2, held out.** Leave-one-out MAE **1.60–1.65 kg** (MAPE 11.8–12.3%) on 10 bunches. The previously reported 1.27 kg was in-sample. A caliper ellipsoid reaches 1.67 kg on the same bunches.
- **Bug found.** Session 2's colour images were recorded before its depth, with no overlap in time, and are not registered to it. v2 keeps only pixels that pass both a depth and a colour test, so for those bunches its mask covers only part of the bunch (Section 5).
- **Fix: a depth-only mask.** It removes the colour test, which gives leave-one-out MAE **1.13 kg** (MAPE 7.9%) with one gain. It beats v2 under every evaluation scheme, and the improvement is entirely in session 2 (1.63 → 0.68 kg), the only session with the bug.
- **Against Aqil's manual workflow** on the same 10 bunches: the depth-only mask scores 1.13 kg against his 1.45 kg. That's 0.33 kg better, not statistically significant (95% CI −1.22 to +0.54). v2 was 0.15 kg worse.
- **Status.** The depth-only mask is exploratory: the bug was found by inspecting this data. It should be fixed as-is and confirmed on new bunches before it replaces v2.
- **Sample size.** With 10 evaluable bunches, only MAE differences of about 0.6 kg can be detected reliably.

---

## 2. Data

| | Session 1 | Session 2 |
|---|---|---|
| Bunches | FFB10, 11, 12, 17, 18, 19 | FFB31, 32, 33, 34, 35 |
| Recording | One bag; depth aligned to colour | Separate depth and RGB bags, recorded at different times |
| Scene | Indoors, cloth tarp | Outdoors, flat board |
| Camera to tarp (measured) | 1.55 m | 1.55–1.57 m |

- **Ground truth.** `ground_truth.csv` has 50 bunches: mass, caliper width, length and thickness, and displaced volume rounded to whole litres.
  - Density is mass ÷ rounded volume, so it is derived, not measured.
  - The rounding alone sets a floor of about 0.24 kg on any method's MAE.
  - With exact volume, mass is almost perfectly linear in volume (R² 0.982). The remaining error is in estimating volume.
- **FFB18** is excluded from scoring. A person is in frame throughout. v2's grid volume also undercounts it less than any other bunch, so the fitted gain overshoots it (Section 7).
- **Camera.** The bag metadata reports a D455 (serial 215122256082), while Aqil's thesis describes a D435i, so these may not be his recordings. The bunch IDs and ground truth match his.
- **Rotations.** Aqil's protocol turns each bunch through four orientations of about 7 s each. That probably explains the disturbance partway through most recordings. v2's all-frame median blends these poses.

---

## 3. Method (v2)

1. **Temporal fusion.** Each depth frame gets a 3 × 3 median filter, zero depth is marked invalid, and the per-pixel median over time is taken.
2. **Mask.** Take the closest surface (the 5th-percentile depth) and find the tarp in a depth histogram behind it. Keep pixels nearer than the tarp that also pass an HSV colour test (saturation above, value below thresholds taken from the image border). The contour nearest the centre is kept. In session 2 (848 × 480) the mask is then grown within a padded box, again using depth and colour.
3. **Volume.** A 2.5D grid with 2 mm cells: V = Σ (z_ref − z_min per cell) × cell area, where z_ref is the 95th-percentile depth of the masked points.
4. **Mass.** mass = gain × V. Density (956.28 kg/m³) cancels once the gain is fitted. v2 fits one gain per session by least squares on all 11 bunches: 2.25 and 2.53.

**Input bugs as run**

| Bug | Effect |
|---|---|
| Session 2 colour not registered to depth | The mask covers only part of the bunch (Section 5). The main cause of the session gap |
| Split-bag reader keeps frame references | Session 2 fused 32 frames, not 120 |
| Median filter before invalid masking | Zero-depth neighbours pull edge pixels nearer or remove them |
| Disturbed frames fused | Session 1 medians blend several poses |
| Session 1 colour in BGR order | Affects only the Grounding DINO/SAM2 comparison (Approach C). The depth mask uses saturation and value, which the swap leaves unchanged |

The "848 vs 1280" branch in v2 is not two cameras; it separates the two sessions.

---

## 4. v2 results

**Held out** (FFB18 used for training, never scored; percentile bootstrap CIs)

| Calibration | LOO MAE (95% CI) | MAPE | Pearson r² | R² | 4/7 splits | 7/4 splits |
|---|---|---|---|---|---|---|
| One gain | 1.60 kg (1.16–2.01) | 11.8% | 0.89 | 0.74 | 1.66 kg | 1.59 kg |
| Per-session gains (as in v2) | 1.65 kg (1.24–2.10) | 12.3% | 0.81 | 0.73 | 1.97 kg | 1.65 kg |

- **Schemes.** LOO predicts each bunch from the other nine. "4/7" and "7/4" use all 330 splits of 4 or 7 training bunches.
- **Per-session gains** fall apart with only 4 training bunches, and 100 of the 4/7 predictions are impossible because a session has no training bunch.
- **Prediction intervals.** 90% jackknife+ intervals have a median width of 5.6 kg; 9 of 10 bunches fall inside.

**In-sample** (the gain is fitted on the evaluated bunches, so this measures fit, not prediction): v1 1.95 kg, Approach A 1.27 kg, Approach C 1.99 kg MAE.

- **Approach C (Grounding DINO → SAM2)** loses mainly on FFB32, where SAM2 covered 3.8 of 10 L.
- **Per-session gains** were previously credited as the biggest improvement. Held out, they are no better than one gain.

**Caliper baseline.** Mass from the caliper ellipsoid volume (π/6 · W · L · T), with an intercept, fitted on the other 40 bunches: 1.67 kg MAE, MAPE 10.7%, Pearson r² 0.74 on the same 10.

---

## 5. The session 2 colour bug and the depth-only mask

### What the data contains

Each session 2 bunch has two bags, one for depth and one for RGB. Frame timestamps (`results/v4/recording_offsets.csv`) show that the RGB recording ends **35–92 s before the depth recording starts**, with no overlap for any bunch. In FFB32's RGB frame a person is still placing the bunch. Nothing in the bags registers the colour sensor to the depth sensor.

### What v2 does with it

v2 resizes the RGB image to the depth grid and keeps pixels that pass both tests:

1. **Depth:** the pixel is nearer than the tarp.
2. **Colour:** the pixel is saturated and dark enough to be fruit.

In session 2, test 2 checks where the bunch sat in a different recording, from an unregistered sensor. The two positions differ by tens of pixels, so the mask keeps only their overlap. For example, FFB31 has 12,388 pixels passing the depth test but only 5,775 after the colour test. The masks are too small, so session 2 volumes are too small.

### Why it went unnoticed

The per-session gain (2.53 against 2.25) scaled session 2 back up, which kept the average error plausible. The difference instead showed up as an unexplained 1.5× gap between sessions in volume above the tarp ÷ true volume. Aqil's manual estimates of the same bunches show no such gap (estimate ÷ truth 1.04 against 0.98). That already put the gap in these recordings or in the segmentation, not in the bunches.

Neither of the obvious repairs works:
- **Timestamp pairing:** no RGB frame overlaps the depth recording.
- **A fixed offset:** the estimated shift differs from bunch to bunch.

### The fix

`segment.plane_height_mask` never reads colour:

1. **Find the tarp:** the most common depth (1 cm bins) in the central half of the frame.
2. **Fit the plane:** RANSAC on the pixels within 3 cm of that depth.
3. **Keep the bunch:** pixels more than 3 cm above the plane; the connected blob nearest the centre is kept.
4. **Volume:** the frustum volume between the depth surface and the same plane.

Both 3 cm thresholds were fixed before any held-out run.

### Results

`scripts/plane_mask_eval.py` and `notebooks/ffb_pipeline_v4.ipynb` reproduce each other; outputs are in `results/plane_mask_*` and `results/v4/`.

| Held-out MAE (kg) | One gain: LOO | 4/7 | 7/4 | Per-session: LOO | 4/7 | 7/4 |
|---|---|---|---|---|---|---|
| v2 | 1.60 | 1.66 | 1.59 | 1.65 | 1.97 | 1.65 |
| Tarp-plane volume, v2 mask | 2.65 | 2.83 | 2.67 | 1.62 | 1.90 | 1.69 |
| **Tarp-plane volume, depth-only mask** | **1.13** | **1.20** | **1.14** | **1.09** | **1.25** | **1.14** |

- **Against v2:** 0.46–0.72 kg lower under every scheme. The difference is significant with per-session gains on the 4/7 splits (−0.72 kg, 95% CI −1.34 to −0.15) and borderline elsewhere (bootstrap P(not better) 0.04–0.10). No other change tested in this project beat v2 in any scheme.
- **Accuracy with one gain, LOO:** MAPE 7.9%, Pearson r² 0.87, R² 0.84, MAE 95% CI 0.66–1.61 kg. 90% jackknife+ intervals narrow from 5.6 to 4.9 kg.
- **Where the gain comes from** (LOO MAE, one gain):

  | Session | v2 | Depth-only | Aqil |
  |---|---|---|---|
  | 1 (colour aligned) | 1.57 kg | 1.57 kg | 1.46 kg |
  | 2 (colour unregistered) | 1.63 kg | **0.68 kg** | 1.44 kg |

  The session without the bug doesn't change on average; the session with it improves 2.4-fold. That is what a real fix to this bug should do, and a change that merely fitted the data would be unlikely to leave session 1 untouched.
- **Session gap:** volume above the tarp ÷ true volume now differs 1.14× between sessions, down from 1.49×.
- **Calibration:** the gains are 0.50 (session 1) and 0.53 (session 2) kg/L, and one gain (0.51 kg/L) works as well as two. Density is about 0.96 kg/L. The ratio of about 1.9 between density and gain is what a top-down view of a resting, spiky bunch should give: a resting ellipsoid alone gives 1.33×.
- **Threshold sensitivity** (`results/plane_mask_sweep.csv`): one-gain LOO stays between 1.01 and 1.13 kg for heights of 3–5 cm and tarp bands of 2–5 cm. At 2 cm tarp noise joins the bunch and MAE reaches 7 kg, so 3 cm is a floor.
- **Masks** (`results/v4/masks.png`):
  - In session 1 the depth-only mask is slightly larger than v2's, adding the fringe that v2's colour test dropped.
  - FFB12's mask includes a thin tail, either the stalk or a tarp fold more than 3 cm high. It may add volume that isn't bunch.

### Session 1: steady frames and unaligned depth

In every session 1 bag the bunch is steady only for the first 21–35 frames, then keeps moving to the end; v2's median blends 42–78% moving frames. Session 2's reader stopped at frame 32, inside its steady opening. Two candidates were fixed before running (`scripts/session1_eval.py`, `results/session1_*`), both with the depth-only mask and the existing steady-frame detector:

| Held-out MAE (kg), one gain | LOO | 4/7 | 7/4 | Session 1 | Session 2 |
|---|---|---|---|---|---|
| All frames (above) | 1.13 | 1.20 | 1.14 | 1.57 | 0.68 |
| Steady frames | 1.04 | 1.11 | 1.04 | 1.33 | 0.76 |
| **Steady frames + unaligned depth** | **0.99** | **1.08** | **1.00** | **1.35** | **0.62** |

- **The gain comes from FFB12 and FFB19,** the bunches whose medians had blended motion. Steady frames + unaligned depth beats all frames by 0.14 kg (LOO, one gain; 95% CI −0.30 to −0.02). With colour no longer used, unaligned depth also lets both sessions be processed identically.
- **FFB11 and FFB17 stay about 2.1 kg over** in every variant. From above they look larger than they are (volume above the tarp 2.2× displaced volume, against 1.7–2.0× for the others). Aqil gets them nearly right, probably because he averages four orientations. The session 1 bags contain only one steady pose, so no pose averaging is possible here.
- **Cloth folds are ruled out.** Thin appendages, FFB12's tail included, hold under 0.1% of volume.

### Caveats

- **Found by inspecting this data.** The bug was found by looking at this data, so the result is exploratory.
- **One post-hoc change.** The tarp search was changed once after the first held-out run. It had located the tarp relative to the 5th-percentile depth, which fails when the bunch fills under 5% of the frame; a synthetic test exposed this. The change moved one-gain LOO from 1.37 to 1.13 kg.
- **Two more candidates.** The session 1 variants were two further candidates stacked on this method.
- **Recommendation:** fix the method as it stands (depth-only mask, steady frames, unaligned depth, one gain) and test it on new bunches.

---

## 6. Comparison with related work

Both theses used the same ground truth, so all methods can be scored on the same bunches. Aqil's per-bunch estimates are in `aqil_table_c.csv` (his Table C). Group 2's were read off their Figure A6.

| Same 10 bunches | MAE | MAPE | Paired difference vs Aqil (95% CI) |
|---|---|---|---|
| Aqil, manual CloudCompare | 1.45 kg | 9.9% | — |
| Depth-only mask, LOO, one gain | **1.13 kg** | **7.9%** | −0.33 kg (−1.22 to +0.54) |
| Depth-only mask, LOO, per-session | 1.09 kg | 7.5% | −0.36 kg (−1.19 to +0.45) |
| v2, LOO, one gain | 1.60 kg | 11.8% | +0.15 kg (−0.36 to +0.68) |
| Caliper ellipsoid, fitted on 40 others | 1.67 kg | 10.7% | — |

- **Aqil's headline** (MAE 1.15 kg and Pearson r² 0.902 over all 50 bunches) includes bunches not in this set. It needs no fitted gain but relies on manual segmentation. His density comes from the same 50 bunches.
- **Group 2** (YOLOv8 + PCA ellipsoid):
  - **Their 73%:** the mean of (1 − |error| ÷ actual) over 6 session 1 bunches, including FFB18, with an unstated "empirical scaling factor". Their own report calls it a projected goal.
  - **On the 5 comparable bunches:** their errors average 4.4 kg (MAPE 25.6%), against 1.57 kg (MAPE 9.9%) for both v2 and the depth-only mask.
  - **Their errors change sign** from bunch to bunch (+3.5 to +4.6 kg on the three largest, −8.2 kg on FFB19), so no single scale factor could correct them.
- **Calibration:**
  - **Aqil:** fits nothing.
  - **Group 2:** apply an unstated factor with no held-out check.
  - **This pipeline:** fits a gain and evaluates it held out.
  - **What the depth-only mask changes:** v2 needed two gains 12% apart. It now needs one, within 7% of the per-session values.

---

## 7. Other changes tested and rejected

`notebooks/ffb_pipeline_v3.ipynb` and `scripts/cpu_eval.py` test v2 changes one at a time; `scripts/grid_pitch_eval.py` covers the grid-volume fixes.

| Held-out MAE (kg), vs v2 | One gain: LOO | 4/7 | 7/4 | Per-session: LOO | 4/7 | 7/4 |
|---|---|---|---|---|---|---|
| v2 | 1.60 | 1.66 | 1.59 | 1.65 | 1.97 | 1.65 |
| Steady frames only | 1.43 | 1.79 | 1.58 | 1.58 | 2.21 | 1.85 |
| + median filter after masking | 2.28 | 2.72 | 2.50 | 2.29 | 3.35 | 2.69 |
| + grid-free (frustum) volume | 5.84 | 5.80 | 5.78 | 2.44 | 3.76 | 2.98 |
| + tarp-plane reference | 3.26 | 3.46 | 3.29 | 1.69 | 2.06 | 1.81 |
| Filter after masking + grid cell = pixel pitch | 6.47 | 6.61 | 6.47 | 3.25 | 4.14 | 3.47 |
| Filter after masking + z_ref from tarp ring | 5.30 | 6.09 | 5.40 | 1.86 | 2.35 | 2.04 |
| Filter after masking + empty-cell interpolation | 7.10 | 7.35 | 7.18 | 3.46 | 4.44 | 3.73 |

Across four calibration models, none of the 60 v3 paired comparisons is significantly better than v2, and 25 are significantly worse.

**Why v2's grid undercounts.** v2's grid recovers only 31–59% of displaced volume, for two reasons:
- **Session 1:** z_ref, the 95th percentile of the masked bunch points, sits 11–20 cm above the tarp.
- **Session 2:** the 2 mm cells are finer than the 3.6 mm depth pixel, so many cells get no point.

Each of these was confirmed per bunch. FFB18 has the smallest z_ref gap, which is why the gain overshoots it.

**Why the fixes failed.** Each fix works on synthetic scenes but fails on real depth:
- **Bigger grid cells** catch near-outlier pixels.
- **Interpolation** can't tell aliasing gaps from real dropouts.
- **A deeper z_ref** overshoots session 1.

These fixes changed the volume calculation while keeping v2's colour-dependent mask, so none of them addressed the session 2 bug.

**How much of the v2 gain is physics.** Predicting mass as an external density (Aqil's 956.28 kg/m³) × volume, with nothing fitted to these bunches:

| Volume used | MAE |
|---|---|
| Grid volume (filter after masking) | 8.98 kg |
| + both grid fixes | 4.52 kg |
| v2 fitted gain, for reference | 1.65 kg |

The two mechanisms explain about half of what the gain corrects.

**Other checks**
- **Steady frames** (fusing only frames before the scene is disturbed) don't change accuracy: every CI spans zero.
- **A MoGe-2 monocular depth cross-check** agreed in direction with the session gap. It used this pipeline's masks, so it isn't independent evidence.
- **Plane-fitting upgrades from recent literature:**
  - MSAC scoring with LO-RANSAC refits changes volumes by at most 1% and MAE by about 0.02 kg (`fit_plane(robust=True)`).
  - A MAD-scaled Tukey refit was rejected: debris on one side of the tarp skews the scale estimate, and depth errors reached 87 mm.
- **Exploratory model mass = a·V + b·H** (v2 volume plus object height): 1.04 kg LOO but 1.65 kg on the 4/7 splits. It was chosen after seeing the data.

---

## 8. Limitations

| Item | Issue |
|---|---|
| Sample size | 10 evaluable bunches; MAE differences under about 0.6 kg cannot be detected. About 30 bunches would detect 0.33 kg |
| Depth-only mask | Exploratory; confirm on new bunches. Needs a flat tarp covering the centre of the frame, and at least 3 cm between bunch and tarp noise |
| Session 2 colour | Not registered to depth, and captured at a different time. Any colour-based step in session 2 is unreliable |
| Session effect | A residual 1.14× remains; with 5–6 bunches per session it cannot be attributed further |
| Ground truth | Displaced volume rounded to whole litres; density is derived from it |
| FFB18 | Person in frame; excluded |
| Protocol | Top-down camera about 1.55 m above a flat tarp. Other distances and backgrounds are untested |

---

## 9. Recommendations for new data

1. **Pre-register the model.** Fix the depth-only mask (3 cm height, 3 cm band), steady frames, unaligned depth and one gain as the primary model before looking at new results. Report LOO and repeated splits with confidence intervals.
2. **Record 25–30 bunches in one setup:** same background, camera height and bag layout. Record depth and colour together (one bag, aligned) if colour is needed at all.
3. **Keep the scene still** for the first ~5 s of each recording, then do any rotations. Set RealSense depth units to 100 µm.
4. **Include a known-volume object** in some recordings to check the geometry independently of bunch ground truth.
5. **Improve the ground truth:** measure displaced volume to 0.1 L, and record whether caliper measurements include spikes.

---

## 10. Corrections from the previous version

| Previous statement | Correction |
|---|---|
| Approach A MAE 1.27 kg, r² 0.857 | In-sample. Held out: 1.60–1.65 kg |
| CV r² 0.886 with per-camera scaling | The CV fitted one gain; with per-session gains, per-bunch r² is 0.770 |
| Scales 2.02 / 2.53 for two cameras | One camera; the split is by session; the gains used were 2.25 / 2.53 |
| Per-camera scaling was the largest improvement | Not supported held out |
| Density refitted per fold from 11 bunches | Mean of all 50 rows; it cancels out of predictions |
| Temporal median over 120 frames | 32 frames in session 2; session 1 includes disturbed frames |
| Session gap from spikier bunches or caliper protocol | Mostly the unregistered session 2 colour (Section 5); Aqil's estimates show no gap |
| Within 0.014 r² of Aqil | Not comparable. Like for like, v2 is +0.15 kg and the depth-only mask −0.33 kg MAE against Aqil |
| MoGe-2 shows the gap is not camera-specific | It used this pipeline's masks, so it isn't independent |
| FFB18: +10 kg systematic error | v1 overestimated its volume by 10 L; Approach A by 3.2 kg |
| Shooting distance 1.2–1.5 m | Measured 1.55–1.57 m |
| GDino/SAM2 fallback used when the depth mask fails | Never triggered |
