# Pre-registered protocol: zero-shot masks scored on mass error, and the calibration-sample-size curve

Frozen when this file is committed, before any mass error from a zero-shot mask has been computed. Architecture
and reasons are in [zeroshot_design.md](zeroshot_design.md). Anything not stated here is exploratory and must be
labelled so in the results.

## 0. Limitation stated up front: n = 11

The dataset holds 11 usable recordings (FFB10, 11, 12, 17, 18, 19 in session 1; FFB31-35 in session 2). FFB18 has a
person in frame and is never scored, so **10 bunches are scored**. Only 5 of them (session 1) have colour that is
registered to depth, so the headline zero-shot comparison scores **5 bunches**. With this n:
- a paired MAE difference of less than about 1 kg will not be separable from zero (the repo's own bound is about 0.6 kg for n = 10);
- percentile bootstrap intervals over 5 bunches are rough and typically too narrow;
- results are a feasibility check on one site, one camera and one tarp set-up, not evidence of general ranking.

The calibration curve cannot reach k = 11 with held-out scoring: FFB18 is not scored, so k = 11 has no test bunch.
See section 6.

## 1. Question

(a) If the segmentation stage of the mass pipeline is replaced by a zero-shot detector or promptable segmenter, how
does held-out mass error change? (b) How does held-out mass error depend on the number of bunches used to fit the gain?

## 2. Fixed pipeline for everything except the mask

- Fusion: `FusionConfig(reader="fixed", frames="steady")` (aligned depth, steady opening frames).
- Tarp plane: `segment.plane_height_mask` on that fused depth, taken once per bunch and shared by all providers.
- Volume: `volume.plane_volume(fused, mask, K, plane)[0]` (the existing code, unchanged).
- Mass: one gain, `mass = g * V` (`evaluate.Model("global_k")`), leave-one-out. A bunch's gain is fitted on the other
  bunches of the fit pool that have a valid volume from the same provider.
- Control: `depth_plane` (the current depth-only mask, no post-processing), run through the same code.

## 3. Providers, prompts, thresholds, checkpoints

| Provider | Detection | Segmentation | Prompt / threshold |
|---|---|---|---|
| `gdino_sam2` | Grounding DINO tiny, `IDEA-Research/grounding-dino-tiny` | SAM 2.1 tiny, `facebook/sam2.1-hiera-tiny`, box prompt, single mask | text `oil palm fruit bunch.`; box threshold 0.30, text threshold 0.25 |
| `yoloworld_sam2` | YOLO-World v2 small, `yolov8s-worldv2.pt` (ultralytics assets v8.4.0) | same SAM 2.1 tiny, box prompt | class `oil palm fruit bunch`; confidence 0.10; default image size 640 |
| `sam2_point` | none | same SAM 2.1 tiny | one positive point at the image centre; 3 candidate masks, keep the highest predicted IoU |
| `sam2_box` | none | same SAM 2.1 tiny | box over the middle half of each axis (0.25-0.75 of width and height), single mask |
| `sam2_oracle_box` | control mask's bounding box | same SAM 2.1 tiny | **diagnostic only, not zero-shot** |
| `yoloworld_lowconf_sam2` | as `yoloworld_sam2` | same | confidence 0.02. **Disclosed deviation:** added after a smoke test (three calls on FFB10 and FFB11, status only, no mass computed) found no YOLO-World detection at 0.10. Reported as a sensitivity arm, never as the primary YOLO-World result. |

Instance choice for detectors: among detected boxes keep those with area between 0.3% and 35% of the image and a
centre within 0.75 of the half-diagonal from the image centre, then take the highest score (`masking.pick_box`).
Checkpoints are the smallest available variants because the study must run on CPU; larger variants are untested
and the results say nothing about them.

Library versions at freeze time: Python 3.10.0, torch 2.10.0+cpu, transformers 4.57.6, ultralytics 8.4.202,
clip-anytorch 2.6.0 (supplies `clip` for YOLO-World text embeddings). Actual versions are written to `run_info.json`.
No fine-tuning, no per-provider tuning, no API keys. Weights are stored outside the repo and are gitignored.

**Post-processing** (all providers except the control): hole fill, keep the largest connected component, discard masks
under 20 pixels (counted as no detection). Masks are saved raw and post-processed at scoring time.

## 4. Inputs (modalities) and which bunches

| Arm | Image given to the provider | Bunches in the fit pool | Bunches scored |
|---|---|---|---|
| **A (primary)** `rgb` | colour frame, same grid as depth | session 1: FFB10, 11, 12, 17, 18, 19 | FFB10, 11, 12, 17, 19 (n = 5) |
| **B (secondary)** `hmap` | height above the tarp plane, 0 m black to 0.35 m white, 3 channels, rendered from depth | all 11 | the 10 other than FFB18 |

Session 2 has no registered colour (colour recorded 35-92 s before depth), so it is absent from arm A: a provider is
not charged a failure for it. Arm B exists to cover those bunches and to test the "3D-aware" input; detectors are
trained on photographs, so weak arm B results do not describe the models on real RGB. In each arm the control is
refitted and scored on the same fit pool and scored bunches. Arm A control numbers will therefore differ from
the repo's 0.99 kg headline (different bunches, aligned depth).

## 5. Metrics and statistics

Primary metric: mass error of the held-out prediction (kg), `pred - actual`.
- MAE, MAPE (%), signed bias (mean `pred - actual`), per provider and arm.
- 95% percentile bootstrap CIs over the scored bunches (`evaluate.bootstrap_ci`, B = 20000, seed 0).
- Paired difference in absolute error against the control with a bootstrap CI (`evaluate.paired_bootstrap`, B = 20000, seed 0).
- A provider is called better or worse than the control only if the paired CI excludes 0; otherwise the result is
  "not distinguishable". With n = 5 this is expected for most comparisons.

**Failures** (no detection, empty mask after post-processing, provider exception): reported as a coverage count and scored two ways.
1. *Detected-only*: failed bunches are dropped from that provider's MAE, and the paired comparison uses the bunches both methods predicted.
2. *Failure-penalised* (headline when coverage is below 100%): the prediction for a failed bunch is the mean mass of the fit pool excluding that bunch, i.e. the no-information guess.
A provider that fails on a fit-pool bunch simply has fewer bunches to fit its gain. If the plane fit fails, the bunch fails for every provider including the control.

Secondary diagnostic: IoU of the post-processed mask against the control mask, per bunch and mean. The control mask
is not ground truth, so IoU is only a description of how different the masks are and is never used to rank providers.

## 6. Calibration-sample-size curve

- Volumes: `results/session1/volumes.csv`, rows `B_steady_native` (the 0.99 kg recipe), with the committed masks and plane. No re-run.
- Model: `global_k` (`mass = g V`). Sensitivity: `aV+b`, reported for k >= 2.
- Main pool: the 10 scored bunches. For every k = 1..9 every subset of k bunches is enumerated (at most C(10,5) = 252),
  the gain is fitted on it, and the other 10 - k bunches are scored. k = 10 has no held-out bunch in the pool and is
  reported **in-sample only**, flagged.
- Sensitivity pool: fit candidates are all 11 bunches (including FFB18), scored bunches are the 10 scored bunches not
  in the subset; k = 1..10. **k = 11 is not measurable** with held-out scoring and is not reported as held-out.
- Per k: MAE, MAPE, signed bias; interval 1 = 95% bootstrap over bunches (B = 20000, seed 0) of the per-bunch mean error
  across all subsets that leave that bunch out; interval 2 = 5th-95th percentile and maximum of subset MAE.
- Reading the curve: with n = 10 the points at large k average very few distinct test bunches, so the curve's right end is noisy and its left end is dominated by which bunch is picked.

## 7. Runtime estimate (CPU vs Kaggle)

The number of mask calls is 6 providers x (6 rgb + 11 hmap images) = 102, plus the cheap control. Measured on this
machine (i7-1355U, 12 threads, no GPU), one smoke call: SAM 2 tiny with a point prompt on a 1280 x 720 image took 8.6 s;
the first Grounding DINO + SAM 2 call took 151 s including a 700 MB download and model load. Expected CPU time
after the downloads: about 10-25 s per call, so roughly 20-45 minutes in total. That is practical, so inference runs
on CPU. A Kaggle notebook (`notebooks/kaggle_zeroshot_infer.ipynb`) running the same script is provided for replication
and for larger checkpoints; on a T4 it should take a few minutes plus model setup (estimate, not measured).
Downloads stay well under 5 GB (about 0.7 GB Grounding DINO, 0.15 GB SAM 2 tiny, 0.03 GB YOLO-World, 0.35 GB CLIP).
Actual times will be reported from the run.

## 8. Deviations log

Any change after this commit is appended here with the reason and date, and the affected result is marked
exploratory. Entries so far: none (the `yoloworld_lowconf_sam2` arm is declared above, before scoring).
