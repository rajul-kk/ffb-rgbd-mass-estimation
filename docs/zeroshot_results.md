# Results: zero-shot masks scored on mass error, and the calibration curve

Run under [PROTOCOL_zeroshot.md](PROTOCOL_zeroshot.md) (deviations are listed there). Raw tables are in
`results/zeroshot/` and `results/calibration_curve/`. **n is tiny: 5 scored bunches in (a), 10 in (b).**

## (a) Zero-shot masks, arm A (colour image, session 1, 5 scored bunches)

Held-out (leave-one-out) mass error, one gain per provider, fit pool FFB10, 11, 12, 17, 18, 19 (FFB18 fit only).
Intervals are 95% percentile bootstrap over the 5 scored bunches. Failure-penalised numbers are shown because two
providers never detected anything.

| Provider | Detected | MAE kg [95% CI] | MAPE % | Bias kg | Paired diff in abs error vs control, kg [CI] | Mean IoU vs control |
|---|---|---|---|---|---|---|
| depth_plane (control) | 5/5 | 1.37 [0.64, 1.95] | 8.9 | +0.20 | - | - |
| Grounding DINO + SAM 2 | 5/5 | 1.28 [0.59, 1.85] | 8.3 | +0.20 | -0.09 [-0.13, -0.05] | 0.83 |
| SAM 2, centre point | 5/5 | 1.26 [0.65, 1.80] | 8.2 | +0.20 | -0.11 [-0.19, 0.00] | 0.84 |
| SAM 2, central box | 5/5 | 18.75 [14.63, 24.61] | 116 | -6.74 | +17.37 [12.95, 22.96] | 0.04 |
| YOLO-World + SAM 2 (conf 0.10) | 0/5 | 3.58 [2.03, 5.44] (penalised) | 20.2 | -1.50 | +2.21 [0.51, 4.64] | n/a |
| YOLO-World + SAM 2 (conf 0.02) | 0/5 | 3.58 (penalised) | 20.2 | -1.50 | +2.21 | n/a |
| SAM 2, oracle box (diagnostic, not zero-shot) | 5/5 | 1.27 [0.55, 1.84] | 8.2 | +0.19 | -0.11 [-0.14, -0.07] | 0.83 |

The control here is 1.37 kg, not the 0.99 kg headline: different bunches (5 of session 1 only) and aligned depth.

What this says, and does not say:
- **Grounding DINO + SAM 2 and a single centre point on SAM 2 match the control**: differences of about 0.1 kg against
  a stated detectability floor of 0.6 kg or more. The paired intervals exclude zero only because all five bunches shift
  the same way; with n = 5 that is not evidence of an improvement and is not claimed as one. SAM 2 masks have about 17% fewer pixels than the
  depth mask (mean IoU 0.83), and the single fitted gain absorbs that constant shrinkage. The method is therefore
  insensitive to this particular boundary difference once calibrated.
- **The localisation was never the problem on these bunches**: the oracle box, which is not zero-shot, gives the same
  error as the point and Grounding DINO prompts.
- **SAM 2 with a central box fails** (mask is mostly tarp, IoU 0.04, errors up to tens of kg). This is a property of the
  pre-registered prompt (box over the middle half of the frame, larger than the bunch), not of SAM 2 in general.
- **YOLO-World never detected the bunch** (0 of 6 session 1 bunches at confidence 0.10 and 0.02). A sanity check showed the
  pipeline works for another class (`person`). Its row is the no-information baseline (predict the training mean).
- **IoU is not mass error.** SAM 2 masks differ from the control by IoU 0.83 yet give the same mass error; the central box
  differs by 0.96 and gives a 17 kg difference. Both statements are about difference from the control mask, not truth.
- Session 2 (5 scored bunches) could not be tested with colour. Arm B (height-map images, all 10 scored bunches) was
  run later on a Kaggle GPU; see section (a2).
- Prior repo result (1.99 kg for Grounding DINO + SAM 2 against 1.27 kg, in-sample) is not reproduced here: with the
  protocol's selection rule and a held-out gain, Grounding DINO + SAM 2 is level with the control.

Runtime on this CPU (i7-1355U, shared with other jobs): Grounding DINO tiny + SAM 2 tiny 60-355 s per image (mean 132 s
in the scored table), SAM 2 alone 3-36 s, YOLO-World 1-33 s (first call includes model load). The section 7 estimate was too optimistic for
Grounding DINO.


## (a2) Zero-shot masks, arm B (height-above-plane image, 10 scored bunches; Kaggle T4 GPU, 2026-11-06)

Same script, prompts, thresholds, post-processing and scoring as arm A; masks from `notebooks/kaggle_zeroshot_infer.ipynb`
(kernel `rajulkabir/ffb-zeroshot-gpu`, private). Run info: torch 2.11.0+cu130, transformers 5.20.2, ultralytics 8.4.203,
Tesla T4 (`zeroshot_cache/masks/run_info_hmap_gpu.json`). The same kernel re-ran arm A on the GPU as a device check:
detections agreed on 11/11 bunches for every provider, 22 of 24 detected masks were byte-identical to the CPU masks,
and the rest differed by 12 pixels in total. Arm A results above are the original CPU masks, unchanged.

| Provider | Detected | MAE kg [95% CI] | MAPE % | Bias kg | Paired diff vs control, kg [CI] | Mean IoU vs control |
|---|---|---|---|---|---|---|
| depth_plane (control) | 10/10 | 1.02 [0.58, 1.50] | 7.5 | -0.03 | - | - |
| Grounding DINO + SAM 2 | 10/10 | 5.18 [4.03, 6.30] | 37.3 | -2.05 | +4.16 [2.96, 5.39] | 0.29 |
| SAM 2, centre point | 10/10 | 2.78 [1.23, 5.09] | 21.3 | -1.20 | +1.76 [0.13, 4.19] | 0.77 |
| SAM 2, central box | 10/10 | 7.94 [3.82, 12.40] | 50.1 | -6.30 | +6.92 [2.81, 11.42] | 0.10 |
| YOLO-World + SAM 2 (both thresholds) | 0/10 | 2.91 [1.51, 4.53] (penalised) | 19.2 | -0.51 | +1.89 [0.25, 3.66] | n/a |
| SAM 2, oracle box (diagnostic, not zero-shot) | 10/10 | 0.96 [0.50, 1.45] | 7.1 | -0.02 | -0.06 [-0.09, -0.03] | 0.94 |

What this says, and does not say:
- **On the rendered height map every zero-shot provider is worse than the control**, by 1.8 to 6.9 kg. As stated in
  advance (design D7), detectors trained on photographs seeing a grey-scale height render is a harsh test; this is
  not a verdict on the models for real colour images.
- **Localisation is the failure, not segmentation**: given the control's box, SAM 2 on the height map matches the
  control (IoU 0.94, -0.06 kg). Grounding DINO's boxes on the render overlap the bunch poorly (IoU 0.29).
- YOLO-World again detected nothing.
- n = 10; intervals are wide. No zero-shot route to all 10 bunches beats the depth-plane mask.
## (b) Calibration-sample-size curve

Volumes from `results/session1/volumes.csv` (row `B_steady_native`), model `mass = g V`. 10 scored bunches; every subset
of k bunches fits the gain and is scored on the other 10 - k bunches. **Limitation: n = 11 recordings, 10 scored; k = 11
cannot be scored held-out, and at k = 9 the held-out set is a single bunch.**

| k | subsets | MAE kg | 95% CI (bootstrap over bunches) | MAPE % | subset MAE 5th-95th pct |
|---|---|---|---|---|---|
| 1 | 10 | 1.42 | 1.11-1.78 | 10.0 | 0.96-2.26 |
| 2 | 45 | 1.23 | 0.90-1.60 | 8.7 | 0.86-1.61 |
| 3 | 120 | 1.14 | 0.77-1.54 | 8.1 | 0.73-1.58 |
| 4 | 210 | 1.08 | 0.68-1.51 | 7.8 | 0.67-1.51 |
| 5 | 252 | 1.05 | 0.63-1.49 | 7.6 | 0.66-1.53 |
| 6 | 210 | 1.03 | 0.60-1.48 | 7.4 | 0.54-1.55 |
| 7 | 120 | 1.01 | 0.57-1.47 | 7.3 | 0.34-1.66 |
| 8 | 45 | 0.99 | 0.54-1.46 | 7.2 | 0.22-1.65 |
| 9 | 10 | 0.99 | 0.53-1.46 | 7.1 | 0.14-2.13 |
| 10 (in-sample only) | 1 | 0.89 | 0.48-1.31 | 6.5 | - |

- k = 9 reproduces the repo's leave-one-out MAE of 0.99 kg, as it should.
- Most of the gain from more calibration bunches arrives by k = 3 to 4 (1.42 to 1.08 kg); beyond k = 5 the curve is flat
  (1.05 to 0.99 kg) and far inside the interval width. Bias stays within 0.14 kg of zero (see CSV).
- The interval narrows only slowly because it is dominated by which bunches are being scored (the report names FFB11 and FFB17 as hard
  cases; not re-checked per k here), not by the calibration size. The widest worry is at k = 1: a single-bunch gain can be off by 2.3 kg MAE in the worst 5% of choices.
- Sensitivity: letting FFB18 into the fit pool (k = 1..10 held-out) changes MAE by at most 0.05 kg at any k. An intercept
  model (`aV+b`) is much worse for small k (2.70 kg at k = 2, 1.23 kg at k = 5, 0.99 kg at k = 9) and only matches the
  one-gain model near k = 9.
- Figure: `results/calibration_curve/curve.png`.
