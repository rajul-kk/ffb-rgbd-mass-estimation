# FFB Mass Estimation: Short Report

Estimating oil palm fresh fruit bunch (FFB) mass from a top-down Intel RealSense D455 depth recording. The full analysis, with every table, check and correction, is in [full-report.md](full-report.md).

## Result

Each bunch is predicted from a gain fitted on the other bunches (leave-one-out, LOO). 10 bunches are scored; FFB18 is excluded because a person is in frame, though it is still used to fit the gain.

| Method | LOO MAE | MAPE |
|---|---|---|
| v2: depth ∩ colour mask, 2.5D grid volume | 1.60 kg | 11.8% |
| Depth-only mask | 1.13 kg | 7.9% |
| **Depth-only mask + steady frames + unaligned depth** | **0.99 kg** | — |
| Aqil, manual CloudCompare (same bunches) | 1.45 kg | 9.9% |
| Caliper ellipsoid, fitted on 40 other bunches | 1.67 kg | 10.7% |

- **Against Aqil:** 0.33 kg better for the depth-only mask, but not statistically significant (95% CI −1.22 to +0.54 kg). With 10 bunches only differences above about 0.6 kg can be detected.
- **Against Group 2** (YOLOv8 + PCA ellipsoid, 5 comparable bunches): their errors average 4.4 kg (25.6%), against 1.57 kg (9.9%) here.
- **Status:** exploratory. The fixes were found and tuned on these 10 bunches, so expect 1.0–1.6 kg on new bunches.

## Method

1. **Fuse depth:** per-pixel median over the steady opening frames, before the bunch is moved.
2. **Segment from depth alone:** find the tarp as the most common depth in the centre of the frame, fit a plane to it with RANSAC, and keep pixels more than 3 cm above the plane.
3. **Volume:** the volume between the depth surface and the plane.
4. **Mass:** one fitted gain × volume (about 0.51 kg/L), always evaluated on held-out bunches. No other training is needed.

## The bug that explained the session gap

The data has two recording sessions. In session 2, the colour and depth were recorded in separate bags, with the colour ending 35–92 s before the depth starts, and the two are not registered. v2 kept only pixels that passed both a depth and a colour test, so its session 2 masks covered only part of each bunch. A per-session gain had hidden this by scaling the volumes back up, leaving an unexplained 1.5× gap between sessions.

Removing the colour test fixes it. Session 2 error falls from 1.63 to 0.68 kg, session 1 is unchanged (1.57 kg), and the gap between sessions falls from 1.49× to 1.14×. That pattern is what a real fix to this bug should produce.

Fusing only the steady frames helps session 1 (1.57 → 1.35 kg), where v2's median had blended in moving frames.

## What did not work

- **Foundation-model segmentation (Grounding DINO → SAM2):** 1.99 kg against 1.27 kg for the plain mask, both in-sample. SAM 3 was blocked on gated model access.
- **Geometry fixes** (frustum volume, tarp-plane reference, pixel-sized cells, empty-cell interpolation): fine on synthetic scenes, worse on real data (2.3–7 kg against 1.6 kg held out).
- **Plane-fit upgrades from the literature:** MSAC/LO-RANSAC changed MAE by about 0.02 kg; a Tukey refit gave errors up to 87 mm.
- **MoGe-2 monocular depth:** agreed with the session gap, but used this pipeline's masks, so it is not independent evidence.
- **Per-session gains:** no better than one gain when held out.
- **Aligning session 2 colour to depth:** not possible, because there is no overlap in time and the offset varies by bunch.

## Limitations

- **Sample size:** 10 bunches, and about a dozen variants were tried on them, so the numbers are optimistic.
- **Hard cases:** FFB11 and FFB17 are overestimated by about 2.1 kg in every variant. From above they look larger than they are, and Aqil probably gets them right by averaging four orientations, which these recordings cannot support.
- **Set-up:** the method assumes a top-down camera about 1.55 m above a flat tarp filling the centre of the frame, with bunches at least 3 cm tall.
- **Ground truth:** displaced volume is rounded to whole litres, which sets a floor of about 0.24 kg MAE.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest
python scripts/plane_mask_eval.py     # depth-only mask: held-out, vs Aqil
python scripts/session1_eval.py       # steady frames, unaligned depth
```
