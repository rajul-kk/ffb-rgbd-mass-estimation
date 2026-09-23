# FFB Mass Estimation from RGB-D

Estimates the mass of oil palm fresh fruit bunches (FFBs) from a short top-down Intel RealSense D455 recording. Segmentation and geometry need no training data; one calibration gain is fitted to weighed bunches.

Full analysis: [report.md](report.md).

## Results

Held out: each bunch is predicted from a gain fitted on the other bunches. 10 bunches are scored (FFB18 is excluded because a person is in frame).

| Method | LOO MAE | 4/7 splits | 7/4 splits | MAPE (LOO) |
|---|---|---|---|---|
| v2 (depth ∩ colour mask, 2.5D grid) | 1.60 kg | 1.66 kg | 1.59 kg | 11.8% |
| **Depth-only mask (exploratory)** | **1.13 kg** | **1.20 kg** | **1.14 kg** | **7.9%** |
| Aqil, manual CloudCompare (same bunches) | 1.45 kg | — | — | 9.9% |
| Caliper ellipsoid, fitted on 40 other bunches | 1.67 kg | — | — | 10.7% |

**The bug behind the gain.**
- **What happened:** session 2's RGB was recorded 35–92 s *before* its depth, with no overlap, and isn't registered to it. v2 keeps only pixels that pass both a depth and a colour test, so for session 2 its mask covered only part of the bunch.
- **The fix:** the depth-only mask ([report §5](report.md#5-the-session-2-colour-bug-and-the-depth-only-mask)) keeps pixels more than 3 cm above a fitted tarp plane and never reads colour.
- **The effect:** session 2's error falls from 1.63 to 0.68 kg, while session 1, which never had the bug, stays at 1.57 kg.
- **Status:** the method was found on this data, so it is exploratory until it's confirmed on new bunches. The comparison with Aqil is not statistically significant with 10 bunches.

## Run

```bash
pip install -r requirements.txt
python -m pytest                      # 55 tests
python scripts/cpu_eval.py            # v2 reproduced on CPU + fix ladder (report §7)
python scripts/grid_pitch_eval.py     # rejected grid-volume fixes (report §7)
python scripts/plane_mask_eval.py     # depth-only mask: held-out, threshold sweep, vs Aqil (report §5–6)
python scripts/make_v4_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/ffb_pipeline_v4.ipynb
```

- **Inputs and cache:** bags are read from `data/FFB{N}/`, and fused depth is cached in `fused_cache/`.
- **Cache rebuilds:** a cache file is rebuilt automatically when the fusion settings or the reading/fusion code change.
- **CI:** the tests run on every push.

**Optional models:** MoGe-2 and SAM 3 run in a separate environment:
```bash
python -m venv --system-site-packages .venv-models
.venv-models\Scripts\python -m pip install -U pip "transformers>=5" git+https://github.com/microsoft/MoGe.git
.venv-models\Scripts\python scripts\moge_check.py
```
SAM 3 also needs approved access to `facebook/sam3`, then `scripts/cpu_eval.py --sam3`.

**v2 notebook (Kaggle):** upload the dataset, set `PROJECT_DIR` in the `preflight` cell, and run top to bottom. It falls back to CPU without a GPU.

## Repository

```
ffb/                      package: bag IO, fusion, segmentation, volume, evaluation, SAM 3
  segment.py              v2 mask (extract_mask) and depth-only mask (plane_height_mask)
  volume.py               2.5D grid, frustum volume, RANSAC tarp plane
  evaluate.py             LOO, exhaustive splits, bootstrap CIs, jackknife+
notebooks/
  kaggle_ffb_pipeline_v2.ipynb   v2 results (v1 baseline in kaggle_ffb_pipeline.ipynb)
  ffb_pipeline_v3.ipynb          CPU re-evaluation of v2 and single changes
  ffb_pipeline_v4.ipynb          depth-only mask: recording offsets, masks, held-out, vs Aqil
scripts/                  reproducible evaluations (see Run)
results/                  outputs of the scripts and notebooks
tests/                    synthetic geometry, segmentation, fusion and evaluation tests
ground_truth.csv          50 bunches: mass, caliper W/L/T, displaced volume
aqil_table_c.csv          Aqil's per-bunch estimates (thesis Table C)
bag_reader.py, perception_pipeline.py   code used by the v2 notebook
docs/archive/             original project plan, compute estimates and v2 parameter notes (superseded)
```

## Limitations

- **Sample size:** 10 evaluable bunches, so only MAE differences of about 0.6 kg can be detected.
- **Protocol:** top-down camera about 1.55 m above a flat tarp; other set-ups are untested.
- **Ground truth:** displaced volume is rounded to whole litres, a floor of about 0.24 kg MAE.

See [report §8](report.md#8-limitations) for the full list.
