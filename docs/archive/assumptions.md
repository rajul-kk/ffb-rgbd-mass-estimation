> Superseded by [report.md](../../report.md); kept as a record of the original plan.

# Pipeline Assumptions & Manually-Tuned Parameters

Parameters that are dataset-specific or empirically fitted. Each one that breaks
will cause silent under/over-estimation rather than a crash.

---

## Hard constants

| Parameter | Value | Where set | Risk |
|---|---|---|---|
| `DENSITY_CONSTANT` | 956.28 kg/m³ | Cell 3, from GT mean | 11 FFBs, one variety, one harvest date. Different ripeness/moisture → different density. ±5% error propagates 1:1 to mass. |
| `SCALE_BY_WIDTH` | `{1280: 2.02, 848: 2.53}` | Cell 3 | Empirically fitted hemisphere correction. Changes with scene geometry (distance, surface orientation). Cell 3b re-derives these from GT — still dataset-specific. |
| `cam_width == 848` branch | hard-coded integer | `_patched_process_frame` | Any other RealSense mode (e.g. 640×480, 1280×800) falls through to `SCALE_2D5 = 2.02` with no bbox expansion. |

---

## Depth segmentation

| Parameter | Value | Function | Risk |
|---|---|---|---|
| `z_front = 5th percentile of depth` | implicit | `_depth_foreground_mask` | Assumes FFB is the closest object. Fails if operator hand, tripod leg, or clutter is nearer. |
| Margin clip | `np.clip(margin, 0.08, 0.22)` | `_auto_margin` | Assumes FFB protrudes 8–22 cm above tarp. Flat or very large/small bunches clip at the bounds. |
| `search_lo / search_hi` | 5–45 cm | `_auto_margin` | Tarp must be within 5–45 cm behind bunch front. Cluttered scenes (e.g. FFB11) land the histogram peak on clutter instead of tarp. |
| `fallback_m` | 10 cm | `_auto_margin` | Default when histogram has <200 points. Arbitrary. |
| `colour_s_min` (tight mask) | 40 | `_depth_foreground_mask` | Tuned for green tarp. High-saturation non-tarp backgrounds leak through; very dark FFBs may be over-filtered. Tested lowering to 20 for FFB35 (S_mean=26) — caused catastrophic leakage in FFB17/FFB31/FFB19 (CV 116%, vol +8L). Depth z-window alone is NOT sufficient tarp exclusion at S<40. |
| `colour_v_max` | 160 | both mask functions | Filters bright objects. Breaks in direct sunlight where tarp pixel values exceed 160. |
| `min_area_frac / max_area_frac` | 0.003 / 0.35 | `_depth_foreground_mask` | FFB must occupy 0.3–35% of frame. Silently returns `None` (→ YOLO fallback) for very close or very distant shots. |
| `0.75 × half_diagonal` centre threshold | implicit | `_depth_foreground_mask` | FFB assumed roughly centred. Off-centre shots fail and fall back to largest contour. |

---

## Bbox expansion (848×480 only)

| Parameter | Value | Function | Risk |
|---|---|---|---|
| `z_window` | 0.25 m | `_expand_mask_bbox` | Tuned for 1.2–1.5 m shooting distance. Too wide at close range (includes tarp); too narrow at >2 m. |
| `pad_px` | 20 px | `_expand_mask_bbox` | Fixed-pixel padding — over-expands at lower resolution or close range. |
| `s_min` | 25 | `_expand_mask_bbox` | Looser colour gate than tight mask. Same saturation-background risk at a lower threshold. |

---

## Volume integration

| Parameter | Value | Function | Risk |
|---|---|---|---|
| `grid_step` | 2 mm | `_compute_2d5_volume` | Not adaptive to point cloud density or shooting distance. |
| `z_window` in `_project_to_3d` | 0.25 m | `_patched_project_to_3d` | Clips point cloud to 25 cm behind bunch front. Consistent with `z_window` in bbox expansion but independently hardcoded. |
| `dome_fraction > 1.4` trigger | 1.4 | `_patched_process_frame` | Adaptive scale fallback threshold tuned by inspection of this dataset's depth distributions. |
| `pct = 90` ellipsoid | 90th percentile | `_compute_ellipsoid_volume` | Removes top/bottom 5% of PCA-projected points to reduce outlier inflation. Not tuned but also not principled. |

---

## What is genuinely general (not tuned)

- 2.5D grid integration formula (physics, no fitting)
- Median blur + morphological open/close (standard preprocessing)
- PCA ellipsoid fallback (no dataset-specific parameters)
- YOLO-SAM2 fallback path (zero-shot, no tuning)
- `nn.Embedding` device patch and `_clear_fn_overloads` patch (PyTorch bug workarounds)

---

## Recalibration checklist for new data

1. Re-derive `DENSITY_CONSTANT` from a fresh sample of ≥10 FFBs (weigh + water-displacement)
2. Re-run Cell 3b after Cell 5 to get new `SCALE_BY_WIDTH` values
3. If new camera resolution: add to `SCALE_BY_WIDTH` dict and add bbox-expansion branch if depth is noisy
4. Verify tarp colour assumption holds (green tarp → `colour_s_min=40`); adjust if background changes
5. Check `np.clip(margin, 0.08, 0.22)` bounds against actual FFB protrusion range
