# Design: zero-shot mask providers and calibration-sample-size curve

Status: written before any zero-shot result exists. The experiment itself is frozen in
[PROTOCOL_zeroshot.md](PROTOCOL_zeroshot.md); this file records architecture and the reasons for it.

## 1. Where masks enter the pipeline today

Current best recipe (report.md): steady-frame fusion, depth-only mask, plane-referenced volume, one gain.

| Stage | Code | Consumes a mask? |
|---|---|---|
| Read + fuse depth | `ffb/io.py`, `ffb/fusion.py` | No. Returns fused depth and one RGB frame. |
| Tarp plane | `segment.plane_height_mask` (RANSAC on the dominant central depth, `volume.fit_plane`) | No. The plane is fitted from depth alone before the mask is cut. |
| **Segmentation** | `segment.plane_height_mask` (current), `segment.extract_mask` (v2) | **Produces** the mask. |
| Volume | `volume.plane_volume(fused, mask, K, plane)` (also `grid_volume`, `ring_mask`, `tarp_points`, which need a mask) | **Yes**: sums pixel pyramids inside the mask. |
| Calibration and scoring | `evaluate.Model`, `evaluate.loo`, `bootstrap_ci` | No. Takes one volume per bunch. |

Consequence: the mask is a single array crossing one boundary, `mask -> plane_volume`. The plane does not depend on
the mask, so swapping the segmenter changes nothing else. A provider interface is therefore cheap, and the comparison
isolates the segmentation stage.

Two data facts shape the design:
1. **Session 2 (FFB31-35) has no usable RGB.** Its colour bag ends 35-92 s before the depth bag starts and is
   unregistered (report.md). RGB-prompted detectors cannot localise a bunch in the depth grid for those five. Session 1
   (FFB10, 11, 12, 17, 18, 19) has colour aligned to depth when read with `reader="fixed"`.
2. **FFB18 has a person in frame.** It is never scored but the existing pipeline uses it to fit the gain.

## 2. Module layout

```
ffb/masking.py           MaskRequest, MaskResult, MaskProvider protocol, registry, postprocess, iou. numpy + cv2 only.
ffb/masking_models.py    Grounding DINO + SAM2, YOLO-World + SAM2, SAM2 prompts. torch imported lazily inside methods.
ffb/zeroshot.py          volume_from_result, score_provider (failures, calibration, metrics, paired CIs, IoU).
ffb/calibration_curve.py calibration-sample-size curve (numpy/pandas only).
scripts/zeroshot_export_inputs.py   bags -> small npz of (rgb, fused depth, K, plane) per bunch. CPU, pyrealsense2.
scripts/zeroshot_infer.py           inputs npz -> mask PNGs per provider. The only step that needs torch (CPU or Kaggle).
scripts/zeroshot_score.py           mask PNGs + inputs -> volumes, mass metrics, CIs, IoU. numpy only.
scripts/calibration_curve.py        volumes csv -> curve csv + summary json.
notebooks/kaggle_zeroshot_infer.ipynb   generated; wraps zeroshot_infer.py for Kaggle.
```

### Mask-provider interface

```python
@dataclass(frozen=True)
class MaskRequest:
    name: str                 # bunch id
    rgb: np.ndarray | None    # H x W x 3 uint8 on the SAME grid as depth, or None if unregistered
    depth: np.ndarray         # fused depth, metres, 0 = no data
    K: Intrinsics
    plane: tuple              # (normal, d) from the depth-only tarp fit; mask-independent

@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray | None   # bool, depth grid; None = no detection
    score: float | None       # provider confidence, diagnostic only
    meta: dict                # prompt, box, seconds, failure reason

class MaskProvider(Protocol):
    name: str
    needs_rgb: bool
    def __call__(self, req: MaskRequest) -> MaskResult: ...
```

Rules that keep the mass code untouched:
- A provider returns a mask on the depth grid or `None`. It never returns volumes, edits depth, or sees ground truth.
- `volume_from_result` is the only consumer: shared post-processing (hole fill, largest component), then the existing `volume.plane_volume`, unchanged.
- Providers register by name in `masking.REGISTRY`; a new segmenter is one class and one registration line. The scorer reads saved mask files by provider name and imports no model library.
- The current segmentation is wrapped as `depth_plane` and is the control, going through the same code path.

## 3. Data flow

```
bags --export--> inputs/{FFB}.npz (rgb, fused, K, plane)           [CPU, pyrealsense2, cached]
inputs --infer--> masks/{provider}/{FFB}.png + meta.json           [torch; CPU or Kaggle]
inputs + masks --score--> volumes.csv, metrics.csv, summary.json   [numpy]
results/session1/volumes.csv --curve--> calibration_curve.csv      [numpy]
```

Splitting inference from scoring is the central decision. Only small npz and PNG files move to Kaggle (never bags),
and heavy dependencies stay out of the tested package and CI.

## 4. Decisions, alternatives, reasons

| # | Decision | Alternatives considered | Why rejected |
|---|---|---|---|
| D1 | Provider = function from a request to a mask on the depth grid | Provider returns a volume; subclass `segment`; a flag inside `measure()` | A volume-returning provider duplicates mass code and lets providers differ in geometry, defeating a mass-error comparison. A flag in `measure()` entangles the experiment with every existing script and cache key. |
| D2 | Inference and scoring are separate scripts joined by files | One script; notebook only | One script forces torch on every scoring run and cannot be split between CPU and Kaggle. A notebook-only flow is untestable. |
| D3 | Export inputs once (rgb + fused depth + K + plane) | Ship bags to Kaggle; re-fuse there | Bags are large and need pyrealsense2. The npz is a few MB per bunch. |
| D4 | `FusionConfig(reader="fixed", frames="steady")` (the "A_steady" row) for every provider and the control | Native unaligned depth (0.99 kg, best) | Unaligned depth is not on the colour grid, so a colour-image mask cannot be applied to it. A_steady is the best aligned variant (session 1 MAE 1.33 vs 1.35 kg). The control is re-run on the same input, so the comparison is fair even though it is not the 0.99 kg headline. |
| D5 | Plane from `plane_height_mask` on the same fused depth for all providers | Refit the plane per mask from a tarp ring | The ring fit depends on the mask and was worse held-out. A shared plane keeps the segmenter the only varying factor. If no plane can be fitted, the bunch is a pipeline failure for every provider including the control. |
| D6 | Primary scope: session 1; control restricted to the same bunches and fit pool | Use session 2's unregistered RGB; mix sessions | Session 2 RGB shows another time and pose: masks would fail for reasons unrelated to the model. |
| D7 | Secondary modality: a height-above-plane map rendered as a 3-channel image, given to the same detectors, all 11 bunches | Skip; add depth as an extra SAM2 channel | It is the only route to all scored bunches and the literal "3D-aware" input. Secondary because detectors are trained on photographs: poor results must not be read as a verdict on the models for real RGB. |
| D8 | SAM2 baselines use fixed, label-free prompts (image-centre point; central box). An "oracle box" from the control mask is a diagnostic only | Use the control's box as the main SAM2 box prompt | That makes SAM2 depend on the control: not zero-shot and circular. The oracle row separates localisation loss from SAM2 boundary quality. |
| D9 | Same post-processing for all RGB providers (fill holes, keep the largest component); control untouched | Per-provider tuning | Tuning on ~10 bunches repeats the over-tuning the report already admits. |
| D10 | Failures scored two ways, both reported: detected-only and failure-penalised (prediction = training-mean mass) | Drop silently; impute zero | Dropping hides coverage and flatters a provider that rarely fires. Zero is arbitrary and huge. The training mean is the no-information baseline. |
| D11 | Mass metrics under LOO with `global_k` (mass = k V): MAE, MAPE, signed bias, percentile bootstrap 95% CIs over bunches (B=20000, seed 0), paired bootstrap vs control | IoU as primary; R2 | The question is mass error. IoU is a secondary diagnostic against the control mask, which is not ground truth. |
| D12 | Calibration curve from stored volumes (`results/session1/volumes.csv`, row `B_steady_native`, the 0.99 kg recipe), exhaustive subset enumeration, scoring on the complement | Re-run bags; random subsets | Volumes already exist and are deterministic. C(10,k) is at most 252, so enumeration is exact and removes Monte Carlo noise. |
| D13 | Run models on CPU if the measured cost is small, else Kaggle | Always Kaggle | See protocol runtime section. |
| D14 | Weights live outside the repo (`D:\Work\models\zeroshot`, via `HF_HOME`), and `.gitignore` is extended anyway | Cache inside the repo | Weights must never be committed. |

## 5. Calibration-curve design

Pool: the 10 scored bunches. For each k in 1..9, every subset of k bunches fits `mass = g V` and is scored on the
other 10 - k bunches. Each bunch's error is averaged over all subsets that leave it out, and the interval is a
percentile bootstrap over bunches. A second interval, the 5th to 95th percentile of subset MAE, shows what a
practitioner who picks k bunches at random might see. k = 10 has no held-out bunch inside the pool and is reported
in-sample only. k = 11 would need FFB18, which is not scored; a sensitivity run lets FFB18 into the fit pool
(k = 1..10 of 11 candidates, scored on the scored bunches not drawn), and k = 11 is stated as unmeasurable.
An `aV+b` model is a sensitivity result for k >= 3.

## 6. Risks

- **n is tiny.** The primary zero-shot comparison has 5 scored bunches. CIs will be wide; the study is a feasibility check.
- **Instance selection** (which box to keep) can dominate results; fixed in advance in the protocol.
- **Tarp pixels in a mask** contribute near-zero volume but noise clipped at zero gives a small positive bias; reported through signed bias.
- **Version drift.** Checkpoints and library versions are in the protocol and in each run's `meta.json`.
- **CPU timing** may differ from the estimate; the Kaggle notebook runs the same script.
- **YOLO-World needs ultralytics**, not a repo requirement; installed outside the repo environment, not in CI.
- **The control is not the 0.99 kg headline** (D4); compare only against the control row of this study.
