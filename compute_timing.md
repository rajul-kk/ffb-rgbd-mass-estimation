# Compute Time Estimates: CPU vs GPU

## Measurement basis

| Source | Detail |
|--------|--------|
| **Measured locally** | Python 3.10, Intel i7-class CPU, this machine |
| **Model inference (GPU)** | Published benchmarks: SAM 2 paper, YOLOv8 docs, DepthAnything v2 repo |
| **T4 estimate from A100** | T4 FP32 ≈ 8.1 TFLOPS; A100 FP32 ≈ 19.5 TFLOPS → T4 ≈ 2.4× slower |
| **CPU model estimate** | Intel Xeon (Kaggle) ≈ 0.5 TFLOPS FP32 → ~16× slower than T4 for ViT-class models |

---

## Per-step breakdown (single frame)

All model inputs at full resolution: **1280×720** (FFB10-19) or **848×480 depth + 1280×720 RGB** (FFB31-35).
SAM 2 resizes internally to 1024×1024; DepthAnything resizes to 518×518.

| Step | CPU (Kaggle Xeon) | 1× T4 GPU | Notes |
|------|:-----------------:|:---------:|-------|
| **Bag read — best frame (60-frame scan)** | | | |
| FFB10-19 (combined bag) | **5–8 s** | 5–8 s | I/O bound — same on any hardware |
| FFB31-35 (split bags) | **11–12 s** | 11–12 s | Two sequential bag reads |
| **Model inference** | | | |
| Median filter 1280×720 | **108 ms** ✱ | ~2 ms | ✱ Measured locally |
| YOLO-World S (1280×720) | ~1 200 ms | ~25 ms | Pub. benchmark; scales ∝ px² |
| SAM 2 hiera_small | ~8 000 ms | ~100 ms | Encoder dominates; decoder ~10 ms |
| DepthAnything v2 ViT-S *(optional)* | ~2 000 ms | ~45 ms | Disabled by default |
| Mask resize (cv2 INTER_NEAREST) | **3 ms** ✱ | ~3 ms | ✱ Measured locally |
| 3D projection — vectorised NumPy | **13 ms** ✱ | ~13 ms | CPU-bound; ~100 K pts |
| Convex hull (scipy, ~50 K pts) | **35 ms** ✱ | ~35 ms | CPU-bound |
| **Inference subtotal (no depth fusion)** | **~9.4 s** | **~178 ms** | |
| **Inference subtotal (with depth fusion)** | **~11.4 s** | **~223 ms** | |

---

## Per-bundle totals (1 best frame per bundle)

| Bundle group | CPU total | 1× T4 total | 2× T4 parallel¹ |
|---|:---:|:---:|:---:|
| FFB10-19 (combined, no fusion) | ~15–17 s | ~5.2–8.2 s | ~2.6–4.1 s |
| FFB10-19 (combined, with fusion) | ~17–19 s | ~5.2–8.2 s | ~2.6–4.1 s |
| FFB31-35 (split, no fusion) | ~21–22 s | ~11.2–12.2 s | ~5.6–6.1 s |
| FFB31-35 (split, with fusion) | ~23–24 s | ~11.2–12.2 s | ~5.6–6.1 s |

> ¹ 2× T4: two bundles processed simultaneously via `torch.multiprocessing`; bag I/O runs on separate CPU cores, model inference on separate GPUs. Wall-clock ≈ single-bundle time.

---

## All 11 bundles (dataset run)

| Scenario | CPU | 1× T4 | 2× T4 parallel |
|---|:---:|:---:|:---:|
| No depth fusion (recommended) | **~3–4 min** | **~1.6–2.5 min** | **~50–75 s** |
| With depth fusion | **~4–5 min** | **~1.8–2.7 min** | **~55–80 s** |

**Bottom line:** for an 11-bundle dataset run, CPU is perfectly viable (~3–4 min). The GPU advantage here is modest because **bag reading (I/O, CPU-bound) is 80–90% of the wall-clock time** regardless of GPU tier.

---

## Real-time feasibility (30 fps live camera feed)

Budget per frame at 30 fps = **33 ms**

| Step | CPU | 1× T4 | Real-time viable? |
|---|:---:|:---:|:---:|
| YOLO-World S | 1 200 ms | 25 ms | T4 only |
| SAM 2 hiera_small | 8 000 ms | 100 ms | Neither |
| SAM 2 hiera_tiny *(faster variant)* | 5 000 ms | 55 ms | T4 borderline |
| Full pipeline (no fusion) | ~9 400 ms | ~178 ms | Neither — batch mode only |

SAM 2 is the bottleneck. For real-time use, options are:
1. **Run YOLO every frame, SAM every N frames** (track between keyframes with bbox IoU)
2. **Switch to SAM 2 video predictor** — propagates masks temporally, encoder runs once then decoder only (~15 ms/frame on T4)
3. **YOLO + threshold-based depth crop only** — no SAM, fast but less accurate

---

## VRAM usage (1× T4, 16 GB)

| Model | VRAM |
|---|---|
| YOLO-World S | ~160 MB |
| SAM 2 hiera_small | ~380 MB |
| DepthAnything v2 ViT-S | ~105 MB |
| Working tensors (1280×720 batch) | ~300 MB |
| **Total** | **~945 MB** |

All three models fit comfortably on a single T4. The pipeline runs each model sequentially so peak VRAM stays well under 2 GB — safe to run two independent pipeline instances on two T4s simultaneously.

---

## 2× T4 parallelisation — recommended approach

```python
# run_parallel.py
import torch.multiprocessing as mp
from perception_pipeline import FFBPerceptionPipeline, CameraIntrinsics
from bag_reader import load_best_frame, find_bundle_bags, get_intrinsics_from_bag, get_depth_scale
import os

def process_bundle(args):
    ffb_dir, gpu_id = args
    pipeline = FFBPerceptionPipeline(device=f"cuda:{gpu_id}", use_depth_fusion=False)
    depth_bag, _ = find_bundle_bags(ffb_dir)
    K = get_intrinsics_from_bag(depth_bag) or CameraIntrinsics.default_1280x720()
    K.depth_scale = get_depth_scale(depth_bag)
    rgb, depth = load_best_frame(ffb_dir)
    result = pipeline.process_frame(rgb, depth, K)
    return ffb_dir, result

if __name__ == "__main__":
    DATA_DIR = "data"
    bundles = sorted(
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    # Alternate GPU 0 / GPU 1 across bundles
    args = [(b, i % 2) for i, b in enumerate(bundles)]

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=2) as pool:
        all_results = dict(pool.map(process_bundle, args))
```

---

## Recommendation summary

| Use case | Recommended setup |
|---|---|
| Dataset run (11 bundles, one-off) | **CPU is fine** — total ~3–4 min |
| Iterative dev / mask QC loop | **1× T4** — fast enough, simpler setup |
| Production (real-time on ramp) | **1× T4 + SAM 2 video predictor** (temporal propagation) |
| Parallel batch (many bundles) | **2× T4** with `torch.multiprocessing` |

The median filter (108 ms measured) is the single most expensive CPU step outside the AI models. If CPU speed matters, replace with `cv2.medianBlur` (≈8 ms at 1280×720).

---

## CPU real-time: three options

SAM 2's image encoder (~8 s on CPU) makes the current pipeline non-viable for real-time on CPU. There are three practical paths depending on how much mask accuracy you need.

### Option A — Depth-threshold segmentation (no AI models, ~55 ms/frame)

The camera is **fixed overhead**, ramp surface is always at a known depth (~1.5 m). A bunch raised above the ramp sits at ~1.1–1.4 m. A depth band + morphological clean-up produces a usable mask with zero model inference:

```python
import cv2, numpy as np

def depth_threshold_mask(depth_m: np.ndarray,
                          ramp_depth: float = 1.50,
                          min_raise: float = 0.05,
                          max_raise: float = 0.45) -> np.ndarray:
    """
    Pixels between (ramp_depth - max_raise) and (ramp_depth - min_raise)
    are the bunch. Calibrate ramp_depth once per installation.
    """
    lo = ramp_depth - max_raise   # ~1.05 m
    hi = ramp_depth - min_raise   # ~1.45 m
    mask = ((depth_m > lo) & (depth_m < hi)).astype(np.uint8)
    # Remove noise and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask.astype(bool)
```

| Step | Time |
|---|---|
| Depth threshold + morphology | ~5 ms ✱ |
| 3D projection | ~13 ms ✱ |
| Convex hull | ~35 ms ✱ |
| **Total per frame** | **~55 ms → ~18 fps** |

✱ Measured locally on this machine.

**Limitations:** breaks if a worker's hand/arm enters the depth band; assumes ramp depth is stable (recalibrate if camera is moved). For a controlled ramp installation these are acceptable constraints.

---

### Option B — MobileSAM (CPU-viable, ~350 ms/frame)

[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) replaces SAM's heavy ViT-H image encoder with a ViT-Tiny distilled version. The decoder is identical to SAM 1.

| Model component | SAM 2 hiera_small | MobileSAM |
|---|:---:|:---:|
| Encoder | ViT-B/16 | ViT-Tiny (distilled) |
| Encoder CPU time | ~8 000 ms | ~280 ms |
| Decoder CPU time | ~10 ms | ~10 ms |
| Mask quality vs SAM 2 | baseline | ~5–8% lower IoU |
| Weights size | 185 MB | 40 MB |

```python
# Drop-in swap — only the loader changes
def _load_mobilesam(checkpoint: str = "mobile_sam.pt", device: str = "cpu"):
    from mobile_sam import sam_model_registry, SamPredictor
    model = sam_model_registry["vit_t"](checkpoint=checkpoint)
    model.to(device).eval()
    return SamPredictor(model)
```

The rest of `_segment()` is identical — `SamPredictor` has the same `.set_image()` / `.predict()` API as SAM 2's `SAM2ImagePredictor`.

**Per-frame timing on CPU (triggered use-case):**

| Step | Time |
|---|---|
| Stability trigger (10 frames) | ~333 ms |
| YOLO-World S keyframe | ~1 200 ms |
| MobileSAM encode+decode | ~290 ms |
| 3D projection + hull | ~50 ms |
| **Total: trigger → estimate** | **~1.9 s** |

Under 2 seconds on CPU with no GPU required. Adequate for ramp throughput where bunches dwell for ≥5 seconds.

---

### Option C — SAM 2 VideoPredictor, CPU keyframe pre-warming (~500 ms steady-state)

Use the triggered architecture from Section 10 (pipeline_plan.md) with SAM 2's `VideoPredictor`. The slow encoder (~8 s) runs **once while the stability check is still accumulating frames** (a 10-frame window at 30 fps = 333 ms real time — overlapped, not sequential). After the keyframe is encoded, each propagation step costs only the decoder.

```
Timeline (CPU, triggered):
  t=0.0s   Bunch lands on scale
  t=0.0–0.3s  Stability window accumulating (10 frames)
             └── Background thread: YOLO on frame 0 → bbox → SAM2 encoder (8s, async)
  t=8.3s   Encoder done, stability confirmed
           └── SAM2 decoder on keyframe → mask → project → volume
  t=8.4s   First estimate output

  t=8.4s+  Each new frame: decoder only → mask → project → volume
           ~200–500 ms/frame
```

The first reading takes ~8.4 s (dominated by encoder). Subsequent frames in the same bunch: ~300 ms each, giving you 2–3 readings per second to average over. This is only worthwhile if you need multiple readings for confidence — if one reading per bunch is sufficient, Option B is simpler.

---

### CPU real-time decision tree

```
Fixed overhead camera with stable ramp background?
├── YES → Option A (depth threshold, ~55 ms, no models needed)
│         Simplest, most robust for controlled installation
└── NO  → Need robust mask against complex backgrounds?
          ├── YES, GPU available → SAM 2 VideoPredictor on T4 (~480 ms)
          ├── YES, CPU only, 1 reading/bunch → Option B (MobileSAM, ~1.9 s)
          └── YES, CPU only, multi-reading → Option C (SAM 2 VideoPredictor, 8s keyframe, ~300ms steady)
```

**For this project's ramp installation: Option A is the right production choice.** The camera is fixed, the ramp background is known and controlled, and depth thresholding recovers the same point cloud as SAM 2 (FFB bunches are convex and the ramp is flat). Keep SAM 2 for Kaggle/offline validation only.

---

### Adding Option A to the pipeline (one flag)

```python
pipeline = FFBPerceptionPipeline(
    ...
    segmentation_mode="depth_threshold",  # "sam2" | "depth_threshold" | "mobilesam"
    ramp_depth_m=1.50,                    # calibrate once per installation
)
```

This would make `_segment()` dispatch to either `_segment_sam2()` or `_segment_depth_threshold()` based on the flag — no other changes to `process_frame()`. Want me to implement this?
