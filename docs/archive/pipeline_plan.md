> Superseded by [report.md](../../report.md); kept as a record of the original plan.

# Zero-Shot 3D FFB Mass Estimation — Project Plan

## 1. Problem Statement

Estimate the **mass of Oil Palm Fresh Fruit Bunches (FFBs)** from a single paired RGB+Depth frame captured by an Intel RealSense camera (planned as a D435i; the recorded bags report a D455, serial 215122256082) mounted over a loading-ramp scale. The legacy system uses brittle bounding-box depth thresholding and PCA ellipsoid approximations; this project replaces it with a modular, zero-shot AI pipeline.

---

## 2. Data

| Item | Detail |
|------|--------|
| Camera | Intel RealSense **D455** (active stereo IR) |
| Depth scale | 0.001 m/unit (confirmed from sensor profile) |
| Format | RealSense `.bag` (ROS-compatible) |
| Samples | FFB10–FFB35 (11 bunches), each with a depth bag, RGB bag, and reference colour PNG |
| Ground truth | Scale weight per bunch (to be linked during regressor training) |

### Dataset layout (validated 2026-06-10)

| Bundle group | Depth res | RGB res | Bag layout | Void depth | Mean depth |
|---|---|---|---|---|---|
| FFB10, 11, 12, 17, 18, 19 | 1280×720 | 1280×720 | Combined bag (depth+colour) | ~7–8% | ~1.48–1.54 m |
| FFB31, 32, 33, 34, 35 | **848×480** | 1280×720 | **Split bags** (DEPTH.bag + RGB.bag) | ~5–6% | ~1.54–1.55 m |

> **Note:** FFB31-35 have a depth/RGB resolution mismatch. The pipeline resizes the SAM 2 mask (generated at RGB resolution) down to 848×480 with nearest-neighbour interpolation before applying it to the depth map. Camera intrinsics must match the **depth** stream resolution.

---

## 3. Pipeline Architecture

```
.bag file
   │
   ▼
[bag_reader.py]  ──────────────────────────────────────────────
   │  aligned RGB (H×W×3 uint8) + Depth (H×W uint16)
   ▼
[FFBPerceptionPipeline.process_frame()]
   │
   ├─ Step 1: YOLO-World (yolov8s-world.pt)
   │     vocabulary = "oil palm fruit bunch"
   │     output: bbox [x1,y1,x2,y2], confidence
   │     fallback: centre 50% crop if no detection
   │
   ├─ Step 2: SAM 2 (sam2_hiera_small)
   │     prompt: bbox from Step 1
   │     output: binary mask (H×W bool)
   │     fallback: filled bbox region if mask empty
   │
   ├─ Step 3: DepthAnything v2 fusion (OPTIONAL toggle)
   │     fills IR-reflectance voids in masked depth region
   │     scale+shift alignment to metric via least-squares
   │
   └─ Step 4: 3D Projection (NumPy vectorised)
         pinhole back-projection per masked pixel
         output: (N,3) float32 point cloud (metres)
         output: convex hull volume (m³) via scipy.spatial
   │
   ▼
[estimate_mass_from_pointcloud()]  ← placeholder
   PointNet regressor (to be trained)
   output: predicted mass (kg)
```

---

## 4. Preprocessing Steps

### 4.1 Bag File Extraction
- Use `pyrealsense2` to open each `.bag` with `repeat_playback=False`.
- Align depth stream to colour stream with `rs.align(rs.stream.color)`.
- Extract per-bundle: up to N frames (typically 30–60 at 30 fps → 1–2 sec capture).
- Save `depth_scale` (metres/unit) from the sensor profile.

### 4.2 Depth Preprocessing
| Issue | Cause | Fix |
|-------|-------|-----|
| Pixels = 0 | IR occlusion, reflective fruitlets | DepthAnything fusion (Step 3) |
| Outlier spikes | Multipath IR interference | Median filter 3×3 before projection |
| Background floor | Ramp surface visible outside bunch | SAM 2 mask eliminates |

Apply a **3×3 median filter** to the raw depth map before projection:
```python
from scipy.ndimage import median_filter
depth_filtered = median_filter(depth_m, size=3)
```

### 4.3 RGB Preprocessing
- Convert BGR → RGB (done in `bag_reader.py`).
- No normalisation needed before YOLO-World (handled internally by ultralytics).
- SAM 2 expects uint8 RGB; ensure `rgb.dtype == np.uint8`.

### 4.4 Representative Frame Selection
When a bag contains multiple frames, pick the frame with the **highest valid depth pixel count within the central 50% crop** — this is typically the sharpest, least-motion frame.

```python
best_frame, best_count = None, 0
for rgb, depth in iter_frames(bag_path, max_frames=60):
    h, w = depth.shape
    crop = depth[h//4:3*h//4, w//4:3*w//4]
    count = int((crop > 0).sum())
    if count > best_count:
        best_count, best_frame = count, (rgb, depth)
```

### 4.5 Camera Intrinsics
- Extract per-device from the bag profile via `bag_reader.get_intrinsics_from_bag(depth_bag)`.
- The intrinsics must match the **depth stream resolution** (not the RGB resolution).
- Fallbacks: `CameraIntrinsics.default_848x480()` for FFB31-35, `CameraIntrinsics.default_1280x720()` for FFB10-19.
- **Always use bag-extracted intrinsics** in production — factory defaults are approximate.

---

## 5. Environment & Dependencies

### 5.1 Python packages
```
pyrealsense2>=2.54
ultralytics>=8.2        # YOLOWorld
sam2                    # pip install git+https://github.com/facebookresearch/segment-anything-2
depth-anything-v2       # pip install git+https://github.com/DepthAnything/Depth-Anything-V2
torch>=2.0
torchvision
open3d>=0.18            # optional visualisation
scipy
numpy
opencv-python
```

### 5.2 Model weights (download once)
| Model | File | ~Size |
|-------|------|-------|
| YOLO-World S | `yolov8s-world.pt` | 50 MB |
| SAM 2 Small | `sam2_hiera_small.pt` | 185 MB |
| DepthAnything v2 ViT-S | `depth_anything_v2_vits.pth` | 98 MB |

### 5.3 Kaggle GPU setup snippet
```python
import os, subprocess
# Install deps (run once per session)
subprocess.run(["pip", "install", "-q",
    "ultralytics",
    "git+https://github.com/facebookresearch/segment-anything-2.git",
    "pyrealsense2",
])
```

---

## 6. Downstream: PointNet Mass Regressor (TODO)

### Input representation
- Sample 1 024 points from the 3D point cloud via **Farthest Point Sampling (FPS)**.
- Centre at centroid and scale to unit sphere (standard PointNet normalisation).

### Architecture suggestion
- **PointNet++** (MSG variant) — handles variable-density FFB point clouds well.
- Input: (B, 1024, 3) — 3D XYZ only; optionally add RGB (B, 1024, 6).
- Output head: single regression neuron → mass in kg.

### Training data
- One labelled sample per bunch (`estimated_convex_volume` + scale reading).
- With 11 bunches, use **leave-one-out cross-validation**.
- Data augmentation: random rotation around Z-axis, small jitter, random point dropout.

### Loss
```python
loss = F.smooth_l1_loss(pred_mass, true_mass)
```

---

## 7. File Structure

```
RGBD-Mass/
├── data/
│   ├── FFB10/
│   │   ├── ffb10Depth_3D.bag
│   │   ├── ffb10RGB_2D.bag
│   │   └── ffb10SC_Color.png
│   └── ... (FFB11 – FFB35)
├── perception_pipeline.py   # Main pipeline class
├── bag_reader.py             # RealSense .bag extraction utilities
├── pipeline_plan.md          # This document
└── notebooks/
    └── kaggle_demo.ipynb     # End-to-end Kaggle GPU demo
```

---

## 8. Execution Order

1. `bag_reader.py` — validate extraction on one bundle.
2. `perception_pipeline.py` smoke-test (`__main__` block).
3. Run full pipeline over all 11 FFB bundles; save `PerceptionResult` per bundle to `.npz`.
4. Inspect point clouds in Open3D; verify SAM 2 mask quality.
5. Build PointNet regressor; train with leave-one-out CV.
6. Evaluate: MAE (kg), MAPE (%), R².

---

## 9. Known Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| YOLO-World misses FFB (low confidence) | Centre-crop fallback; tune confidence threshold |
| SAM 2 leaks onto ramp surface | Use stricter bbox padding; add point prompt at bunch centre |
| Depth voids > 50% of bunch | Enable DepthAnything fusion; ensure ≥10 valid anchor pixels for alignment |
| VRAM OOM on T4 with all 3 models | Load models sequentially; del + torch.cuda.empty_cache() between steps |
| Small dataset (11 samples) | FPS augmentation; transfer-learn from pre-trained PointNet++ |

---

## 10. Real-Time Inference — What Would Need to Change

The current pipeline is **batch / offline**: it reads from `.bag` files, scans all frames to find the best one, and processes each bundle independently. Making it real-time (live camera at 30 fps) requires changes at every layer.

### 10.1 What "real-time" means here

The ramp use-case is not strict 30 fps video — a bunch sits stationary on the scale for several seconds. "Real-time" means:

- Camera streams continuously at 30 fps
- Pipeline detects when a bunch is present and stable
- Produces a mass estimate within ~1–2 seconds of the bunch settling
- Outputs to the scale display / logging system without human intervention

This is a **triggered single-shot** pattern, not continuous tracking. That distinction drives most of the design decisions below.

---

### 10.2 Change 1 — Replace bag reader with live RealSense stream

**Current:** `iter_bundle()` opens a `.bag` file, decodes all frames, picks the best one.

**Real-time:** Open a live pipeline with `rs.pipeline()` and stream directly.

```python
# realtime_camera.py
import pyrealsense2 as rs
import numpy as np

class LiveRealSense:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        rgb   = np.asanyarray(aligned.get_color_frame().get_data())
        depth = np.asanyarray(aligned.get_depth_frame().get_data())
        return rgb, depth

    def stop(self):
        self.pipeline.stop()
```

**Nothing else in the pipeline changes** — `process_frame()` already takes a numpy RGB + depth pair.

---

### 10.3 Change 2 — Add a bunch-presence trigger

Without a trigger, the pipeline runs on every frame including empty ramp, conveyor motion, and workers passing through.

**Approach:** compare consecutive depth frames. A bunch settling on the scale produces a large sudden change in the depth centroid of the ramp region, then stabilises.

```python
def is_bunch_stable(depth_history: list[np.ndarray], roi: tuple, 
                    n_stable: int = 10, threshold_m: float = 0.01) -> bool:
    """
    Returns True when the mean depth in the ROI has been stable
    (std < threshold_m) across the last n_stable frames.
    roi = (y1, y2, x1, x2) pixels
    """
    y1, y2, x1, x2 = roi
    means = [d[y1:y2, x1:x2][d[y1:y2, x1:x2] > 0].mean() * 0.001
             for d in depth_history[-n_stable:]]
    return len(means) == n_stable and np.std(means) < threshold_m
```

Only run the expensive pipeline (YOLO + SAM 2) once the scene is confirmed stable — this eliminates ~99% of frames.

---

### 10.4 Change 3 — Replace SAM 2 ImagePredictor with VideoPredictor

This is the biggest performance change. The current `SAM2ImagePredictor` runs the full image encoder (~100 ms on T4) on every call to `set_image()`. The `SAM2VideoPredictor` runs the encoder **once on a keyframe**, then propagates the mask using only the lightweight decoder (~15 ms/frame).

```
Current (ImagePredictor, every frame):
  set_image() → full encoder → decoder → mask   ~100 ms/frame

Real-time (VideoPredictor, after keyframe):
  init_state(keyframe) → full encoder once
  propagate_in_video() → decoder only            ~15 ms/frame
```

**When to re-run the full encoder (new keyframe):**
- When the trigger fires (new bunch detected)
- If mask IoU vs previous frame drops below ~0.7 (bunch shifted)

```python
# Sketch of VideoPredictor integration
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(config, checkpoint, device="cuda")

# On trigger — run once
with torch.inference_mode():
    state = predictor.init_state(video_path=None, offload_to_cpu=False)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=bbox)

# Each subsequent frame
with torch.inference_mode():
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        current_mask = masks[0][0].cpu().numpy() > 0.0
```

---

### 10.5 Change 4 — Run YOLO only on keyframes

YOLO-World (~25 ms on T4) is needed to generate the initial bbox prompt for SAM 2. Once the VideoPredictor is tracking, YOLO is redundant until the next bunch.

**Strategy:**
- Run YOLO once on trigger frame
- If VideoPredictor loses the mask (IoU drop), re-run YOLO to re-anchor
- Skip YOLO on all other frames

This reduces YOLO from every-frame to once-per-bunch.

---

### 10.6 Change 5 — Producer-consumer threading

Bag reading / camera I/O and model inference run at different rates. Without threading, inference blocks the camera buffer and frames are dropped silently.

```
Thread A (camera thread)          Thread B (inference thread)
────────────────────────          ──────────────────────────
get_frame() @ 30 fps  ──queue──▶  stability check
if stable: put to queue            if triggered: YOLO + SAM2 keyframe
                                   else: SAM2 propagate
                                   project + volume
                                   output to display/log
```

```python
import threading, queue

frame_queue = queue.Queue(maxsize=5)   # drop old frames if inference falls behind

def camera_worker(cam, trigger_fn, q):
    history = []
    while True:
        rgb, depth = cam.get_frame()
        history.append(depth)
        if len(history) > 15: history.pop(0)
        if trigger_fn(history):
            try:
                q.put_nowait((rgb, depth))
            except queue.Full:
                pass   # drop frame — inference is busy

def inference_worker(pipeline, K, q, output_fn):
    while True:
        rgb, depth = q.get()
        result = pipeline.process_frame(rgb, depth, K)
        output_fn(result)
```

---

### 10.7 Change 6 — Remove/replace the median filter

`cv2.medianBlur` at 5 ms is fast but still a synchronous CPU call inside the GPU inference path. In streaming mode, move it to the camera thread so it runs in parallel with inference.

Alternatively, for a static scene (bunch is still) skip the filter entirely — temporal averaging across frames suppresses IR spikes better than a single-frame spatial filter.

---

### 10.8 Change 7 — Output interface

**Current:** Returns a `BundleResult` dict, printed to console.

**Real-time:** Needs to write to wherever the ramp scale system reads from:

| Output target | Mechanism |
|---|---|
| Serial/RS-232 to scale display | `pyserial` write |
| Modbus PLC | `pymodbus` holding register |
| Local web dashboard | FastAPI WebSocket push |
| CSV/database log | `sqlite3` / append to file |

---

### 10.9 Summary: batch vs real-time component map

| Component | Batch (current) | Real-time |
|---|---|---|
| Frame source | `iter_bundle()` from `.bag` | `LiveRealSense.get_frame()` |
| Frame selection | Scan 60 frames, pick best | Stability trigger on depth ROI |
| SAM 2 mode | `SAM2ImagePredictor` (encoder every frame) | `SAM2VideoPredictor` (encoder once, decoder per frame) |
| YOLO cadence | Every frame | Once per bunch (keyframe only) |
| Depth filter | `cv2.medianBlur` in inference thread | Move to camera thread; or skip (temporal avg) |
| Aggregation | Median of top-5 frames | Single triggered frame (bunch is stable) |
| Threading | Single-threaded | Camera thread + inference thread + queue |
| Output | `PerceptionResult` dict | Serial / Modbus / WebSocket / DB |
| Trigger | None (explicit call) | Depth-stability detector on ramp ROI |

**Estimated real-time latency (T4, after trigger fires):**

| Step | Time |
|---|---|
| Stability confirmation (10 frames @ 30 fps) | ~333 ms |
| YOLO keyframe detection | ~25 ms |
| SAM 2 keyframe encode | ~100 ms |
| 3D projection + volume | ~20 ms |
| **Total from bunch-stable to estimate** | **~480 ms** |

Under 0.5 seconds from bunch settling to mass estimate — well within practical ramp throughput requirements.
