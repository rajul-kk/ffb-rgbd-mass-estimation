"""
perception_pipeline.py
Zero-shot 3D point cloud extraction for Oil Palm FFB mass estimation.
Pipeline: YOLO-World -> SAM 2 -> (optional) DepthAnything v2 -> 3D projection
Target: Kaggle T4/P100 GPU (16 GB VRAM)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """
    RealSense D455 intrinsic parameters.
    Actual values should always be read from the bag via bag_reader.get_intrinsics_from_bag().
    The class-method defaults are approximate fallbacks only.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 848
    height: int = 480
    depth_scale: float = 0.001  # RealSense raw uint16 -> metres

    @classmethod
    def default_848x480(cls) -> "CameraIntrinsics":
        # D455 @ 848x480 — approximate; calibrate per device for production.
        return cls(fx=424.0, fy=424.0, cx=424.0, cy=240.0,
                   width=848, height=480)

    @classmethod
    def default_1280x720(cls) -> "CameraIntrinsics":
        # D455 @ 1280x720
        return cls(fx=637.0, fy=637.0, cx=640.0, cy=360.0,
                   width=1280, height=720)


@dataclass
class PerceptionResult:
    mask: np.ndarray                  # (H, W) bool
    point_cloud: np.ndarray           # (N, 3) float32, metric metres
    estimated_convex_volume: float    # m^3
    detection_confidence: float = 0.0
    depth_fill_ratio: float = 0.0     # fraction of mask pixels where depth was interpolated


# ---------------------------------------------------------------------------
# Model lazy-loaders (called once in __init__)
# ---------------------------------------------------------------------------

def _load_yolo_world(weights: str = "yolov8s-world.pt") -> object:
    from ultralytics import YOLOWorld
    model = YOLOWorld(weights)
    model.set_classes(["oil palm fruit bunch"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _load_sam2(config: str = "sam2_hiera_small.yaml",
               checkpoint: str = "sam2_hiera_small.pt",
               device: str = "cpu") -> object:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # SAM 2 v1.0 uses "sam2_hiera_small.yaml";
    # SAM 2 v1.1 (sam2.1) renamed configs to "sam2.1_hiera_small.yaml".
    # Try the supplied name first, then the v1.1 equivalent automatically.
    def _try_build(cfg_name):
        import os as _os
        for candidate in [cfg_name, _os.path.basename(cfg_name)]:
            try:
                return build_sam2(candidate, checkpoint, device=device)
            except Exception:
                pass
        return None

    alt_config = config.replace("sam2_", "sam2.1_")
    model = _try_build(config) or _try_build(alt_config)
    if model is None:
        raise RuntimeError(
            f"Could not load SAM 2 with config '{config}' or '{alt_config}'. "
            "Check that the checkpoint and installed SAM 2 version match."
        )

    predictor = SAM2ImagePredictor(model)
    # SAM 2 defaults to bfloat16/autocast which is unsupported on CPU.
    if device == "cpu":
        predictor.model = predictor.model.float()
    return predictor


def _load_depth_anything(encoder: str = "vits") -> object:
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder=encoder, features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(
        torch.load(f"depth_anything_v2_{encoder}.pth", map_location="cpu")
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------

class FFBPerceptionPipeline:
    """
    Zero-shot FFB perception: detect -> segment -> (fuse depth) -> 3D project.

    Args:
        yolo_weights:      Path to YOLOWorld weights file.
        sam2_config:       SAM 2 model config name (must match checkpoint).
        sam2_checkpoint:   Path to SAM 2 checkpoint.
        use_depth_fusion:  Enable DepthAnything v2 void-filling.
        da_encoder:        DepthAnything backbone ('vits' | 'vitb' | 'vitl').
    """

    def __init__(
        self,
        yolo_weights: str = "yolov8s-world.pt",
        sam2_config: str = "sam2_hiera_small.yaml",
        sam2_checkpoint: str = "sam2_hiera_small.pt",
        use_depth_fusion: bool = False,
        da_encoder: str = "vits",
        device: Optional[str] = None,  # None = auto-detect; pass "cpu" to force CPU
    ) -> None:
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Pipeline] Initialising on device: {self.device}")

        self.yolo = _load_yolo_world(yolo_weights)
        # YOLO-World respects the device it was loaded on; re-send if forced
        if device is not None:
            self.yolo.to(self.device)
        print("[Pipeline] YOLO-World loaded.")

        self.sam2 = _load_sam2(sam2_config, sam2_checkpoint, device=self.device)
        print("[Pipeline] SAM 2 loaded.")

        self.depth_model: Optional[object] = None
        self.use_depth_fusion = use_depth_fusion
        if use_depth_fusion:
            self.depth_model = _load_depth_anything(da_encoder)
            print("[Pipeline] DepthAnything v2 loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        rgb_image: np.ndarray,          # (H, W, 3) uint8 RGB
        depth_map: np.ndarray,          # (H, W) uint16 or float32 (raw sensor)
        camera_intrinsics: CameraIntrinsics,
    ) -> PerceptionResult:
        """
        Full inference pass for one frame.

        Handles RGB/depth size mismatch: YOLO-World and SAM 2 run on the
        full-resolution RGB; the resulting mask is downsampled to match the
        depth map before 3D projection.

        Returns a PerceptionResult with mask, point cloud and convex volume.
        """
        import cv2

        # 1. Ensure float depth in metres
        depth_m = self._to_metric_depth(depth_map, camera_intrinsics.depth_scale)
        dh, dw = depth_m.shape

        # 1b. Median filter to suppress IR multipath spikes before projection
        # cv2.medianBlur (~8 ms) is ~13× faster than scipy.ndimage at 1280×720
        import cv2 as _cv2
        depth_m = _cv2.medianBlur(depth_m, 3)

        # 2. Detect FFB with YOLO-World (runs on full-res RGB)
        bbox, confidence = self._detect(rgb_image)

        # 3. Segment with SAM 2 (runs on full-res RGB, yields full-res mask)
        mask_rgb_res = self._segment(rgb_image, bbox)

        # 4. Resize mask to depth resolution if they differ
        rh, rw = rgb_image.shape[:2]
        if (rh, rw) != (dh, dw):
            mask_depth_res = cv2.resize(
                mask_rgb_res.astype(np.uint8),
                (dw, dh),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_depth_res = mask_rgb_res

        # 5. Optional: fill depth voids via DepthAnything v2 fusion
        fill_ratio = 0.0
        if self.use_depth_fusion and self.depth_model is not None:
            # Resize RGB to depth resolution for DepthAnything inference
            if (rh, rw) != (dh, dw):
                rgb_for_fusion = cv2.resize(rgb_image, (dw, dh))
            else:
                rgb_for_fusion = rgb_image
            depth_m, fill_ratio = self._fuse_depth(rgb_for_fusion, depth_m, mask_depth_res)

        # 6. Project to 3D point cloud
        point_cloud = self._project_to_3d(depth_m, mask_depth_res, camera_intrinsics)

        # 7. Compute convex hull volume
        volume = self._compute_convex_volume(point_cloud)

        return PerceptionResult(
            mask=mask_depth_res,
            point_cloud=point_cloud,
            estimated_convex_volume=volume,
            detection_confidence=float(confidence),
            depth_fill_ratio=float(fill_ratio),
        )

    # ------------------------------------------------------------------
    # Step 1 – Detection
    # ------------------------------------------------------------------

    def _detect(self, rgb: np.ndarray) -> tuple[list[float], float]:
        """
        Run YOLO-World and return best [x1,y1,x2,y2] bbox + confidence.
        Falls back to a centre-crop box if no detections.
        """
        h, w = rgb.shape[:2]
        with torch.inference_mode():
            results = self.yolo.predict(rgb, verbose=False)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # Fallback: centre 50 % crop
            margin_x, margin_y = w * 0.25, h * 0.25
            return [margin_x, margin_y, w - margin_x, h - margin_y], 0.0

        # Pick highest-confidence detection
        confidences = boxes.conf.cpu().numpy()
        best = int(np.argmax(confidences))
        bbox = boxes.xyxy[best].cpu().numpy().tolist()
        return bbox, float(confidences[best])

    # ------------------------------------------------------------------
    # Step 2 – Segmentation
    # ------------------------------------------------------------------

    def _segment(self, rgb: np.ndarray, bbox: list[float]) -> np.ndarray:
        """
        Prompt SAM 2 with bbox, return (H, W) bool mask.
        """
        self.sam2.set_image(rgb)
        input_box = np.array(bbox, dtype=np.float32)[None]  # (1, 4)

        with torch.inference_mode():
            masks, scores, _ = self.sam2.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )

        # masks shape: (1, H, W) or (num_masks, H, W)
        mask = masks[int(np.argmax(scores))].astype(bool)

        if not mask.any():
            # Emergency fallback: fill bbox region
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = (int(v) for v in bbox)
            mask[y1:y2, x1:x2] = True

        return mask

    # ------------------------------------------------------------------
    # Step 3 – Depth fusion (optional)
    # ------------------------------------------------------------------

    def _fuse_depth(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Use DepthAnything v2 relative depth to fill zero/NaN voids within mask.
        Alignment: scale+shift relative depth to match valid metric depth pixels.
        """
        import torchvision.transforms.functional as TF

        void_mask = mask & (depth_m <= 0)
        n_void = int(void_mask.sum())
        n_mask = int(mask.sum())
        if n_void == 0 or n_mask == 0:
            return depth_m, 0.0

        # Infer relative depth
        rgb_tensor = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            rel_depth = self.depth_model(rgb_tensor)  # (1, 1, H, W) or (1, H, W)

        rel_depth = rel_depth.squeeze().cpu().numpy().astype(np.float32)

        # Align scale/shift using valid (non-void) mask pixels via least-squares
        valid = mask & (depth_m > 0)
        if valid.sum() < 10:
            return depth_m, 0.0

        A = np.stack([rel_depth[valid], np.ones(valid.sum())], axis=1)
        b = depth_m[valid]
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        scale, shift = result

        aligned = rel_depth * scale + shift
        fused = depth_m.copy()
        fused[void_mask] = np.clip(aligned[void_mask], 0, None)

        fill_ratio = n_void / n_mask
        return fused, fill_ratio

    # ------------------------------------------------------------------
    # Step 4 – 3D projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project_to_3d(
        depth_m: np.ndarray,
        mask: np.ndarray,
        K: CameraIntrinsics,
    ) -> np.ndarray:
        """
        Vectorised pinhole back-projection. Returns (N, 3) float32 array.
        Z = depth_m, X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy
        """
        h, w = depth_m.shape
        # Pixel grid
        us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))

        valid = mask & (depth_m > 0)
        Z = depth_m[valid].astype(np.float32)
        u = us[valid]
        v = vs[valid]

        # Remove background bleed-through: keep points within 0.5 m of median depth.
        # The FFB is a compact object (<0.4 m deep); background pixels at 1-2 m
        # further inflate the convex hull by 10-30x without this filter.
        if Z.size > 0:
            med_z = float(np.median(Z))
            keep = np.abs(Z - med_z) < 0.5
            Z, u, v = Z[keep], u[keep], v[keep]

        X = (u - K.cx) * Z / K.fx
        Y = (v - K.cy) * Z / K.fy

        return np.stack([X, Y, Z], axis=-1)  # (N, 3)

    # ------------------------------------------------------------------
    # Step 5 – Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_convex_volume(point_cloud: np.ndarray) -> float:
        """
        Convex hull volume (m^3) of the 3D point cloud.
        Returns 0.0 if cloud is too sparse for a valid hull.
        """
        if point_cloud.shape[0] < 4:
            return 0.0
        try:
            hull = ConvexHull(point_cloud)
            return float(hull.volume)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_metric_depth(depth: np.ndarray, scale: float) -> np.ndarray:
        """Convert raw sensor depth (uint16 or float) to float32 metres."""
        d = depth.astype(np.float32)
        if depth.dtype == np.uint16:
            d = d * scale
        return d


# ---------------------------------------------------------------------------
# Multi-frame aggregation (batch / offline mode)
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc

@_dc
class BundleResult:
    """Aggregated result across N frames of a single FFB bundle."""
    median_volume: float        # m³ — robust central estimate
    std_volume: float           # m³ — spread across frames (quality proxy)
    mean_volume: float          # m³
    merged_volume: float        # m³ — hull of all N merged point clouds
    per_frame_volumes: list     # [float] one per processed frame
    best_point_cloud: np.ndarray  # (N,3) from the highest-confidence frame
    best_mask: np.ndarray         # (H,W) bool from the highest-confidence frame
    n_frames_used: int


def process_bundle_multi_frame(
    ffb_dir: str,
    pipeline: "FFBPerceptionPipeline",
    camera_intrinsics: "CameraIntrinsics",
    n_frames: int = 5,
    max_scan: int = 60,
) -> BundleResult:
    """
    Process the top-N frames (by valid depth count) from a bundle and aggregate.

    Strategy:
      1. Scan up to max_scan frames from the bag, rank by valid depth pixels
         in the centre 50% crop (same heuristic as load_best_frame).
      2. Run the full pipeline on the top-N candidates.
      3. Return median/mean/std of per-frame convex volumes, plus a merged
         point cloud hull as an alternative estimate.

    Args:
        ffb_dir:            Path to the FFB bundle directory.
        pipeline:           Initialised FFBPerceptionPipeline instance.
        camera_intrinsics:  CameraIntrinsics for this bundle.
        n_frames:           How many top frames to process (default 5).
        max_scan:           Max frames to scan when ranking (default 60).

    Returns:
        BundleResult with median/std/merged volumes and per-frame breakdown.
    """
    from bag_reader import iter_bundle

    # ---- Rank frames by valid depth count in centre crop ----
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for rgb, depth in iter_bundle(ffb_dir, max_frames=max_scan):
        h, w = depth.shape
        crop = depth[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        score = int((crop > 0).sum())
        candidates.append((score, rgb.copy(), depth.copy()))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_n = candidates[: min(n_frames, len(candidates))]

    # ---- Run pipeline on each candidate ----
    volumes: list[float] = []
    all_pts: list[np.ndarray] = []
    best_result = None
    best_score = -1

    for score, rgb, depth in top_n:
        result = pipeline.process_frame(rgb, depth, camera_intrinsics)
        volumes.append(result.estimated_convex_volume)
        if result.point_cloud.shape[0] > 0:
            all_pts.append(result.point_cloud)
        if score > best_score:
            best_score = score
            best_result = result

    # ---- Aggregate ----
    vols = np.array(volumes, dtype=np.float32)
    merged_vol = 0.0
    if all_pts:
        merged_pts = np.concatenate(all_pts, axis=0)
        merged_vol = FFBPerceptionPipeline._compute_convex_volume(merged_pts)

    return BundleResult(
        median_volume=float(np.median(vols)),
        std_volume=float(np.std(vols)),
        mean_volume=float(np.mean(vols)),
        merged_volume=merged_vol,
        per_frame_volumes=volumes,
        best_point_cloud=best_result.point_cloud if best_result else np.zeros((0, 3)),
        best_mask=best_result.mask if best_result else np.zeros((0, 0), dtype=bool),
        n_frames_used=len(top_n),
    )


# ---------------------------------------------------------------------------
# Placeholder: downstream PointNet mass regressor
# ---------------------------------------------------------------------------

def estimate_mass_from_pointcloud(
    point_cloud: np.ndarray,
    model_path: Optional[str] = None,
) -> float:
    """
    Placeholder for PointNet-based mass regression.

    Args:
        point_cloud:  (N, 3) metric point cloud from PerceptionResult.
        model_path:   Path to trained PointNet weights (.pth).

    Returns:
        Predicted mass in kilograms (float). Returns -1.0 until implemented.
    """
    # TODO: load PointNet model, normalise point cloud, run inference
    # suggested steps:
    #   1. fps_sample(point_cloud, n=1024)  – farthest point sampling
    #   2. centre + scale to unit sphere
    #   3. model.forward(pts_tensor) -> mass_kg
    return -1.0


# ---------------------------------------------------------------------------
# CLI smoke-test helper (not for production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from bag_reader import load_best_frame, get_intrinsics_from_bag, find_bundle_bags, get_depth_scale

    ffb_dir = sys.argv[1] if len(sys.argv) > 1 else "data/FFB10"
    depth_bag, _ = find_bundle_bags(ffb_dir)

    K = get_intrinsics_from_bag(depth_bag) or CameraIntrinsics.default_1280x720()
    K.depth_scale = get_depth_scale(depth_bag)

    rgb, depth = load_best_frame(ffb_dir)

    pipeline = FFBPerceptionPipeline(use_depth_fusion=False, device="cpu")
    result = pipeline.process_frame(rgb, depth, K)

    print(f"Points extracted : {result.point_cloud.shape[0]:,}")
    print(f"Convex volume    : {result.estimated_convex_volume:.6f} m³")
    print(f"Detection conf   : {result.detection_confidence:.3f}")
    print(f"Depth fill ratio : {result.depth_fill_ratio:.3f}")
