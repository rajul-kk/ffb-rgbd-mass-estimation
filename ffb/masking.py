"""Mask-provider interface: any segmenter that maps a request to a mask on the depth grid plugs into the mass code.

The mass code (volume.plane_volume and evaluate.*) only ever sees a boolean mask on the depth grid, so
providers never touch volumes, depth or ground truth. This module needs numpy and cv2 only; model-backed
providers live in masking_models.py and import torch lazily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from . import segment, volume
from .camera import Intrinsics


@dataclass(frozen=True)
class MaskRequest:
    name: str
    depth: np.ndarray                      # fused depth, metres, 0 = no data
    K: Intrinsics
    plane: Optional[tuple]                 # (normal, d) from the depth-only tarp fit; independent of any mask
    rgb: Optional[np.ndarray] = None       # H x W x 3 uint8 on the depth grid, or None if not registered
    ref_mask: Optional[np.ndarray] = None  # control mask; read ONLY by oracle diagnostics, never by zero-shot providers


@dataclass(frozen=True)
class MaskResult:
    mask: Optional[np.ndarray]             # bool, depth grid; None = no detection
    score: Optional[float] = None          # provider confidence, diagnostic only
    meta: dict = field(default_factory=dict)


class MaskProvider(Protocol):
    name: str
    needs_rgb: bool

    def __call__(self, req: MaskRequest) -> MaskResult: ...


REGISTRY: dict[str, Callable[[], MaskProvider]] = {}


def register(name: str):
    """Class decorator: make a provider constructible by name without importing model libraries up front."""
    def deco(factory):
        if name in REGISTRY:
            raise ValueError(f"provider {name!r} already registered")
        REGISTRY[name] = factory
        return factory
    return deco


def make(name: str) -> MaskProvider:
    if name not in REGISTRY:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def postprocess(mask: Optional[np.ndarray], min_px: int = 20) -> Optional[np.ndarray]:
    """Shared clean-up for every non-control provider: fill holes, keep the largest component. None if empty."""
    if mask is None:
        return None
    m = np.asarray(mask, bool)
    if int(m.sum()) < min_px:
        return None
    from scipy.ndimage import binary_fill_holes
    m = binary_fill_holes(m)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1:
        return None
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    out = lab == keep
    return out if int(out.sum()) >= min_px else None


def pick_box(boxes, scores, shape, max_centre_dist: float = 0.75, area_range=(0.003, 0.35)):
    """Highest-scoring (x0, y0, x1, y1) box whose centre is within max_centre_dist of the half-diagonal from the
    image centre and whose area is within area_range of the image. Returns (box, score) or (None, 0.0)."""
    h, w = shape
    half = float(np.hypot(w / 2, h / 2))
    best, best_score = None, 0.0
    for b, s in zip(boxes, scores):
        x0, y0, x1, y1 = (float(v) for v in b)
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (h * w)
        if not area_range[0] <= area <= area_range[1]:
            continue
        if np.hypot((x0 + x1) / 2 - w / 2, (y0 + y1) / 2 - h / 2) > max_centre_dist * half:
            continue
        if float(s) > best_score:
            best, best_score = (x0, y0, x1, y1), float(s)
    return best, best_score


def iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Intersection over union; NaN if either mask is missing, 0 if both are empty."""
    if a is None or b is None:
        return float("nan")
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 0.0


def bbox(mask: np.ndarray):
    """(x0, y0, x1, y1) with exclusive upper edges, or None for an empty mask."""
    ys, xs = np.nonzero(mask)
    return None if xs.size == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def height_image(depth: np.ndarray, K: Intrinsics, plane, h_max: float = 0.35) -> np.ndarray:
    """Height above the tarp plane as a 3-channel uint8 image: 0 m = black, h_max = white, no data = black."""
    n, d = plane
    valid = depth > 0
    h = np.zeros(depth.shape, np.float64)
    h[valid] = volume.project(depth, valid, K).astype(np.float64) @ n + d
    g = np.clip(h / h_max, 0, 1) * 255
    return np.repeat(g.astype(np.uint8)[..., None], 3, axis=2)


def volume_from_result(req: MaskRequest, result: MaskResult) -> Optional[float]:
    """The only bridge to the mass code: plane-referenced volume (m^3) of a mask, or None on failure."""
    if result.mask is None or req.plane is None:
        return None
    if result.mask.shape != req.depth.shape:
        raise ValueError(f"mask {result.mask.shape} is not on the depth grid {req.depth.shape}")
    return float(volume.plane_volume(req.depth, result.mask, req.K, req.plane)[0])


@register("depth_plane")
class DepthPlaneProvider:
    """Control: the current depth-only mask (segment.plane_height_mask), unchanged and not post-processed."""
    name, needs_rgb = "depth_plane", False

    def __call__(self, req: MaskRequest) -> MaskResult:
        mask, plane = segment.plane_height_mask(req.depth, req.K)
        return MaskResult(mask, None, {"plane_found": plane is not None})
