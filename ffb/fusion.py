"""Temporal depth fusion over a bundle, cached by its parameters."""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics
from .io import Bundle, iter_frames


@dataclass(frozen=True)
class FusionConfig:
    max_frames: int = 120
    reader: str = "notebook"         # "fixed" copies frames, avoiding bag_reader's 32-frame cap on split bags
    spatial: str = "blur_then_mask"  # notebook order; "mask_then_nanmedian" keeps invalid pixels out of the median

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def nan_median3(a: np.ndarray) -> np.ndarray:
    """3x3 median over finite neighbours; windows with none stay NaN."""
    w = np.lib.stride_tricks.sliding_window_view(np.pad(a, 1, constant_values=np.nan), (3, 3))
    w = w.reshape(*a.shape, 9)
    s = np.sort(w, axis=-1)
    k = np.isfinite(w).sum(-1)
    lo = np.take_along_axis(s, np.maximum((k - 1) // 2, 0)[..., None], -1)[..., 0]
    hi = np.take_along_axis(s, (k // 2)[..., None], -1)[..., 0]
    out = (lo + hi) / 2
    out[k == 0] = np.nan
    return out.astype(a.dtype)


def to_metric(depth: np.ndarray, scale: float, spatial: str) -> np.ndarray:
    """Raw uint16 depth to float32 metres with NaN for invalid pixels."""
    dm = depth.astype(np.float32) * scale
    if spatial == "blur_then_mask":
        dm = cv2.medianBlur(dm, 3)
        dm[dm == 0] = np.nan
    elif spatial in ("mask_then_nanmedian", "none"):
        dm[depth == 0] = np.nan
        if spatial == "mask_then_nanmedian":
            dm = nan_median3(dm)
    else:
        raise ValueError(f"unknown spatial mode {spatial!r}")
    return dm


def nanmedian_rows(stack: np.ndarray, rows: int = 64) -> np.ndarray:
    """Per-pixel nanmedian over time in row blocks (same result, lower peak memory); no data -> 0."""
    out = np.empty(stack.shape[1:], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in range(0, stack.shape[1], rows):
            out[r:r + rows] = np.nanmedian(stack[:, r:r + rows], axis=0)
    return np.nan_to_num(out, nan=0.0)


def fuse(bundle: Bundle, K: Intrinsics, cfg: FusionConfig, cache_dir="fused_cache"):
    """Return (fused depth in metres with 0 = no data, best rgb, frames used)."""
    path = Path(cache_dir) / f"{bundle.name}_{cfg.key()}.npz"
    if path.exists():
        d = np.load(path)
        return d["fused"], d["rgb"], int(d["n_frames"])
    frames, best_score, best_rgb = [], -1, None
    for rgb, depth in iter_frames(bundle, cfg.max_frames, cfg.reader):
        dm = to_metric(depth, K.depth_scale, cfg.spatial)
        h, w = dm.shape
        score = int((dm[h // 4:3 * h // 4, w // 4:3 * w // 4] > 0).sum())
        if rgb is not None and score > best_score:
            best_score, best_rgb = score, rgb
        frames.append(dm)
    if not frames:
        raise RuntimeError(f"no frames read from {bundle.name}")
    fused = nanmedian_rows(np.stack(frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fused=fused, rgb=best_rgb, n_frames=len(frames))
    return fused, best_rgb, len(frames)
