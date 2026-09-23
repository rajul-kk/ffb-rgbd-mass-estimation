"""Temporal depth fusion over a bundle, cached by its parameters."""
from __future__ import annotations

import ast
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Intrinsics
from .io import Bundle, iter_frames

_STEADY_FIELDS = ("frames", "ref_frames", "tol_m", "max_changed")
_ROOT = Path(__file__).resolve().parents[1]


def _code_only(path: Path) -> str:
    """Module AST without docstrings, so comment and docstring edits do not invalidate caches."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)                 and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def code_stamp(reader: str) -> str:
    """Hash of the code that reads and fuses frames; a cache built by different code is rebuilt, not reused."""
    files = [_ROOT / "ffb" / "io.py", Path(__file__)] + ([_ROOT / "bag_reader.py"] if reader == "notebook" else [])
    return hashlib.sha1("".join(_code_only(f) for f in files if f.exists()).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class FusionConfig:
    max_frames: int = 120
    reader: str = "notebook"         # "fixed" copies frames (no 32-frame cap); "native" also leaves combined depth unaligned
    spatial: str = "blur_then_mask"  # "mask_then_nanmedian" keeps invalid pixels out of the 3x3 median
    frames: str = "all"              # "steady": only the opening run of frames that still match frames 0..ref_frames-1
    ref_frames: int = 8
    tol_m: float = 0.02
    max_changed: float = 0.08

    def key(self) -> str:
        d = asdict(self)
        if self.frames == "all":  # keeps cache names from before steady selection existed
            for k in _STEADY_FIELDS:
                d.pop(k)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


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


def frame_changes(depths, scale: float, ref_frames: int = 8, tol_m: float = 0.02) -> np.ndarray:
    """Per frame: share of reference-valid pixels that moved more than tol_m, plus share that lost depth."""
    ref = nanmedian_rows(np.stack([np.where(d > 0, d.astype(np.float32) * scale, np.nan) for d in depths[:ref_frames]]))
    rv = ref > 0
    r = ref[rv]
    out = np.empty(len(depths))
    for i, d in enumerate(depths):
        raw = d[rv]
        fin = raw > 0
        moved = np.abs(raw[fin].astype(np.float32) * scale - r[fin]) > tol_m
        out[i] = (moved.mean() if moved.size else 1.0) + (~fin).mean()
    return out


def steady_prefix(changed: np.ndarray, max_changed: float) -> int:
    """Length of the opening run of frames whose change stays at or below max_changed."""
    bad = np.flatnonzero(np.asarray(changed) > max_changed)
    return int(bad[0]) if bad.size else len(changed)


def fuse(bundle: Bundle, K: Intrinsics, cfg: FusionConfig, cache_dir="fused_cache", return_info: bool = False):
    """Return (fused depth m with 0 = no data, best rgb, frames used), plus {n_read, changed} if return_info."""
    path = Path(cache_dir) / f"{bundle.name}_{cfg.key()}.npz"
    stamp = code_stamp(cfg.reader)
    d = np.load(path) if path.exists() else None
    if d is not None and ("code" not in d or str(d["code"]) != stamp):
        warnings.warn(f"{path.name}: built by different reader/fusion code; rebuilding")
        d = None
    if d is not None:
        n = int(d["n_frames"])
        info = dict(n_read=int(d["n_read"]) if "n_read" in d else n,
                    changed=d["changed"] if "changed" in d else np.array([]))
        return (d["fused"], d["rgb"], n, info) if return_info else (d["fused"], d["rgb"], n)
    stream, changed, n_read = iter_frames(bundle, cfg.max_frames, cfg.reader), np.array([]), None
    if cfg.frames == "steady":
        if cfg.reader == "notebook":
            raise ValueError("steady frame selection needs a copying reader")
        stream = list(stream)
        n_read = len(stream)
        changed = frame_changes([d for _, d in stream], K.depth_scale, cfg.ref_frames, cfg.tol_m)
        stream = stream[:steady_prefix(changed, cfg.max_changed)]
    elif cfg.frames != "all":
        raise ValueError(f"unknown frames mode {cfg.frames!r}")
    frames, best_score, best_rgb = [], -1, None
    for rgb, depth in stream:
        dm = to_metric(depth, K.depth_scale, cfg.spatial)
        h, w = dm.shape
        score = int((dm[h // 4:3 * h // 4, w // 4:3 * w // 4] > 0).sum())
        if rgb is not None and score > best_score:
            best_score, best_rgb = score, rgb
        frames.append(dm)
    if not frames or best_rgb is None:
        raise RuntimeError(f"no usable frames from {bundle.name}")
    n_read = len(frames) if n_read is None else n_read
    fused = nanmedian_rows(np.stack(frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fused=fused, rgb=best_rgb, n_frames=len(frames), n_read=n_read, changed=changed, code=stamp)
    info = dict(n_read=n_read, changed=changed)
    return (fused, best_rgb, len(frames), info) if return_info else (fused, best_rgb, len(frames))
