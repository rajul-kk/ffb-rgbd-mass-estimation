"""Depth-first FFB segmentation, ported unchanged from the v2 notebook."""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


def auto_margin(depth_m, z_front, valid, search_lo=0.05, search_hi=0.45, n_bins=80,
                fallback_m=0.10, clip_lo=0.08, clip_hi=0.22):
    band = depth_m[valid & (depth_m > z_front + search_lo) & (depth_m < z_front + search_hi)]
    if band.size < 200:
        return fallback_m
    hist, edges = np.histogram(band, bins=n_bins)
    z_tarp = float(edges[int(np.argmax(hist))])
    return float(np.clip(z_tarp - z_front - 0.05, clip_lo, clip_hi))


def _at_depth_res(rgb, h, w):
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR) if rgb.shape[:2] != (h, w) else rgb


def depth_foreground_mask(depth_m, rgb=None, min_area_frac=0.003, max_area_frac=0.35,
                          colour_s_min=40, colour_v_max=160):
    H, W = depth_m.shape
    valid = (depth_m > 0.1) & (depth_m < 10.0)
    if not valid.any():
        return None, None, None
    z_front = float(np.percentile(depth_m[valid], 5))
    margin_m = auto_margin(depth_m, z_front, valid)
    fg = (valid & (depth_m <= z_front + margin_m)).astype(np.uint8)
    if rgb is not None:
        hsv = cv2.cvtColor(_at_depth_res(rgb, H, W), cv2.COLOR_RGB2HSV)
        fg = (fg.astype(bool) & (hsv[..., 1] >= colour_s_min) & (hsv[..., 2] <= colour_v_max)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    img_area = H * W
    cx, cy = W * 0.5, H * 0.5
    half_diag = (cx ** 2 + cy ** 2) ** 0.5

    def dist(c):
        m = cv2.moments(c)
        if m["m00"] == 0:
            return float("inf")
        return ((m["m10"] / m["m00"] - cx) ** 2 + (m["m01"] / m["m00"] - cy) ** 2) ** 0.5

    sized = [c for c in contours if min_area_frac * img_area < cv2.contourArea(c) < max_area_frac * img_area]
    if not sized:
        return None, None, None
    c = min(sized, key=dist)
    if dist(c) > 0.75 * half_diag:
        c = max(sized, key=cv2.contourArea)
        if dist(c) > 0.75 * half_diag:
            return None, None, None
    mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
    if mask.mean() > max_area_frac:
        return None, None, None
    x, y, w, h = cv2.boundingRect(c)
    return mask.astype(bool), [float(x), float(y), float(x + w), float(y + h)], margin_m


def expand_mask_bbox(fg_mask, depth_m, rgb_image, z_front, z_window=0.25, s_min=25, v_max=160, pad_px=20):
    dh, dw = depth_m.shape
    if not fg_mask.any():
        return fg_mask
    ys, xs = np.where(fg_mask)
    y1, y2 = max(0, ys.min() - pad_px), min(dh, ys.max() + pad_px)
    x1, x2 = max(0, xs.min() - pad_px), min(dw, xs.max() + pad_px)
    search = np.zeros((dh, dw), bool)
    search[y1:y2, x1:x2] = True
    depth_ok = (depth_m > 0) & (depth_m <= z_front + z_window)
    if rgb_image is not None:
        hsv = cv2.cvtColor(_at_depth_res(rgb_image, dh, dw), cv2.COLOR_RGB2HSV)
        colour_ok = (hsv[..., 1] >= s_min) & (hsv[..., 2] <= v_max)
    else:
        colour_ok = np.ones((dh, dw), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded = cv2.morphologyEx((search & depth_ok & colour_ok).astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    return expanded if expanded.any() else fg_mask


def border_hsv_stats(rgb, border_px=25):
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    border = np.concatenate([
        hsv[:border_px, :].reshape(-1, 3), hsv[h - border_px:, :].reshape(-1, 3),
        hsv[:, :border_px].reshape(-1, 3), hsv[:, w - border_px:].reshape(-1, 3),
    ])
    s_min = float(np.clip(np.percentile(border[:, 1], 85), 15, 70))
    v_max = float(np.clip(np.percentile(border[:, 2], 95) + 40, 120, 220))
    return s_min, v_max


def extract_mask(fused, rgb, width: int, fallback: Optional[Callable] = None):
    """Approach A mask rules; returns (mask or None, z_front, z_window, source)."""
    s_min, v_max = border_hsv_stats(rgb)
    fg, _, margin = depth_foreground_mask(fused, rgb=rgb, colour_s_min=s_min, colour_v_max=v_max)
    valid_depth = fused[fused > 0.1]
    if valid_depth.size == 0:
        return None, None, None, "failed"
    z_front = float(np.percentile(valid_depth, 5))
    if fg is not None:
        z_window = float(margin) + 0.05
        if width != 848:
            return fg, z_front, z_window, "depth"
        ys, xs = np.where(fg)
        diag = float(((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2) ** 0.5)
        pad = int(np.clip(0.08 * diag, 8, 40))
        return expand_mask_bbox(fg, fused, rgb, z_front, z_window=z_window, pad_px=pad), z_front, z_window, "depth"
    mask = fallback(rgb) if fallback is not None else None
    if mask is None:
        return None, z_front, 0.25, "failed"
    dh, dw = fused.shape
    if mask.shape != (dh, dw):
        mask = cv2.resize(mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, z_front, 0.25, "semantic"


def plane_height_mask(fused, K, h_min=0.03, band=0.03, min_area_frac=0.003, max_area_frac=0.35, robust=False):
    """Depth-only mask: pixels more than h_min above a RANSAC plane fit to the dominant far surface (the tarp).
    Needs no colour, so it works when RGB is unregistered or recorded at a different time. Returns (mask, plane)."""
    from . import volume
    valid = (fused > 0.1) & (fused < 10.0)
    if not valid.any():
        return None, None
    H, W = fused.shape
    centre = np.zeros_like(valid)
    centre[H // 4:3 * H // 4, W // 4:3 * W // 4] = True
    z = fused[valid & centre]  # the tarp dominates the central half of the frame in this protocol
    if z.size < 200:
        return None, None
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.02, 0.01))
    z_tarp = float(edges[np.argmax(hist)] + 0.005)
    n, d, _ = volume.fit_plane(volume.project(fused, valid & (np.abs(fused - z_tarp) < band), K), robust=robust)
    h = np.zeros_like(fused, dtype=np.float64)
    pts = volume.project(fused, valid, K).astype(np.float64)
    h[valid] = pts @ n + d
    fg = (valid & (h > h_min)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n_lab, lab, st, cen = cv2.connectedComponentsWithStats(fg)
    ok = [i for i in range(1, n_lab) if min_area_frac * H * W < st[i, 4] < max_area_frac * H * W]
    if not ok:
        return None, (n, d)
    best = min(ok, key=lambda i: np.hypot(cen[i][0] - W / 2, cen[i][1] - H / 2))
    if np.hypot(cen[best][0] - W / 2, cen[best][1] - H / 2) > 0.75 * np.hypot(W / 2, H / 2):
        return None, (n, d)
    mask = lab == best
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool), (n, d)
