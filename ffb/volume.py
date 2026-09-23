"""Back-projection, the notebook's 2.5D grid volume, and grid-free frustum volume above a tarp plane."""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Intrinsics


def project(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Masked valid pixels to (N, 3) float32 camera-frame metres."""
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    valid = mask & (depth_m > 0)
    Z = depth_m[valid].astype(np.float32)
    return np.stack([(us[valid] - K.cx) * Z / K.fx, (vs[valid] - K.cy) * Z / K.fy, Z], axis=-1)


def clip_depth(depth_m: np.ndarray, z_front: float, z_window: float) -> np.ndarray:
    d = depth_m.copy()
    d[(d > 0) & (d > z_front + z_window)] = 0
    return d


def pixel_pitch(K: Intrinsics, depth_m: float) -> float:
    """Projected xy size of one pixel at depth_m; grid cells finer than this are aliased empty."""
    return float(depth_m) / K.fx


def ring_z_ref(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics, percentile: float = 50) -> float | None:
    """Ground depth from the tarp ring around the mask, not the bunch itself; None if too few ring points."""
    ring_pts = tarp_points(depth_m, mask, K)
    if ring_pts.shape[0] < 20:
        return None
    return float(np.percentile(ring_pts[:, 2], percentile))


def grid_volume(points: np.ndarray, grid_step: float = 0.002, z_ref: float | None = None,
                 fill_radius: int = 0) -> tuple[float, float]:
    """Notebook integral: sum over 2 mm xy cells of (z_ref - min z). z_ref defaults to the 95th-percentile depth
    of `points` (bunch-only, so it can sit above the true tarp); pass a ring-derived z_ref to fix that.
    fill_radius > 0 fills empty cells from their nearest filled neighbour (CloudCompare's own 2.5D volume
    approach), capped at that many cells away so it patches aliasing gaps, not real occlusion."""
    if points.shape[0] < 20:
        return 0.0, 0.0
    x, y, z = points.T.astype(np.float64)
    if z_ref is None:
        z_ref = float(np.percentile(z, 95))
    xi = np.floor((x - x.min()) / grid_step).astype(np.int32)
    yi = np.floor((y - y.min()) / grid_step).astype(np.int32)
    nx, ny = int(xi.max()) + 1, int(yi.max()) + 1
    z_top = np.full(nx * ny, z_ref, dtype=np.float64)
    np.minimum.at(z_top, xi * ny + yi, z)
    has = z_top < z_ref - 1e-4
    if fill_radius > 0 and not has.all():
        from scipy.ndimage import distance_transform_edt
        grid, has2d = z_top.reshape(nx, ny), has.reshape(nx, ny)
        dist, idx = distance_transform_edt(~has2d, return_indices=True)
        filled = grid[tuple(idx)]
        grid = np.where((~has2d) & (dist <= fill_radius), filled, grid)
        z_top, has = grid.ravel(), (grid.ravel() < z_ref - 1e-4)
    return float(np.sum(z_ref - z_top[has]) * grid_step ** 2), z_ref


def frustum_volume(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics, z_back) -> float:
    """Sum of each masked pixel's pyramid volume between surface depth and z_back: (zb^3 - zs^3) / (3 fx fy)."""
    valid = mask & (depth_m > 0)
    zs = depth_m[valid].astype(np.float64)
    zb = np.broadcast_to(np.asarray(z_back, np.float64), depth_m.shape)[valid]
    dv = np.nan_to_num(zb ** 3 - zs ** 3, nan=0.0)
    return float(np.sum(np.clip(dv, 0, None)) / (3 * K.fx * K.fy))


def plane_depth(plane, K: Intrinsics, shape) -> np.ndarray:
    """Depth at which each pixel's ray meets the plane n.p + d = 0 (NaN where it never does)."""
    n, d = plane
    h, w = shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -d / (n[0] * (u - K.cx) / K.fx + n[1] * (v - K.cy) / K.fy + n[2])
    z[~(z > 0)] = np.nan
    return z


def fit_plane(points: np.ndarray, thresh: float = 0.01, iters: int = 500, seed: int = 0, robust: bool = False):
    """RANSAC plane refit by SVD on inliers; normal faces the camera. Returns (n, d, inlier rmse m).
    robust=True scores hypotheses by MSAC (truncated squared residuals) and adds LO-RANSAC refits."""
    rng = np.random.default_rng(seed)
    P = points.astype(np.float64)
    if len(P) < 100:
        raise ValueError(f"only {len(P)} points for plane fit")
    if len(P) > 30000:
        P = P[rng.choice(len(P), 30000, replace=False)]
    best, best_cost = None, np.inf
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        r = np.abs((P - a) @ (n / norm))
        cost = float((np.minimum(r, thresh) ** 2).sum()) if robust else -float((r < thresh).sum())
        if cost < best_cost:
            best, best_cost = r < thresh, cost
    if best is None:
        raise ValueError("plane fit: every sampled triplet was collinear")
    Q = P[best]
    centroid = Q.mean(0)
    n = np.linalg.svd(Q - centroid, full_matrices=False)[2][-1]
    for _ in range(3 if robust else 0):
        Q = P[np.abs((P - centroid) @ n) < thresh]
        centroid = Q.mean(0)
        n = np.linalg.svd(Q - centroid, full_matrices=False)[2][-1]
    if n[2] > 0:
        n = -n
    d = -float(n @ centroid)
    return n, d, float(np.sqrt(np.mean((Q @ n + d) ** 2)))


def ring_mask(mask: np.ndarray, grow: float = 0.5, gap_px: int = 15) -> np.ndarray:
    """Pixels inside the mask's enlarged bounding box but outside the dilated mask."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    bh, bw = ys.max() - ys.min(), xs.max() - xs.min()
    ring = np.zeros_like(mask, dtype=bool)
    ring[max(0, int(ys.min() - grow * bh)):min(h, int(ys.max() + grow * bh) + 1),
         max(0, int(xs.min() - grow * bw)):min(w, int(xs.max() + grow * bw) + 1)] = True
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    return ring & ~cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def tarp_points(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics, grow: float = 0.5, gap_px: int = 15):
    """Valid points in the tarp ring around the mask."""
    return project(depth_m, ring_mask(mask, grow, gap_px), K)


def plane_volume(depth_m: np.ndarray, mask: np.ndarray, K: Intrinsics, plane):
    """Frustum volume above the plane, 98th-percentile height (m) and footprint area (m^2) of parts >2 cm tall."""
    n, d = plane
    zp = plane_depth(plane, K, depth_m.shape)
    valid = mask & (depth_m > 0)
    h = project(depth_m, mask, K).astype(np.float64) @ n + d
    tall = np.zeros_like(valid)
    tall[valid] = h > 0.02
    height = float(np.percentile(h, 98)) if h.size else 0.0
    return frustum_volume(depth_m, mask, K, zp), height, float(np.nansum(zp[tall] ** 2) / (K.fx * K.fy))
