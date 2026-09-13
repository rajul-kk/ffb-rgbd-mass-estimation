"""Ray-cast depth of an ellipsoid on a flat tarp, for geometry checks with exact answers."""
from __future__ import annotations

import numpy as np

from .camera import Intrinsics


def render(K: Intrinsics, D: float, a: float, b: float, c: float, resting: bool, s: int = 1, crop: bool = False):
    """Depth (m), object mask and intrinsics for a camera looking down +z at a tarp at z=D.
    resting=True puts a full ellipsoid on the tarp; False renders a half-ellipsoid dome."""
    fx, fy = K.fx * s, K.fy * s
    cx, cy = (K.cx + 0.5) * s - 0.5, (K.cy + 0.5) * s - 0.5
    W, H = K.width * s, K.height * s
    u0, u1, v0, v1 = 0, W, 0, H
    if crop:
        top = D - (2 * c if resting else c)
        hu, hv = 1.2 * fx * a / top + 2, 1.2 * fy * b / top + 2
        u0, u1 = max(0, int(cx - hu)), min(W, int(cx + hu) + 1)
        v0, v1 = max(0, int(cy - hv)), min(H, int(cy + hv) + 1)
    u, v = np.meshgrid(np.arange(u0, u1, dtype=float), np.arange(v0, v1, dtype=float))
    rx, ry = (u - cx) / fx, (v - cy) / fy
    cz = D - c if resting else D
    A = (rx / a) ** 2 + (ry / b) ** 2 + 1 / c ** 2
    B = -2 * cz / c ** 2
    C = (cz / c) ** 2 - 1
    disc = B * B - 4 * A * C
    t = (-B - np.sqrt(np.clip(disc, 0, None))) / (2 * A)
    hit = (disc >= 0) & (t < D)
    return np.where(hit, t, D), hit, Intrinsics(fx, fy, cx - u0, cy - v0, u1 - u0, v1 - v0)


def ellipsoid_volume(a: float, b: float, c: float, resting: bool) -> float:
    return (4 / 3 if resting else 2 / 3) * np.pi * a * b * c
