"""Pinhole camera intrinsics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_scale: float = 0.001
