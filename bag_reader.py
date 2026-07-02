"""
bag_reader.py
RealSense .bag file reader — extracts aligned RGB + depth frame pairs.

Supports two dataset layouts:
  - Combined bag (FFB10-19): one bag contains both depth + colour streams.
  - Split bags   (FFB31-35): separate DEPTH.bag (depth-only) and RGB.bag
                              (colour-only), paired by frame index.

Requires: pyrealsense2, numpy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bag_has_stream(bag_path: str, stream_type) -> bool:
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    has = any(s.stream_type() == stream_type for s in profile.get_streams())
    pipeline.stop()
    return has


def _iter_single_bag(
    bag_path: str,
    max_frames: Optional[int],
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Iterate a combined bag that contains both colour and depth streams."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    align = rs.align(rs.stream.color)
    count = 0
    try:
        while True:
            if max_frames is not None and count >= max_frames:
                break
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            rgb = np.asanyarray(color_frame.get_data())[..., ::-1].copy()
            depth = np.asanyarray(depth_frame.get_data())
            yield rgb, depth
            count += 1
    finally:
        pipeline.stop()


def _extract_depth_frames(
    depth_bag: str,
    max_frames: Optional[int],
) -> list[np.ndarray]:
    """Extract all depth frames from a depth-only bag."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(depth_bag), repeat_playback=False)
    config.enable_stream(rs.stream.depth)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    frames = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break
            d = frameset.get_depth_frame()
            if not d:
                continue
            frames.append(np.asanyarray(d.get_data()))
    finally:
        pipeline.stop()
    return frames


def _extract_rgb_frames(
    rgb_bag: str,
    max_frames: Optional[int],
) -> list[np.ndarray]:
    """Extract all colour frames from a colour-only bag."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(rgb_bag), repeat_playback=False)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    frames = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break
            c = frameset.get_color_frame()
            if not c:
                continue
            rgb = np.asanyarray(c.get_data())
            # handle both bgr8 and rgb8 formats
            if c.get_profile().format().name == "bgr8":
                rgb = rgb[..., ::-1].copy()
            frames.append(rgb)
    finally:
        pipeline.stop()
    return frames


# ---------------------------------------------------------------------------
# Bundle layout detection
# ---------------------------------------------------------------------------

def find_bundle_bags(ffb_dir: str) -> tuple[str, Optional[str]]:
    """
    Given a bundle folder, return (depth_bag_path, rgb_bag_path_or_None).

    If the depth bag also contains colour, rgb_bag_path is None.
    """
    import pyrealsense2 as rs

    files = os.listdir(ffb_dir)
    bags = [os.path.join(ffb_dir, f) for f in files if f.endswith(".bag")]

    depth_bag = rgb_bag = None
    for b in bags:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, b, repeat_playback=False)
        profile = pipeline.start(config)
        streams = {s.stream_type() for s in profile.get_streams()}
        pipeline.stop()
        has_depth = rs.stream.depth in streams
        has_color = rs.stream.color in streams
        if has_depth and has_color:
            return b, None          # combined bag
        if has_depth:
            depth_bag = b
        if has_color:
            rgb_bag = b

    if depth_bag is None:
        raise FileNotFoundError(f"No depth bag found in {ffb_dir}")
    return depth_bag, rgb_bag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iter_frames(
    bag_path: str,
    max_frames: Optional[int] = None,
    rgb_bag_path: Optional[str] = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Iterate over aligned RGB+depth frame pairs.

    For combined bags (FFB10-19):  pass only bag_path.
    For split bags   (FFB31-35):   pass depth bag as bag_path, colour as rgb_bag_path.

    Yields:
        (rgb uint8 H×W×3 RGB, depth uint16 H×W)
    """
    if rgb_bag_path is None:
        yield from _iter_single_bag(bag_path, max_frames)
        return

    # Split-bag path: extract both independently then zip by frame index
    depth_frames = _extract_depth_frames(bag_path, max_frames)
    rgb_frames = _extract_rgb_frames(rgb_bag_path, max_frames)
    n = min(len(depth_frames), len(rgb_frames))
    for i in range(n):
        yield rgb_frames[i], depth_frames[i]


def iter_bundle(
    ffb_dir: str,
    max_frames: Optional[int] = None,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Auto-detect layout and iterate frames from an FFB bundle directory.

    Handles both combined and split-bag layouts automatically.

    Yields:
        (rgb uint8 H×W×3 RGB, depth uint16 H×W)
    """
    depth_bag, rgb_bag = find_bundle_bags(ffb_dir)
    yield from iter_frames(depth_bag, max_frames=max_frames, rgb_bag_path=rgb_bag)


def load_best_frame(ffb_dir: str, max_frames: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the single best (highest valid depth pixel count) frame from a bundle.

    Returns:
        (rgb uint8 H×W×3, depth uint16 H×W)
    """
    best_rgb = best_depth = None
    best_count = 0
    for rgb, depth in iter_bundle(ffb_dir, max_frames=max_frames):
        h, w = depth.shape
        crop = depth[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        count = int((crop > 0).sum())
        if count > best_count:
            best_count = count
            best_rgb = rgb.copy()
            best_depth = depth.copy()
    if best_rgb is None:
        raise RuntimeError(f"No valid frames found in {ffb_dir}")
    return best_rgb, best_depth


def load_frame_from_bag(
    bag_path: str,
    frame_index: int = 0,
    rgb_bag_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a single frame by index. See iter_frames for layout details."""
    for i, (rgb, depth) in enumerate(iter_frames(bag_path, rgb_bag_path=rgb_bag_path)):
        if i == frame_index:
            return rgb, depth
    raise RuntimeError(f"Frame {frame_index} not found in {bag_path}")


def get_depth_scale(bag_path: str) -> float:
    """Return the sensor depth scale (raw units -> metres) from a .bag file."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.depth)
    profile = pipeline.start(config)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    pipeline.stop()
    return float(scale)


def get_intrinsics_from_bag(bag_path: str):
    """
    Extract camera intrinsics from a bag file.

    Tries colour stream first (higher resolution, preferred for RGB-based
    models). Falls back to depth stream intrinsics when the bag contains
    depth only (e.g. FFB31-35 DEPTH.bag).

    Returns a CameraIntrinsics instance whose width/height match the stream
    that was read, or None on failure.
    """
    import pyrealsense2 as rs
    from perception_pipeline import CameraIntrinsics

    def _extract(stream_type):
        pl = rs.pipeline()
        cfg = rs.config()
        rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
        cfg.enable_stream(stream_type)
        try:
            prof = pl.start(cfg)
        except Exception:
            return None
        try:
            for s in prof.get_streams():
                if s.stream_type() == stream_type:
                    intr = s.as_video_stream_profile().get_intrinsics()
                    return CameraIntrinsics(
                        fx=intr.fx, fy=intr.fy,
                        cx=intr.ppx, cy=intr.ppy,
                        width=intr.width, height=intr.height,
                    )
        finally:
            pl.stop()
        return None

    return _extract(rs.stream.color) or _extract(rs.stream.depth)
