"""RealSense bag reading for combined and split FFB bundles."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .camera import Intrinsics


def _rs():
    import pyrealsense2 as rs
    return rs


@dataclass
class Bundle:
    name: str
    folder: str
    depth_bag: str
    rgb_bag: Optional[str]  # None when the depth bag also carries colour

    @property
    def layout(self) -> str:
        return "combined" if self.rgb_bag is None else "split"


def _start(path: str, streams=()):
    rs = _rs()
    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, path, repeat_playback=False)
    for s in streams:
        cfg.enable_stream(getattr(rs.stream, s))
    profile = pipe.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    return pipe, profile


def find_bundle(folder: str) -> Bundle:
    name, depth_bag, rgb_bag = os.path.basename(os.path.normpath(str(folder))), None, None
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".bag"):
            continue
        path = os.path.join(folder, f)
        pipe, profile = _start(path)
        types = {s.stream_type().name for s in profile.get_streams()}
        pipe.stop()
        if {"depth", "color"} <= types:
            return Bundle(name, str(folder), path, None)
        if "depth" in types:
            depth_bag = path
        elif "color" in types:
            rgb_bag = path
    if depth_bag is None:
        raise FileNotFoundError(f"no depth bag in {folder}")
    return Bundle(name, str(folder), depth_bag, rgb_bag)


def intrinsics(bundle: Bundle, stream: str = "auto") -> Intrinsics:
    """Intrinsics of the depth grid: "auto" uses colour when the bag has it (aligned depth), "depth" never does."""
    pipe, profile = _start(bundle.depth_bag)
    try:
        by_type = {s.stream_type().name: s for s in profile.get_streams()}
        s = by_type["color"] if stream == "auto" and "color" in by_type else by_type["depth"]
        i = s.as_video_stream_profile().get_intrinsics()
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
    finally:
        pipe.stop()
    return Intrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height, float(scale))


def color_intrinsics(bundle: Bundle) -> Intrinsics:
    """Intrinsics of the colour stream (read from the RGB bag for split bundles)."""
    pipe, profile = _start(bundle.rgb_bag or bundle.depth_bag)
    try:
        s = next(s for s in profile.get_streams() if s.stream_type().name == "color")
        i = s.as_video_stream_profile().get_intrinsics()
    finally:
        pipe.stop()
    return Intrinsics(i.fx, i.fy, i.ppx, i.ppy, i.width, i.height)


def _rgb(frame) -> np.ndarray:
    img = np.asanyarray(frame.get_data())
    return (img[..., ::-1] if frame.get_profile().format().name == "bgr8" else img).copy()


def _stream(path: str, stream: str, max_frames: Optional[int]) -> Iterator[np.ndarray]:
    pipe, _ = _start(path, [stream])
    n = 0
    try:
        while max_frames is None or n < max_frames:
            try:
                fs = pipe.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                return
            f = fs.get_depth_frame() if stream == "depth" else fs.get_color_frame()
            if f:
                n += 1
                yield np.asanyarray(f.get_data()).copy() if stream == "depth" else _rgb(f)
    finally:
        pipe.stop()


def _combined(path: str, max_frames: Optional[int], align: bool):
    rs = _rs()
    pipe, _ = _start(path, ["color", "depth"])
    aligner, n = (rs.align(rs.stream.color) if align else None), 0
    try:
        while max_frames is None or n < max_frames:
            try:
                fs = pipe.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                return
            if aligner is not None:
                fs = aligner.process(fs)
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if c and d:
                n += 1
                yield _rgb(c), np.asanyarray(d.get_data()).copy()
    finally:
        pipe.stop()


def iter_frames(bundle: Bundle, max_frames: Optional[int] = None, reader: str = "fixed"):
    """Yield (rgb or None, depth uint16). "notebook" reuses bag_reader; "native" leaves combined-bag depth unaligned."""
    if reader == "notebook":
        import bag_reader
        yield from bag_reader.iter_bundle(bundle.folder, max_frames=max_frames)
        return
    if reader not in ("fixed", "native"):
        raise ValueError(f"unknown reader {reader!r}")
    if bundle.rgb_bag is None:
        yield from _combined(bundle.depth_bag, max_frames, align=reader == "fixed")
        return
    depth = list(_stream(bundle.depth_bag, "depth", max_frames))
    rgb = list(_stream(bundle.rgb_bag, "color", max_frames))
    for i, d in enumerate(depth):
        yield (rgb[i] if i < len(rgb) else None), d
