"""Step 1 of the zero-shot benchmark: bags -> one small npz per bunch (rgb, fused depth, intrinsics, plane, control mask).

Uses the A_steady fusion (aligned depth, steady opening frames) for every provider and the control, so an RGB mask
lands on the depth grid. Split-bag bunches (session 2) are flagged rgb_registered=False: their colour is unusable.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import io, segment  # noqa: E402
from ffb.fusion import FusionConfig, fuse  # noqa: E402

FUSION = FusionConfig(reader="fixed", frames="steady")


def main(data=ROOT / "data", out=ROOT / "zeroshot_cache" / "inputs"):
    warnings.simplefilter("ignore")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for folder in sorted(p for p in Path(data).iterdir() if p.is_dir() and p.name.startswith("FFB")):
        b = io.find_bundle(str(folder))
        K = io.intrinsics(b, "auto")
        fused, rgb, n_used, info = fuse(b, K, FUSION, str(ROOT / "fused_cache"), return_info=True)
        mask, plane = segment.plane_height_mask(fused, K)
        ok = rgb is not None and rgb.shape[:2] == fused.shape and b.layout == "combined"
        np.savez_compressed(
            out / f"{b.name}.npz", fused=fused, rgb=rgb if rgb is not None else np.zeros((1, 1, 3), np.uint8),
            K=np.array([K.fx, K.fy, K.cx, K.cy, K.width, K.height, K.depth_scale]),
            plane_n=plane[0] if plane else np.full(3, np.nan), plane_d=plane[1] if plane else np.nan,
            ref_mask=mask if mask is not None else np.zeros(fused.shape, bool),
            rgb_registered=ok, layout=b.layout, frames_used=n_used)
        print(b.name, b.layout, "fused", fused.shape, "rgb", None if rgb is None else rgb.shape, "registered", ok,
              "plane", plane is not None, "mask_px", 0 if mask is None else int(mask.sum()), flush=True)


if __name__ == "__main__":
    main()
