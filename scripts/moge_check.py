"""MoGe-2 monocular geometry as an independent check on the RealSense tarp-plane volumes (CPU)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import evaluate as ev  # noqa: E402
from ffb import experiment as ex  # noqa: E402
from ffb import io, segment, volume  # noqa: E402
from ffb.fusion import FusionConfig, fuse  # noqa: E402

CFG = FusionConfig(reader="native", spatial="mask_then_nanmedian", frames="steady")
MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def ring_pixels(mask, grow=0.5, gap_px=30):
    """Pixels in the mask's enlarged bounding box but outside the dilated mask."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    bh, bw = ys.max() - ys.min(), xs.max() - xs.min()
    ring = np.zeros_like(mask)
    ring[max(0, int(ys.min() - grow * bh)):min(h, int(ys.max() + grow * bh) + 1),
         max(0, int(xs.min() - grow * bw)):min(w, int(xs.max() + grow * bw) + 1)] = True
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    return ring & ~cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def plane_measure(depth, mask, K):
    n, d, rmse = volume.fit_plane(volume.tarp_points(depth, mask, K))
    v, height, _ = volume.plane_volume(depth, mask, K, (n, d))
    return v, height, 1000 * rmse


def main():
    from moge.model.v2 import MoGeModel

    torch.set_num_threads(os.cpu_count())
    model = MoGeModel.from_pretrained(MODEL_ID).eval()
    gt = ex.ground_truth(ROOT)
    rs = pd.read_csv(ROOT / "results" / "v3" / "volumes.csv")
    rs = rs[rs["fusion"] == "steady_native"].set_index("ffb")
    rows = []
    for folder in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir() and p.name.startswith("FFB")):
        b = io.find_bundle(str(folder))
        Kd, Kc = io.intrinsics(b, "depth"), io.color_intrinsics(b)
        fused, rgb, _ = fuse(b, Kd, CFG, ROOT / "fused_cache")
        mask_d = segment.extract_mask(fused, rgb, Kd.width)[0]
        h, w = rgb.shape[:2]
        # Same naive colour registration as the depth pipeline, widened to cover parallax; tarp pixels add ~0 volume.
        mask = cv2.resize(mask_d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = cv2.dilate(mask, np.ones((25, 25), np.uint8)).astype(bool)
        rs_depth = cv2.resize(fused, (w, h), interpolation=cv2.INTER_NEAREST)

        t0 = time.time()
        with torch.inference_mode():
            out = model.infer(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                              fov_x=float(np.degrees(2 * np.arctan(w / (2 * Kc.fx)))), use_fp16=False)
        depth = out["depth"].cpu().numpy().astype(np.float32)
        depth[~out["mask"].cpu().numpy().astype(bool) | ~np.isfinite(depth)] = 0
        seconds = time.time() - t0

        tarp = ring_pixels(mask) & (depth > 0) & (rs_depth > 0)
        scale = float(np.median(rs_depth[tarp] / depth[tarp]))
        v_raw, h_raw, rmse_raw = plane_measure(depth, mask, Kc)
        v_scaled, h_scaled, rmse_scaled = plane_measure(depth * scale, mask, Kc)
        rows.append(dict(ffb=b.name, layout=b.layout, seconds=round(seconds, 1), rs_over_moge_tarp_depth=scale,
                         V_moge=v_raw, V_moge_scaled=v_scaled, V_rs=rs.loc[b.name, "V_plane"],
                         H_moge_scaled=h_scaled, H_rs=rs.loc[b.name, "H"], plane_rmse_moge_mm=rmse_scaled,
                         true_L=gt.loc[b.name, "Actual_Volume_L"], T_cal_cm=gt.loc[b.name, "Thickness_cm"]))
        r = rows[-1]
        print(f"{b.name} {b.layout:8s} {seconds:5.1f}s scale={scale:.3f} moge={1000 * v_scaled:6.2f}L rs={1000 * r['V_rs']:6.2f}L "
              f"true={r['true_L']:.0f}L H={100 * h_scaled:.1f}/{100 * r['H_rs']:.1f}cm T={r['T_cal_cm']}", flush=True)

    t = pd.DataFrame(rows).set_index("ffb")
    out_dir = ROOT / "results" / "v3"
    t.to_csv(out_dir / "moge_check.csv")
    s = t.drop(index=ex.EXCLUDED)
    for col in ("V_moge", "V_moge_scaled", "V_rs"):
        s[f"{col}/true"] = 1000 * s[col] / s["true_L"]
    pd.set_option("display.width", 220)
    print(s.groupby("layout")[["rs_over_moge_tarp_depth", "V_moge/true", "V_moge_scaled/true", "V_rs/true", "plane_rmse_moge_mm"]]
          .agg(["mean", "std"]).round(3).to_string())
    print("corr(H_moge_scaled, caliper T):", round(np.corrcoef(s["H_moge_scaled"], s["T_cal_cm"])[0, 1], 3),
          "| corr(V_moge_scaled, V_rs):", round(np.corrcoef(s["V_moge_scaled"], s["V_rs"])[0, 1], 3))
    df = pd.DataFrame({"group": s["layout"], "mass": gt.loc[s.index, "Actual_Mass_kg"]})
    for col in ("V_moge_scaled", "V_rs"):
        for model in ex.BASE_MODELS[:3]:
            m = ev.metrics(df["mass"], ev.loo(df.assign(V=s[col]), model))
            print(f"LOO {col:14s} {model.name:14s} MAE={m['mae']:.2f} kg  R2={m['r2']:.2f}")


if __name__ == "__main__":
    main()
