"""Held-out check of three rejected grid-volume fixes: pixel-pitch cells, tarp-ring z_ref, empty-cell interpolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import experiment as ex  # noqa: E402
from ffb import volume  # noqa: E402
from ffb.fusion import FusionConfig  # noqa: E402

FUSIONS = {"notebook": FusionConfig(reader="notebook", spatial="blur_then_mask"),
           "nan_blur": FusionConfig(reader="fixed", spatial="mask_then_nanmedian")}
STEPS = [("S0_notebook", "notebook", "V_grid"), ("S2_nan_blur", "nan_blur", "V_grid"),
         ("S2b_pitch_grid", "nan_blur", "V_grid_pitch"), ("S2c_ring_zref", "nan_blur", "V_grid_ring"),
         ("S2d_ring_zref_pitch", "nan_blur", "V_grid_ring_pitch"), ("S2e_fill_radius", "nan_blur", "V_grid_fill")]


def grid_variants(fused, mask, K, z_front, z_window):
    """The rejected grid volumes, computed on v2's clipped depth and mask."""
    pts = volume.project(volume.clip_depth(fused, z_front, z_window), mask, K)
    pitch = max(0.002, volume.pixel_pitch(K, z_front + z_window))
    ring = volume.ring_z_ref(fused, mask, K)
    return dict(V_grid_pitch=volume.grid_volume(pts, grid_step=pitch)[0], grid_pitch_mm=1000 * pitch, ring_z_ref=ring,
                V_grid_ring=volume.grid_volume(pts, z_ref=ring)[0] if ring else np.nan,
                V_grid_ring_pitch=volume.grid_volume(pts, grid_step=pitch, z_ref=ring)[0] if ring else np.nan,
                V_grid_fill=volume.grid_volume(pts, fill_radius=3)[0])


DENSITY = 0.95628  # kg/L, Aqil's mean over his 50 bunches; not fit to any bunch in this dataset


def zero_fit_mae(vol, gt, col):
    """MAE of density * volume with no parameter fit to our own masses; excludes FFB18 like everywhere else."""
    df = vol[vol.fusion == "nan_blur"].set_index("ffb")
    pred = DENSITY * df[col] * 1000
    actual = gt.loc[df.index, "Actual_Mass_kg"]
    err = (pred - actual).drop(index=ex.EXCLUDED, errors="ignore")
    return float(err.abs().mean())


def main():
    out = ROOT / "results" / "grid_fixes"
    out.mkdir(parents=True, exist_ok=True)
    gt = ex.ground_truth(ROOT)
    vol, _ = ex.run_bundles(str(ROOT / "data"), FUSIONS, str(ROOT / "fused_cache"), extra=grid_variants)
    vol.to_csv(out / "volumes.csv", index=False)
    t = ex.heldout_comparison(vol, gt, STEPS, models=ex.BASE_MODELS[:2])
    t.to_csv(out / "heldout.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.pivot_table(index=["model", "step"], columns="scheme", values="mae", aggfunc="first").round(2).to_string())
    zero_fit = {col: round(zero_fit_mae(vol, gt, col), 2) for col in ("V_grid", "V_grid_pitch", "V_grid_ring_pitch")}
    print("zero-fit density*volume MAE (n=10, no parameter fit to our masses):", zero_fit)
    (out / "summary.json").write_text(json.dumps(
        {"note": "isolated pixel-pitch/ring-z_ref grid checks, not part of the pre-registered cpu_eval steps",
         "zero_fit_density_mae_kg": zero_fit, "density_kg_l": DENSITY}, indent=2))


if __name__ == "__main__":
    main()
