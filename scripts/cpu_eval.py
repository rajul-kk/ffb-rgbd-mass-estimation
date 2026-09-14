"""Local CPU evaluation of Approach A: notebook parity, cumulative fixes, calibration models, optional SAM 3."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import experiment as ex  # noqa: E402
from ffb.fusion import FusionConfig  # noqa: E402

# report.md Approach A litres after the notebook's per-camera refit; used only to check parity.
REPORT_A_L = {"FFB10": 18.70, "FFB11": 15.01, "FFB12": 22.56, "FFB17": 11.53, "FFB18": 13.19, "FFB19": 18.61,
              "FFB31": 13.24, "FFB32": 7.95, "FFB33": 14.90, "FFB34": 12.17, "FFB35": 14.38}
FUSIONS = {
    "notebook": FusionConfig(reader="notebook", spatial="blur_then_mask"),
    "all_frames": FusionConfig(reader="fixed", spatial="blur_then_mask"),
    "nan_blur": FusionConfig(reader="fixed", spatial="mask_then_nanmedian"),
}
# One fix per step. Primary test, fixed before any run: S4 vs S0 LOO MAE (n=10) under both primary models.
STEPS = [
    ("S0_notebook", "notebook", "V_grid"),
    ("S1_all_frames", "all_frames", "V_grid"),
    ("S2_nan_blur", "nan_blur", "V_grid"),
    ("S3_frustum", "nan_blur", "V_frustum"),
    ("S4_tarp_plane", "nan_blur", "V_plane"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=str(ROOT / "fused_cache"))
    ap.add_argument("--sam3", action="store_true", help="also run SAM 3 (needs transformers>=5 and facebook/sam3 access)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, gt = time.time(), ex.ground_truth(ROOT)

    vol, keep = ex.run_bundles(args.data, FUSIONS, args.cache, keep_fusion="nan_blur")
    vol.to_csv(out / "cpu_eval_volumes.csv", index=False)
    lad, preds = ex.ladder(vol, gt, STEPS)
    lad.to_csv(out / "cpu_eval_ladder.csv", index=False)
    ffbs = [f for f in ex.step_frame(vol, gt, "notebook", "V_grid").index if f not in ex.EXCLUDED]
    per_ffb = pd.DataFrame({"actual_kg": gt.loc[ffbs, "Actual_Mass_kg"]})
    for (step, model), p in preds.items():
        if step in ("S0_notebook", "S4_tarp_plane") and model in ex.PRIMARY_MODELS:
            per_ffb[f"{step}:{model}"] = p
    per_ffb.round(3).to_csv(out / "cpu_eval_per_ffb_loo.csv")

    tarp = float(np.nanmedian(vol.loc[vol["fusion"] == "nan_blur", "tarp_depth"]))
    cameras = {f"{K.width}x{K.height}": K for K in (keep["FFB10"][0], keep["FFB31"][0])}
    summary = dict(parity=ex.parity(ex.step_frame(vol, gt, "notebook", "V_grid"), REPORT_A_L),
                   jackknife_plus=ex.intervals(vol, gt, [STEPS[0], STEPS[-1]]),
                   caliper=ex.caliper(gt, ffbs), synthetic=ex.synthetic_check(tarp, cameras))
    summary["cv_S4_tarp_plane"] = [ex.cv_summary(ex.step_frame(vol, gt, "nan_blur", "V_plane"), m)
                                   for m in ex.BASE_MODELS[:2]]
    summary["sam3"] = ex.sam3_eval(keep) if args.sam3 else dict(status="not requested")
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "cpu_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(lad[lad["primary"]].round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("parity", "caliper", "jackknife_plus", "runtime_s")}, indent=1, default=float))


if __name__ == "__main__":
    main()
