"""Held-out check of the pixel-pitch grid fix, isolated from cpu_eval.py's pre-registered steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import experiment as ex  # noqa: E402
from ffb.fusion import FusionConfig  # noqa: E402

FUSIONS = {"notebook": FusionConfig(reader="notebook", spatial="blur_then_mask"),
           "nan_blur": FusionConfig(reader="fixed", spatial="mask_then_nanmedian")}
STEPS = [("S0_notebook", "notebook", "V_grid"), ("S2_nan_blur", "nan_blur", "V_grid"),
         ("S2b_pitch_grid", "nan_blur", "V_grid_pitch")]


def main():
    out = ROOT / "results"
    gt = ex.ground_truth(ROOT)
    vol, _ = ex.run_bundles(str(ROOT / "data"), FUSIONS, str(ROOT / "fused_cache"))
    vol.to_csv(out / "grid_pitch_volumes.csv", index=False)
    t = ex.heldout_comparison(vol, gt, STEPS, models=ex.BASE_MODELS[:2])
    t.to_csv(out / "grid_pitch_heldout.csv", index=False)
    pd_opts = {"display.width": 200}
    import pandas as pd
    with pd.option_context(*[x for kv in pd_opts.items() for x in kv]):
        print(t.pivot_table(index=["model", "step"], columns="scheme", values="mae",
                             aggfunc="first").round(2).to_string())
    (out / "grid_pitch_summary.json").write_text(json.dumps(
        {"note": "isolated pixel-pitch grid check, not part of the pre-registered cpu_eval steps"}, indent=2))


if __name__ == "__main__":
    main()
