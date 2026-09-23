"""Held-out check of the depth-only plane-height mask against v2, Aqil, and a threshold sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import evaluate as ev  # noqa: E402
from ffb import experiment as ex  # noqa: E402
from ffb import io, segment, volume  # noqa: E402
from ffb.fusion import FusionConfig, fuse  # noqa: E402

CFG = FusionConfig(reader="notebook")
PRESPECIFIED = dict(h_min=0.03, band=0.03)  # fixed before any held-out run
SWEEP = [(h, b) for h in (0.02, 0.03, 0.04, 0.05) for b in (0.02, 0.03, 0.05)]


def main():
    out = ROOT / "results"
    gt = ex.ground_truth(ROOT)
    scenes = {}
    for folder in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir() and p.name.startswith("FFB")):
        b = io.find_bundle(str(folder))
        K = io.intrinsics(b)
        scenes[b.name] = (b.layout, K, fuse(b, K, CFG, str(ROOT / "fused_cache"))[0])

    vol, _ = ex.run_bundles(str(ROOT / "data"), {"notebook": CFG}, str(ROOT / "fused_cache"), verbose=False)
    for robust in (False, True):
        col = "V_plane_ph_msac" if robust else "V_plane_ph"
        for name, (_, K, fused) in scenes.items():
            m, pl = segment.plane_height_mask(fused, K, **PRESPECIFIED, robust=robust)
            vol.loc[vol.ffb == name, col] = volume.plane_volume(fused, m, K, pl)[0] if m is not None else np.nan
    vol.to_csv(out / "plane_mask_volumes.csv", index=False)

    steps = [("v2_grid", "notebook", "V_grid"), ("tarp_plane_v2_mask", "notebook", "V_plane"),
             ("plane_height_mask", "notebook", "V_plane_ph"), ("plane_height_mask_msac_lo", "notebook", "V_plane_ph_msac")]
    held = ex.heldout_comparison(vol, gt, steps, models=ex.BASE_MODELS[:2])
    held.to_csv(out / "plane_mask_heldout.csv", index=False)

    sweep = []
    for h_min, band in SWEEP:
        rows = []
        for name, (layout, K, fused) in scenes.items():
            m, pl = segment.plane_height_mask(fused, K, h_min=h_min, band=band)
            rows.append(dict(ffb=name, fusion="x", layout=layout, V=volume.plane_volume(fused, m, K, pl)[0] if m is not None else np.nan))
        t = ex.heldout_comparison(pd.DataFrame(rows), gt, [("s", "x", "V")], models=ex.BASE_MODELS[:2], B=200)
        mae = t.set_index(["model", "scheme"])["mae"]
        sweep.append(dict(h_min=h_min, band=band, global_LOO=mae["global_k", "LOO"], global_4_7=mae["global_k", "4/7"],
                          per_session_LOO=mae["per_session_k", "LOO"]))
    pd.DataFrame(sweep).to_csv(out / "plane_mask_sweep.csv", index=False)

    a_tab = pd.read_csv(ROOT / "aqil_table_c.csv")
    aqil = pd.Series(a_tab["Est_Mass_kg"].to_numpy(), index="FFB" + a_tab["FFB_No"].astype(str))
    vs_aqil = {}
    for step, col in (("v2_grid", "V_grid"), ("plane_height_mask", "V_plane_ph")):
        df = ex.step_frame(vol, gt, "notebook", col).drop(index=ex.EXCLUDED)
        for model in ex.BASE_MODELS[:2]:
            pred = ev.loo(df, model)
            actual = df["mass"]
            a = aqil.loc[df.index]
            vs_aqil[f"{step}:{model.name}"] = dict(mae=float((pred - actual).abs().mean()),
                                                   mape=float(100 * ((pred - actual).abs() / actual).mean()),
                                                   aqil_mae=float((a - actual).abs().mean()),
                                                   **ev.paired_bootstrap((pred - actual).abs(), (a - actual).abs()))
    g = ex.step_frame(vol, gt, "notebook", "V_plane_ph").drop(index=ex.EXCLUDED)
    gains = dict(global_kg_per_L=float(ex.BASE_MODELS[1].fit(g)[0]) / 1000,
                 per_session_kg_per_L={k: float(v[0]) / 1000 for k, v in ex.BASE_MODELS[0].fit(g).items()})
    ratio = (vol.set_index("ffb")[["V_plane", "V_plane_ph"]].mul(1000).div(gt["Actual_Volume_L"], axis=0)
             .drop(index=ex.EXCLUDED).groupby(vol.set_index("ffb")["layout"]).median())
    summary = dict(prespecified=PRESPECIFIED, vs_aqil_same_10=vs_aqil, gains=gains,
                   volume_over_true_by_session=ratio.round(3).to_dict(),
                   session_ratio={c: float(ratio.loc["combined", c] / ratio.loc["split", c]) for c in ratio})
    (out / "plane_mask_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(held.pivot_table(index=["model", "step"], columns="scheme", values="mae", aggfunc="first").round(2).to_string())
    print(pd.DataFrame(sweep).round(2).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("vs_aqil_same_10", "gains", "session_ratio")}, indent=1, default=float))


if __name__ == "__main__":
    main()
