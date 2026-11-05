"""Calibration-sample-size curve from stored volumes (docs/PROTOCOL_zeroshot.md section 6). No bags, no models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import calibration_curve as cc  # noqa: E402
from ffb import experiment as ex  # noqa: E402

ROW = "B_steady_native"


def main():
    out = ROOT / "results" / "calibration_curve"
    out.mkdir(parents=True, exist_ok=True)
    gt = ex.ground_truth(ROOT)
    vol = pd.read_csv(ROOT / "results" / "session1" / "volumes.csv")
    d = vol[vol.fusion == ROW].set_index("ffb")
    df = pd.DataFrame({"V": 1000 * d["V"], "mass": gt.loc[d.index, "Actual_Mass_kg"]})  # litres
    scored = [b for b in df.index if b not in ex.EXCLUDED]
    runs = {
        "global_k_pool10": dict(model=cc.GLOBAL_K, fit_pool=scored, score_pool=scored),
        "global_k_pool11_fit": dict(model=cc.GLOBAL_K, fit_pool=list(df.index), score_pool=scored),
        "affine_pool10": dict(model=cc.AFFINE, fit_pool=scored, score_pool=scored),
    }
    tabs = {}
    for name, kw in runs.items():
        tabs[name] = cc.summarise(cc.subset_errors(df, **kw)).assign(run=name)
    pd.concat(tabs.values()).round(4).to_csv(out / "curve.csv", index=False)
    pd.set_option("display.width", 200)
    for name, t in tabs.items():
        print(name)
        print(t[["k", "held_out", "n_subsets", "mae", "mae_lo", "mae_hi", "mape", "bias", "subset_mae_p05", "subset_mae_p95"]]
              .round(2).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(
        {n: t.round(3).to_dict("records") for n, t in tabs.items()}, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = tabs["global_k_pool10"]
        t = t[t.held_out]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.fill_between(t.k, t.subset_mae_p05, t.subset_mae_p95, alpha=0.2, label="subset MAE, 5th-95th pct")
        ax.errorbar(t.k, t.mae, yerr=[t.mae - t.mae_lo, t.mae_hi - t.mae], marker="o", capsize=3,
                    label="mean MAE, 95% bootstrap over bunches")
        ax.set_xlabel("bunches used to fit the gain (k)")
        ax.set_ylabel("held-out MAE (kg)")
        ax.set_title("Calibration-sample-size curve (10 scored bunches)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=130)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
