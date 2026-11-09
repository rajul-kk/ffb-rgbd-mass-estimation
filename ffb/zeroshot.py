"""Score mask providers on held-out mass error (docs/PROTOCOL_zeroshot.md sections 2 and 5). numpy/pandas only.

Input is one volume per (provider, bunch), NaN where the provider failed. Calibration is leave-one-out with the
global gain (mass = g * V) fitted on the other fit-pool bunches that have a volume from the same provider.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as ev

MODEL = ev.Model("global_k", ("V",))


def loo_predictions(vol: pd.Series, mass: pd.Series, fit_pool, scored) -> pd.DataFrame:
    """Per scored bunch: actual, pred (NaN on failure) and pred_pen (training-mean mass on failure).

    vol: volume by bunch for one provider (NaN = failed). A bunch also fails if no other bunch has a volume to fit on."""
    rows = []
    for b in scored:
        others = [o for o in fit_pool if o != b]
        fit = pd.DataFrame({"V": vol.reindex(others), "mass": mass.reindex(others)}).dropna()
        pred = np.nan
        if len(fit) and np.isfinite(vol.get(b, np.nan)):
            pred = float(MODEL.predict(MODEL.fit(fit), pd.DataFrame({"V": [vol[b]]}))[0])
        fallback = float(mass.reindex(others).mean())
        rows.append(dict(ffb=b, actual=float(mass[b]), pred=pred, pred_pen=fallback if np.isnan(pred) else pred))
    return pd.DataFrame(rows).set_index("ffb")


def _stats(err: pd.Series, actual: pd.Series, B: int) -> dict:
    out = {}
    for name, v in (("mae", err.abs()), ("mape", 100 * err.abs() / actual), ("bias", err)):
        m, lo, hi = ev.bootstrap_ci(v, B=B)
        out |= {name: m, f"{name}_lo": lo, f"{name}_hi": hi}
    return out


def summarise_provider(pb: pd.DataFrame, control: pd.DataFrame, B: int = 20000) -> dict:
    """Coverage, detected-only and failure-penalised stats, and paired differences against the control."""
    det = pb.dropna(subset=["pred"])
    row = dict(n_scored=len(pb), n_detected=len(det))
    if len(det):
        row |= {f"det_{k}": v for k, v in _stats(det["pred"] - det["actual"], det["actual"], B).items()}
        both = det.index.intersection(control.dropna(subset=["pred"]).index)
        if len(both) > 1:
            d = ev.paired_bootstrap((det.loc[both, "pred"] - det.loc[both, "actual"]).abs(),
                                    (control.loc[both, "pred"] - control.loc[both, "actual"]).abs(), B=B)
            row |= {f"det_vs_control_{k}": v for k, v in d.items()} | {"det_vs_control_n": len(both)}
    row |= {f"pen_{k}": v for k, v in _stats(pb["pred_pen"] - pb["actual"], pb["actual"], B).items()}
    if len(pb) > 1:
        d = ev.paired_bootstrap((pb["pred_pen"] - pb["actual"]).abs(),
                                (control["pred_pen"] - control["actual"]).abs().reindex(pb.index), B=B)
        row |= {f"pen_vs_control_{k}": v for k, v in d.items()}
    return row
