"""Calibration-sample-size curve: held-out mass error as a function of how many bunches the gain was fitted on.

Every subset of k bunches from the fit pool is enumerated (or sampled when there are too many), a model is fitted
on it, and the other scoring bunches are predicted. Two intervals are reported per k: the percentile spread of
subset MAE (what one practitioner who picks k bunches could see) and a bootstrap over bunches of the per-bunch
mean error (uncertainty in which bunches exist, with the test set held fixed across subsets).
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from . import evaluate as ev

GLOBAL_K = ev.Model("global_k", ("V",))
AFFINE = ev.Model("aV+b", ("V",), intercept=True)


def _subsets(n: int, k: int, max_subsets: int, rng):
    if comb(n, k) <= max_subsets:
        return list(combinations(range(n), k)), True
    return [tuple(sorted(rng.choice(n, k, replace=False))) for _ in range(max_subsets)], False


def subset_errors(df: pd.DataFrame, model=GLOBAL_K, fit_pool=None, score_pool=None, ks=None,
                  max_subsets: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Long table: k, subset id, scored bunch, actual, pred, held_out. df needs columns V and mass.

    A bunch is scored for a subset only if it is in score_pool and not in the subset (held_out=True). If no subset
    of size k leaves a scored bunch out, the k's own scored members are scored in-sample (held_out=False) instead."""
    fit_pool = list(df.index if fit_pool is None else fit_pool)
    score_pool = list(df.index if score_pool is None else score_pool)
    n_par = len(model.features) + model.intercept
    ks = list(range(n_par, len(fit_pool) + 1)) if ks is None else [k for k in ks if k >= n_par]
    rng, rows = np.random.default_rng(seed), []
    for k in ks:
        subs, _ = _subsets(len(fit_pool), k, max_subsets, rng)
        held_rows, in_rows = [], []
        for sid, s in enumerate(subs):
            names = [fit_pool[i] for i in s]
            coef = model.fit(df.loc[names])
            test = [b for b in score_pool if b not in names]
            if test:
                pred = model.predict(coef, df.loc[test])
                held_rows += [(k, sid, b, df.at[b, "mass"], p, True) for b, p in zip(test, pred)]
            else:
                inside = [b for b in score_pool if b in names]
                pred = model.predict(coef, df.loc[inside])
                in_rows += [(k, sid, b, df.at[b, "mass"], p, False) for b, p in zip(inside, pred)]
        rows += held_rows or in_rows  # in-sample rows only when no subset of this size leaves a scored bunch out
    return pd.DataFrame(rows, columns=["k", "subset", "ffb", "actual", "pred", "held_out"])


def summarise(rec: pd.DataFrame, B: int = 20000, seed: int = 0) -> pd.DataFrame:
    """One row per k: MAE, MAPE, signed bias with bunch-bootstrap 95% CIs, and the 5th-95th percentile of subset MAE."""
    rec = rec.assign(err=rec["pred"] - rec["actual"])
    rec["ae"], rec["ape"] = rec["err"].abs(), 100 * rec["err"].abs() / rec["actual"]
    out = []
    for k, g in rec.groupby("k"):
        per = g.groupby("ffb").agg(mae=("ae", "mean"), mape=("ape", "mean"), bias=("err", "mean"))
        row = dict(k=int(k), held_out=bool(g["held_out"].all()), n_subsets=int(g["subset"].nunique()),
                   n_bunches_scored=len(per))
        for col in ("mae", "mape", "bias"):
            m, lo, hi = ev.bootstrap_ci(per[col], B=B, seed=seed)
            row |= {col: m, f"{col}_lo": lo, f"{col}_hi": hi}
        sub = g.groupby("subset")["ae"].mean()
        row |= dict(subset_mae_p05=float(sub.quantile(0.05)), subset_mae_p95=float(sub.quantile(0.95)),
                    subset_mae_max=float(sub.max()))
        out.append(row)
    return pd.DataFrame(out)
