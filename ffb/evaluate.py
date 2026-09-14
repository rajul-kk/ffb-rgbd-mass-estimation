"""Small-sample evaluation: calibration models, LOO, exhaustive CV, bootstrap CIs and jackknife+."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Model:
    name: str
    features: tuple
    intercept: bool = False
    by_group: bool = False  # separate coefficients per df["group"]

    def _X(self, df):
        cols = [df[f].to_numpy(float) for f in self.features]
        if self.intercept:
            cols.append(np.ones(len(df)))
        return np.column_stack(cols)

    def fit(self, df, target="mass"):
        if not self.by_group:
            return np.linalg.lstsq(self._X(df), df[target].to_numpy(float), rcond=None)[0]
        return {g: np.linalg.lstsq(self._X(s), s[target].to_numpy(float), rcond=None)[0]
                for g, s in df.groupby("group")}

    def predict(self, coef, df):
        if not self.by_group:
            return self._X(df) @ coef
        out = np.full(len(df), np.nan)
        groups = df["group"].to_numpy()
        for g, c in coef.items():
            m = groups == g
            if m.any():
                out[m] = self._X(df[m]) @ c
        return out


def metrics(actual, pred) -> dict:
    """MAE, MAPE, squared Pearson r and coefficient of determination over finite predictions."""
    y, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    err = p - y
    r = np.corrcoef(y, p)[0, 1] if len(y) > 2 else np.nan
    return dict(n=int(ok.sum()), mae=float(np.abs(err).mean()), mape=float(100 * np.mean(np.abs(err) / y)),
                pearson_r2=float(r ** 2), r2=float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)))


def loo(df, model, target="mass") -> pd.Series:
    pred = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        coef = model.fit(df.drop(df.index[i]), target)
        pred.iloc[i] = model.predict(coef, df.iloc[[i]])[0]
    return pred


def exhaustive_cv(df, model, n_train: int, target="mass") -> pd.DataFrame:
    """Every n_train-subset as training set; one row per held-out prediction."""
    rows, idx = [], np.arange(len(df))
    for fold, train in enumerate(combinations(idx, n_train)):
        test = np.setdiff1d(idx, train)
        pred = model.predict(model.fit(df.iloc[list(train)], target), df.iloc[test])
        rows += [(fold, df.index[j], df[target].iloc[j], p) for j, p in zip(test, pred)]
    return pd.DataFrame(rows, columns=["fold", "ffb", "actual", "pred"])


def bootstrap_ci(values, B: int = 20000, seed: int = 0, level: float = 0.95):
    """Mean with a percentile bootstrap CI."""
    v = np.asarray(values, float)
    stats = v[np.random.default_rng(seed).integers(0, len(v), (B, len(v)))].mean(1)
    a = 50 * (1 - level)
    lo, hi = np.percentile(stats, [a, 100 - a])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(err_new, err_ref, B: int = 20000, seed: int = 0) -> dict:
    """Mean of err_new - err_ref over matched rows, CI, and bootstrap P(new is not better)."""
    d = np.asarray(err_new, float) - np.asarray(err_ref, float)
    stats = d[np.random.default_rng(seed).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return dict(diff=float(d.mean()), lo=float(lo), hi=float(hi), p_not_better=float((stats >= 0).mean()))


def jackknife_plus(df, model, alpha: float = 0.1, target="mass") -> pd.DataFrame:
    """Nested jackknife+ intervals: each row's interval uses only the other rows (coverage >= 1 - 2*alpha)."""
    out = []
    for j in range(len(df)):
        rest, test = df.drop(df.index[j]), df.iloc[[j]]
        lows, highs = [], []
        for i in range(len(rest)):
            coef = model.fit(rest.drop(rest.index[i]), target)
            resid = abs(rest[target].iloc[i] - model.predict(coef, rest.iloc[[i]])[0])
            mu = model.predict(coef, test)[0]
            lows.append(mu - resid)
            highs.append(mu + resid)
        n = len(rest)
        k_lo, k_hi = int(np.floor(alpha * (n + 1))), int(np.ceil((1 - alpha) * (n + 1)))
        lo = np.sort(lows)[k_lo - 1] if k_lo >= 1 else -np.inf
        hi = np.sort(highs)[k_hi - 1] if k_hi <= n else np.inf
        y = test[target].iloc[0]
        out.append((df.index[j], y, lo, hi, bool(lo <= y <= hi)))
    return pd.DataFrame(out, columns=["ffb", "actual", "lo", "hi", "covered"])
