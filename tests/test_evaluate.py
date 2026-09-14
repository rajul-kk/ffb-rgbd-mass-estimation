import numpy as np
import pandas as pd
import pytest

from ffb import evaluate as ev


def frame():
    rng = np.random.default_rng(0)
    V = rng.uniform(0.008, 0.02, 10)
    group = np.array(["combined"] * 5 + ["split"] * 5)
    mass = 1000 * V * np.where(group == "split", 2.5, 2.0) + rng.normal(0, 0.8, 10)
    return pd.DataFrame({"V": V, "H": rng.uniform(0.1, 0.2, 10), "group": group, "mass": mass},
                        index=[f"FFB{i}" for i in range(10)])


def test_density_constant_cancels_in_notebook_calibration():
    df = frame()
    r, m = df["V"].to_numpy(), df["mass"].to_numpy()
    preds = [r * (m @ r / (D * (r @ r))) * D for D in (500.0, 956.28, 2000.0)]
    assert np.allclose(preds[0], preds[1]) and np.allclose(preds[1], preds[2])


def test_per_group_through_origin_loo_ignores_per_group_rescaling():
    df, scaled = frame(), frame()
    scaled.loc[scaled["group"] == "split", "V"] *= 2.53
    model = ev.Model("k", ("V",), by_group=True)
    assert np.allclose(ev.loo(df, model), ev.loo(scaled, model))


def test_exhaustive_cv_counts_and_flags_missing_groups():
    cv = ev.exhaustive_cv(frame(), ev.Model("k", ("V",), by_group=True), 4)
    assert len(cv) == 210 * 6
    assert cv["pred"].isna().sum() == 10 * 5


def test_metrics_separates_correlation_from_fit():
    m = ev.metrics([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert m["pearson_r2"] == pytest.approx(1.0) and m["r2"] < 0


def test_jackknife_plus_is_exact_on_noiseless_data():
    df = frame()
    df["mass"] = 1500 * df["V"]
    jk = ev.jackknife_plus(df, ev.Model("k", ("V",)), alpha=0.1)
    assert np.allclose(jk["lo"], jk["actual"]) and np.allclose(jk["hi"], jk["actual"])


def test_bootstraps():
    assert ev.bootstrap_ci([2.0] * 8) == (2.0, 2.0, 2.0)
    pb = ev.paired_bootstrap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert pb["diff"] == -1.0 and pb["p_not_better"] == 0.0
