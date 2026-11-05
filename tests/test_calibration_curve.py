from math import comb

import numpy as np
import pandas as pd
import pytest

from ffb import calibration_curve as cc


def frame(n=8, noise=0.0, gain=0.5, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.uniform(10, 30, n)
    return pd.DataFrame({"V": V, "mass": gain * V * (1 + noise * rng.standard_normal(n))},
                        index=[f"B{i}" for i in range(n)])


def test_noise_free_data_gives_zero_error_at_every_k():
    s = cc.summarise(cc.subset_errors(frame(8, noise=0.0)))
    assert (s["mae"] < 1e-9).all() and (s["bias"].abs() < 1e-9).all()


def test_subsets_are_enumerated_exactly_and_scored_on_the_complement():
    df = frame(6, noise=0.1)
    rec = cc.subset_errors(df, ks=[2])
    assert rec["subset"].nunique() == comb(6, 2)
    assert (rec.groupby("subset").size() == 4).all()
    assert rec["held_out"].all()


def test_a_bunch_is_never_scored_by_a_subset_that_contains_it():
    df = frame(6, noise=0.1)
    pool = list(df.index)
    rec = cc.subset_errors(df, ks=[3])
    from itertools import combinations
    subs = list(combinations(pool, 3))
    for sid, ffb in zip(rec["subset"], rec["ffb"]):
        assert ffb not in subs[sid]


def test_error_falls_with_k_on_noisy_data():
    s = cc.summarise(cc.subset_errors(frame(10, noise=0.15, seed=3)), B=2000)
    assert s["mae"].iloc[0] > s["mae"].iloc[-1]
    assert s["subset_mae_p95"].iloc[0] > s["subset_mae_p95"].iloc[-1]


def test_full_pool_k_is_flagged_in_sample():
    s = cc.summarise(cc.subset_errors(frame(5, noise=0.2)), B=500)
    assert s.loc[s.k == 5, "held_out"].item() is False
    assert s.loc[s.k < 5, "held_out"].all()


def test_fit_pool_can_include_bunches_that_are_never_scored():
    df = frame(7, noise=0.1)
    rec = cc.subset_errors(df, fit_pool=list(df.index), score_pool=list(df.index[:5]), ks=[2])
    assert set(rec["ffb"]) <= set(df.index[:5])


def test_affine_model_skips_k_below_its_parameter_count():
    rec = cc.subset_errors(frame(6, noise=0.1), model=cc.AFFINE)
    assert rec["k"].min() == 2


def test_sampling_is_used_and_reproducible_when_too_many_subsets():
    df = frame(12, noise=0.1)
    a = cc.subset_errors(df, ks=[6], max_subsets=50, seed=1)
    b = cc.subset_errors(df, ks=[6], max_subsets=50, seed=1)
    assert a["subset"].nunique() == 50
    pd.testing.assert_frame_equal(a, b)


def test_interval_contains_the_mean_and_bias_sign_is_right():
    df = frame(8, noise=0.0)
    df["mass"] = df["mass"] * 1.2   # truth is 1.2x; a gain fitted on k bunches is exact -> still zero bias
    s = cc.summarise(cc.subset_errors(df), B=500)
    assert (s["bias"].abs() < 1e-9).all()
    df2 = frame(8, noise=0.1, seed=5)
    s2 = cc.summarise(cc.subset_errors(df2), B=2000)
    assert ((s2["mae_lo"] <= s2["mae"]) & (s2["mae"] <= s2["mae_hi"])).all()


def test_in_sample_rows_never_mix_into_a_held_out_k():
    df = frame(6, noise=0.2)
    # 6 fit candidates, 5 scored: at k = 5 the subset made of the 5 scored bunches has no test bunch, the others do
    rec = cc.subset_errors(df, fit_pool=list(df.index), score_pool=list(df.index[:5]), ks=[5, 6])
    assert rec[rec.k == 5]["held_out"].all()
    assert not rec[rec.k == 6]["held_out"].any()
