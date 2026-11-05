import numpy as np
import pandas as pd
import pytest

from ffb import zeroshot as zs

MASS = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0})


def vols(scale=0.1, **over):
    v = MASS * scale
    for k, x in over.items():
        v[k] = x
    return v


def test_perfect_volumes_give_zero_error_and_full_coverage():
    pb = zs.loo_predictions(vols(), MASS, list(MASS.index), list(MASS.index))
    assert np.allclose(pb["pred"], pb["actual"])
    assert pb["pred"].notna().all()


def test_failed_bunch_is_nan_in_detected_only_and_training_mean_in_penalised():
    pb = zs.loo_predictions(vols(C=np.nan), MASS, list(MASS.index), list(MASS.index))
    assert np.isnan(pb.loc["C", "pred"])
    assert pb.loc["C", "pred_pen"] == pytest.approx((10 + 20 + 40) / 3)
    assert pb.loc["A", "pred"] == pytest.approx(10.0)  # gain still fitted on the bunches that worked


def test_a_failed_fit_pool_bunch_is_excluded_from_the_gain_not_zeroed():
    pb = zs.loo_predictions(vols(D=np.nan), MASS, list(MASS.index), ["A", "B", "C"])
    assert np.allclose(pb["pred"], pb["actual"])


def test_scoring_set_can_be_smaller_than_fit_pool():
    pb = zs.loo_predictions(vols(), MASS, list(MASS.index), ["A", "B"])
    assert list(pb.index) == ["A", "B"]


def test_fit_pool_without_any_other_volume_fails_the_bunch():
    pb = zs.loo_predictions(pd.Series({"A": 1.0, "B": np.nan}), MASS[["A", "B"]], ["A", "B"], ["A"])
    assert np.isnan(pb.loc["A", "pred"])


def test_summary_reports_coverage_bias_and_paired_difference():
    ctrl = zs.loo_predictions(vols(), MASS, list(MASS.index), list(MASS.index))
    noisy = vols(A=1.3, B=1.8, C=3.3, D=3.6)             # alternating +30% and -10% volume errors
    pb = zs.loo_predictions(noisy, MASS, list(MASS.index), list(MASS.index))
    s = zs.summarise_provider(pb, ctrl, B=500)
    assert s["n_detected"] == 4 and s["n_scored"] == 4
    assert s["det_mae"] > 0 and s["det_vs_control_diff"] > 0   # worse than the perfect control
    assert s["det_mae_lo"] <= s["det_mae"] <= s["det_mae_hi"]


def test_summary_with_failures_has_both_scorings():
    ctrl = zs.loo_predictions(vols(), MASS, list(MASS.index), list(MASS.index))
    pb = zs.loo_predictions(vols(A=np.nan, B=np.nan), MASS, list(MASS.index), list(MASS.index))
    s = zs.summarise_provider(pb, ctrl, B=500)
    assert s["n_detected"] == 2
    assert s["pen_mae"] > s["det_mae"]          # penalising failures cannot look better than ignoring them here
