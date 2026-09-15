import numpy as np
import pandas as pd

from ffb import evaluate as ev
from ffb import experiment as ex


def test_heldout_comparison_covers_schemes_and_pairs_against_first_step():
    rng = np.random.default_rng(0)
    names = [f"FFB{i}" for i in range(11)]
    layout = ["combined"] * 6 + ["split"] * 5
    V = np.linspace(0.008, 0.02, 11)
    rows = [dict(ffb=n, fusion="exact", layout=l, V_grid=v, H=0.2) for n, l, v in zip(names, layout, V)]
    rows += [dict(ffb=n, fusion="noisy", layout=l, V_grid=v * (1 + rng.normal(0, 0.1)), H=0.2) for n, l, v in zip(names, layout, V)]
    gt = pd.DataFrame({"Actual_Mass_kg": 1000 * V}, index=names)
    t = ex.heldout_comparison(pd.DataFrame(rows), gt, [("S0", "exact", "V_grid"), ("S1", "noisy", "V_grid")],
                              models=[ev.Model("k", ("V",))], excluded=["FFB3"], B=200)
    assert set(t["scheme"]) == {"LOO", "4/7", "7/4"}
    assert t.loc[t["step"] == "S0", "mae"].max() < 1e-9
    assert t.loc[t["step"] == "S0", "vs_first_diff"].isna().all()
    assert (t.loc[t["step"] == "S1", "vs_first_diff"] > 0).all()
