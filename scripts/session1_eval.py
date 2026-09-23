"""Session 1 check: fuse only the steady opening frames (and optionally unaligned depth) under the depth-only mask.
Candidates were fixed before any result: A = steady frames, B = steady frames + native depth for both sessions."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import evaluate as ev  # noqa: E402
from ffb import experiment as ex  # noqa: E402
from ffb import io, segment, volume  # noqa: E402
from ffb.fusion import FusionConfig, fuse  # noqa: E402

CANDIDATES = {
    "all_frames": FusionConfig(reader="notebook"),                       # current depth-only result
    "A_steady": FusionConfig(reader="fixed", frames="steady"),
    "B_steady_native": FusionConfig(reader="native", frames="steady"),
}


def main():
    warnings.simplefilter("ignore")
    out = ROOT / "results"
    gt = ex.ground_truth(ROOT)
    rows = []
    for folder in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir() and p.name.startswith("FFB")):
        b = io.find_bundle(str(folder))
        for name, cfg in CANDIDATES.items():
            K = io.intrinsics(b, "depth" if cfg.reader == "native" else "auto")
            fused, _, n_used, info = fuse(b, K, cfg, str(ROOT / "fused_cache"), return_info=True)
            m, pl = segment.plane_height_mask(fused, K)
            rows.append(dict(ffb=b.name, fusion=name, layout=b.layout, frames_used=n_used, frames_read=info["n_read"],
                             V=volume.plane_volume(fused, m, K, pl)[0] if m is not None else np.nan))
            print(b.name, name, n_used, "/", info["n_read"], flush=True)
    vol = pd.DataFrame(rows)
    vol.to_csv(out / "session1_volumes.csv", index=False)

    steps = [(n, n, "V") for n in CANDIDATES]
    held = ex.heldout_comparison(vol, gt, steps, models=ex.BASE_MODELS[:2])
    held.to_csv(out / "session1_heldout.csv", index=False)

    per = {}
    for n in CANDIDATES:
        df = ex.step_frame(vol, gt, n, "V").drop(index=ex.EXCLUDED)
        p = ev.loo(df, ex.BASE_MODELS[1])
        per[n] = p - df["mass"]
    per = pd.DataFrame(per).assign(session=ex.step_frame(vol, gt, "all_frames", "V").drop(index=ex.EXCLUDED)["group"])
    per.round(3).to_csv(out / "session1_per_bunch_errors.csv")
    ratio = vol.assign(true_L=gt.loc[vol.ffb, "Actual_Volume_L"].to_numpy())
    ratio["ratio"] = 1000 * ratio.V / ratio.true_L
    spread = ratio[~ratio.ffb.isin(ex.EXCLUDED)].groupby(["fusion", "layout"])["ratio"].agg(["median", "std"]).round(3)

    pd.set_option("display.width", 200)
    print(held.pivot_table(index=["model", "step"], columns="scheme", values="mae", aggfunc="first").round(2).to_string())
    print(held[held.scheme == "LOO"][["model", "step", "vs_first_diff", "vs_first_lo", "vs_first_hi"]].round(2).to_string(index=False))
    print(per.round(2).to_string())
    print(per.drop(columns="session").abs().groupby(per.session).mean().round(2).to_string())
    print(spread.to_string())
    (out / "session1_summary.json").write_text(json.dumps(dict(
        mae_by_session=per.drop(columns="session").abs().groupby(per.session).mean().round(3).to_dict(),
        ratio_spread={f"{a}:{b}": v for (a, b), v in spread["std"].items()}), indent=2))


if __name__ == "__main__":
    main()
