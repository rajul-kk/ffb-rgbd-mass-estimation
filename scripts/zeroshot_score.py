"""Step 3 of the zero-shot benchmark: saved masks -> volumes -> held-out mass error (docs/PROTOCOL_zeroshot.md).

No model libraries. Reads zeroshot_cache/inputs and zeroshot_cache/masks, writes results/zeroshot/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import experiment as ex  # noqa: E402
from ffb import masking, zeroshot as zs  # noqa: E402
from zeroshot_infer import load_request  # noqa: E402

S1 = ["FFB10", "FFB11", "FFB12", "FFB17", "FFB18", "FFB19"]
ARMS_ALL = {  # modality -> (fit pool, scored); None = every bunch found
    "rgb": (S1, [b for b in S1 if b not in ex.EXCLUDED]),
    "hmap": (None, None),
}
ARMS = ARMS_ALL  # narrowed in main() to the arms that have saved masks
CONTROL = "depth_plane"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "zeroshot_cache")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "zeroshot")
    ap.add_argument("--B", type=int, default=20000)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    arms = {m: v for m, v in ARMS_ALL.items() if (a.cache / "masks" / m).is_dir()}
    gt = ex.ground_truth(ROOT)
    mass = gt["Actual_Mass_kg"]
    names = sorted((p.stem for p in (a.cache / "inputs").glob("FFB*.npz")), key=lambda s: int(s[3:]))
    rows, meta_rows = [], []
    for modality, (fit_pool, scored) in arms.items():
        fit_pool = fit_pool or names
        scored = scored or [b for b in names if b not in ex.EXCLUDED]
        providers = [CONTROL] + sorted(p.name for p in (a.cache / "masks" / modality).glob("*") if p.is_dir())
        for prov in providers:
            meta = {}
            if prov != CONTROL:
                meta = json.loads((a.cache / "masks" / modality / prov / "meta.json").read_text())
            for b in fit_pool:
                req = load_request(a.cache / "inputs" / f"{b}.npz", modality)
                if prov == CONTROL:
                    res, ref = masking.MaskResult(req.ref_mask), req.ref_mask
                else:
                    png = a.cache / "masks" / modality / prov / f"{b}.png"
                    raw = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) if png.exists() else None
                    res = masking.MaskResult(masking.postprocess(None if raw is None else raw > 0))
                    ref = req.ref_mask
                v = masking.volume_from_result(req, res)
                rows.append(dict(arm=modality, provider=prov, ffb=b, V_L=np.nan if v is None else 1000 * v,
                                 detected=res.mask is not None, iou_vs_control=masking.iou(res.mask, ref),
                                 mask_px=0 if res.mask is None else int(res.mask.sum()),
                                 seconds=meta.get(b, {}).get("seconds"), score=meta.get(b, {}).get("score")))
    vol = pd.DataFrame(rows)
    vol.to_csv(a.out / "volumes.csv", index=False)

    per_rows, sum_rows = [], []
    for modality, (fit_pool, scored) in arms.items():
        fit_pool = fit_pool or names
        scored = scored or [b for b in names if b not in ex.EXCLUDED]
        arm = vol[vol.arm == modality]
        pbs = {p: zs.loo_predictions(g.set_index("ffb")["V_L"], mass, fit_pool, scored) for p, g in arm.groupby("provider")}
        for p, pb in pbs.items():
            per_rows.append(pb.assign(arm=modality, provider=p).reset_index())
            s = zs.summarise_provider(pb, pbs[CONTROL], B=a.B)
            g = arm[(arm.provider == p) & arm.ffb.isin(scored)]
            s |= dict(arm=modality, provider=p, mean_iou_vs_control=float(g["iou_vs_control"].mean()) if p != CONTROL else np.nan,
                      mean_seconds=float(g["seconds"].mean()) if g["seconds"].notna().any() else np.nan)
            sum_rows.append(s)
    pd.concat(per_rows).round(4).to_csv(a.out / "per_bunch.csv", index=False)
    summ = pd.DataFrame(sum_rows)
    lead = ["arm", "provider", "n_scored", "n_detected"]
    summ = summ[lead + [c for c in summ.columns if c not in lead]]
    summ.round(4).to_csv(a.out / "metrics.csv", index=False)
    (a.out / "summary.json").write_text(json.dumps(summ.round(4).to_dict("records"), indent=1, default=float))
    pd.set_option("display.width", 250)
    for arm, g in summ.groupby("arm"):
        print(f"arm {arm}")
        cols = ["provider", "n_detected", "n_scored", "det_mae", "det_mae_lo", "det_mae_hi", "det_bias", "pen_mae", "pen_mae_lo",
                "pen_mae_hi", "pen_mape", "pen_bias", "pen_vs_control_diff", "pen_vs_control_lo", "pen_vs_control_hi", "mean_iou_vs_control"]
        print(g[[c for c in cols if c in g.columns]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
