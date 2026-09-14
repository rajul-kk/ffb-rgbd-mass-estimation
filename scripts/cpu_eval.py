"""Local CPU evaluation of Approach A: notebook parity, cumulative fixes, calibration models, optional SAM 3."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import evaluate as ev  # noqa: E402
from ffb import io, segment, synthetic, volume  # noqa: E402
from ffb.fusion import FusionConfig, fuse  # noqa: E402

# report.md Approach A litres after the notebook's per-camera refit; used only to check parity.
REPORT_A_L = {"FFB10": 18.70, "FFB11": 15.01, "FFB12": 22.56, "FFB17": 11.53, "FFB18": 13.19, "FFB19": 18.61,
              "FFB31": 13.24, "FFB32": 7.95, "FFB33": 14.90, "FFB34": 12.17, "FFB35": 14.38}
FUSIONS = {
    "notebook": FusionConfig(reader="notebook", spatial="blur_then_mask"),
    "all_frames": FusionConfig(reader="fixed", spatial="blur_then_mask"),
    "nan_blur": FusionConfig(reader="fixed", spatial="mask_then_nanmedian"),
}
# One fix per step. Primary test, fixed before any run: S4 vs S0 LOO MAE (n=10) under both primary models.
STEPS = [
    ("S0_notebook", "notebook", "V_grid"),
    ("S1_all_frames", "all_frames", "V_grid"),
    ("S2_nan_blur", "nan_blur", "V_grid"),
    ("S3_frustum", "nan_blur", "V_frustum"),
    ("S4_tarp_plane", "nan_blur", "V_plane"),
]
MODELS = [
    ev.Model("per_session_k", ("V",), by_group=True),
    ev.Model("global_k", ("V",)),
    ev.Model("aV+b", ("V",), intercept=True),
    ev.Model("aV+bH", ("V", "H")),
]
PRIMARY_MODELS = ("per_session_k", "global_k")
EXCLUDED = ["FFB18"]


def ground_truth() -> pd.DataFrame:
    g = pd.read_csv(ROOT / "ground_truth.csv")
    g.index = "FFB" + g["FFB_No"].astype(str)
    g["ellipsoid_L"] = np.pi / 6 * g["Width_cm"] * g["Length_cm"] * g["Thickness_cm"] / 1000
    return g


def measure(bundle, K, cfg, cache):
    t0 = time.time()
    fused, rgb, n_frames = fuse(bundle, K, cfg, cache)
    mask, z_front, z_window, source = segment.extract_mask(fused, rgb, K.width)
    row = dict(n_frames=n_frames, mask_source=source, z_front=z_front, z_window=z_window)
    if mask is not None:
        clipped = volume.clip_depth(fused, z_front, z_window)
        v_grid, z_ref = volume.grid_volume(volume.project(clipped, mask, K))
        row.update(V_grid=v_grid, z_ref=z_ref, V_frustum=volume.frustum_volume(clipped, mask, K, z_ref),
                   mask_px=int(mask.sum()), mask_holes=float((fused[mask] <= 0).mean()))
        try:
            n, d, rmse = volume.fit_plane(volume.tarp_points(fused, mask, K))
            v_plane, height, area = volume.plane_volume(fused, mask, K, (n, d))
            row.update(V_plane=v_plane, H=height, area=area, tarp_depth=float(-d / n[2]),
                       tilt_deg=float(np.degrees(np.arccos(abs(n[2])))), plane_rmse_mm=1000 * rmse)
        except ValueError as e:
            row["plane_error"] = str(e)
    row["seconds"] = round(time.time() - t0, 1)
    return row, mask, rgb, fused


def run_bundles(data, cache):
    rows, keep = [], {}
    for folder in sorted(p for p in Path(data).iterdir() if p.is_dir() and p.name.startswith("FFB")):
        bundle = io.find_bundle(str(folder))
        K = io.intrinsics(bundle)
        for name, cfg in FUSIONS.items():
            row, mask, rgb, fused = measure(bundle, K, cfg, cache)
            rows.append(dict(ffb=bundle.name, fusion=name, layout=bundle.layout, width=K.width, fx=round(K.fx, 1)) | row)
            if name == "nan_blur":
                keep[bundle.name] = (K, mask, rgb, fused)
            print(f"{bundle.name:6s} {name:10s} frames={row['n_frames']:3d} mask={row['mask_source']:6s} "
                  f"grid={1000 * row.get('V_grid', np.nan):6.2f}L plane={1000 * row.get('V_plane', np.nan):6.2f}L "
                  f"H={100 * row.get('H', np.nan):5.1f}cm tilt={row.get('tilt_deg', np.nan):4.1f} {row['seconds']}s",
                  flush=True)
    return pd.DataFrame(rows), keep


def step_frame(vol, gt, fusion, vcol):
    d = vol[vol["fusion"] == fusion].set_index("ffb")
    df = pd.DataFrame({"V": d[vcol], "H": d.get("H"), "group": d["layout"]}).dropna(subset=["V"])
    df["mass"] = gt.loc[df.index, "Actual_Mass_kg"].to_numpy()
    return df


def cv_summary(df, model):
    cv = ev.exhaustive_cv(df, model, 4)
    ex = cv[~cv["ffb"].isin(EXCLUDED)]
    allp = ev.metrics(ex["actual"], ex["pred"])
    per = ex.dropna().groupby("ffb").agg(actual=("actual", "first"), pred=("pred", "mean"))
    return dict(model=model.name, n_pred=int(ex["pred"].notna().sum()), n_missing=int(ex["pred"].isna().sum()),
                mae=allp["mae"], mape=allp["mape"], pearson_r2_all=allp["pearson_r2"], r2_all=allp["r2"],
                pearson_r2_per_ffb=ev.metrics(per["actual"], per["pred"])["pearson_r2"])


def parity(s0):
    ratio = pd.Series(REPORT_A_L).reindex(s0.index) / (1000 * s0["V"])
    tab = ratio.groupby(s0["group"]).agg(["count", "mean", "std"])
    tab["cv_pct"] = 100 * tab["std"] / tab["mean"]
    model = MODELS[0]
    pred = model.predict(model.fit(s0), s0)
    keep = ~s0.index.isin(EXCLUDED)
    return dict(ratio_by_group=tab.reset_index().to_dict("records"),
                insample_per_session=ev.metrics(s0["mass"][keep], pred[keep]),
                cv_global=cv_summary(s0, MODELS[1]), cv_per_session=cv_summary(s0, MODELS[0]))


def ladder(vol, gt):
    rows, errs, preds = [], {}, {}
    for step, fusion, vcol in STEPS:
        df = step_frame(vol, gt, fusion, vcol).drop(index=EXCLUDED, errors="ignore")
        for model in MODELS:
            if df[list(model.features)].isna().any().any():
                continue
            pred = ev.loo(df, model)
            err = (pred - df["mass"]).abs()
            _, lo, hi = ev.bootstrap_ci(err)
            row = dict(step=step, model=model.name, primary=model.name in PRIMARY_MODELS) | ev.metrics(df["mass"], pred)
            row |= dict(mae_lo=lo, mae_hi=hi)
            base = errs.get(("S0_notebook", model.name))
            if base is not None:
                common = err.index.intersection(base.index)
                row |= {f"vs_S0_{k}": v for k, v in ev.paired_bootstrap(err[common], base[common]).items()}
            errs[(step, model.name)], preds[(step, model.name)] = err, pred
            rows.append(row)
    return pd.DataFrame(rows), preds


def intervals(vol, gt):
    rows = []
    for step, fusion, vcol in (STEPS[0], STEPS[-1]):
        df = step_frame(vol, gt, fusion, vcol).drop(index=EXCLUDED, errors="ignore")
        for model in MODELS[:2]:
            jk = ev.jackknife_plus(df, model, alpha=0.1)
            rows.append(dict(step=step, model=model.name, coverage=float(jk["covered"].mean()),
                             median_width_kg=float((jk["hi"] - jk["lo"]).median())))
    return rows


def caliper(gt, ffbs):
    g = gt.assign(V=gt["ellipsoid_L"], mass=gt["Actual_Mass_kg"])
    model = ev.Model("ellipsoid aV+b", ("V",), intercept=True)
    df = g.loc[ffbs]
    return dict(loo_same_10=ev.metrics(df["mass"], ev.loo(df, model)),
                fit_other_40=ev.metrics(df["mass"], model.predict(model.fit(g.drop(index=ffbs)), df)))


def synthetic_check(vol, keep):
    D = float(np.nanmedian(vol.loc[vol["fusion"] == "nan_blur", "tarp_depth"]))
    shape = dict(D=D, a=0.17, b=0.20, c=0.13)
    rows = []
    for ffb in ("FFB10", "FFB31"):
        K = keep[ffb][0]
        for resting in (False, True):
            dref, href, Kc = synthetic.render(K, **shape, resting=resting, s=8, crop=True)
            ref = volume.frustum_volume(dref, href, Kc, D)
            depth, hit, _ = synthetic.render(K, **shape, resting=resting)
            depth = depth.astype(np.float32)
            ring = cv2.dilate(hit.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
            rows.append(dict(
                depth_grid=f"{K.width}x{K.height}", shape="resting ellipsoid" if resting else "dome", tarp_m=round(D, 3),
                ref_over_true=ref / synthetic.ellipsoid_volume(0.17, 0.20, 0.13, resting),
                frustum_over_ref=volume.frustum_volume(depth, hit, K, D) / ref,
                grid_plane_ref_over_ref=volume.grid_volume(volume.project(depth, ring, K))[0] / ref,
                grid_tight_mask_over_ref=volume.grid_volume(volume.project(depth, hit, K))[0] / ref))
    return rows


def sam3_eval(keep, gt):
    from ffb.semantic import Sam3Segmenter

    seg = Sam3Segmenter()
    try:
        seg.load()
    except Exception as e:  # gated weights, missing transformers>=5, or no network
        return dict(status=f"blocked: {type(e).__name__}: {e}"[:400])
    rows = []
    for ffb, (K, depth_mask, rgb, fused) in keep.items():
        for mode in ("text", "text+depth_box"):
            box = None
            if mode != "text" and depth_mask is not None:
                ys, xs = np.nonzero(depth_mask)
                sx, sy = rgb.shape[1] / fused.shape[1], rgb.shape[0] / fused.shape[0]
                box = [xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx, (ys.max() + 1) * sy]
            t0 = time.time()
            mask, score = seg.segment(rgb, box)
            row = dict(ffb=ffb, mode=mode, score=score, seconds=round(time.time() - t0, 1))
            if mask is not None:
                mask = cv2.resize(mask.astype(np.uint8), fused.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = (fused > 0.1) & (fused < 10.0)
                z_front = float(np.percentile(fused[valid], 5))
                clipped = volume.clip_depth(fused, z_front, segment.auto_margin(fused, z_front, valid) + 0.05)
                n, d, _ = volume.fit_plane(volume.tarp_points(fused, mask, K))
                row.update(V_grid=volume.grid_volume(volume.project(clipped, mask, K))[0],
                           V_plane=volume.plane_volume(fused, mask, K, (n, d))[0], layout=keep[ffb][0].width)
                if depth_mask is not None:
                    row["iou_vs_depth_mask"] = float((mask & depth_mask).sum() / (mask | depth_mask).sum())
            rows.append(row)
            print(f"SAM3 {ffb} {mode}: score={score:.2f} {row.get('V_plane', np.nan) * 1000:.2f}L", flush=True)
    return dict(status="ok", rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--cache", default=str(ROOT / "fused_cache"))
    ap.add_argument("--sam3", action="store_true", help="also run SAM 3 (needs transformers>=5 and facebook/sam3 access)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, gt = time.time(), ground_truth()

    vol, keep = run_bundles(args.data, args.cache)
    vol.to_csv(out / "cpu_eval_volumes.csv", index=False)
    lad, preds = ladder(vol, gt)
    lad.to_csv(out / "cpu_eval_ladder.csv", index=False)
    ffbs = [f for f in step_frame(vol, gt, "notebook", "V_grid").index if f not in EXCLUDED]
    per_ffb = pd.DataFrame({"actual_kg": gt.loc[ffbs, "Actual_Mass_kg"]})
    for (step, model), p in preds.items():
        if step in ("S0_notebook", "S4_tarp_plane") and model in PRIMARY_MODELS:
            per_ffb[f"{step}:{model}"] = p
    per_ffb.round(3).to_csv(out / "cpu_eval_per_ffb_loo.csv")

    summary = dict(parity=parity(step_frame(vol, gt, "notebook", "V_grid")), jackknife_plus=intervals(vol, gt),
                   caliper=caliper(gt, ffbs), synthetic=synthetic_check(vol, keep))
    summary["cv_S4_tarp_plane"] = [cv_summary(step_frame(vol, gt, "nan_blur", "V_plane"), m) for m in MODELS[:2]]
    summary["sam3"] = sam3_eval(keep, gt) if args.sam3 else dict(status="not requested")
    summary["runtime_s"] = round(time.time() - t0, 1)
    (out / "cpu_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(lad[lad["primary"]].round(3).to_string(index=False))
    print(json.dumps({k: summary[k] for k in ("parity", "caliper", "jackknife_plus", "runtime_s")}, indent=1, default=float))


if __name__ == "__main__":
    main()
