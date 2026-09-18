"""Experiment steps shared by scripts/cpu_eval.py and the v3 notebook."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import evaluate as ev
from . import io, segment, synthetic, volume
from .fusion import FusionConfig, fuse

EXCLUDED = ["FFB18"]
BASE_MODELS = [
    ev.Model("per_session_k", ("V",), by_group=True),
    ev.Model("global_k", ("V",)),
    ev.Model("aV+b", ("V",), intercept=True),
    ev.Model("aV+bH", ("V", "H")),
]
PRIMARY_MODELS = ("per_session_k", "global_k")


def ground_truth(root) -> pd.DataFrame:
    g = pd.read_csv(Path(root) / "ground_truth.csv")
    g.index = "FFB" + g["FFB_No"].astype(str)
    g["ellipsoid_L"] = np.pi / 6 * g["Width_cm"] * g["Length_cm"] * g["Thickness_cm"] / 1000
    return g


def measure(bundle, K, cfg: FusionConfig, cache):
    t0 = time.time()
    fused, rgb, n_frames, info = fuse(bundle, K, cfg, cache, return_info=True)
    mask, z_front, z_window, source = segment.extract_mask(fused, rgb, K.width)
    row = dict(n_read=info["n_read"], n_frames=n_frames, mask_source=source, z_front=z_front, z_window=z_window)
    if mask is not None:
        clipped = volume.clip_depth(fused, z_front, z_window)
        v_grid, z_ref = volume.grid_volume(volume.project(clipped, mask, K))
        pitch = max(0.002, volume.pixel_pitch(K, z_front + z_window))
        v_grid_pitch, _ = volume.grid_volume(volume.project(clipped, mask, K), grid_step=pitch)
        ring_ref = volume.ring_z_ref(fused, mask, K)
        v_grid_ring = volume.grid_volume(volume.project(clipped, mask, K), z_ref=ring_ref)[0] if ring_ref else np.nan
        v_grid_ring_pitch = (volume.grid_volume(volume.project(clipped, mask, K), grid_step=pitch, z_ref=ring_ref)[0]
                              if ring_ref else np.nan)
        row.update(V_grid=v_grid, z_ref=z_ref, V_frustum=volume.frustum_volume(clipped, mask, K, z_ref),
                   V_grid_pitch=v_grid_pitch, grid_pitch_mm=1000 * pitch,
                   V_grid_ring=v_grid_ring, V_grid_ring_pitch=v_grid_ring_pitch, ring_z_ref=ring_ref,
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


def run_bundles(data, fusions: dict, cache, keep_fusion=None, verbose=True):
    """Measure every FFB folder under every fusion; returns (table, {ffb: (K, mask, rgb, fused)} for keep_fusion)."""
    rows, keep = [], {}
    for folder in sorted(p for p in Path(data).iterdir() if p.is_dir() and p.name.startswith("FFB")):
        bundle = io.find_bundle(str(folder))
        for name, cfg in fusions.items():
            K = io.intrinsics(bundle, "depth" if cfg.reader == "native" else "auto")
            row, mask, rgb, fused = measure(bundle, K, cfg, cache)
            rows.append(dict(ffb=bundle.name, fusion=name, layout=bundle.layout, width=K.width, fx=round(K.fx, 1)) | row)
            if name == keep_fusion:
                keep[bundle.name] = (K, mask, rgb, fused)
            if verbose:
                print(f"{bundle.name:6s} {name:15s} frames={row['n_frames']:3d}/{row['n_read']:3d} mask={row['mask_source']:6s} "
                      f"grid={1000 * row.get('V_grid', np.nan):6.2f}L plane={1000 * row.get('V_plane', np.nan):6.2f}L "
                      f"H={100 * row.get('H', np.nan):5.1f}cm tilt={row.get('tilt_deg', np.nan):4.1f} {row['seconds']}s",
                      flush=True)
    return pd.DataFrame(rows), keep


def step_frame(vol, gt, fusion, vcol):
    d = vol[vol["fusion"] == fusion].set_index("ffb")
    df = pd.DataFrame({"V": d[vcol], "H": d.get("H"), "group": d["layout"]}).dropna(subset=["V"])
    df["mass"] = gt.loc[df.index, "Actual_Mass_kg"].to_numpy()
    return df


def cv_summary(df, model, excluded=EXCLUDED):
    cv = ev.exhaustive_cv(df, model, 4)
    ex = cv[~cv["ffb"].isin(excluded)]
    allp = ev.metrics(ex["actual"], ex["pred"])
    per = ex.dropna().groupby("ffb").agg(actual=("actual", "first"), pred=("pred", "mean"))
    return dict(model=model.name, n_pred=int(ex["pred"].notna().sum()), n_missing=int(ex["pred"].isna().sum()),
                mae=allp["mae"], mape=allp["mape"], pearson_r2_all=allp["pearson_r2"], r2_all=allp["r2"],
                pearson_r2_per_ffb=ev.metrics(per["actual"], per["pred"])["pearson_r2"])


def parity(s0, report_litres: dict, excluded=EXCLUDED):
    """Compare notebook-reader grid volumes with report.md litres, and recompute v2's headline metrics."""
    ratio = pd.Series(report_litres).reindex(s0.index) / (1000 * s0["V"])
    tab = ratio.groupby(s0["group"]).agg(["count", "mean", "std"])
    tab["cv_pct"] = 100 * tab["std"] / tab["mean"]
    model = BASE_MODELS[0]
    pred = model.predict(model.fit(s0), s0)
    keep = ~s0.index.isin(excluded)
    return dict(ratio_by_group=tab.reset_index().to_dict("records"),
                insample_per_session=ev.metrics(s0["mass"][keep], pred[keep]),
                cv_global=cv_summary(s0, BASE_MODELS[1], excluded),
                cv_per_session=cv_summary(s0, BASE_MODELS[0], excluded))


def ladder(vol, gt, steps, models=BASE_MODELS, primary=PRIMARY_MODELS, excluded=EXCLUDED):
    """LOO metrics per (step, model), with a paired bootstrap against the first step."""
    rows, errs, preds = [], {}, {}
    base_step = steps[0][0]
    for step, fusion, vcol in steps:
        df = step_frame(vol, gt, fusion, vcol).drop(index=excluded, errors="ignore")
        for model in models:
            if df[list(model.features)].isna().any().any():
                continue
            pred = ev.loo(df, model)
            err = (pred - df["mass"]).abs()
            _, lo, hi = ev.bootstrap_ci(err)
            row = dict(step=step, model=model.name, primary=model.name in primary) | ev.metrics(df["mass"], pred)
            row |= dict(mae_lo=lo, mae_hi=hi)
            base = errs.get((base_step, model.name))
            if base is not None:
                common = err.index.intersection(base.index)
                row |= {f"vs_S0_{k}": v for k, v in ev.paired_bootstrap(err[common], base[common]).items()}
            errs[(step, model.name)], preds[(step, model.name)] = err, pred
            rows.append(row)
    return pd.DataFrame(rows), preds


def intervals(vol, gt, steps, models=BASE_MODELS[:2], excluded=EXCLUDED):
    rows = []
    for step, fusion, vcol in steps:
        df = step_frame(vol, gt, fusion, vcol).drop(index=excluded, errors="ignore")
        for model in models:
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


def synthetic_check(tarp_depth: float, cameras: dict):
    """Grid and frustum volume against a supersampled reference for each {label: Intrinsics}."""
    shape = dict(D=tarp_depth, a=0.17, b=0.20, c=0.13)
    rows = []
    for label, K in cameras.items():
        for resting in (False, True):
            dref, href, Kc = synthetic.render(K, **shape, resting=resting, s=8, crop=True)
            ref = volume.frustum_volume(dref, href, Kc, tarp_depth)
            depth, hit, _ = synthetic.render(K, **shape, resting=resting)
            depth = depth.astype(np.float32)
            ring = cv2.dilate(hit.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
            rows.append(dict(
                depth_grid=label, shape="resting ellipsoid" if resting else "dome", tarp_m=round(tarp_depth, 3),
                ref_over_true=ref / synthetic.ellipsoid_volume(0.17, 0.20, 0.13, resting),
                frustum_over_ref=volume.frustum_volume(depth, hit, K, tarp_depth) / ref,
                grid_plane_ref_over_ref=volume.grid_volume(volume.project(depth, ring, K))[0] / ref,
                grid_tight_mask_over_ref=volume.grid_volume(volume.project(depth, hit, K))[0] / ref))
    return rows


def heldout_comparison(vol, gt, steps, models=BASE_MODELS, excluded=EXCLUDED, B: int = 20000):
    """MAE per (model, step, scheme) for LOO and every 4-train and 7-train split, paired against the first step.
    Split schemes train on excluded bunches but never score them, as v2 did; errors are per-bunch means."""
    rows = []
    for model in models:
        base = None
        for step, fusion, vcol in steps:
            df = step_frame(vol, gt, fusion, vcol)
            if df[list(model.features)].isna().any().any():
                continue
            scored = df.drop(index=excluded, errors="ignore")
            pred = ev.loo(scored, model)
            schemes = {"LOO": ((pred - scored["mass"]).abs(), ev.metrics(scored["mass"], pred)["pearson_r2"], 0)}
            for n_train in (4, 7):
                cv = ev.exhaustive_cv(df, model, n_train)
                cv = cv[~cv["ffb"].isin(excluded)]
                missing = int(cv["pred"].isna().sum())
                cv = cv.dropna()
                per = cv.assign(ae=(cv["pred"] - cv["actual"]).abs()).groupby("ffb").agg(
                    actual=("actual", "first"), pred=("pred", "mean"), ae=("ae", "mean"))
                schemes[f"{n_train}/{len(df) - n_train}"] = (per["ae"], ev.metrics(per["actual"], per["pred"])["pearson_r2"], missing)
            if base is None:
                base = schemes
            for scheme, (err, r2, missing) in schemes.items():
                row = dict(model=model.name, step=step, scheme=scheme, mae=float(err.mean()), per_bunch_r2=r2, missing=missing)
                if schemes is not base:
                    common = err.index.intersection(base[scheme][0].index)
                    pb = ev.paired_bootstrap(err[common], base[scheme][0][common], B=B)
                    row |= {f"vs_first_{k}": v for k, v in pb.items()}
                rows.append(row)
    return pd.DataFrame(rows)


def sam3_eval(keep):
    """SAM 3 masks (text, and text plus the depth-mask box) turned into volumes; reports why if unavailable."""
    from .semantic import Sam3Segmenter

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
                           V_plane=volume.plane_volume(fused, mask, K, (n, d))[0], layout=K.width)
                if depth_mask is not None:
                    row["iou_vs_depth_mask"] = float((mask & depth_mask).sum() / (mask | depth_mask).sum())
            rows.append(row)
            print(f"SAM3 {ffb} {mode}: score={score:.2f} {row.get('V_plane', np.nan) * 1000:.2f}L", flush=True)
    return dict(status="ok", rows=rows)
