"""Step 2 of the zero-shot benchmark: run mask providers over exported inputs and save raw masks as PNG.

The only step that needs torch. Runs on CPU or on Kaggle (same script). Outputs, per modality and provider:
  <out>/<modality>/<provider>/<FFB>.png   raw mask on the depth grid (absent = no detection)
  <out>/<modality>/<provider>/meta.json   per-bunch prompts, boxes, scores, seconds, failure reasons
  <out>/run_info.json                     library versions and settings
Modalities: rgb (colour image, registered bunches only) and hmap (height-above-plane image, all bunches).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ffb import masking, masking_models  # noqa: E402,F401  (import registers the providers)
from ffb.camera import Intrinsics  # noqa: E402

MODEL_PROVIDERS = ["gdino_sam2", "yoloworld_sam2", "yoloworld_lowconf_sam2", "sam2_point", "sam2_box", "sam2_oracle_box"]


def load_request(npz: Path, modality: str):
    d = np.load(npz)
    K = Intrinsics(*[float(v) for v in d["K"][:4]], int(d["K"][4]), int(d["K"][5]), float(d["K"][6]))
    plane = None if np.isnan(d["plane_d"]) else (d["plane_n"], float(d["plane_d"]))
    depth = d["fused"]
    if modality == "rgb":
        rgb = d["rgb"] if bool(d["rgb_registered"]) else None
    else:
        rgb = masking.height_image(depth, K, plane) if plane is not None else None
    ref = d["ref_mask"] if d["ref_mask"].any() else None
    return masking.MaskRequest(name=npz.stem, depth=depth, K=K, plane=plane, rgb=rgb, ref_mask=ref)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=ROOT / "zeroshot_cache" / "inputs")
    ap.add_argument("--out", type=Path, default=ROOT / "zeroshot_cache" / "masks")
    ap.add_argument("--providers", nargs="+", default=MODEL_PROVIDERS)
    ap.add_argument("--modalities", nargs="+", default=["rgb", "hmap"])
    ap.add_argument("--bunches", nargs="*", default=None)
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    files = sorted(a.inputs.glob("FFB*.npz"), key=lambda p: int(p.stem[3:]))
    if a.bunches:
        files = [f for f in files if f.stem in a.bunches]
    import torch, transformers
    info = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
                gdino=masking_models.GDINO_ID, sam2=masking_models.SAM2_ID, yolow=masking_models.YOLOW_WEIGHTS,
                text=masking_models.TEXT_QUERY, device="cpu", threads=torch.get_num_threads())
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_info.json").write_text(json.dumps(info, indent=2))
    for modality in a.modalities:
        for pname in a.providers:
            prov, meta = masking.make(pname), {}
            odir = a.out / modality / pname
            odir.mkdir(parents=True, exist_ok=True)
            for f in files:
                req = load_request(f, modality)
                t0 = time.time()
                try:
                    res = prov(req)
                except Exception as e:  # a provider crash is a failure of that bunch, recorded, not hidden
                    res = masking.MaskResult(None, None, {"reason": f"{type(e).__name__}: {e}"[:300]})
                sec = round(time.time() - t0, 1)
                meta[f.stem] = dict(score=res.score, seconds=sec, detected=res.mask is not None, **{
                    k: v for k, v in res.meta.items() if k != "seconds"})
                png = odir / f"{f.stem}.png"
                if res.mask is not None:
                    cv2.imwrite(str(png), res.mask.astype(np.uint8) * 255)
                elif png.exists():
                    png.unlink()
                print(f"{modality:4s} {pname:16s} {f.stem:6s} detected={res.mask is not None} {sec}s "
                      f"{res.meta.get('reason', '')}", flush=True)
            (odir / "meta.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    main()
