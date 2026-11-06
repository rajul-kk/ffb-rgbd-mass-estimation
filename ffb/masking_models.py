"""Model-backed mask providers. torch, transformers and ultralytics are imported lazily, never at module import.

Every provider here uses fixed, pre-registered prompts and thresholds (docs/PROTOCOL_zeroshot.md) and returns the
raw mask on the depth grid; masking.postprocess is applied by the caller, identically for all of them.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import masking
from .masking import MaskRequest, MaskResult

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-tiny"
YOLOW_WEIGHTS = "yolov8s-worldv2.pt"
TEXT_QUERY = "oil palm fruit bunch"      # Grounding DINO wants lowercase text ending in a full stop
GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD = 0.30, 0.25
YOLOW_CONF = 0.10
YOLOW_CONF_LOW = 0.02                    # sensitivity arm, added after a smoke test found no detections at 0.10
CENTRE_POINT = (0.5, 0.5)                # fractions of (width, height)
CENTRE_BOX = (0.25, 0.25, 0.75, 0.75)    # fractions of (x0, y0, x1, y1)

DEVICE = "cpu"  # set once by the driver via set_device(); models are moved there on first load


def resolve_device(requested: str = "auto", cuda_available: Optional[bool] = None) -> str:
    """'auto' -> 'cuda' when a GPU is visible, else 'cpu'. An explicit 'cuda' without a GPU is an error, not a silent fallback."""
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("device 'cuda' requested but no GPU is visible")
    if requested not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {requested!r}")
    return requested


def set_device(device: str) -> None:
    global DEVICE
    DEVICE = device


def _to(inputs, device):
    """Move a processor's BatchFeature/dict of tensors to device, leaving non-tensors alone."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


class _Sam2:
    """Shared lazily loaded SAM 2 (transformers Sam2Model); providers reuse one instance."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self, model_id: str = SAM2_ID):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).eval().to(DEVICE)

    def segment(self, rgb: np.ndarray, box=None, point=None):
        """Return (mask bool at rgb resolution, predicted IoU). Box: one mask. Point: best of three by predicted IoU."""
        from PIL import Image
        kw = dict(images=Image.fromarray(rgb), return_tensors="pt")
        if box is not None:
            kw["input_boxes"] = [[[float(v) for v in box]]]
        else:
            kw["input_points"] = [[[[float(point[0]), float(point[1])]]]]
            kw["input_labels"] = [[[1]]]
        inputs = self.processor(**kw)
        with self.torch.no_grad():
            out = self.model(**_to(inputs, DEVICE), multimask_output=box is None)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # (1, n, H, W)
        ious = out.iou_scores[0, 0].detach().cpu().numpy()
        best = int(np.argmax(ious))
        return masks[0, best].numpy().astype(bool), float(ious[best])


def _no_rgb(req: MaskRequest) -> Optional[MaskResult]:
    return None if req.rgb is not None else MaskResult(None, None, {"reason": "no registered rgb"})


@masking.register("gdino_sam2")
class GroundingDinoSam2:
    name, needs_rgb = "gdino_sam2", True

    def __init__(self):
        self._m = self._p = None

    def _load(self):
        if self._m is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self._p = AutoProcessor.from_pretrained(GDINO_ID)
            self._m = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).eval().to(DEVICE)

    def __call__(self, req: MaskRequest) -> MaskResult:
        if (r := _no_rgb(req)) is not None:
            return r
        import torch
        from PIL import Image
        self._load()
        t0 = time.time()
        h, w = req.rgb.shape[:2]
        inputs = self._p(images=Image.fromarray(req.rgb), text=TEXT_QUERY + ".", return_tensors="pt")
        with torch.no_grad():
            out = self._m(**_to(inputs, DEVICE))
        res = self._p.post_process_grounded_object_detection(
            out, inputs["input_ids"].to(DEVICE), threshold=GDINO_BOX_THRESHOLD, text_threshold=GDINO_TEXT_THRESHOLD,
            target_sizes=[(h, w)])[0]
        box, score = masking.pick_box(res["boxes"].cpu().tolist(), res["scores"].cpu().tolist(), (h, w))
        meta = dict(n_detections=len(res["scores"]), box=box, seconds_detect=round(time.time() - t0, 1))
        if box is None:
            return MaskResult(None, None, meta | {"reason": "no detection"})
        mask, iou_pred = _Sam2.get().segment(req.rgb, box=box)
        return MaskResult(mask, score, meta | {"sam_iou_pred": iou_pred, "seconds": round(time.time() - t0, 1)})


@masking.register("yoloworld_sam2")
class YoloWorldSam2:
    name, needs_rgb = "yoloworld_sam2", True

    def __init__(self, weights: str = YOLOW_WEIGHTS, conf: float = YOLOW_CONF):
        self.weights, self.conf, self._m = weights, conf, None

    def _load(self):
        if self._m is None:
            from ultralytics import YOLOWorld
            self._m = YOLOWorld(self.weights)
            self._m.set_classes([TEXT_QUERY])

    def __call__(self, req: MaskRequest) -> MaskResult:
        if (r := _no_rgb(req)) is not None:
            return r
        self._load()
        t0 = time.time()
        h, w = req.rgb.shape[:2]
        det = self._m.predict(req.rgb[..., ::-1], conf=self.conf, verbose=False, device=DEVICE)[0].boxes  # BGR input
        box, score = masking.pick_box(det.xyxy.cpu().tolist(), det.conf.cpu().tolist(), (h, w))
        meta = dict(n_detections=len(det), box=box, seconds_detect=round(time.time() - t0, 1))
        if box is None:
            return MaskResult(None, None, meta | {"reason": "no detection"})
        mask, iou_pred = _Sam2.get().segment(req.rgb, box=box)
        return MaskResult(mask, score, meta | {"sam_iou_pred": iou_pred, "seconds": round(time.time() - t0, 1)})


@masking.register("yoloworld_lowconf_sam2")
class YoloWorldLowConfSam2(YoloWorldSam2):
    """Sensitivity arm: same model and prompt, confidence threshold 0.02 instead of 0.10."""
    name = "yoloworld_lowconf_sam2"

    def __init__(self):
        super().__init__(conf=YOLOW_CONF_LOW)


@masking.register("sam2_point")
class Sam2Point:
    """Label-free: one positive point at the image centre."""
    name, needs_rgb = "sam2_point", True

    def __call__(self, req: MaskRequest) -> MaskResult:
        if (r := _no_rgb(req)) is not None:
            return r
        h, w = req.rgb.shape[:2]
        pt = (CENTRE_POINT[0] * w, CENTRE_POINT[1] * h)
        mask, iou_pred = _Sam2.get().segment(req.rgb, point=pt)
        return MaskResult(mask, iou_pred, {"point": pt})


@masking.register("sam2_box")
class Sam2Box:
    """Label-free: the central box covering the middle half of each axis."""
    name, needs_rgb = "sam2_box", True

    def __call__(self, req: MaskRequest) -> MaskResult:
        if (r := _no_rgb(req)) is not None:
            return r
        h, w = req.rgb.shape[:2]
        box = (CENTRE_BOX[0] * w, CENTRE_BOX[1] * h, CENTRE_BOX[2] * w, CENTRE_BOX[3] * h)
        mask, iou_pred = _Sam2.get().segment(req.rgb, box=box)
        return MaskResult(mask, iou_pred, {"box": box})


@masking.register("sam2_oracle_box")
class Sam2OracleBox:
    """DIAGNOSTIC, not zero-shot: SAM 2 prompted with the control mask's bounding box. Upper bound for localisation."""
    name, needs_rgb = "sam2_oracle_box", True

    def __call__(self, req: MaskRequest) -> MaskResult:
        if (r := _no_rgb(req)) is not None:
            return r
        box = None if req.ref_mask is None else masking.bbox(req.ref_mask)
        if box is None:
            return MaskResult(None, None, {"reason": "no control mask"})
        mask, iou_pred = _Sam2.get().segment(req.rgb, box=box)
        return MaskResult(mask, iou_pred, {"box": box})
