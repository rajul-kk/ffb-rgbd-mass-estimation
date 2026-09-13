"""SAM 3 concept segmentation, replacing the Grounding DINO -> SAM 2 -> YOLO-World cascade."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def pick_instance(masks, scores, shape, max_centre_dist: float = 0.75):
    """Highest-scoring non-empty instance whose centroid is within max_centre_dist of the half-diagonal."""
    h, w = shape
    half = np.hypot(w / 2, h / 2)
    best_mask, best_score = None, 0.0
    for m, score in zip(masks, scores):
        m = np.asarray(m, bool)
        if m.shape != (h, w) or not m.any():
            continue
        ys, xs = np.nonzero(m)
        if np.hypot(xs.mean() - w / 2, ys.mean() - h / 2) > max_centre_dist * half:
            continue
        if float(score) > best_score:
            best_mask, best_score = m, float(score)
    return best_mask, best_score


class Sam3Segmenter:
    """Text prompt, optionally with a positive box in the same image; loads lazily on first use."""

    def __init__(self, model_id: str = "facebook/sam3", prompt: str = "oil palm fruit bunch",
                 device: str = "cpu", threshold: float = 0.3, mask_threshold: float = 0.5):
        self.model_id, self.prompt, self.device = model_id, prompt, device
        self.threshold, self.mask_threshold = threshold, mask_threshold
        self._model = self._processor = None

    def load(self):
        if self._model is not None:
            return
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise RuntimeError("SAM 3 needs transformers>=5 with Sam3Model") from e
        self._processor = Sam3Processor.from_pretrained(self.model_id)
        self._model = Sam3Model.from_pretrained(self.model_id).to(self.device).eval()

    def segment(self, rgb: np.ndarray, box: Optional[Sequence[float]] = None):
        """Return (mask at rgb resolution or None, score)."""
        import torch
        from PIL import Image

        self.load()
        kwargs = dict(images=Image.fromarray(rgb), text=self.prompt, return_tensors="pt")
        if box is not None:
            kwargs.update(input_boxes=[[[float(v) for v in box]]], input_boxes_labels=[[1]])
        inputs = self._processor(**kwargs).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        res = self._processor.post_process_instance_segmentation(
            outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist())[0]
        masks = [m.cpu().numpy() for m in res["masks"]]
        return pick_instance(masks, res["scores"].cpu().numpy(), rgb.shape[:2])

    def __call__(self, rgb: np.ndarray):
        return self.segment(rgb)[0]
