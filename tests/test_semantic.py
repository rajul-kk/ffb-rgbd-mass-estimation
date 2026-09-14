import sys
import types

import numpy as np
import pytest

from ffb.semantic import Sam3Segmenter, pick_instance


def blob(shape, cy, cx, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2


def test_pick_instance_ignores_high_scoring_corner_detection():
    shape = (480, 640)
    mask, score = pick_instance([blob(shape, 20, 20, 15), blob(shape, 240, 320, 60)], [0.9, 0.6], shape)
    assert score == 0.6 and mask[240, 320]


def test_pick_instance_returns_none_without_candidates():
    assert pick_instance([np.zeros((10, 10), bool)], [0.8], (10, 10)) == (None, 0.0)


def test_segmenter_explains_missing_sam3_support(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    with pytest.raises(RuntimeError, match="transformers>=5"):
        Sam3Segmenter().load()
