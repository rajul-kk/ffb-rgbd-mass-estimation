import hashlib
import json
import warnings

import cv2
import numpy as np

from ffb.fusion import FusionConfig, frame_changes, nan_median3, nanmedian_rows, steady_prefix, to_metric


def test_nan_median3_matches_bruteforce():
    rng = np.random.default_rng(0)
    a = rng.uniform(1, 2, (17, 23)).astype(np.float32)
    a[rng.random(a.shape) < 0.35] = np.nan
    a[:3, :3] = np.nan
    got, padded = nan_median3(a), np.pad(a, 1, constant_values=np.nan)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            window = padded[y:y + 3, x:x + 3]
            if np.isnan(window).all():
                assert np.isnan(got[y, x])
            else:
                assert np.isclose(got[y, x], np.nanmedian(window))


def test_blur_before_masking_pulls_edge_pixels_nearer():
    # Centre pixel sees 3 invalid, 2 near (1.0 m) and 4 far (1.3 m) neighbours.
    depth = np.pad(np.array([[0, 0, 0], [1000, 1000, 1300], [1300, 1300, 1300]], np.uint16), 2, mode="edge")
    assert np.isclose(to_metric(depth, 0.001, "blur_then_mask")[3, 3], 1.0)
    assert np.isclose(to_metric(depth, 0.001, "mask_then_nanmedian")[3, 3], 1.3)


def test_median_filter_commutes_with_disparity():
    z = np.random.default_rng(1).uniform(0.8, 2.0, (40, 60)).astype(np.float32)
    assert np.allclose(1 / cv2.medianBlur(1 / z, 3), cv2.medianBlur(z, 3), rtol=1e-5)


def test_nanmedian_rows_matches_single_call():
    rng = np.random.default_rng(2)
    stack = rng.uniform(1, 2, (15, 130, 20)).astype(np.float32)
    stack[rng.random(stack.shape) < 0.4] = np.nan
    stack[:, 0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
    assert np.allclose(nanmedian_rows(stack, rows=7), expected, rtol=0, atol=0)


def test_steady_prefix_stops_at_first_disturbed_frame():
    base = np.random.default_rng(3).integers(1200, 1500, (48, 64)).astype(np.uint16)
    depths = [base.copy() for _ in range(30)]
    for d in depths[12:20]:
        d[10:40, 10:50] = 900  # something 30-60 cm nearer covers ~40% of the view
    changed = frame_changes(depths, 0.001, ref_frames=8, tol_m=0.02)
    assert changed[:12].max() == 0 and changed[12] > 0.3
    assert steady_prefix(changed, 0.08) == 12


def test_all_frames_cache_keys_match_v2_names():
    old = {"max_frames": 120, "reader": "notebook", "spatial": "blur_then_mask"}
    expected = hashlib.sha1(json.dumps(old, sort_keys=True).encode()).hexdigest()[:10]
    assert FusionConfig(reader="notebook").key() == expected
    assert FusionConfig(frames="steady").key() != FusionConfig().key()
