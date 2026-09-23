import cv2
import numpy as np
import pytest

from ffb import segment

H, W = 240, 320


def scene(radius=40, z_tarp=1.5, z_blob=1.3, cy=None, cx=None):
    """A flat tarp with a closer, saturated circular bunch on it; matches depth_foreground_mask's assumptions."""
    cy, cx = H // 2 if cy is None else cy, W // 2 if cx is None else cx
    depth = np.full((H, W), z_tarp, np.float32)
    mask = np.zeros((H, W), bool)
    ys, xs = np.ogrid[:H, :W]
    mask[(ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2] = True
    depth[mask] = z_blob
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 1], hsv[..., 2] = 10, 210          # tarp: low saturation, bright
    hsv[mask] = [0, 200, 140]                    # bunch: saturated, darker (below the 160 default v_max)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return depth, rgb, mask


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_depth_foreground_mask_finds_the_bunch():
    depth, rgb, truth = scene()
    mask, box, margin = segment.depth_foreground_mask(depth, rgb=rgb)
    assert mask is not None
    assert iou(mask, truth) > 0.9
    assert margin == pytest.approx(0.15, abs=0.02)  # z_tarp - z_blob - 0.05


def test_depth_foreground_mask_none_when_depth_is_all_invalid():
    depth = np.zeros((H, W), np.float32)
    mask, box, margin = segment.depth_foreground_mask(depth)
    assert mask is None and box is None and margin is None


def test_depth_foreground_mask_rejects_a_blob_above_max_area_frac():
    depth, rgb, _ = scene(radius=140)  # ~96% of the frame, far above max_area_frac=0.35
    mask, box, margin = segment.depth_foreground_mask(depth, rgb=rgb)
    assert mask is None


def test_depth_foreground_mask_rejects_a_blob_off_to_one_side():
    depth, rgb, _ = scene(radius=25, cy=20, cx=20)  # small and in a corner, far from centre
    mask, box, margin = segment.depth_foreground_mask(depth, rgb=rgb)
    assert mask is None


def test_border_hsv_stats_reads_tarp_not_bunch():
    _, rgb, _ = scene()
    s_min, v_max = segment.border_hsv_stats(rgb)
    assert s_min < 40    # tarp saturation (~10) should pull this well below the bunch's 200
    assert v_max > 200   # tarp value (~210) should pull this above the bunch's 170


def test_auto_margin_matches_the_true_protrusion():
    depth, _, truth = scene()
    valid = depth > 0.1
    z_front = float(np.percentile(depth[valid], 5))
    margin = segment.auto_margin(depth, z_front, valid)
    assert margin == pytest.approx(0.15, abs=0.02)


def test_expand_mask_bbox_grows_a_tight_mask_without_reaching_the_tarp():
    depth, rgb, truth = scene(radius=40)
    shrunk = np.zeros_like(truth)
    ys, xs = np.nonzero(truth)
    cy, cx = int(ys.mean()), int(xs.mean())
    shrunk[cy - 10:cy + 10, cx - 10:cx + 10] = True
    z_front = float(np.percentile(depth[truth], 5))
    expanded = segment.expand_mask_bbox(shrunk, depth, rgb, z_front, z_window=0.2, pad_px=30)
    assert expanded.sum() > shrunk.sum()
    assert iou(expanded, truth) > 0.8


def test_extract_mask_end_to_end():
    depth, rgb, truth = scene()
    mask, z_front, z_window, source = segment.extract_mask(depth, rgb, width=1280)
    assert source == "depth"
    assert iou(mask, truth) > 0.85
    assert z_front == pytest.approx(1.3, abs=0.02)


def test_extract_mask_falls_back_when_the_depth_mask_rejects_the_blob():
    # off-centre blob: depth is real (z_front is computable), but depth_foreground_mask rejects it
    depth, rgb, _ = scene(radius=25, cy=20, cx=20)
    called = {}

    def fallback(img):
        called["ran"] = True
        m = np.zeros((H, W), bool)
        m[50:100, 50:100] = True
        return m

    mask, z_front, z_window, source = segment.extract_mask(depth, rgb, width=1280, fallback=fallback)
    assert called.get("ran") and source == "semantic"
    assert mask.sum() == 2500
    # z_front is the 5th percentile over the whole frame; this blob is too small a share to pull it down
    assert z_front == pytest.approx(1.5, abs=0.02)


def test_extract_mask_fails_cleanly_with_no_valid_depth_anywhere():
    # previously crashed with a numpy IndexError instead of returning "failed"
    depth = np.zeros((H, W), np.float32)
    rgb = np.zeros((H, W, 3), np.uint8)
    called = {}

    def fallback(img):
        called["ran"] = True
        return None

    mask, z_front, z_window, source = segment.extract_mask(depth, rgb, width=1280, fallback=fallback)
    assert mask is None and z_front is None and source == "failed"
    assert "ran" not in called  # no usable depth means no usable volume either way; fallback is skipped


def test_plane_height_mask_finds_the_bunch_without_colour():
    from ffb import synthetic
    from ffb.camera import Intrinsics
    K = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)
    depth, hit, _ = synthetic.render(K, 1.55, a=0.17, b=0.20, c=0.13, resting=False)
    mask, (n, d) = segment.plane_height_mask(depth.astype(np.float32), K)
    assert iou(mask, hit) > 0.85
    assert -d / n[2] == pytest.approx(1.55, abs=0.003)


def test_plane_height_mask_ignores_an_rgb_shift_by_construction():
    # the session 2 failure: colour recorded minutes apart and unregistered; this mask never reads colour
    from ffb import synthetic
    from ffb.camera import Intrinsics
    K = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)
    depth, hit, _ = synthetic.render(K, 1.55, a=0.17, b=0.20, c=0.13, resting=False)
    m1, _ = segment.plane_height_mask(depth.astype(np.float32), K)
    assert m1 is not None and m1.sum() > 0.85 * hit.sum()


def test_plane_height_mask_none_without_valid_depth():
    from ffb.camera import Intrinsics
    K = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)
    assert segment.plane_height_mask(np.zeros((480, 848), np.float32), K) == (None, None)
