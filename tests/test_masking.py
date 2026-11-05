import numpy as np
import pytest

from ffb import masking, synthetic, volume
from ffb.camera import Intrinsics
from ffb.masking import MaskRequest, MaskResult

K = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)
D = 1.3


@pytest.fixture(scope="module")
def scene():
    depth, hit, _ = synthetic.render(K, D, 0.17, 0.20, 0.13, resting=True)
    depth = depth.astype(np.float32)
    plane = volume.fit_plane(volume.tarp_points(depth, hit, K))[:2]
    return depth, hit, plane


def request(scene, **kw):
    depth, hit, plane = scene
    return MaskRequest(name="syn", depth=depth, K=K, plane=plane, **kw)


class FixedProvider:
    """A third-party segmenter written against the interface only."""
    name, needs_rgb = "fixed", False

    def __init__(self, mask):
        self.mask = mask

    def __call__(self, req):
        return MaskResult(self.mask, 0.9, {"note": "test"})


def test_any_provider_feeds_the_unchanged_mass_code(scene):
    depth, hit, plane = scene
    req = request(scene)
    v = masking.volume_from_result(req, FixedProvider(hit)(req))
    assert v == pytest.approx(volume.plane_volume(depth, hit, K, plane)[0])
    assert v == pytest.approx(volume.frustum_volume(depth, hit, K, D), rel=1e-4)


def test_failed_detection_or_missing_plane_gives_none(scene):
    req = request(scene)
    assert masking.volume_from_result(req, MaskResult(None)) is None
    no_plane = MaskRequest(name="x", depth=req.depth, K=K, plane=None)
    assert masking.volume_from_result(no_plane, MaskResult(scene[1])) is None


def test_mask_off_the_depth_grid_is_an_error(scene):
    with pytest.raises(ValueError, match="depth grid"):
        masking.volume_from_result(request(scene), MaskResult(np.ones((10, 10), bool)))


def test_control_provider_recovers_the_synthetic_bunch(scene):
    res = masking.make("depth_plane")(request(scene))
    assert res.mask is not None
    assert masking.iou(res.mask, scene[1]) > 0.8


def test_registry_rejects_duplicates_and_unknown_names():
    with pytest.raises(KeyError):
        masking.make("nope")
    with pytest.raises(ValueError):
        masking.register("depth_plane")(lambda: None)


def test_registered_provider_is_constructible_by_name():
    @masking.register("_test_plugin")
    def factory():
        return FixedProvider(None)
    try:
        assert masking.make("_test_plugin").name == "fixed"
    finally:
        del masking.REGISTRY["_test_plugin"]


def test_postprocess_fills_holes_and_keeps_largest_component():
    m = np.zeros((60, 60), bool)
    m[10:40, 10:40] = True
    m[20:25, 20:25] = False      # hole
    m[50:54, 50:54] = True       # stray blob
    out = masking.postprocess(m)
    assert out[22, 22] and not out[51, 51]
    assert out.sum() == 30 * 30


def test_postprocess_empty_or_tiny_is_none():
    assert masking.postprocess(None) is None
    assert masking.postprocess(np.zeros((30, 30), bool)) is None
    tiny = np.zeros((30, 30), bool)
    tiny[5:7, 5:7] = True
    assert masking.postprocess(tiny) is None


def test_iou_and_bbox():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[2:7] = True
    assert masking.iou(a, b) == pytest.approx(3 / 7)
    assert np.isnan(masking.iou(a, None))
    assert masking.iou(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 0.0
    assert masking.bbox(b) == (0, 2, 10, 7)
    assert masking.bbox(np.zeros((3, 3), bool)) is None


def test_height_image_is_bright_on_the_bunch_and_dark_on_the_tarp(scene):
    depth, hit, plane = scene
    img = masking.height_image(depth, K, plane)
    assert img.shape == depth.shape + (3,) and img.dtype == np.uint8
    assert img[hit].mean() > 100 and img[~hit].mean() < 5
