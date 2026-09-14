import numpy as np
import pytest

from ffb import synthetic, volume
from ffb.camera import Intrinsics

K848 = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)
K1280 = Intrinsics(646.5, 646.5, 640.0, 360.0, 1280, 720)
ABC = dict(a=0.17, b=0.20, c=0.13)


def reference(D, resting):
    depth, hit, Kc = synthetic.render(K1280, D, **ABC, resting=resting, s=8, crop=True)
    return volume.frustum_volume(depth, hit, Kc, D)


@pytest.mark.parametrize("D", [1.0, 1.3, 1.6])
@pytest.mark.parametrize("resting", [False, True])
@pytest.mark.parametrize("K", [K848, K1280], ids=["848", "1280"])
def test_frustum_volume_is_resolution_independent(K, resting, D):
    depth, hit, _ = synthetic.render(K, D, **ABC, resting=resting)
    assert volume.frustum_volume(depth.astype(np.float32), hit, K, D) == pytest.approx(reference(D, resting), rel=0.005)


def test_dome_reference_matches_analytic_volume():
    assert reference(1.3, False) == pytest.approx(synthetic.ellipsoid_volume(**ABC, resting=False), rel=0.005)


def test_resting_ellipsoid_is_overstated_not_halved_by_top_view():
    # The volume under the visible surface already exceeds the true volume, so mirroring (x2) is wrong.
    ratio = reference(1.3, True) / synthetic.ellipsoid_volume(**ABC, resting=True)
    assert 1.3 < ratio < 1.4


def test_notebook_grid_volume_loses_cells_when_pixels_are_coarser_than_grid():
    depth, hit, _ = synthetic.render(K848, 1.3, **ABC, resting=False)
    depth = depth.astype(np.float32)
    ring = np.zeros_like(hit)
    ys, xs = np.nonzero(hit)
    ring[ys.min() - 30:ys.max() + 30, xs.min() - 30:xs.max() + 30] = True
    grid, z_ref = volume.grid_volume(volume.project(depth, ring, K848))
    assert z_ref == pytest.approx(1.3, abs=1e-6)
    assert grid / reference(1.3, False) < 0.6
