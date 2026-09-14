import numpy as np
import pytest

from ffb import synthetic, volume
from ffb.camera import Intrinsics

K = Intrinsics(432.1, 432.1, 424.0, 240.0, 848, 480)


def test_fit_plane_recovers_tilted_tarp_despite_bunch_outliers():
    rng = np.random.default_rng(0)
    n_true = np.array([0.05, -0.08, -1.0]) / np.linalg.norm([0.05, -0.08, -1.0])
    d_true = -1.3 * n_true[2]
    xy = rng.uniform(-0.5, 0.5, (6000, 2))
    z = -(n_true[0] * xy[:, 0] + n_true[1] * xy[:, 1] + d_true) / n_true[2]
    pts = np.column_stack([xy, z + rng.normal(0, 0.003, len(z))])
    near = rng.random(len(z)) < 0.3
    pts[near, 2] -= rng.uniform(0.03, 0.3, near.sum())
    n, d, _ = volume.fit_plane(pts)
    assert np.degrees(np.arccos(np.clip(n @ n_true, -1, 1))) < 0.5
    assert d == pytest.approx(d_true, abs=0.003)


def test_plane_volume_matches_frustum_volume_on_flat_tarp():
    depth, hit, _ = synthetic.render(K, 1.3, 0.17, 0.20, 0.13, resting=True)
    depth = depth.astype(np.float32)
    plane = volume.fit_plane(volume.tarp_points(depth, hit, K))[:2]
    v_plane, height, _ = volume.plane_volume(depth, hit, K, plane)
    assert v_plane == pytest.approx(volume.frustum_volume(depth, hit, K, 1.3), rel=1e-4)
    assert height == pytest.approx(0.26, abs=0.02)
