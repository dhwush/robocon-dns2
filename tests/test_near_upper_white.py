"""近处斜视上层白：深度孔洞、下层出画以及背景约束。"""
import numpy as np
import pytest
from cube.stack_look import coarse_bands, _upper_metric_face, _upper_white_by_color
from geom.transform import deproject_pixels
from ground.plane import Plane
from sim.scene import intrinsics_for

COLORS = [{'name': 'white', 'hsv_low': [0, 0, 140], 'hsv_high': [180, 110, 255]}]


def scene():
    intr = intrinsics_for(320, 240)
    plane = Plane(np.array([0., -1., 0.]), -.7)
    yy, xx = np.indices((240, 320))
    rays = deproject_pixels(np.column_stack([xx.ravel(), yy.ravel()]), np.ones(xx.size), intr)
    # 斜面近端与远端深度不同，正面宽约 0.26 m。
    z = .4 / (rays[:, 2] - .6 * rays[:, 0])
    points = rays * z[:, None]
    heights = plane.signed_distance(points)
    mask = ((np.abs(points[:, 0]) < .13) & (heights > .48) & (heights < .69)
            & (z > .15)).reshape(yy.shape)
    color = np.full((240, 320, 3), 40, np.uint8)
    color[mask] = 210
    depth = np.full(yy.shape, np.nan)
    depth[mask] = z.reshape(yy.shape)[mask]
    return intr, plane, np.array([0., .7, .4]), color, depth, mask


@pytest.mark.parametrize('mode', ['dense', 'sparse', 'missing'])
def test_upper_only_oblique_view_is_recognized_with_remembered_anchor(mode):
    intr, plane, anchor, color, depth, mask = scene()
    if mode == 'sparse':
        yy, xx = np.indices(mask.shape)
        depth[(yy % 2 != 0) | (xx % 2 != 0)] = np.nan
    elif mode == 'missing':
        depth[:] = np.nan
    bands = coarse_bands(color, depth, plane, intr, COLORS, .05, 1.2, anchor, pitch=0.)
    assert not bands['lower']
    assert bands['upper_white']
    assert not bands['cap']
    assert len(bands['upper_uv']) >= 40
    # 没有补造深度或污染原图。
    assert np.all(color[~mask] == 40)
    if mode == 'missing':
        assert np.isnan(depth).all()


def test_sparse_metric_support_uses_measured_points_only():
    intr, plane, anchor, color, depth, mask = scene()
    yy, xx = np.indices(mask.shape)
    selected = mask & (yy % 2 == 0) & (xx % 2 == 0)
    pixels = np.column_stack([xx[selected], yy[selected]])
    points = deproject_pixels(pixels, depth[selected], intr)
    high = np.ones(len(points), bool)
    kept = _upper_metric_face(points, pixels, high, plane, mask.shape, mask.astype(np.uint8)*255)
    assert kept.sum() >= 40
    assert len(kept) == len(points)
    # 同一批像素只有细高度带时不能冒充上层侧面。
    points[:, 1] = .1
    assert not _upper_metric_face(points, pixels, high, plane, mask.shape, mask.astype(np.uint8)*255).any()


def test_color_rescue_requires_anchor_and_respects_contradictory_depth():
    intr, plane, anchor, color, depth, mask = scene()
    empty = np.zeros((0, 2), int)
    assert _upper_white_by_color(color, depth, intr, plane, COLORS, .05, 1.2,
                                 None, None, empty) is None
    depth[mask] = 1.1
    assert _upper_white_by_color(color, depth, intr, plane, COLORS, .05, 1.2,
                                 anchor, None, empty) is None


def test_upper_metric_rejects_distant_surface():
    intr, plane, anchor, color, depth, mask = scene()
    yy, xx = np.nonzero(mask)
    pixels = np.column_stack([xx, yy])
    points = deproject_pixels(pixels, depth[mask], intr)
    points[:, 2] += .6
    assert not _upper_metric_face(points, pixels, np.ones(len(points), bool), plane,
                                  mask.shape, mask.astype(np.uint8)*255).any()


def test_upward_anchor_with_small_z_still_supports_upper_color():
    from cube.stack_look import _at_tower_face
    intr = intrinsics_for(320, 240)
    angle = np.radians(-15.)
    up = np.array([0., -np.cos(angle), -np.sin(angle)])
    forward = np.cross(up, [1., 0., 0.])
    plane = Plane(up, -.55)
    anchor = forward * .22 - up * .55
    assert anchor[2] < .15
    yy, xx = np.indices((240, 320))
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    hits = _at_tower_face(pixels, intr, plane, anchor)
    heights = plane.signed_distance(hits)
    mask = ((heights > .50) & (heights < .69) & (np.abs(hits[:, 0]) < .13)).reshape(yy.shape)
    color = np.full((240, 320, 3), 40, np.uint8)
    color[mask] = 210
    depth = np.full(mask.shape, np.nan)
    bands = coarse_bands(color, depth, plane, intr, COLORS, .05, 1.2, anchor, pitch=-15.)
    assert bands['upper_white']
    assert not bands['lower']


def test_coarse_bands_retains_supported_clipped_blue():
    import cv2
    from dataclasses import replace
    intr = replace(intrinsics_for(320, 240), fx=500., fy=500.)
    plane = Plane(np.array([0., -1., 0.]), -.82)
    hsv = np.zeros((240, 320, 3), np.uint8)
    hsv[43:210, 70:250] = (174, 255, 220)
    hsv[:40, 70:250] = (111, 255, 200)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    depth = np.full((240, 320), .4)
    bands = coarse_bands(color, depth, plane, intr, COLORS, .05, 1.2,
                         np.array([0., .82, .4]), pitch=0.)
    assert bands['cap']
    assert np.any(bands['cap_uv'][:, 1] < 40)
