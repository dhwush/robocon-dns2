"""Regression checks for transition bands, pose validation and retained tower anchors."""
from dataclasses import replace

import numpy as np

from cube.cap_pose import _accept, _colored_mask, camera_to_car_world, optical_from_car_world
from cube.color_boundary import adjacent_color_pairs, boundary_pixels
from cube.pose_projection import pose_matches_color
from cube.stack_look import StackSearch
from ground.plane import Plane
from main import GroundMemory, load_config
from sim.scene import intrinsics_for
from view_tower import TowerWatch, tower_ground, measure_cap


def masks(gap=12, sideways=False):
    red = np.zeros((240, 320), np.uint8)
    blue = np.zeros_like(red)
    blue[30:100, 50:250] = 255
    if sideways:
        red[30:100, 260:310] = 255
    else:
        red[100+gap:190+gap, 50:250] = 255
    return red, blue


def test_transition_band_joins_target_but_not_remote_or_sideways_colors():
    red, blue = masks()
    uv = boundary_pixels(red, blue)
    assert len(uv) == 200
    assert np.all((uv[:, 1] >= 99) & (uv[:, 1] <= 112))
    red, blue = masks(gap=50)
    assert not adjacent_color_pairs(red, blue)
    red, blue = masks(sideways=True)
    assert not adjacent_color_pairs(red, blue)


def test_existing_touching_boundary_retains_multi_row_depth_support():
    red, blue = masks(gap=0)
    uv = boundary_pixels(red, blue)
    assert len(np.unique(uv[:, 1])) > 1


def scene():
    intr = intrinsics_for(320, 240)
    color = np.full((240, 320, 3), 40, np.uint8)
    depth = np.full((240, 320), np.nan, np.float32)
    uu, vv = np.meshgrid(np.arange(320), np.arange(240))
    xx = (uu - intr.cx) / intr.fx * .55
    yy = (vv - intr.cy) / intr.fy * .55
    face = (np.abs(xx) <= .10) & (yy >= -.22) & (yy <= -.02)
    color[face & (yy < -.12)] = (255, 0, 0)
    color[face & (yy >= -.12)] = (0, 0, 255)
    depth[face] = .55
    return color, depth, intr, Plane(np.array([0., -1., 0.]), -.68)


def test_projection_accepts_aligned_cube_rejects_shifted_and_oversized_cube():
    color, depth, intr, plane = scene()
    pose = _accept(np.array([0., -.12, .65]), np.array([1., 0., 0.]), plane, None)
    assert pose_matches_color(pose, color, intr, plane, _colored_mask(color))
    shifted = replace(pose, center=pose.center + [-.16, 0, 0], corners=pose.corners + [-.16, 0, 0])
    assert not pose_matches_color(shifted, color, intr, plane, _colored_mask(color))
    huge = replace(pose, corners=(pose.corners - pose.center) * 4 + pose.center)
    assert not pose_matches_color(huge, color, intr, plane, _colored_mask(color))
    actual, reason = measure_cap(color, depth, intr, plane, load_config('config/default.yaml'))
    assert actual is not None and reason is None


def test_bad_measurement_is_rejected_before_fusion(monkeypatch):
    import view_tower
    color, depth, intr, plane = scene()
    wrong = _accept(np.array([-.16, -.12, .65]), np.array([1., 0., 0.]), plane, None)
    monkeypatch.setattr(view_tower, 'measure_cap_pose', lambda *a, **k: (wrong, None))
    pose, reason = measure_cap(color, depth, intr, plane, load_config('config/default.yaml'))
    assert pose is None and reason == 'projection'


def test_visible_cap_preserves_anchor_after_reporting_then_missing_target_expires():
    search = StackSearch(kind='two_cap', settled=True, anchor=np.array([0., .68, .5]))
    watch = TowerWatch(hold_s=.5)
    watch.line = 'reported'
    visible = {'lower': False, 'upper_white': False, 'cap': True}
    for now in (0., .6, 5., 10.):
        assert not watch.depart(now, search, visible)
        np.testing.assert_array_equal(watch.anchor, search.anchor)
    assert not watch.depart(10.1, search, {})
    assert not watch.depart(10.5, search, {})
    assert watch.depart(10.7, search, {})


def test_known_installation_height_is_shared_by_ground_and_world(monkeypatch):
    import view_tower
    from ground.plane import GroundFix
    color, depth, intr, plane = scene()
    fitted = GroundFix(plane=Plane(plane.normal, -.646), source='fit', height=.646)
    monkeypatch.setattr(view_tower, '_prior_ground', lambda *a, **k: fitted)
    memory = GroundMemory()
    fix = tower_ground(depth, color, intr, {}, np.array([0., 1., 0.]), .68, memory)
    assert fix.height == .68 and fix.plane.offset == -.68 and memory.prior is fix
    assert load_config('config/default.yaml')['camera_height'] == .669
    pose = _accept(np.array([0., -.12, .65]), np.array([1., 0., 0.]), fix.plane, None)
    assert abs(pose.world[2] - .80) < 1e-9
    world = camera_to_car_world(np.array([0., -.131, .65]))
    assert abs(world[2] - .80) < 1e-9
    np.testing.assert_allclose(optical_from_car_world(world), [0., .65, .131])
