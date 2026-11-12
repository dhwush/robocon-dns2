"""Reject plausible fixed-size cubes at the wrong distance without rejecting clipping."""
from dataclasses import replace

import numpy as np

from cube.cap_pose import _accept, _colored_mask
from cube.pose_projection import pose_matches_color
from test_tower_stability_fix import scene


def pose_for(plane, z=.65, via='depth'):
    return replace(_accept(np.array([0., -.12, z]), np.array([1., 0., 0.]), plane, None), via=via)


def check(pose, color, depth, intr, plane, mask=None):
    detail = {}
    accepted = pose_matches_color(pose, color, intr, plane,
                                 _colored_mask(color) if mask is None else mask,
                                 depth=depth, diagnostics=detail)
    return accepted, detail


def test_wrong_distance_fixed_twenty_cm_cube_is_rejected_even_without_depth():
    color, depth, intr, plane = scene()
    # Every edge is still exactly 20 cm. The old coarse coverage test accepts this
    # oversized projection because it surrounds the real target.
    wrong = pose_for(plane, z=.45, via='close')
    accepted, detail = check(wrong, color, None, intr, plane)
    assert not accepted and detail['reason'] == 'visible_edge_mismatch'


def test_correct_visible_edges_do_not_override_contradictory_reliable_depth():
    color, depth, intr, plane = scene()
    depth[np.isfinite(depth)] += .12
    accepted, detail = check(pose_for(plane), color, depth, intr, plane)
    assert not accepted and detail['reason'] == 'depth_mismatch'
    assert detail['depth']['median_error_m'] > .10


def test_rgb_fallback_does_not_reuse_depth_already_classified_unreliable():
    color, depth, intr, plane = scene()
    depth[np.isfinite(depth)] += .12
    accepted, detail = check(pose_for(plane, via='close'), color, depth, intr, plane)
    assert accepted
    assert detail['depth']['status'] == 'unavailable_or_unreliable'


def test_clipped_left_edge_is_not_treated_as_a_real_cube_edge():
    color, depth, intr, plane = scene()
    crop = 150
    color, depth = color[:, crop:], depth[:, crop:]
    intr = replace(intr, width=color.shape[1], cx=intr.cx - crop)
    accepted, detail = check(pose_for(plane), color, depth, intr, plane)
    assert accepted
    assert not any(e['axis'] == 1 and e['side'] == 0 for e in detail['edges'])


def test_mask_holes_and_interior_threshold_cutoffs_are_not_physical_edges():
    color, depth, intr, plane = scene()
    mask = _colored_mask(color)
    ys, xs = np.nonzero(mask)
    left, right = xs.min(), xs.max()
    mask[:, :left + (right-left)//4] = 0
    mask[100:115, 150:165] = 0
    accepted, detail = check(pose_for(plane), color, depth, intr, plane, mask)
    assert accepted and detail['depth']['status'] == 'consistent'


def test_millimeter_depth_noise_is_tolerated():
    color, depth, intr, plane = scene()
    valid = np.isfinite(depth)
    depth[valid] += np.random.default_rng(10).normal(0, .004, valid.sum())
    assert check(pose_for(plane), color, depth, intr, plane)[0]


def test_without_reliable_depth_or_physical_edges_color_coverage_is_insufficient():
    color, depth, intr, plane = scene()
    mask = _colored_mask(color)
    # A threshold boundary with no supporting image contrast cannot establish scale.
    color[:] = (40, 40, 40)
    accepted, detail = check(pose_for(plane, via='side'), color, None, intr, plane, mask)
    assert not accepted and detail['reason'] == 'insufficient_validation'


def test_wrapper_rejects_overlarge_rgb_measurement_before_it_reaches_fusion(monkeypatch):
    import view_tower
    from main import load_config
    color, depth, intr, plane = scene()
    wrong = pose_for(plane, z=.45, via='close')
    monkeypatch.setattr(view_tower, 'measure_cap_pose', lambda *a, **k: (wrong, None))
    measured, reason = view_tower.measure_cap(color, depth, intr, plane, load_config('config/default.yaml'))
    assert measured is None and reason == 'projection'
