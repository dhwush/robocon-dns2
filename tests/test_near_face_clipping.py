"""Clipped near colors may support a seam without pretending to be a full face."""
from types import SimpleNamespace
import numpy as np
import pytest
from cube.stack_look import _cap_faces


def scene():
    red = np.zeros((240, 320), np.uint8)
    blue = red.copy()
    red[43:210, 70:250] = 255
    blue[:40, 70:250] = 255
    depth = np.full(red.shape, .4)
    intr = SimpleNamespace(fx=500., fy=500.)
    return red, blue, depth, intr


def test_near_clipped_blue_survives_only_with_pair_context():
    red, blue, depth, intr = scene()
    mask = red | blue
    strict = _cap_faces(mask, depth, intr, .05, 1.2)
    assert np.any(strict & red)
    assert not np.any(strict & blue)
    actual = _cap_faces(mask, depth, intr, .05, 1.2, red=red, blue=blue)
    np.testing.assert_array_equal(actual, mask)
    assert not np.any(actual[mask == 0])


@pytest.mark.parametrize('case', ['far', 'unclipped', 'unrelated', 'no_depth', 'no_supported_face'])
def test_partial_rescue_requires_near_clipped_supported_pair(case):
    red, blue, depth, intr = scene()
    if case == 'far': depth[:] = .9
    if case == 'unclipped':
        red = np.roll(red, 10, axis=0)
        blue = np.roll(blue, 10, axis=0)
    if case == 'unrelated': blue = np.roll(blue, 150, axis=1)
    if case == 'no_depth': depth[:] = np.nan
    if case == 'no_supported_face': red[80:] = 0
    mask = red | blue
    strict = _cap_faces(mask, depth, intr, .05, 1.2)
    actual = _cap_faces(mask, depth, intr, .05, 1.2, red=red, blue=blue)
    np.testing.assert_array_equal(actual, strict)
