import numpy as np
import pytest
from cube.stack_look import coarse_bands, _upper_top_support
from ground.plane import Plane
from sim.scene import intrinsics_for
from tests.test_stack_look import _cfg, _fill_height, _level_plane


@pytest.mark.parametrize('background_depth', [float('nan'), 0., 1.1, 2., 5.])
@pytest.mark.parametrize('layers', [1, 2])
def test_background_colors_do_not_make_cap(background_depth, layers):
    intr = intrinsics_for(160, 120)
    color = np.full((120, 160, 3), 40, np.uint8)
    depth = np.full((120, 160), np.nan)
    _fill_height(color, depth, intr, .6, .20, .40, 60, 100, (255, 255, 255))
    if layers == 2:
        _fill_height(color, depth, intr, .6, .50, .70, 60, 100, (255, 255, 255))
    for left, right, bgr in [(60, 80, (0, 0, 255)), (80, 100, (180, 50, 0))]:
        _fill_height(color, depth, intr, .6, .74, .90, left, right, bgr)
        mask = np.all(color == bgr, axis=2)
        depth[mask] = background_depth
    result = coarse_bands(color, depth, _level_plane(), intr, _cfg()['colors'], .05, 1.2)
    assert result['lower']
    assert result['upper_white'] == (layers == 2)
    assert not result['cap']


@pytest.mark.parametrize('height,span,expected', [(.70, .30, True), (.35, .30, False), (.70, .04, False)])
def test_downward_second_white_top_support(height, span, expected):
    # 顶面只有几行像素，仍按实际二维尺寸判断；第一层顶面和小噪点不算。
    x, z = np.meshgrid(np.linspace(-span/2, span/2, 30), np.linspace(.6, .6+span, 5))
    points = np.column_stack([x.ravel(), np.full(x.size, .9-height), z.ravel()])
    u, v = np.meshgrid(np.arange(30), np.arange(5))
    pixels = np.column_stack([u.ravel(), v.ravel()])
    keep = _upper_top_support(points, pixels, np.ones(x.size, bool), Plane(np.array([0., -1., 0.]), -.9), (5, 30))
    assert bool(keep.any()) == expected
