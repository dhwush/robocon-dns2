"""Candidate-only frames cannot manufacture an outline pose or draw cached measurements."""
import sys

import numpy as np
import pytest

from camera.source import Frame
from cube.cap_pose import _accept
from cube.stack_look import StackSearch
from ground.plane import GroundFix, Plane
from sim.scene import intrinsics_for


@pytest.mark.parametrize('with_history', [False, True])
def test_candidate_only_has_no_outline_completion_or_historical_green_box(monkeypatch, with_history):
    import view_tower as view
    intr = intrinsics_for(160, 120)
    plane = Plane(np.array([0., -1., 0.]), -.68)
    pose = _accept(np.array([0., -.12, .65]), np.array([1., 0., 0.]), plane, None)
    color = np.full((120, 160, 3), 40, np.uint8)
    frame = Frame(color, np.full((120, 160), np.nan), np.array([0., 1., 0.]))
    index, rendered = [-1], []
    class Source:
        def __init__(self, *a, **k):
            pass
        def read(self):
            index[0] += 1
            return frame
        def close(self):
            pass
    def bands(*a, **k):
        candidate = not with_history or index[0] > 0
        return {'cap': True, 'cap_rgb_candidate': candidate, 'cap_partial': candidate,
                'lower': False, 'upper_white': False, 'anchor': None, 'covers': False,
                'lower_uv': None, 'upper_uv': None, 'cap_uv': None}
    def measured(*a, **k):
        return (pose, None) if with_history and index[0] == 0 else (None, 'short')
    def forbidden_outline(*a, **k):
        raise AssertionError('candidate pixels must not be completed into a cube')
    def paint(color, bands, lines, intr, plane, cap_pose):
        rendered.append((cap_pose, lines))
        return color
    monkeypatch.setattr(sys, 'argv', ['view_tower.py'])
    monkeypatch.setattr(view, 'RealSenseSource', Source)
    monkeypatch.setattr(Source, 'intrinsics', intr, raising=False)
    monkeypatch.setattr(view, 'StackSearch', lambda: StackSearch(kind='two_cap', settled=True))
    monkeypatch.setattr(StackSearch, 'update', lambda *a, **k: 'upstairs')
    monkeypatch.setattr(view, 'tower_ground', lambda *a: GroundFix(plane, 'imu', height=.68))
    monkeypatch.setattr(view, 'coarse_bands', bands)
    monkeypatch.setattr(view, 'measure_cap', measured)
    monkeypatch.setattr(view, 'measure_outline', forbidden_outline)
    monkeypatch.setattr(view, 'cap_top_color', lambda *a, **k: 'blue')
    monkeypatch.setattr(view, 'close_depth_unusable', lambda *a, **k: False)
    monkeypatch.setattr(view, 'pose_matches_color', lambda *a, **k: True)
    monkeypatch.setattr(view, 'paint', paint)
    monkeypatch.setattr(view.cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(view.cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(view.cv2, 'waitKey', lambda *a: ord('q') if index[0] >= int(with_history) else -1)
    view.main()
    if with_history:
        assert rendered[0][0] is not None
    assert rendered[-1][0] is None
    assert '目标可见，本帧无法精测' in '\n'.join(rendered[-1][1])
