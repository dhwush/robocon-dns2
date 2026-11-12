"""方块框取多帧中值，跳开的一帧不换框。抬头后不再画下层。"""

from __future__ import annotations

import numpy as np

from cube.outline import OutlineTrack, square_corners
from ground.plane import Plane


def _plane() -> Plane:
    return Plane(np.array([0.0, -1.0, 0.0]), -0.7)


def _sample(bottom, yaw=0.0, layer=1):
    edge = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
    return {
        "layer": layer,
        "bottom": np.asarray(bottom, dtype=float),
        "edge": edge,
        "x": float(bottom[0]),
        "y": float(bottom[1]),
        "z": float(bottom[2]),
        "yaw": yaw,
        "quarter": True,
        "long_side": 0.36,
        "short_side": 0.34,
        "inliers": 20,
    }


def test_outline_waits_for_several_frames_then_is_a_square():
    track = OutlineTrack(need=5, jump=0.03)
    plane = _plane()
    drawn = []
    for index in range(4):
        shift = 0.01 * (index - 1.5)
        drawn = track.update([_sample([0.10 + shift, 0.70, 0.80])], "ground", plane, 0.35)
        assert drawn == []
    drawn = track.update([_sample([0.10, 0.70, 0.80])], "ground", plane, 0.35)
    assert len(drawn) == 1
    corners = drawn[0]["corners"]
    spans = np.linalg.norm(corners - corners[0], axis=1)
    assert abs(float(np.max(spans)) - 0.35 * np.sqrt(2.0)) < 1e-6
    assert abs(float(np.linalg.norm(drawn[0]["bottom"] - np.array([0.10, 0.70, 0.80])))) < 0.02


def test_a_jumped_frame_does_not_move_the_box():
    track = OutlineTrack(need=5, jump=0.03)
    plane = _plane()
    for _ in range(5):
        drawn = track.update([_sample([0.10, 0.70, 0.80])], "ground", plane, 0.35)
    kept = drawn[0]["bottom"].copy()
    drawn = track.update([_sample([0.40, 0.70, 0.80])], "ground", plane, 0.35)
    assert np.allclose(drawn[0]["bottom"], kept)


def test_looking_up_drops_the_lower_box():
    track = OutlineTrack(need=5, jump=0.03)
    plane = _plane()
    for _ in range(5):
        track.update([_sample([0.10, 0.70, 0.80], layer=1)], "ground", plane, 0.35)
    drawn = track.update([_sample([0.10, 0.70, 0.80], layer=1)], "upstairs", plane, 0.35)
    assert drawn == []
    for _ in range(5):
        drawn = track.update([_sample([0.12, 0.70, 0.70], layer=2)], "upstairs", plane, 0.35)
    assert len(drawn) == 1
    assert drawn[0]["layer"] == 2
    back = track.update([_sample([0.10, 0.70, 0.80], layer=1)], "one", plane, 0.35)
    assert back == []


def test_square_corners_match_one_edge():
    normal = np.array([0.0, -1.0, 0.0])
    edge = np.array([1.0, 0.0, 0.0])
    corners = square_corners(np.array([0.0, 0.7, 1.0]), edge, normal, 0.35)
    assert abs(float(np.linalg.norm(corners[1] - corners[0])) - 0.35) < 1e-9
    assert abs(float(np.linalg.norm(corners[3] - corners[0])) - 0.35) < 1e-9
