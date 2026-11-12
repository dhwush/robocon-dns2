"""Capture round trips and instrumentation preserve actual recognition outputs."""
from pathlib import Path
import sys

import numpy as np

from camera.source import Frame
from cube.cap_pose import color_body_pixels, measure_cap
from ground.plane import Plane
from sim.scene import intrinsics_for
from tower_diagnostics import DiagnosticRecorder, load_frame


def test_raw_round_trip_tracing_and_recognition_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[1]
    intr = intrinsics_for(160, 120)
    color = np.full((120, 160, 3), 40, dtype=np.uint8)
    color[20:60, 50:110] = (150, 60, 40)
    color[60:100, 50:110] = (30, 20, 160)
    depth = np.full((120, 160), np.nan, dtype=np.float32)
    depth[20:100, 50:110] = 0.6
    frame = Frame(color, depth, np.array([0., 1., 0.]), [(1.2, [0., 0., 0.])])
    plane = Plane(np.array([0., -1., 0.]), -0.7)
    expected_pixels = color_body_pixels(color)
    expected, reason = measure_cap(color, depth, intr, plane, 0.05, 1.2)
    original_color, original_depth = color.copy(), depth.copy()
    previous = sys.getprofile()
    recorder = DiagnosticRecorder(tmp_path / 'capture', 20, {'min_samples': 5}, intr, .7, root)
    assert recorder.begin()
    recorder.read_at = recorder.capture_at
    actual_pixels = color_body_pixels(color)
    actual, actual_reason = measure_cap(color, depth, intr, plane, .05, 1.2)
    recorder.save(frame, frame, {'measured': actual, 'reason': actual_reason, 'plane': plane})
    assert sys.getprofile() is previous
    np.testing.assert_array_equal(actual_pixels, expected_pixels)
    assert reason == actual_reason
    assert (expected is None) == (actual is None)
    if actual is not None:
        np.testing.assert_array_equal(actual.center, expected.center)
        np.testing.assert_array_equal(actual.corners, expected.corners)
        assert actual.yaw == expected.yaw
    np.testing.assert_array_equal(color, original_color)
    np.testing.assert_array_equal(depth, original_depth)
    restored = load_frame(recorder.path, 0)
    np.testing.assert_array_equal(restored['raw_color'], color)
    np.testing.assert_array_equal(restored['raw_depth'], depth)
    assert restored['raw_depth'].dtype == np.float32
    np.testing.assert_array_equal(restored['raw_gravity'], frame.gravity)
    assert restored['gyro'] == [[1.2, [0., 0., 0.]]]
    assert restored['trace']
    assert any(e['function'] == 'measure_cap' for e in restored['trace'])
    assert all(e['return_line'] > 0 for e in restored['trace'])
    assert (recorder.path / 'source/cube/cap_pose.py').exists()


def test_duration_stops_and_restores_profile(tmp_path):
    import time
    root = Path(__file__).resolve().parents[1]
    recorder = DiagnosticRecorder(tmp_path / 'capture', 1, {}, intrinsics_for(16, 12), .7, root)
    previous = sys.getprofile()
    assert recorder.begin()
    recorder.stop_trace()
    assert sys.getprofile() is previous
    recorder.start = time.monotonic() - 2
    assert not recorder.begin()
    assert recorder.finished
    assert sys.getprofile() is previous


def test_existing_capture_not_overwritten(tmp_path):
    import pytest
    with pytest.raises(FileExistsError):
        DiagnosticRecorder(tmp_path, 20, {}, intrinsics_for(16, 12), .7, tmp_path)
