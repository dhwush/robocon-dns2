"""Device calibration contracts, without requiring a connected camera."""
from types import SimpleNamespace

import numpy as np
import pytest

from camera.source import RealSenseSource, device_color_intrinsics
from geom.transform import deproject_pixels, project_points


def profile_for(model, coeffs):
    raw = SimpleNamespace(width=640, height=480, fx=601., fy=602.,
                          ppx=319., ppy=241., model=model, coeffs=coeffs)
    video = SimpleNamespace(get_intrinsics=lambda: raw)
    stream = SimpleNamespace(as_video_stream_profile=lambda: video)
    requested = []
    profile = SimpleNamespace(get_stream=lambda kind: requested.append(kind) or stream)
    return profile, raw, requested


@pytest.mark.parametrize('model,expected', [
    ('distortion.brown_conrady', 'brown'),
    ('distortion.inverse_brown_conrady', 'inverse_brown_conrady'),
])
def test_active_color_calibration_preserves_coefficients_and_geometry(model, expected):
    profile, raw, requested = profile_for(model, [.03, -.01, .001, -.002, .004])
    intr = device_color_intrinsics(profile, SimpleNamespace(stream=SimpleNamespace(color='color')))
    assert requested == ['color']
    assert (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy) == (640, 480, 601., 602., 319., 241.)
    assert intr.distortion_model == expected
    np.testing.assert_allclose(intr.D, raw.coeffs)
    points = np.array([[.2, -.1, .8], [-.15, .2, 1.]])
    uv = project_points(points, intr)
    np.testing.assert_allclose(deproject_pixels(uv, points[:, 2], intr), points, atol=1e-10)


def test_zero_distortion_and_unsupported_nonzero_model():
    profile, raw, _ = profile_for('distortion.modified_brown_conrady', [0.] * 5)
    rs = SimpleNamespace(stream=SimpleNamespace(color='color'))
    assert device_color_intrinsics(profile, rs).distortion_model == 'none'
    raw.coeffs[0] = .1
    with pytest.raises(ValueError, match='不支持'):
        device_color_intrinsics(profile, rs)
    raw.fx = float('nan')
    with pytest.raises(ValueError, match='无效'):
        device_color_intrinsics(profile, rs)


def test_imu_column_major_rotation_transforms_acceleration_and_gyro():
    # A non-symmetric rotation exposes transpose errors hidden by identity mocks.
    angle = np.deg2rad(30.)
    rotation = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                         [0., np.sin(angle), np.cos(angle)]])
    ext = SimpleNamespace(rotation=rotation.flatten(order='F').tolist())
    motion = SimpleNamespace(get_extrinsics_to=lambda color: ext)
    stream = SimpleNamespace(as_motion_stream_profile=lambda: motion)
    obj = RealSenseSource.__new__(RealSenseSource)
    obj._rs = SimpleNamespace(stream=SimpleNamespace(color='color'))
    obj.profile = SimpleNamespace(get_stream=lambda kind: stream)
    for kind in ('accel', 'gyro'):
        actual = obj._motion_rotation(kind)
        np.testing.assert_allclose(actual @ [0., 1., 0.], [0., np.sqrt(3.) / 2., .5])
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-15)
    obj._accel_to_color = obj._motion_rotation('accel')
    np.testing.assert_allclose(obj._gravity_from_specific(np.array([0., -9.81, 0.])),
                               [0., np.sqrt(3.) / 2., .5])
