"""RealSense callbacks simulated without a device, including composite motion frames."""
from collections import deque
from types import SimpleNamespace
import threading

import numpy as np

from camera.source import Frame, RealSenseSource
from view_tower import attitude_line, frame_attitude_valid


class Motion:
    def __init__(self, kind, stamp, sample):
        self.profile = SimpleNamespace(stream_type=lambda: kind)
        self.timestamp = stamp * 1000
        self.sample = sample

    def is_frameset(self):
        return False

    def is_motion_frame(self):
        return True

    def as_motion_frame(self):
        return self

    def get_motion_data(self):
        return SimpleNamespace(**dict(zip(('x', 'y', 'z'), self.sample)))


class Composite:
    def __init__(self, motions=(), video=True):
        self.motions = {m.profile.stream_type(): m for m in motions}
        self.video = video

    def is_frameset(self):
        return True

    def as_frameset(self):
        return self

    def first_or_default(self, kind):
        return self.motions.get(kind)

    def get_depth_frame(self):
        return SimpleNamespace(get_data=lambda: np.full((4, 6), 500, np.uint16)) if self.video else None

    def get_color_frame(self):
        return SimpleNamespace(get_data=lambda: np.zeros((4, 6, 3), np.uint8)) if self.video else None


def source(monkeypatch):
    import camera.source as module
    clock = [10.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    obj = RealSenseSource.__new__(RealSenseSource)
    obj._rs = SimpleNamespace(stream=SimpleNamespace(accel='accel', gyro='gyro'))
    obj._lock, obj._event = threading.Lock(), threading.Event()
    obj._pending = None
    obj._gyro = deque(maxlen=400)
    obj._accel = obj._accel_timestamp_s = obj._accel_received_at = None
    obj._gyro_timestamp_s = None
    obj._gravity_timeout_s = .5
    obj._gyro_ready = obj._video_ready = True
    obj._gyro_to_color = obj._accel_to_color = np.eye(3)
    obj.depth_scale = np.float32(.001)
    # Emulate the real failure condition: aligned output contains no IMU frames.
    obj.align = SimpleNamespace(process=lambda original: Composite(video=original.video))
    obj.temporal_rs = SimpleNamespace(process=lambda depth: depth)
    return obj, clock


def accel(stamp, pitch=0):
    angle = np.deg2rad(pitch)
    gravity = np.array([0., np.cos(angle), np.sin(angle)])
    return Motion('accel', stamp, -gravity * 9.81)


def test_composite_acceleration_updates_before_alignment(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(Composite([accel(1., 42.8)]))
    first = obj._take()
    clock[0] += .1
    obj._handle_frame(Composite([accel(1.1, -13)]))
    second = obj._take()
    assert first.gravity_status == second.gravity_status == 'fresh'
    assert second.gravity_timestamp_s == 1.1
    assert abs(np.degrees(np.arcsin(first.gravity[2])) - 42.8) < 1e-8
    assert abs(np.degrees(np.arcsin(second.gravity[2])) + 13.) < 1e-8
    np.testing.assert_allclose(second.depth, .5)


def test_standalone_and_motion_only_composites_both_update_cache(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(accel(1., 20))
    obj._handle_frame(Composite())
    assert obj._take().gravity_status == 'fresh'
    clock[0] += .1
    obj._handle_frame(Composite([accel(1.1, -10)], video=False))
    assert obj._pending is None
    obj._handle_frame(Composite())
    frame = obj._take()
    assert frame.gravity_timestamp_s == 1.1
    assert abs(np.degrees(np.arcsin(frame.gravity[2])) + 10) < 1e-8


def test_duplicate_timestamp_cannot_keep_old_gravity_alive_and_fresh_sample_recovers(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(Composite([accel(1., 42.8)]))
    obj._take()
    clock[0] += 1.
    obj._handle_frame(Composite([accel(1., 42.8)]))
    stale = obj._take()
    assert stale.gravity is None and stale.gravity_status == 'stale'
    assert stale.gravity_age_s == 1. and stale.gravity_timestamp_s == 1.
    assert not frame_attitude_valid(stale)
    assert '姿态失效' in attitude_line(stale)
    obj._handle_frame(Composite([accel(2., -13)]))
    recovered = obj._take()
    assert frame_attitude_valid(recovered) and recovered.gravity_status == 'fresh'


def test_fresh_stationary_samples_remain_valid_even_if_values_identical(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(Composite([accel(1.)]))
    obj._take()
    clock[0] += 2.
    obj._handle_frame(Composite([accel(3.)]))
    assert obj._take().gravity_status == 'fresh'


def test_pending_image_is_checked_again_at_consumption(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(Composite([accel(1.)]))
    clock[0] += .8
    assert obj._take().gravity_status == 'stale'


def test_missing_and_invalid_acceleration_are_not_replaced_by_a_fake_pose(monkeypatch):
    obj, clock = source(monkeypatch)
    for value in ([np.nan, 0, 0], [np.inf, 0, 0], [0., 0., 0.]):
        obj._handle_frame(Motion('accel', 1., value))
    obj._handle_frame(Composite())
    frame = obj._take()
    assert frame.gravity is None and frame.gravity_status == 'missing'
    assert frame.gravity_timestamp_s is None and frame.gravity_age_s is None
    assert '未收到有效加速度' in attitude_line(frame)


def test_gyro_samples_survive_overwritten_images_and_duplicates_are_skipped(monkeypatch):
    obj, clock = source(monkeypatch)
    for t in (1., 1.1, 1.2):
        motion = Motion('gyro', t, [1., 0., 0.])
        obj._handle_frame(motion)
        obj._handle_frame(Composite([motion, accel(t)]))
    frame = obj._take()
    np.testing.assert_allclose([item[0] for item in frame.gyro], [1., 1.1, 1.2])
    obj._handle_frame(Composite())
    assert obj._take().gyro == []


def test_reopening_clears_old_pose_and_device_timestamps(monkeypatch):
    obj, clock = source(monkeypatch)
    obj._handle_frame(Composite([accel(100.)]))
    device = SimpleNamespace(get_info=lambda key: 'test',
                             first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: .001))
    profile = SimpleNamespace(get_device=lambda: device)
    obj.pipeline = SimpleNamespace(start=lambda *args: profile)
    obj._config = object()
    obj._rs.camera_info = SimpleNamespace(serial_number='serial')
    obj._accel_rotation = lambda: np.eye(3)
    obj._motion_rotation = lambda stream: np.eye(3)
    obj._open_pipeline()
    assert obj._accel is None and obj._accel_timestamp_s is None and obj._pending is None
    assert obj._accel_received_at is None and obj._gyro_timestamp_s is None
    obj._handle_frame(Composite([accel(1.)]))
    assert obj._take().gravity_status == 'fresh'


def test_view_rejects_explicit_stale_gravity_even_if_a_vector_is_present():
    frame = Frame(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)),
                  np.array([0., 1., 0.]), gravity_status='stale', gravity_age_s=1.)
    assert not frame_attitude_valid(frame)
    assert '暂停高度筛选' in attitude_line(frame)


def test_main_pauses_on_stale_pose_recovers_and_records_metadata(monkeypatch, tmp_path):
    import sys
    import view_tower as view
    from cube.stack_look import StackSearch
    from ground.plane import GroundFix, Plane
    from sim.scene import intrinsics_for
    from tower_diagnostics import load_frame

    intr = intrinsics_for(160, 120)
    color = np.full((120, 160, 3), 40, np.uint8)
    depth = np.full((120, 160), np.nan, np.float32)
    images = [Frame(color, depth, np.array([0., 1., 0.]), gravity_timestamp_s=t,
                    gravity_age_s=age, gravity_status=status)
              for t, age, status in [(1., .01, 'fresh'), (1., 1., 'stale'), (2., .01, 'fresh')]]
    index = [-1]
    ground_calls, measure_calls, rendered = [], [], []
    class Source:
        closed = False
        def __init__(self, *a, **k):
            pass
        def read(self):
            index[0] += 1
            return images[index[0]]
        def close(self):
            Source.closed = True

    def ground(*a):
        ground_calls.append(index[0])
        return GroundFix(Plane(np.array([0., -1., 0.]), -.68), 'imu', height=.68)
    def measure(*a):
        measure_calls.append(index[0])
        return None, 'seam'
    def paint(color, bands, lines, intr, plane, cap_pose):
        rendered.append((plane, cap_pose, lines.copy()))
        return color

    monkeypatch.setattr(sys, 'argv', ['view_tower.py', '--diagnostics', str(tmp_path / 'capture')])
    monkeypatch.setattr(view, 'RealSenseSource', Source)
    monkeypatch.setattr(view, 'load_intrinsics', lambda *a: intr)
    monkeypatch.setattr(view, 'StackSearch', lambda: StackSearch(kind='two_cap', settled=True))
    monkeypatch.setattr(StackSearch, 'update', lambda *a, **k: 'upstairs')
    monkeypatch.setattr(view, 'tower_ground', ground)
    monkeypatch.setattr(view, 'measure_cap', measure)
    monkeypatch.setattr(view, 'paint', paint)
    monkeypatch.setattr(view.cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(view.cv2, 'destroyAllWindows', lambda: None)
    monkeypatch.setattr(view.cv2, 'waitKey', lambda *a: ord('q') if index[0] == 2 else -1)
    previous = sys.getprofile()
    view.main()
    assert sys.getprofile() is previous and Source.closed
    assert ground_calls == [0, 2] and measure_calls == [0, 2]
    assert rendered[1][0] is None and rendered[1][1] is None
    assert any('姿态失效' in line for line in rendered[1][2])
    assert any('姿态有效' in line for line in rendered[2][2])
    stale = load_frame(tmp_path / 'capture', 1)
    assert stale['gravity_status'] == 'stale'
    assert stale['gravity_age_s'] == 1. and stale['gravity_timestamp_s'] == 1.
    assert not stale['state']['attitude_valid'] and stale['state']['ground'] is None
    assert stale['state']['reset_reason'] == 'attitude_invalid'
