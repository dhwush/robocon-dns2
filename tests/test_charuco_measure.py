from pathlib import Path
import cv2
import numpy as np
import yaml
from measure_charuco import make_board, estimate, measure


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/charuco_measure.yaml').read_text())


def test_printed_layout_detection_and_pose():
    cfg = config()
    K = np.array([[1000., 0, 400], [0, 1000., 350], [0, 0, 1]])
    for name in ('board1', 'board2'):
        board = make_board(cfg, name)
        image = np.full((700, 800), 255, np.uint8)
        image[170:530, 160:640] = board.generateImage((480, 360))
        pose = estimate(board, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), K, np.zeros(5), cfg)
        assert pose is not None
        np.testing.assert_allclose(pose[1], [-.12, -.09, .5], atol=.002)
        other = make_board(cfg, 'board2' if name == 'board1' else 'board1')
        assert estimate(other, image, K, np.zeros(5), cfg) is None


def test_center_and_relative_transform_with_rotated_reference():
    cfg = config()
    cfg.update(cube_center_from_board_center=[0, 0, .1],
               R_car_board2=np.eye(3).tolist(), t_car_board2=[1, 2, 3])
    R = cv2.Rodrigues(np.array([0., 0., np.pi/2]))[0]
    pose1 = (R, np.array([.4, .5, 1.]), .1, 30)
    pose2 = (R, np.array([.1, .2, .3]), .1, 30)
    result = measure(pose1, pose2, cfg)
    optical = R @ np.array([.12, .09, .1]) + pose1[1]
    expected = R.T @ (optical - pose2[1]) + [1, 2, 3]
    np.testing.assert_allclose(result['car_xyz_m'], expected)
    assert np.isclose(result['optical_distance_m'], np.linalg.norm(optical))
    assert result['car_source'] == 'board2'


def test_missing_reference_has_no_stale_car_result():
    cfg = config()
    result = measure((np.eye(3), np.array([0, 0, 1.]), .1, 20), None, cfg)
    assert result['car_xyz_m'] is None
    assert result['car_distance_m'] is None
    assert not result['provisional']


def test_device_intrinsics_uses_active_color_stream():
    from types import SimpleNamespace
    from measure_charuco import device_intrinsics, pose_intrinsics
    i = SimpleNamespace(width=640, height=480, fx=600., fy=601., ppx=320., ppy=240.,
                        coeffs=[.1, .01, 0., 0., 0.], model='distortion.brown_conrady')
    video = SimpleNamespace(get_intrinsics=lambda: i)
    def get_stream(kind):
        assert kind == 'color'
        return SimpleNamespace(as_video_stream_profile=lambda: video)
    profile = SimpleNamespace(get_stream=get_stream,
                              get_device=lambda: SimpleNamespace(get_info=lambda kind: 'test-device'))
    rs = SimpleNamespace(stream=SimpleNamespace(color='color'), camera_info=SimpleNamespace(serial_number='serial'))
    intr = device_intrinsics(profile, rs)
    K, D = pose_intrinsics(intr)
    assert intr['serial'] == 'test-device'
    assert intr['width'] == 640
    np.testing.assert_allclose(K, [[600, 0, 320], [0, 601, 240], [0, 0, 1]])
    np.testing.assert_allclose(D, i.coeffs)


def test_nonzero_inverse_distortion_is_not_misused():
    import pytest
    from measure_charuco import pose_intrinsics
    with pytest.raises(ValueError, match='畸变模型'):
        pose_intrinsics(dict(fx=1, fy=1, cx=0, cy=0, D=[.1, 0, 0, 0, 0],
                             distortion_model='inverse_brown_conrady'))


def test_both_overlays_share_canvas_without_contaminating_detection(monkeypatch):
    import measure_charuco as module
    raw = np.zeros((30, 40, 3), np.uint8)
    pose = (np.eye(3), np.array([0., 0., 1.]), .2, 12)
    def detect(board, image, K, D, cfg, *, draw_image):
        assert not image.any(), 'overlay must never enter detection input'
        draw_image[board, board] = 255
        return pose
    monkeypatch.setattr(module, 'estimate', detect)
    poses, shown = module.detect_boards([1, 2], raw, None, None, config())
    assert len(poses) == 2 and all(p is pose for p in poses)
    assert shown[1, 1].all() and shown[2, 2].all()
    assert not raw.any()


def test_reference_status_is_visible_without_target_and_does_not_stick():
    from measure_charuco import board_status, board_status_lines
    cfg = config()
    cfg.update(R_car_board2=None, t_car_board2=None)
    pose = (np.eye(3), np.array([0., 0., 1.]), .2, 12)
    states = board_status([None, pose], cfg)
    assert not states['board1']['detected'] and states['board2']['detected']
    lines = board_status_lines(states)
    assert any('Board2 (car): OK' in line for line in lines)
    assert any('transform missing' in line for line in lines)
    missing = board_status([pose, None], cfg)
    assert not missing['board2']['detected']
    assert missing['board2']['reprojection_px'] is None
    assert any('Board2 (car): missing' in line for line in board_status_lines(missing))
    cfg.update(R_car_board2=np.eye(3).tolist(), t_car_board2=[0., 0., 0.])
    assert not any('transform missing' in line for line in board_status_lines(board_status([pose, pose], cfg)))


def test_confirmed_mount_maps_paper_center_and_board_axes():
    cfg = config()
    R, t = np.array(cfg['R_car_board2']), np.array(cfg['t_car_board2'])
    # CAD图中的纸中心在右前角原点的后方、左侧、上方。
    np.testing.assert_allclose(R @ [.120, .090, 0.] + t, [-.175, -.1348, .570])
    np.testing.assert_allclose(R @ [0., 1., 0.], [0., 1., 0.])  # 纸下边朝车前
    np.testing.assert_allclose(R @ [1., 0., 0.], [-1., 0., 0.])
    np.testing.assert_allclose(R @ [0., 0., 1.], [0., 0., -1.])
    assert np.isclose(np.linalg.det(R), 1.)
    # 200mm顶块；最近顶角在左上，顶面在印刷面背后3mm。
    np.testing.assert_allclose(cfg['cube_center_from_board_center'],
                               np.array([-.125, -.1685, .003]) + [.100, .100, .100])


def test_dual_board_car_center_is_invariant_to_camera_position_and_rotation():
    cfg = config()
    board2_R = np.array(cfg['R_car_board2'])
    board2_t = np.array(cfg['t_car_board2'])
    board1_R = cv2.Rodrigues(np.array([0., 0., .7]))[0] @ board2_R
    local_center = np.array([.120, .090, 0.]) + cfg['cube_center_from_board_center']
    expected_car = np.array([.20, .65, .80])
    board1_t = expected_car - board1_R @ local_center
    for omega, shift in (([0., 0., 0.], [.1, .2, 1.2]),
                         ([.3, -.6, .8], [-.2, .1, 1.5])):
        camera_R = cv2.Rodrigues(np.array(omega))[0]
        camera_t = np.array(shift)
        pose1 = (camera_R @ board1_R, camera_R @ board1_t + camera_t, .1, 20)
        pose2 = (camera_R @ board2_R, camera_R @ board2_t + camera_t, .2, 25)
        result = measure(pose1, pose2, cfg, up=np.array([0., -1., 0.]))
        expected_optical = camera_R @ expected_car + camera_t
        np.testing.assert_allclose(result['car_xyz_m'], expected_car, atol=1e-12)
        np.testing.assert_allclose(result['optical_xyz_m'], expected_optical, atol=1e-12)
        assert np.isclose(result['car_distance_m'], np.linalg.norm(expected_car))
        assert np.isclose(result['car_horizontal_m'], np.linalg.norm(expected_car[:2]))
        assert result['car_source'] == 'board2' and not result['provisional']
        cfg['board2_mount_confirmed'] = False
        assert measure(pose1, pose2, cfg)['provisional']
        cfg['board2_mount_confirmed'] = True
    # 参照板丢失后不会保留上一帧的双板车坐标；重力路径仍标记未确认。
    missing = measure(pose1, None, cfg)
    assert missing['car_xyz_m'] is None
    fallback = measure(pose1, None, cfg, up=np.array([0., -1., 0.]))
    assert fallback['car_source'] == 'camera_mount_gravity' and fallback['provisional']


def test_realsense_measurement_requests_only_color_and_closes(monkeypatch):
    import sys
    from types import SimpleNamespace
    from measure_charuco import RealSenseColorSource
    requests, stops = [], []
    raw = np.zeros((2, 3, 3), np.uint8)
    class Config:
        def enable_device(self, serial): requests.append(('device', serial))
        def enable_stream(self, *args): requests.append(args)
    class Pipeline:
        def start(self, cfg): return 'active-profile'
        def wait_for_frames(self, timeout):
            return SimpleNamespace(get_color_frame=lambda: SimpleNamespace(get_data=lambda: raw))
        def stop(self): stops.append(True)
    monkeypatch.setitem(sys.modules, 'pyrealsense2', SimpleNamespace(
        pipeline=Pipeline, config=Config, stream=SimpleNamespace(color='color'),
        format=SimpleNamespace(bgr8='bgr8')))
    source = RealSenseColorSource(serial='test-camera')
    assert requests == [('device', 'test-camera'), ('color', 1280, 720, 'bgr8', 30)]
    frame = source.read()
    assert frame.gravity is None and frame.color.shape == raw.shape
    frame.color[:] = 255
    assert not raw.any()
    source.close()
    source.close()
    assert stops == [True]


def test_yaw_is_board_heading_in_car_frame_with_positive_right_turn():
    cfg = config()
    R2 = np.asarray(cfg['R_car_board2'])
    for expected in (0., 30., -45., 90.):
        # 右转为正，与常规绕Z逆时针的旋转符号相反。
        R1 = cv2.Rodrigues(np.array([0., 0., -np.radians(expected)]))[0] @ R2
        # 上述R1是车系方向，转换为2号板系以构造相机看到的方向。
        camera_R1 = R2.T @ R1
        result = measure((camera_R1, np.array([0., 0., 1.]), .1, 20),
                         (np.eye(3), np.zeros(3), .1, 20), cfg)
        assert np.isclose(result['car_yaw_deg'], expected)


def test_chinese_output_and_degenerate_yaw():
    from measure_charuco import chinese_lines, board_status, draw_chinese, projected_yaw
    cfg = config()
    pose = (np.eye(3), np.array([0., 0., 1.]), .1, 20)
    result = measure(pose, None, cfg)
    result.update(board_status([pose, None], cfg))
    assert result['optical_yaw_deg'] is None  # 纸下方向沿相机Y，无XZ方向。
    assert projected_yaw(0., 1.) == 0.
    assert projected_yaw(1., 0.) == 90.
    lines = chinese_lines(result)
    assert any('相对摄像头光心' in line and 'X=' in line for line in lines)
    assert any('相对小车右前角：暂无有效结果' in line for line in lines)
    shown = draw_chinese(np.zeros((720, 1280, 3), np.uint8), lines)
    assert shown.shape == (720, 1280, 3) and shown.any()
