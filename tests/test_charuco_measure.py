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
    assert result['provisional']
