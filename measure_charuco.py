"""双ChArUco板实时测量；运行说明见 CHARUCO_MEASURE.md。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def make_board(cfg, name):
    lo, hi = cfg[f'{name}_ids']
    ids = np.arange(lo, hi, dtype=np.int32)
    if len(ids) != int(np.prod(cfg['squares'])) // 2:
        raise ValueError('板ID数量与棋盘规格不一致')
    return cv2.aruco.CharucoBoard(tuple(cfg['squares']), cfg['square_length'],
                                cfg['marker_length'],
                                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100), ids)


def estimate(board, image, K, D, cfg):
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids) < cfg['min_corners']:
        return None
    obj = board.getChessboardCorners()[ids.ravel()]
    # 多行、多列角点才可解平面位姿。
    if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0)) < 2:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, D)
    if not ok:
        return None
    R = cv2.Rodrigues(rvec)[0]
    if np.any((obj @ R.T + tvec.reshape(3))[:, 2] <= 0):
        return None
    predicted = cv2.projectPoints(obj, rvec, tvec, K, D)[0]
    error = float(np.sqrt(np.mean(np.sum((predicted - corners) ** 2, axis=2))))
    if error > cfg['max_reprojection_px']:
        return None
    cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
    cv2.drawFrameAxes(image, K, D, rvec, tvec, .06)
    return R, tvec.reshape(3), error, len(ids)


def measure(pose1, pose2, cfg, up=None):
    R1, t1 = pose1[:2]
    center = np.array([cfg['squares'][0], cfg['squares'][1], 0.]) * cfg['square_length'] / 2
    local = center + np.asarray(cfg['board_center_to_a4_center']) + np.asarray(cfg['cube_center_from_board_center'])
    optical = R1 @ local + t1
    result = dict(optical_xyz_m=optical.tolist(), optical_distance_m=float(np.linalg.norm(optical)),
                  car_xyz_m=None, car_distance_m=None, car_horizontal_m=None,
                  car_source=None, provisional=not cfg['cube_mount_confirmed'],
                  reprojection_px=pose1[2], corners=pose1[3])
    car = None
    if pose2 is not None and cfg['R_car_board2'] is not None and cfg['t_car_board2'] is not None:
        R = np.asarray(cfg['R_car_board2'], dtype=float)
        t = np.asarray(cfg['t_car_board2'], dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError('R_car_board2必须为旋转矩阵，t必须为三维向量')
        car = R @ (pose2[0].T @ (optical - pose2[1])) + t
        result['car_source'] = 'board2'
    elif up is not None:
        from cube.cap_pose import _level_from_camera
        car = _level_from_camera(optical, up) + np.asarray(cfg['optical_origin_car'])
        result['car_source'] = 'camera_mount_gravity'
        result['provisional'] |= not cfg['camera_mount_confirmed']
    if car is not None:
        result.update(car_xyz_m=car.tolist(), car_distance_m=float(np.linalg.norm(car)),
                      car_horizontal_m=float(np.linalg.norm(car[:2])))
    return result


def main():
    parser = argparse.ArgumentParser(description='顶块中心相对车右前角、彩色光心的实时坐标与距离')
    parser.add_argument('--config', default=str(ROOT / 'config/charuco_measure.yaml'))
    parser.add_argument('--intrinsics', default=str(ROOT / 'config/intrinsics.yaml'))
    parser.add_argument('--source', default='realsense', help='realsense、USB索引或视频路径')
    parser.add_argument('--serial', default='244622071936')
    parser.add_argument('--headless', action='store_true', help='仅输出JSON，不开窗口')
    parser.add_argument('--jsonl', help='保存逐帧JSON结果')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    intr = yaml.safe_load(Path(args.intrinsics).read_text())
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]], dtype=float)
    D = np.asarray(intr['D'], dtype=float)
    if intr.get('distortion_model') == 'inverse_brown_conrady' and np.any(D):
        raise SystemExit('非零逆Brown畸变不能直接用于solvePnP，请提供整流图像对应内参')
    boards = [make_board(cfg, name) for name in ('board1', 'board2')]
    source = None
    log = None
    try:
        if args.source == 'realsense':
            from camera.source import RealSenseSource
            if args.serial != '244622071936' and Path(args.intrinsics).resolve() == ROOT / 'config/intrinsics.yaml':
                raise ValueError('更换相机必须用--intrinsics提供对应标定文件')
            source = RealSenseSource(width=intr['width'], height=intr['height'], serial=args.serial)
        else:
            source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            if not source.isOpened():
                raise RuntimeError('无法打开视频源')
            source.set(cv2.CAP_PROP_FRAME_WIDTH, intr['width'])
            source.set(cv2.CAP_PROP_FRAME_HEIGHT, intr['height'])
        log = open(args.jsonl, 'a', encoding='utf-8') if args.jsonl else None
        last_print = 0.
        while True:
            up = None
            if args.source == 'realsense':
                frame = source.read()
                image = frame.color.copy()
                if frame.gravity is not None:
                    up = -frame.gravity
            else:
                ok, image = source.read()
                if not ok:
                    break
            if image.shape[:2] != (intr['height'], intr['width']):
                raise ValueError('图像分辨率与内参不一致，禁止直接套用')
            # 两块板都在原图检测，避免第一块的绘制影响第二块。
            views = [image.copy(), image.copy()]
            poses = [estimate(b, v, K, D, cfg) for b, v in zip(boards, views)]
            image = views[0]
            result = measure(poses[0], poses[1], cfg, up) if poses[0] is not None else {'status': 'board1_not_detected'}
            result['timestamp_s'] = time.time()
            if log:
                log.write(json.dumps(result, allow_nan=False) + '\n')
                log.flush()
            if time.monotonic() - last_print > .2:
                print(json.dumps(result, allow_nan=False), flush=True)
                last_print = time.monotonic()
            if not args.headless:
                lines = ['TASK2: cube geometric center | q / Esc: quit']
                if poses[0] is None:
                    lines += ['Board 1 missing / insufficient corners / bad pose']
                else:
                    p = result['optical_xyz_m']
                    lines += [f'Optical XYZ mm: {np.round(np.array(p)*1000,1)}',
                              f'Optical distance: {result["optical_distance_m"]*1000:.1f} mm']
                    if result['car_xyz_m'] is not None:
                        lines += [f'Car XYZ mm: {np.round(np.array(result["car_xyz_m"])*1000,1)}',
                                  f'Car distance: {result["car_distance_m"]*1000:.1f} mm | XY: {result["car_horizontal_m"]*1000:.1f} mm',
                                  f'Source: {result["car_source"]}']
                    else:
                        lines += ['Car: unavailable (need board2 mounting or valid gravity)']
                    lines += [f'Board1 RMS: {poses[0][2]:.2f}px | Board2: {poses[1] is not None}',
                              'PROVISIONAL: verify mounting offsets' if result['provisional'] else 'Mounting confirmed']
                for i, line in enumerate(lines):
                    cv2.putText(image, line, (15, 28 + i*28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,0,0), 4)
                    cv2.putText(image, line, (15, 28 + i*28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,255,255), 1)
                cv2.imshow('Task2 ChArUco measurement', image)
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    break
    except (RuntimeError, ValueError, ImportError, cv2.error) as exc:
        raise SystemExit(f'测量停止：{exc}') from exc
    finally:
        if source is not None:
            source.close() if args.source == 'realsense' else source.release()
        if log:
            log.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
