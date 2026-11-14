"""双ChArUco板实时测量；运行说明见 CHARUCO_MEASURE.md。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache

ROOT = Path(__file__).resolve().parent


def make_board(cfg, name):
    lo, hi = cfg[f'{name}_ids']
    ids = np.arange(lo, hi, dtype=np.int32)
    if len(ids) != int(np.prod(cfg['squares'])) // 2:
        raise ValueError('板ID数量与棋盘规格不一致')
    return cv2.aruco.CharucoBoard(tuple(cfg['squares']), cfg['square_length'],
                                cfg['marker_length'],
                                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100), ids)


def estimate(board, image, K, D, cfg, *, draw_image=None):
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
    error = float(np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    if error > cfg['max_reprojection_px']:
        return None
    canvas = image if draw_image is None else draw_image
    cv2.aruco.drawDetectedCornersCharuco(canvas, corners.reshape(-1, 1, 2), ids.reshape(-1, 1))
    cv2.drawFrameAxes(canvas, K, D, rvec, tvec, .06)
    return R, tvec.reshape(3), error, len(ids)


def detect_boards(boards, image, K, D, cfg):
    """两次检测均读取未绘制原图，所有成功位姿绘制到同一显示图。"""
    canvas = image.copy()
    poses = [estimate(board, image, K, D, cfg, draw_image=canvas) for board in boards]
    return poses, canvas


def board_status(poses, cfg):
    states = {}
    for name, pose in zip(('board1', 'board2'), poses):
        states[name] = dict(detected=pose is not None,
                            corners=0 if pose is None else int(pose[3]),
                            reprojection_px=None if pose is None else float(pose[2]))
    states['board2_mount_configured'] = (cfg['R_car_board2'] is not None
                                        and cfg['t_car_board2'] is not None)
    return states


def board_status_lines(states):
    lines = []
    for name, label in (('board1', 'Board1 (cube)'), ('board2', 'Board2 (car)')):
        state = states[name]
        lines.append(f'{label}: OK | {state["corners"]} corners | RMS {state["reprojection_px"]:.2f}px'
                     if state['detected'] else f'{label}: missing / insufficient corners / bad pose')
    if not states['board2_mount_configured']:
        lines.append('Board2-to-car transform missing: dual-board car coordinates unavailable')
    return lines


def measure(pose1, pose2, cfg, up=None):
    R1, t1 = pose1[:2]
    center = np.array([cfg['squares'][0], cfg['squares'][1], 0.]) * cfg['square_length'] / 2
    local = center + np.asarray(cfg['board_center_to_a4_center']) + np.asarray(cfg['cube_center_from_board_center'])
    optical = R1 @ local + t1
    result = dict(optical_xyz_m=optical.tolist(), optical_distance_m=float(np.linalg.norm(optical)),
                  car_xyz_m=None, car_distance_m=None, car_horizontal_m=None,
                  car_source=None, provisional=not cfg['cube_mount_confirmed'],
                  reprojection_px=pose1[2], corners=pose1[3],
                  optical_axes=['right', 'down', 'forward'], car_axes=['right', 'forward', 'up'])
    heading = R1[:, 1]  # 文字正向看纸下方为顶块朝向基准，与2号板纸下方一致时yaw为0。
    result['optical_yaw_deg'] = projected_yaw(heading[0], heading[2])
    result['car_yaw_deg'] = None
    car = None
    car_heading = None
    if pose2 is not None and cfg['R_car_board2'] is not None and cfg['t_car_board2'] is not None:
        R = np.asarray(cfg['R_car_board2'], dtype=float)
        t = np.asarray(cfg['t_car_board2'], dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError('R_car_board2必须为旋转矩阵，t必须为三维向量')
        car = R @ (pose2[0].T @ (optical - pose2[1])) + t
        car_heading = R @ pose2[0].T @ heading
        result['car_source'] = 'board2'
        result['provisional'] |= not cfg.get('board2_mount_confirmed', False)
    elif up is not None:
        from cube.cap_pose import _level_from_camera
        car = _level_from_camera(optical, up) + np.asarray(cfg['optical_origin_car'])
        car_heading = _level_from_camera(heading, up)
        result['car_source'] = 'camera_mount_gravity'
        result['provisional'] |= not cfg['camera_mount_confirmed']
    if car is not None:
        result['car_yaw_deg'] = projected_yaw(car_heading[0], car_heading[1])
        result.update(car_xyz_m=car.tolist(), car_distance_m=float(np.linalg.norm(car)),
                      car_horizontal_m=float(np.linalg.norm(car[:2])))
    return result


def projected_yaw(right, forward):
    if np.hypot(right, forward) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(right, forward)))


def chinese_lines(result):
    lines = ['任务2：顶块体心位置（毫米）与 yaw（度）；Q / Esc 退出']
    for name, label in (('board1', '1号顶块板'), ('board2', '2号车上板')):
        state = result[name]
        lines.append(f'{label}：已识别，角点 {state["corners"]}，误差 {state["reprojection_px"]:.2f} 像素'
                     if state['detected'] else f'{label}：未识别、角点不足或位姿不合格')
    if not result['board2_mount_configured']:
        lines.append('缺少2号板到小车的安装变换，无法使用双板计算车坐标')
    for prefix, label, axes in (('optical', '相对摄像头光心', 'X右 / Y下 / Z前'),
                                ('car', '相对小车右前角', 'X右 / Y前 / Z上')):
        xyz = result.get(prefix + '_xyz_m')
        if xyz is None:
            lines.append(f'{label}：暂无有效结果')
            continue
        x, y, z = np.asarray(xyz) * 1000
        lines.append(f'{label}（{axes}）：X={x:.1f}，Y={y:.1f}，Z={z:.1f}')
        yaw = result.get(prefix + '_yaw_deg')
        yaw_text = '方向投影不足，无法确定' if yaw is None else f'{yaw:+.1f} 度'
        lines.append(f'{label}：yaw={yaw_text}，直线距离={result[prefix + "_distance_m"]*1000:.1f} 毫米')
    if result.get('car_source'):
        source = '双标定板' if result['car_source'] == 'board2' else '相机安装参数与重力'
        lines.append(f'车坐标来源：{source}')
    lines.append('yaw基准：1号纸下方；车前为0，向车右为正；光心yaw为相机XZ平面投影')
    if result.get('provisional'):
        lines.append('当前使用的安装参数尚未全部确认')
    return lines


@lru_cache(maxsize=1)
def chinese_font():
    return ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)


def draw_chinese(image, lines):
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    font = chinese_font()
    y = 8
    for line in lines:
        # 按实际像素宽度换行，避免长坐标和说明被窗口裁掉。
        row = ''
        for char in line:
            if row and draw.textlength(row + char, font=font) > image.shape[1] - 24:
                draw.text((12, y), row, font=font, fill=(255,255,0), stroke_width=1, stroke_fill=(0,0,0))
                y += 25
                row = ''
            row += char
        draw.text((12, y), row, font=font, fill=(255,255,0), stroke_width=1, stroke_fill=(0,0,0))
        y += 25
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def device_intrinsics(profile, rs):
    """从实际启用的彩色流读取，而非查询另一个分辨率的候选profile。"""
    i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    model = str(i.model).split('.')[-1]
    return dict(width=i.width, height=i.height, fx=i.fx, fy=i.fy,
                cx=i.ppx, cy=i.ppy, D=list(i.coeffs), distortion_model=model,
                serial=profile.get_device().get_info(rs.camera_info.serial_number),
                source='device')


def pose_intrinsics(intr):
    D = np.asarray(intr['D'], dtype=float)
    if np.any(D) and intr.get('distortion_model') not in ('brown_conrady', 'brown'):
        raise ValueError('该畸变模型不能直接用于solvePnP，请使用整流图像及对应内参')
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']], [0, 0, 1.]], dtype=float)
    return K, D


class RealSenseColorSource:
    """双板PnP只需要彩色流，避免深度/IMU组合无法匹配设备。"""

    def __init__(self, width=1280, height=720, serial=None):
        import pyrealsense2 as rs
        self._rs = rs
        self.pipeline = rs.pipeline()
        self.started = False
        config = rs.config()
        if serial:
            config.enable_device(str(serial))
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, 30)
        try:
            self.profile = self.pipeline.start(config)
            self.started = True
        except RuntimeError as exc:
            raise RuntimeError(f'彩色流 {width}x{height}@30 启动失败：{exc}；'
                               '请关闭其他相机程序，并检查设备支持的视频模式和USB连接') from exc

    def read(self):
        from types import SimpleNamespace
        frames = self.pipeline.wait_for_frames(5000)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError('未收到RealSense彩色图像')
        return SimpleNamespace(color=np.asanyarray(color.get_data()).copy(), gravity=None)

    def close(self):
        if self.started:
            self.started = False
            self.pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description='顶块中心相对车右前角、彩色光心的实时坐标与距离')
    parser.add_argument('--config', default=str(ROOT / 'config/charuco_measure.yaml'))
    parser.add_argument('--intrinsics', default=str(ROOT / 'config/intrinsics.yaml'))
    parser.add_argument('--source', default='realsense', help='realsense、USB索引或视频路径')
    parser.add_argument('--serial', default=None, help='可选：指定RealSense序列号')
    parser.add_argument('--intrinsics-source', choices=['device', 'file'], default='device', help='RealSense默认读取设备内参')
    parser.add_argument('--read-intrinsics', action='store_true', help='仅读取并显示设备内参后退出')
    parser.add_argument('--save-intrinsics', help='将本次使用的内参保存为YAML')
    parser.add_argument('--headless', action='store_true', help='仅终端中文输出，不开窗口；--jsonl保存数值JSON')
    parser.add_argument('--jsonl', help='保存逐帧JSON结果')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    intr = None
    boards = [make_board(cfg, name) for name in ('board1', 'board2')]
    source = None
    log = None
    try:
        if args.source == 'realsense':
            file_intr = (yaml.safe_load(Path(args.intrinsics).read_text())
                         if args.intrinsics_source == 'file' else None)
            if file_intr is not None and (not args.serial or str(file_intr.get('serial', '244622071936')) != args.serial):
                raise ValueError('文件内参模式需--serial明确指定与文件匹配的相机')
            source = RealSenseColorSource(width=file_intr['width'] if file_intr else 1280,
                                         height=file_intr['height'] if file_intr else 720,
                                         serial=args.serial)
            intr = device_intrinsics(source.profile, source._rs) if file_intr is None else file_intr

        else:
            if args.read_intrinsics:
                raise ValueError('USB/视频不能通过RealSense接口读取内参，请提供标定文件')
            intr = yaml.safe_load(Path(args.intrinsics).read_text())
            source = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
            if not source.isOpened():
                raise RuntimeError('无法打开视频源')
            source.set(cv2.CAP_PROP_FRAME_WIDTH, intr['width'])
            source.set(cv2.CAP_PROP_FRAME_HEIGHT, intr['height'])
        print(json.dumps({'intrinsics': intr}, ensure_ascii=False), flush=True)
        if args.save_intrinsics:
            Path(args.save_intrinsics).write_text(yaml.safe_dump(intr, allow_unicode=True), encoding='utf-8')
        if args.read_intrinsics:
            return
        K, D = pose_intrinsics(intr)
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
            poses, image = detect_boards(boards, image, K, D, cfg)
            result = measure(poses[0], poses[1], cfg, up) if poses[0] is not None else {'status': 'board1_not_detected'}
            states = board_status(poses, cfg)
            result.update(states)
            result['timestamp_s'] = time.time()
            if log:
                log.write(json.dumps(result, allow_nan=False) + '\n')
                log.flush()
            if time.monotonic() - last_print > .2:
                print(' | '.join(chinese_lines(result)), flush=True)
                last_print = time.monotonic()
            if not args.headless:
                image = draw_chinese(image, chinese_lines(result))
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
        if not args.headless and not args.read_intrinsics:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
