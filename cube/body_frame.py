"""当前车身右前点坐标；X右、Y前、Z上，不累计车身运动。"""
import numpy as np
from geom.transform import fold_yaw_quarter

DEFAULT_OPTICAL_CENTER = np.array([-.211, .05, .669])
LEVEL_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def body_mount(up, cfg=None):
    cfg = cfg or {}
    upward = np.asarray(up, dtype=float)
    if upward.shape != (3,) or not np.isfinite(upward).all() or np.linalg.norm(upward) < 1e-6:
        raise ValueError('姿态无效，不能输出车身坐标')
    upward = upward / np.linalg.norm(upward)
    right = np.array([1., 0., 0.])
    right -= (right @ upward) * upward
    if np.linalg.norm(right) < .05:
        raise ValueError('相机姿态无法确定车身轴向')
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(upward, right), upward))
    origin = np.asarray(cfg.get('camera_optical_center_m', DEFAULT_OPTICAL_CENTER), dtype=float).copy()
    lever = np.asarray(cfg.get('camera_pivot_to_optical_m', [0., 0., 0.]), dtype=float)
    if origin.shape != (3,) or lever.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(lever).all():
        raise ValueError('相机安装位置必须是三个有限数值')
    # 零位光心=转轴+lever；相机俯仰后按同一安装变换更新光心。
    origin += (rotation.T @ LEVEL_ROTATION - np.eye(3)) @ lever
    return rotation, origin


def body_point(point, up, cfg=None):
    rotation, origin = body_mount(up, cfg)
    return rotation.T @ np.asarray(point, dtype=float).reshape(3) + origin


def body_record(record, up, cfg=None):
    result = dict(record)
    point = body_point([record['x'], record['y'], record['z']], up, cfg)
    result.update(zip(('x', 'y', 'z'), map(float, point)))
    if record.get('edge') is not None:
        rotation, _ = body_mount(up, cfg)
        edge = rotation.T @ np.asarray(record['edge'], dtype=float)
        result['yaw'] = float(fold_yaw_quarter(np.arctan2(edge[1], edge[0])))
    result['frame'] = 'car_front_right'
    return result


def calibrate_cap(pose, up, cfg=None):
    if pose is not None:
        pose.world = body_point(pose.center, up, cfg)
        rotation, _ = body_mount(up, cfg)
        edge = rotation.T @ np.asarray(pose.edge, dtype=float)
        pose.yaw = float(fold_yaw_quarter(np.arctan2(edge[1], edge[0])))
    return pose
