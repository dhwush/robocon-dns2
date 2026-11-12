"""多帧融合，以及队内赛评分说明（2026-10-28）的绝对误差计分。

位置用中值。无标记方块的 yaw 先按 90° 展开成连续角再取中值，输出前折回主值区间，
避免 ±45° 两侧被当成两个朝向。离群帧用绝对中位差丢掉。
计分不再用相对误差分档：X/Y 得分 = 5 − 误差(cm)，角度得分 = (5 − 误差(°)) ÷ 2。
分数连续、保留负分。单次每个小项都得正分这次才达标；任务通过要求每次每个小项都为正。
方块 yaw 仍考虑 90° 旋转对称，误差取直观上最小需要旋转的角度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geom.transform import fold_yaw_quarter, wrap_angle


@dataclass
class FusedPose:
    """多帧中值之后的世界系位姿。yaw 已折回主值区间。"""
    x: float
    y: float
    z: float
    yaw: float


# --- 多帧中值 -----------------------------------------------------------------

def _circular_median(angles: np.ndarray) -> float:
    # 普通中值会把 -179° 和 179° 算成差很远。这里在每个样本上量环形距离，取中位代价最小的那个。
    values = np.asarray(angles, dtype=float)
    best = values[0]
    best_cost = np.inf
    for candidate in values:
        delta = np.arctan2(np.sin(values - candidate), np.cos(values - candidate))
        cost = float(np.median(np.abs(delta)))
        if cost < best_cost:
            best_cost = cost
            best = candidate
    return float(best)


def _align_period(angles: np.ndarray, period: float) -> np.ndarray:
    # 以环形中值为参考，把每个角加减若干个周期，使它们落在同一段连续角上。
    ref = _circular_median(angles)
    turns = np.round((angles - ref) / period)
    return angles - turns * period


def fuse_poses(samples: list[dict], mad_scale: float = 3.0) -> FusedPose | None:
    # 丢掉位置和 yaw 的离群帧后取中值。一帧都没有时返回空，调用方继续等待。
    if not samples:
        return None
    xyz = np.array([[s["x"], s["y"], s["z"]] for s in samples], dtype=float)
    yaw = np.array([s["yaw"] for s in samples], dtype=float)
    quarter = any(bool(s.get("quarter")) for s in samples)
    # 位置离群：到中位点的距离，再用这组距离的绝对中位差做门槛。
    center = np.median(xyz, axis=0)
    offset = np.linalg.norm(xyz - center, axis=1)
    med = float(np.median(offset))
    mad = float(np.median(np.abs(offset - med))) + 1e-6
    if quarter:
        # 无标记方块每 90° 重合。先展开再比偏差，±45° 两侧才不会互相把对方当成离群。
        aligned = _align_period(yaw, np.pi / 2.0)
        yaw_delta = np.abs(aligned - float(np.median(aligned)))
    else:
        aligned = yaw
        yaw_ref = _circular_median(yaw)
        yaw_delta = np.abs(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref)))
    yaw_med = float(np.median(yaw_delta))
    yaw_mad = float(np.median(np.abs(yaw_delta - yaw_med))) + 1e-6
    keep = (offset <= med + mad_scale * mad) & (yaw_delta <= yaw_med + mad_scale * yaw_mad)
    if int(keep.sum()) < 1:
        keep = np.ones(len(samples), dtype=bool)
    xyz = xyz[keep]
    fused_xyz = np.median(xyz, axis=0)
    if quarter:
        fused_yaw = fold_yaw_quarter(float(np.median(aligned[keep])))
    else:
        fused_yaw = _circular_median(yaw[keep])
    return FusedPose(x=float(fused_xyz[0]), y=float(fused_xyz[1]), z=float(fused_xyz[2]), yaw=fused_yaw)


# --- 评分 ---------------------------------------------------------------------
# 2026-10-28 队内赛评分说明：X 得分 = 5 − X误差(cm)，Y 得分 = 5 − Y误差(cm)，
# 角度得分 = (5 − 角度误差(°)) ÷ 2。误差恰好 5 cm 或 5° 时该小项 0 分，不满足
# "每次都得正分"。多次测试每个小项单独平均，任务通过要求每次每个小项都为正。

# 得分过线归零的误差。
_PASS_LIMIT_CM = 5.0
_PASS_LIMIT_DEG = 5.0


def angle_delta(est: float, gt: float, quarter: bool) -> float:
    # 正方形棱每 90° 重合。有朝向标记时不再折返，只把差折进 (-π, π]。
    if quarter:
        return min(abs(wrap_angle(est - gt - k * np.pi / 2.0)) for k in range(-2, 3))
    return abs(wrap_angle(est - gt))


def position_score(error_cm: float) -> float:
    # X 或 Y 某一项的得分。误差单位是厘米，5 − 误差，保留负分。
    return float(_PASS_LIMIT_CM - abs(float(error_cm)))


def yaw_score(error_deg: float) -> float:
    # 角度小项的得分。误差单位是度，(5 − 误差) ÷ 2，保留负分。
    return float(0.5 * (_PASS_LIMIT_DEG - abs(float(error_deg))))


def score_trial(
    est_xy: np.ndarray,
    est_yaw: float,
    gt_xy: np.ndarray,
    gt_yaw: float,
    quarter: bool = True,
) -> dict:
    # 一次测试三个小项的误差与得分。位置输入单位是米，误差按厘米计；yaw 输入是弧度。
    # 得分连续、保留负分。三个小项都为正，这一次才达标。
    est = np.asarray(est_xy, dtype=float)
    gt = np.asarray(gt_xy, dtype=float)
    error_x_cm = abs(float(est[0] - gt[0])) * 100.0
    error_y_cm = abs(float(est[1] - gt[1])) * 100.0
    error_yaw_deg = float(np.degrees(angle_delta(est_yaw, gt_yaw, quarter)))
    score_x = position_score(error_x_cm)
    score_y = position_score(error_y_cm)
    score_yaw = yaw_score(error_yaw_deg)
    return {
        "error_x_cm": error_x_cm,
        "error_y_cm": error_y_cm,
        "error_yaw_deg": error_yaw_deg,
        "score_x": score_x,
        "score_y": score_y,
        "score_yaw": score_yaw,
        "passed": score_x > 0.0 and score_y > 0.0 and score_yaw > 0.0,
    }


def average_scores(trials: list[dict]) -> dict:
    # 每个小项单独求平均：平均分 = 各次得分之和 ÷ 次数。位置等价于 5 − 平均绝对误差。
    # 通过看每次，不看平均：任何一次有小项不大于 0，passed 就是 False。
    if not trials:
        return {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "mean_error_x_cm": 0.0,
            "mean_error_y_cm": 0.0,
            "mean_error_yaw_deg": 0.0,
            "passed": False,
            "trials": 0,
        }
    return {
        "x": float(np.mean([t["score_x"] for t in trials])),
        "y": float(np.mean([t["score_y"] for t in trials])),
        "yaw": float(np.mean([t["score_yaw"] for t in trials])),
        "mean_error_x_cm": float(np.mean([t["error_x_cm"] for t in trials])),
        "mean_error_y_cm": float(np.mean([t["error_y_cm"] for t in trials])),
        "mean_error_yaw_deg": float(np.mean([t["error_yaw_deg"] for t in trials])),
        "passed": all(bool(t["passed"]) for t in trials),
        "trials": len(trials),
    }
