"""白盒贴到画面边时，给出还要转的俯仰和原地旋转。

俯仰绕相机右轴：低头是镜头下俯，抬头是上仰。旋转绕竖直轴，小车不平移：
右转、左转。两个角度都是还要再转的大小。白盒完整留在画面里就不再转。
被切掉的一侧按方块对角线补回去，再让补全后的中心落到光轴上。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cube.detect import _color_layers
from geom.transform import deproject_pixels


_DEADBAND_RAD = np.deg2rad(3.0)
# 顶块离开画面边不到这么多，才把它收回去。已经更靠里就保持，不转到画面中心。
_FRAME_CLEARANCE_RAD = np.deg2rad(3.0)
# 顶块贴边时，3 度以内停止，再大就写成 6 度。
_CAP_IGNORE_RAD = np.deg2rad(3.0)
_CAP_COMMAND_RAD = np.deg2rad(6.0)
# 顶面正方形对角线。补到这个跨度，被切掉的一截才算回来。
_DIAGONAL = float(np.sqrt(2.0))


@dataclass(frozen=True)
class AxisMove:
    """一个轴上的调整。direction 为 none 时角度为 0。"""

    direction: str
    radians: float
    degrees: float
    signed_radians: float

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "radians": self.radians,
            "degrees": self.degrees,
            "signed_radians": self.signed_radians,
        }


def _none_move() -> AxisMove:
    return AxisMove("none", 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ViewAdjustment:
    """这一帧要的俯仰和旋转。ready 为真时两个轴都不用再转。"""

    visible: bool
    ready: bool
    reason: str
    pitch: AxisMove
    yaw: AxisMove
    clipped: dict

    def as_dict(self) -> dict:
        return {
            "visible": self.visible,
            "ready": self.ready,
            "reason": self.reason,
            "pitch": self.pitch.as_dict(),
            "yaw": self.yaw.as_dict(),
            "clipped": dict(self.clipped),
        }


def _cap_command(signed: float) -> float:
    """顶块贴边的输出。3 度以内停止，超过就写成 6 度，方向不变。"""
    if abs(signed) <= _CAP_IGNORE_RAD + 1e-12:
        return 0.0
    return float(np.copysign(_CAP_COMMAND_RAD, signed))


def _move(kind: str, signed: float) -> AxisMove:
    # 不超过 3° 就当成已经对准，避免在 0 附近来回给相反方向。
    if abs(signed) <= _DEADBAND_RAD + 1e-12:
        return _none_move()
    if kind == "pitch":
        direction = "down" if signed > 0.0 else "up"
    else:
        direction = "right" if signed > 0.0 else "left"
    magnitude = float(abs(signed))
    return AxisMove(direction, magnitude, float(np.degrees(magnitude)), float(signed))


def _level_axes(gravity: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 重力在彩色相机系里指向下。没有重力时按光轴水平：图像向下就是重力方向。
    down = np.array([0.0, 1.0, 0.0]) if gravity is None else np.asarray(gravity, dtype=float)
    norm = float(np.linalg.norm(down))
    if norm < 1e-6:
        down = np.array([0.0, 1.0, 0.0])
    else:
        down = down / norm
    up = -down
    optical_x = np.array([1.0, 0.0, 0.0])
    right = optical_x - float(np.dot(optical_x, up)) * up
    if float(np.linalg.norm(right)) < 0.25:
        right = np.cross(np.array([0.0, 0.0, 1.0]), up)
    right = right / np.linalg.norm(right)
    forward = np.cross(up, right)
    forward = forward / np.linalg.norm(forward)
    return right, forward, up


def _largest_mask(strict: np.ndarray, min_pixels: int) -> np.ndarray | None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = None
    best_area = min_pixels - 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best = label
            best_area = area
    if best is None:
        return None
    return np.where(labels == best, np.uint8(255), np.uint8(0))


def _clipped_sides(mask: np.ndarray, margin: int) -> dict:
    height, width = mask.shape
    band = max(int(margin), 1)
    on = mask > 0
    return {
        "left": bool(np.any(on[:, :band])),
        "right": bool(np.any(on[:, width - band :])),
        "top": bool(np.any(on[:band, :])),
        "bottom": bool(np.any(on[height - band :, :])),
    }


def _extend_side(
    start: float,
    end: float,
    clipped_low: bool,
    clipped_high: bool,
    expected: float,
    limit: float,
) -> tuple[float, float]:
    # 两侧都被切住时，方块比画面宽，往两边补会把中心留在原地。
    if clipped_low == clipped_high:
        return start, end
    span = end - start
    missing = max(0.0, expected - span)
    if missing == 0.0:
        missing = 0.15 * expected
    missing = min(missing, limit)
    if clipped_high:
        return start, end + missing
    return start - missing, end


def _mask_distance(mask: np.ndarray, depth: np.ndarray, zmin: float, zmax: float) -> float | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(valid.sum()) < 10:
        return None
    distance = float(np.median(z[valid]))
    if not np.isfinite(distance) or distance <= 1e-3:
        return None
    return distance


def _too_close(
    mask: np.ndarray,
    depth: np.ndarray,
    intr,
    cube_size: float,
    zmin: float,
    zmax: float,
    margin: int,
) -> bool:
    # 一个边长在画面上已经比画面更高或更宽，整块放不进去。这时不再补全、不再转。
    # 没有深度时，白块已经顶到对边并且占满这一向，同样算近处。
    height, width = mask.shape[:2]
    distance = _mask_distance(mask, depth, zmin, zmax)
    if distance is not None:
        side_u = float(cube_size) * float(intr.fx) / distance
        side_v = float(cube_size) * float(intr.fy) / distance
        return side_u >= float(width) or side_v >= float(height)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return False
    clipped = _clipped_sides(mask, margin)
    span_u = float(xs.max() - xs.min() + 1)
    span_v = float(ys.max() - ys.min() + 1)
    vertical = (clipped["top"] or clipped["bottom"]) and span_v >= 0.85 * float(height)
    horizontal = (clipped["left"] or clipped["right"]) and span_u >= 0.85 * float(width)
    return vertical or horizontal


def _aim_pixel(mask, depth, intr, cube_size: float, zmin: float, zmax: float, margin: int):
    ys, xs = np.nonzero(mask)
    clipped = _clipped_sides(mask, margin)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    distance = float(np.median(z[valid])) if int(valid.sum()) >= 10 else None
    u0 = float(xs.min())
    u1 = float(xs.max())
    v0 = float(ys.min())
    v1 = float(ys.max())
    if distance is None:
        expect_u = 2.0 * max(u1 - u0, 1.0)
        expect_v = 2.0 * max(v1 - v0, 1.0)
    else:
        expect_u = _DIAGONAL * cube_size * float(intr.fx) / distance
        expect_v = _DIAGONAL * cube_size * float(intr.fy) / distance
    u0, u1 = _extend_side(u0, u1, clipped["left"], clipped["right"], expect_u, float(intr.width))
    v0, v1 = _extend_side(v0, v1, clipped["top"], clipped["bottom"], expect_v, float(intr.height))
    return (u0 + u1) * 0.5, (v0 + v1) * 0.5, clipped


def _angles_toward(point: np.ndarray, gravity: np.ndarray | None) -> tuple[float, float]:
    # 正值：俯仰是低头，旋转是向右。
    right, forward, up = _level_axes(gravity)
    x_l = float(np.dot(point, right))
    y_l = float(np.dot(point, forward))
    z_l = float(np.dot(point, up))
    yaw = float(np.arctan2(x_l, y_l))
    down = -up
    current_down = float(np.arcsin(np.clip(float(down[2]), -1.0, 1.0)))
    desired_down = float(np.arctan2(-z_l, np.hypot(x_l, y_l)))
    return desired_down - current_down, yaw


def _blank() -> dict:
    return {"left": False, "right": False, "top": False, "bottom": False}


def plan_view(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    colors: list[dict],
    cube_size: float = 0.35,
    gravity: np.ndarray | None = None,
    zmin: float = 0.2,
    zmax: float = 3.0,
    min_pixels: int = 40,
    edge_margin: int = 4,
) -> ViewAdjustment:
    """白盒贴到画面边时，给出把补全后的中心收到光轴上要转的俯仰和旋转。"""
    strict, _loose, _front = _color_layers(color_bgr, colors)
    # 视线距离以外的深度已经是空洞。没有这段深度的白不拿来对准，避免对着远处白板转。
    depth_ok = np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)
    strict = np.where(depth_ok, strict, np.uint8(0))
    mask = _largest_mask(strict, min_pixels)
    if mask is None:
        return ViewAdjustment(False, True, "no_cube", _none_move(), _none_move(), _blank())
    u, v, clipped = _aim_pixel(mask, depth, intr, cube_size, zmin, zmax, edge_margin)
    if not any(clipped.values()):
        return ViewAdjustment(True, True, "centered", _none_move(), _none_move(), clipped)
    if _too_close(mask, depth, intr, cube_size, zmin, zmax, edge_margin):
        return ViewAdjustment(True, True, "close", _none_move(), _none_move(), clipped)
    point = deproject_pixels(np.array([[u, v]]), np.array([1.0]), intr)[0]
    pitch_signed, yaw_signed = _angles_toward(point, gravity)
    pitch = _move("pitch", pitch_signed)
    yaw = _move("yaw", yaw_signed)
    ready = pitch.direction == "none" and yaw.direction == "none"
    reason = "fills_frame" if ready else "partial"
    return ViewAdjustment(True, ready, reason, pitch, yaw, clipped)


def _pixel_rows(pixels) -> np.ndarray:
    if pixels is None:
        return np.zeros((0, 2), dtype=float)
    uv = np.asarray(pixels, dtype=float)
    if uv.size == 0:
        return np.zeros((0, 2), dtype=float)
    return uv.reshape(-1, 2)


def _bearing(u: float, v: float, intr, gravity, axis: str) -> float:
    point = deproject_pixels(np.array([[u, v]], dtype=float), np.array([1.0]), intr)[0]
    pitch, yaw = _angles_toward(point, gravity)
    return yaw if axis == "yaw" else pitch


def keep_in_frame(pixels, intr, gravity: np.ndarray | None = None) -> ViewAdjustment:
    """顶块留在画面里要转的俯仰和旋转。已经在画面里就保持，不把体心转到光轴上。

    看的是看见的红蓝像素。某一侧离画面边不到大约 3 度才考虑转。
    算出还要补的在 2 度以内就保持；超过 2 度，方向不变，角度写成 6 度。
    左右或上下同时贴边时，这一轴保持。
    """
    uv = _pixel_rows(pixels)
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0 or intr is None:
        return ViewAdjustment(False, True, "no_pixels", _none_move(), _none_move(), _blank())
    right = uv[int(np.argmax(uv[:, 0]))]
    left = uv[int(np.argmin(uv[:, 0]))]
    top = uv[int(np.argmin(uv[:, 1]))]
    bottom = uv[int(np.argmax(uv[:, 1]))]
    clear = _FRAME_CLEARANCE_RAD

    def row(value: float) -> float:
        return float(np.clip(value, 0.0, intr.height - 1))

    def col(value: float) -> float:
        return float(np.clip(value, 0.0, intr.width - 1))

    right_gap = _bearing(float(right[0]), float(right[1]), intr, gravity, "yaw") - (
        _bearing(float(intr.width - 1), row(float(right[1])), intr, gravity, "yaw") - clear
    )
    left_gap = (
        _bearing(0.0, row(float(left[1])), intr, gravity, "yaw") + clear
    ) - _bearing(float(left[0]), float(left[1]), intr, gravity, "yaw")
    top_gap = (
        _bearing(col(float(top[0])), 0.0, intr, gravity, "pitch") + clear
    ) - _bearing(float(top[0]), float(top[1]), intr, gravity, "pitch")
    bottom_gap = _bearing(float(bottom[0]), float(bottom[1]), intr, gravity, "pitch") - (
        _bearing(col(float(bottom[0])), float(intr.height - 1), intr, gravity, "pitch") - clear
    )
    clipped = {
        "left": bool(left_gap > 0.0),
        "right": bool(right_gap > 0.0),
        "top": bool(top_gap > 0.0),
        "bottom": bool(bottom_gap > 0.0),
    }
    if clipped["left"] and clipped["right"]:
        yaw_signed = 0.0
    elif clipped["right"]:
        yaw_signed = float(right_gap)
    elif clipped["left"]:
        yaw_signed = -float(left_gap)
    else:
        yaw_signed = 0.0
    if clipped["top"] and clipped["bottom"]:
        pitch_signed = 0.0
    elif clipped["top"]:
        pitch_signed = -float(top_gap)
    elif clipped["bottom"]:
        pitch_signed = float(bottom_gap)
    else:
        pitch_signed = 0.0
    pitch = _move("pitch", _cap_command(pitch_signed))
    yaw = _move("yaw", _cap_command(yaw_signed))
    ready = pitch.direction == "none" and yaw.direction == "none"
    reason = "inside" if ready else "edge"
    return ViewAdjustment(True, ready, reason, pitch, yaw, clipped)


def aim_at(point: np.ndarray, gravity: np.ndarray | None = None) -> ViewAdjustment:
    """把这个点收到光轴上，还要再转的俯仰和原地旋转。已经在光轴上就保持。"""
    xyz = np.asarray(point, dtype=float).reshape(3)
    if not np.all(np.isfinite(xyz)) or float(xyz[2]) <= 1e-4:
        return ViewAdjustment(False, True, "no_point", _none_move(), _none_move(), _blank())
    pitch_signed, yaw_signed = _angles_toward(xyz, gravity)
    pitch = _move("pitch", pitch_signed)
    yaw = _move("yaw", yaw_signed)
    ready = pitch.direction == "none" and yaw.direction == "none"
    reason = "center" if ready else "offset"
    return ViewAdjustment(True, ready, reason, pitch, yaw, _blank())
