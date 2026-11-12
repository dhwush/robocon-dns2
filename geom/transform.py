"""坐标变换：世界系 → 相机系 → 图像平面 → 像素。

按课件印刷体的右手系旋转矩阵实现。手写稿里绕 X 轴的公式和印刷体差一个 θ 的符号，
这里以印刷体为准。畸变使用正向径向多项式；课件正文把“畸变前/后”的名字写反了。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


# --- 单轴旋转与外参复合 -------------------------------------------------------
# Rx / Ry / Rz 是绕相机或世界坐标轴的旋转。外参旋转按课件写成 R = Rz(φ) Ry(θ) Rx(ψ)。
# 方块自身只有 yaw，对应 Rz，不再做三轴复合。

def Rx(theta: float) -> np.ndarray:
    # 印刷体：绕 X 时 Y/Z 用 [[cos, -sin], [sin, cos]]。手写稿把 sin 的符号写反了。
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def Ry(theta: float) -> np.ndarray:
    # 绕 Y：X 与 Z 耦合，中间一行保持 Y 不变。
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def Rz(theta: float) -> np.ndarray:
    # 绕 Z：方块 yaw 只用这一项。X/Y 在水平面内转，Z 不动。
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def compose_R(phi: float, theta: float, psi: float) -> np.ndarray:
    # 课件复合顺序是先 Rx(ψ)、再 Ry(θ)、最后 Rz(φ)。矩阵右乘，所以写成 Rz @ Ry @ Rx。
    # 展开后左上角是 cosφ·cosθ，与课件一致。
    return Rz(phi) @ Ry(theta) @ Rx(psi)


# --- 世界系与相机系 -----------------------------------------------------------
# X_c = R X_w + t。反变换 X_w = R^T (X_c - t)。基础模式 R = I、t = 0。

def world_to_camera(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # 单点和 N×3 都收成二维再乘，避免 (3,) @ (3,3) 的形状错误。
    # t 是世界原点在相机系里的位置。
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = (rotation @ pts.T).T + np.asarray(translation, dtype=float)
    return out[0] if single else out


def camera_to_world(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # R 是旋转矩阵，逆等于转置。先减去 t，再左乘 R^T。
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = (rotation.T @ (pts - np.asarray(translation, dtype=float)).T).T
    return out[0] if single else out


# --- 投影矩阵 -----------------------------------------------------------------
# M2 是 4×4 外参，M1 是 3×4 内参。无畸变时 s[u,v,1]^T = M1 M2 X_w。

def extrinsic_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # 齐次外参 M2 = [[R, t], [0, 0, 0, 1]]，方便和内参连乘。
    m2 = np.eye(4)
    m2[:3, :3] = rotation
    m2[:3, 3] = np.asarray(translation, dtype=float)
    return m2


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    # M1 是 3×4。最后一列是 0：这一步只做小孔缩放和平移，不使用第四个齐次分量。
    # fx = f/dX，fy = f/dY，(cx, cy) 是主点，对应课件里的 (u0, v0)。
    return np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def projection_matrix(fx: float, fy: float, cx: float, cy: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # 合成投影 M = M1 M2。得到的第三行是比例因子 s，像素要再除以 s。
    return intrinsics_matrix(fx, fy, cx, cy) @ extrinsic_matrix(rotation, translation)


# --- 已知内参 -----------------------------------------------------------------
# fx、fy、cx、cy 和 D = (k1, k2, p1, p2, k3) 只从配置文件读取，运行时不改。

@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    distortion_model: str

    @property
    def D(self) -> np.ndarray:
        # 顺序与课件一致：D = (k1, k2, p1, p2, k3)。k 是径向，p 是切向。
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=float)


def _intrinsics_from_raw(raw: dict, fx: float, fy: float, cx: float, cy: float) -> Intrinsics:
    k1, k2, p1, p2, k3 = raw["D"]
    return Intrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        k1=float(k1),
        k2=float(k2),
        p1=float(p1),
        p2=float(p2),
        k3=float(k3),
        distortion_model=str(raw["distortion_model"]),
    )


def load_intrinsics(path: str | Path) -> Intrinsics:
    # 读一份已经写明焦距和主点的文件。实机按序列号走 load_intrinsics_for_serial。
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _intrinsics_from_raw(raw, raw["fx"], raw["fy"], raw["cx"], raw["cy"])


def load_intrinsics_for_serial(serial: str, path: str | Path = "config/cameras.yaml") -> Intrinsics:
    # 序列号对上表里的一台。表外的号码不拿另一台的内参顶上。
    serial = str(serial)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cameras = {str(key): value for key, value in (raw.get("cameras") or {}).items()}
    spec = cameras.get(serial)
    if spec is None:
        known = ", ".join(sorted(cameras))
        raise KeyError(f"序列号 {serial} 没有彩色内参。表内有: {known}")
    return _intrinsics_from_raw(raw, spec["fx"], spec["fy"], spec["cx"], spec["cy"])


# --- 径向畸变 -----------------------------------------------------------------
# 正向：理想归一化坐标乘 (1 + k1 r^2 + k2 r^4 + k3 r^6)，再加切向项。
# brown：投影时正向加畸变，反投影时迭代求逆。
# inverse_brown_conrady（D435i）：反投影时直接套多项式，投影时迭代求逆。

def _distort(x: np.ndarray, y: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    # 输入是理想归一化坐标（X/Z, Y/Z），输出是畸变后的归一化坐标。
    r2 = x * x + y * y
    # 径向因子。k3 对应 r^6，D435i 在视场边缘才比较明显。
    radial = 1.0 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2
    # 切向。p1、p2 为 0 时这两项消失，符合“以径向为主”。
    x_t = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
    y_t = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
    return x * radial + x_t, y * radial + y_t


def _undistort_iter(xd: np.ndarray, yd: np.ndarray, intr: Intrinsics, steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    # 求理想坐标，使 _distort(x, y) 接近观测到的 (xd, yd)。
    # 每步用当前估计重算径向和切向，再 x = (xd - 切向) / 径向。
    x = np.array(xd, dtype=float, copy=True)
    y = np.array(yd, dtype=float, copy=True)
    for _ in range(steps):
        r2 = x * x + y * y
        radial = 1.0 + intr.k1 * r2 + intr.k2 * r2 * r2 + intr.k3 * r2 * r2 * r2
        x_t = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
        y_t = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
        x = (xd - x_t) / radial
        y = (yd - y_t) / radial
    return x, y


# --- 像素投影与反投影 ---------------------------------------------------------
# 小孔：x = X/Z，y = Y/Z，再 u = fx x_d + cx，v = fy y_d + cy。
# 反投影：x = (u - cx) z / fx，y = (v - cy) z / fy，z 为该像素深度。

def project_points(points_cam: np.ndarray, intr: Intrinsics) -> np.ndarray:
    # 相机系点 → 像素。先除以 Z 得到归一化平面，再按模型处理畸变，最后乘内参。
    pts = np.atleast_2d(np.asarray(points_cam, dtype=float))
    z = pts[:, 2]
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    model = intr.distortion_model
    if model == "brown":
        # OpenCV：像素上看到的是畸变后的坐标，所以投影时正向加畸变。
        xd, yd = _distort(x, y, intr)
    elif model == "inverse_brown_conrady":
        # RealSense：多项式写在反投影里。投影要先求逆，使反投影能回到原点。
        xd, yd = _undistort_iter(x, y, intr)
    elif model == "none":
        xd, yd = x, y
    else:
        raise ValueError(f"unknown distortion model {model}")
    u = intr.fx * xd + intr.cx
    v = intr.fy * yd + intr.cy
    return np.column_stack([u, v])


def deproject_pixels(pixels: np.ndarray, depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    # 像素 + 深度（米）→ 相机系点。深度来自 D435i 视差，这里不再重做立体匹配。
    pix = np.atleast_2d(np.asarray(pixels, dtype=float))
    z = np.asarray(depth, dtype=float).reshape(-1)
    # 先消掉主点和焦距，得到畸变模型所在的归一化坐标。
    xd = (pix[:, 0] - intr.cx) / intr.fx
    yd = (pix[:, 1] - intr.cy) / intr.fy
    model = intr.distortion_model
    if model == "brown":
        x, y = _undistort_iter(xd, yd, intr)
    elif model == "inverse_brown_conrady":
        # 该模型把多项式直接作用在像素归一化坐标上，得到理想方向。
        x, y = _distort(xd, yd, intr)
    elif model == "none":
        x, y = xd, yd
    else:
        raise ValueError(f"unknown distortion model {model}")
    # 射线方向 (x, y, 1) 乘上该像素的深度。
    return np.column_stack([x * z, y * z, z])


def project_linear(points_world: np.ndarray, intr: Intrinsics, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    # 无畸变的课件公式，用来核对 M1 M2。第三分量是 s，像素坐标要除掉它。
    pts = np.atleast_2d(np.asarray(points_world, dtype=float))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    m = projection_matrix(intr.fx, intr.fy, intr.cx, intr.cy, rotation, translation)
    proj = (m @ hom.T).T
    return proj[:, :2] / proj[:, 2:3]


# --- yaw 角 -------------------------------------------------------------------
# 方块只绕竖直轴转。纯色方块有 90° 对称，折到 (-45°, 45°]；多色时用正面消掉这层歧义。

def wrap_angle(angle: float) -> float:
    # 折到 (-π, π]，避免 359° 和 1° 被当成差 358°。
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def fold_yaw_quarter(angle: float) -> float:
    # 正方形棱每 90° 看起来一样。无标记时收到 [-π/4, π/4)。有朝向标记时不要调用。
    half = np.pi / 4.0
    period = np.pi / 2.0
    return float((angle + half) % period - half)


def yaw_about_z(direction_world: np.ndarray) -> float:
    # 棱在世界 XY 上的方向。atan2(Y, X) 是相对世界 X 轴的角，不是相机光轴角。
    d = np.asarray(direction_world, dtype=float)
    return float(np.arctan2(d[1], d[0]))
