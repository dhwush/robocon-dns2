"""合成彩色图和深度图，供没有相机时的自检使用。"""

from __future__ import annotations

import numpy as np

from geom.transform import Intrinsics, project_points, world_to_camera


# --- 与作业视场一致的内参 -----------------------------------------------------
# 水平 87°、垂直 58°，主点在图像中心。自检不使用配置文件里的 1280×720 数值。

def intrinsics_for(width: int, height: int, model: str = "inverse_brown_conrady") -> Intrinsics:
    # 半幅 / tan(半视场角) 就是焦距（像素）。和作业给出的 87°×58° 一致。
    fx = (width / 2.0) / np.tan(np.deg2rad(87.0 / 2.0))
    fy = (height / 2.0) / np.tan(np.deg2rad(58.0 / 2.0))
    return Intrinsics(width, height, float(fx), float(fy), width / 2.0, height / 2.0, 0, 0, 0, 0, 0, model)


# --- 光栅化 -------------------------------------------------------------------
# 同一像素保留更近的深度。面由均匀采样的点拼出。

def _paint(points: np.ndarray, colors: np.ndarray, intr: Intrinsics, depth: np.ndarray, image: np.ndarray) -> None:
    if len(points) == 0:
        return
    pixels = project_points(points, intr)
    z = points[:, 2]
    ui = np.rint(pixels[:, 0]).astype(int)
    vi = np.rint(pixels[:, 1]).astype(int)
    # 相机后方和画面外的点不画。
    inside = (z > 0.05) & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
    if not np.any(inside):
        return
    ui = ui[inside]
    vi = vi[inside]
    z = z[inside]
    cols = colors[inside]
    pix = vi * intr.width + ui
    # 按像素分组、组内按深度从小到大，unique 保留每组第一个，也就是最近的点。
    order = np.lexsort((z, pix))
    _, first = np.unique(pix[order], return_index=True)
    chosen = order[first]
    flat = vi[chosen] * intr.width + ui[chosen]
    depth.ravel()[flat] = z[chosen]
    image.reshape(-1, 3)[flat] = cols[chosen]


def _face(origin, axis_a, axis_b, color, samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    # axis 向量的长度已经是边长，参数从 -0.5 到 0.5 正好铺满这个面。
    aa = np.linspace(-0.5, 0.5, samples)
    bb = np.linspace(-0.5, 0.5, samples)
    grid_a, grid_b = np.meshgrid(aa, bb)
    pts = origin + grid_a.ravel()[:, None] * axis_a + grid_b.ravel()[:, None] * axis_b
    cols = np.repeat(np.array(color, dtype=np.uint8)[None, :], len(pts), axis=0)
    return pts, cols


# --- 俯视方块 -----------------------------------------------------------------
# 相机 +Z 指向地面，相机系就是世界系。可只画顶面或正面，用来试“看不清面”的分支。

def render_cube(
    intr: Intrinsics,
    bottom_center: np.ndarray,
    yaw: float,
    size: float = 0.2,
    ground_z: float = 0.7,
    top_bgr: tuple[int, int, int] = (255, 80, 40),
    front_bgr: tuple[int, int, int] | None = None,
    noise_m: float = 0.0,
    seed: int = 0,
    faces: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """俯视：相机 +Z 指向地面，这一帧的世界系就是相机系，不用文档 2.3 的水平向前旋转。"""
    image = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    # 地面和方块都要铺到大约一个像素，否则近处的开运算会把有空洞的顶面抹掉，地面点也会少到让平面贴上顶面。
    ground_step = max(float(ground_z) / float(intr.fx), 0.003)
    xs = np.arange(-1.4, 1.4, ground_step)
    ys = np.arange(-1.0, 1.0, ground_step)
    gx, gy = np.meshgrid(xs, ys)
    ground_pts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, ground_z)])
    ground_col = np.full((len(ground_pts), 3), 40, dtype=np.uint8)
    _paint(ground_pts, ground_col, intr, depth, image)

    c, s = np.cos(yaw), np.sin(yaw)
    axis_x = np.array([c, s, 0.0]) * size
    axis_y = np.array([-s, c, 0.0]) * size
    axis_z = np.array([0.0, 0.0, -size])
    near_z = max(float(ground_z) - float(size), 0.3)
    face_samples = max(120, int(np.ceil(2.0 * float(size) / (near_z / float(intr.fx)))))
    center = np.asarray(bottom_center, dtype=float).copy()
    center[2] = ground_z
    top_origin = center + np.array([0.0, 0.0, -size])
    parts = []
    if faces in ("all", "top"):
        parts.append(_face(top_origin, axis_x, axis_y, top_bgr, face_samples))
    side_color = front_bgr if front_bgr is not None else top_bgr
    if faces in ("all", "front"):
        parts.append(_face(center + axis_x * 0.5, axis_y, axis_z, side_color, face_samples))
    if faces == "all":
        parts.append(_face(center - axis_x * 0.5, axis_y, axis_z, top_bgr, face_samples))
        parts.append(_face(center + axis_y * 0.5, axis_x, axis_z, top_bgr, face_samples))
        parts.append(_face(center - axis_y * 0.5, axis_x, axis_z, top_bgr, face_samples))
    for pts, cols in parts:
        _paint(pts, cols, intr, depth, image)
    if noise_m > 0.0:
        rng = np.random.default_rng(seed)
        valid = np.isfinite(depth)
        depth[valid] += rng.normal(0.0, noise_m, size=int(valid.sum()))
        holes = rng.random(depth.shape) < 0.02
        depth[holes] = np.nan
    return image, depth


# --- 斜视场地 -----------------------------------------------------------------
# 地面点在世界系 z = 0，四条边画成深色直线，给外参标定用。

def render_field(
    intr: Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
    field_length: float,
    field_width: float,
    ground_samples: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((intr.height, intr.width, 3), 180, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    xs = np.linspace(-0.3, field_length + 0.3, ground_samples)
    ys = np.linspace(-0.3, field_width + 0.3, ground_samples)
    gx, gy = np.meshgrid(xs, ys)
    world = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    cam = world_to_camera(world, rotation, translation)
    cols = np.full((len(cam), 3), 180, dtype=np.uint8)
    _paint(cam[cam[:, 2] > 0.05], cols[cam[:, 2] > 0.05], intr, depth, image)
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [field_length, 0.0, 0.0],
            [field_length, field_width, 0.0],
            [0.0, field_width, 0.0],
        ]
    )
    cam_corners = world_to_camera(corners, rotation, translation)
    import cv2

    for i in range(4):
        pair = project_points(cam_corners[[i, (i + 1) % 4]], intr)
        p0 = tuple(np.rint(pair[0]).astype(int))
        p1 = tuple(np.rint(pair[1]).astype(int))
        cv2.line(image, p0, p1, (0, 0, 0), 4)
    return image, depth
