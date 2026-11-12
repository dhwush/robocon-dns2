"""地面平面。法向必须接近重力的反方向，从而把方块姿态收成只有 yaw。

方块点投影到这块地面上求足迹，不另拟合上表面。上表面 RMS、顶面与地面的 12° 夹角、
正方形残差都依赖顶面平面，这里不算。足迹的长边、短边和面积在方块检测里丢帧。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- 平面表示 -----------------------------------------------------------------
# 单位法向 normal 指向上，平面方程为 normal · x = offset。

@dataclass
class Plane:
    normal: np.ndarray
    offset: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        # 正值表示点在法向一侧。法向朝上时，方块上的点距离为正。
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts @ self.normal - self.offset

    def project(self, points: np.ndarray) -> np.ndarray:
        # 沿法向落到平面上，用来把可见点收成地面足迹。
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist = self.signed_distance(pts)
        return pts - dist[:, None] * self.normal

    def origin(self) -> np.ndarray:
        # 法向是单位向量时，normal * offset 正好落在平面上，当作平面坐标原点。
        return self.normal * self.offset

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        # 在平面内造一对正交基，供最小外接矩形使用。helper 避开与法向平行的轴。
        up = self.normal
        helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_x = np.cross(helper, up)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(up, axis_x)
        return axis_x, axis_y


# --- RANSAC 拟合 --------------------------------------------------------------
# 三点定一个平面，丢掉和重力夹角过大的假设。倾角仍不得超过 8°。
# 票数按地面上的格子计，近处挤在一起的像素不能把法向拉斜。
# 方块顶面和地面平行，所以在通过重力检查的候选里取最低的一层，避免把顶面当成地面。

def _plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    # 三点叉乘得到法向。三点共线时叉乘接近 0，这个假设作废。
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = float(normal @ points[0])
    return normal, offset


def _orient_up(normal: np.ndarray, offset: float, gravity: np.ndarray) -> tuple[np.ndarray, float, float]:
    # 重力向下，up 取反。法向如果朝下，连同 offset 一起翻过来，保证距离符号稳定。
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    if normal @ up < 0.0:
        normal = -normal
        offset = -offset
    cosine = float(np.clip(normal @ up, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    return normal, offset, angle


def _refit(points: np.ndarray) -> tuple[np.ndarray, float]:
    # 内点再用 SVD 重拟合。最小奇异向量是法向，比三点叉乘稳。
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    offset = float(normal @ center)
    return normal, offset


# 地面上大约 8 cm 一格。同一格里的像素再多，也只算一票，近处地面就不能盖过方块旁边那一片。
_CELL_M = 0.08


def _up_vector(gravity: np.ndarray) -> np.ndarray:
    up = -np.asarray(gravity, dtype=float)
    return up / np.linalg.norm(up)


def _cell_weights(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    # 投影到水平面后按格子计数。一格的总权重是 1，格子里的点平分这一票。
    height = points @ up
    flat = points - height[:, None] * up
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(up[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_x = np.cross(helper, up)
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(up, axis_x)
    u = np.floor(flat @ axis_x / _CELL_M).astype(np.int64)
    v = np.floor(flat @ axis_y / _CELL_M).astype(np.int64)
    key = u * np.int64(73856093) ^ v * np.int64(19349663)
    _unique, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(float)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    total = float(np.asarray(weights, dtype=float).sum())
    if total <= 1e-9:
        return float(np.median(values))
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(values[order][min(index, values.size - 1)])


def _prior_offset(points: np.ndarray, up: np.ndarray, weights: np.ndarray) -> float:
    # 低处那一层的加权中位高度。方块顶面更高，进不了这一层，密集的近处点也不能靠数量把高度拉走。
    height = points @ up
    cutoff = float(np.percentile(height, 40))
    low = height <= cutoff
    if int(low.sum()) < 30:
        low = np.ones(len(points), dtype=bool)
    return _weighted_median(height[low], weights[low])


def _ground_band(points: np.ndarray, up: np.ndarray, offset: float, max_angle_deg: float) -> np.ndarray:
    # 相对先验高度，倾角在 8° 以内时点还可能落在地面上。带外的墙和方块顶面不参与比例。
    height = points @ up
    horizontal = np.linalg.norm(points - height[:, None] * up, axis=1)
    slack = np.maximum(0.08, horizontal * np.sin(np.deg2rad(max_angle_deg)))
    return np.abs(height - offset) <= slack


def _accept_inliers(
    inliers: np.ndarray,
    band: np.ndarray,
    weights: np.ndarray,
    point_count: int,
) -> bool:
    # 至少 30 个点，并且格子票数达到地面带的 5%。只贴住近处一条的平面过不了后面的票数比较。
    count = int(inliers.sum())
    if count < max(30, int(0.05 * point_count)):
        return False
    band_mass = float(weights[band].sum())
    if band_mass <= 1e-9:
        return True
    return float(weights[inliers & band].sum()) >= 0.05 * band_mass


def fit_ground(
    points: np.ndarray,
    gravity: np.ndarray,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
    iterations: int = 200,
    seed: int = 0,
) -> Plane | None:
    # 法向以重力的反方向为先验，倾角仍不得超过 8°。票数按地面格子计，不按像素计。
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None
    up = _up_vector(gravity)
    weights = _cell_weights(pts, up)
    offset0 = _prior_offset(pts, up, weights)
    band = _ground_band(pts, up, offset0, max_angle_deg)
    band_index = np.flatnonzero(band)
    if band_index.size < 3:
        band_index = np.arange(len(pts))
        band = np.ones(len(pts), dtype=bool)
    candidates: list[tuple[float, float, np.ndarray]] = []

    def consider(normal: np.ndarray, offset: float, inliers: np.ndarray) -> None:
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle > max_angle_deg or not _accept_inliers(inliers, band, weights, len(pts)):
            return
        # 偏移用内点的中位高度，倾斜的假平面不能把锚点抛远。
        offset = float(np.median(pts[inliers] @ normal))
        inliers = np.abs(pts @ normal - offset) < thresh
        if not _accept_inliers(inliers, band, weights, len(pts)):
            return
        height = float(np.median(pts[inliers] @ up))
        mass = float(weights[inliers & band].sum())
        candidates.append((height, mass, inliers))

    horizontal = np.abs(pts @ up - offset0) < thresh
    consider(up, offset0, horizontal)
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        choice = rng.choice(band_index, size=3, replace=False)
        fitted = _plane_from_three(pts[choice])
        if fitted is None:
            continue
        normal, offset = fitted
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle > max_angle_deg:
            continue
        inliers = np.abs(pts @ normal - offset) < thresh
        consider(normal, offset, inliers)
    if not candidates:
        return None
    # 顶面和地面都水平。取最低的一层，5 cm 以内的候选里再选格子票数最多的。
    lowest = min(item[0] for item in candidates)
    near_floor = [item for item in candidates if item[0] <= lowest + 0.05]
    near_floor.sort(key=lambda item: item[1], reverse=True)
    for _height, _mass, inliers in near_floor:
        normal, offset = _refit(pts[inliers])
        normal, offset, angle = _orient_up(normal, offset, gravity)
        if angle <= max_angle_deg:
            offset = float(np.median(pts[inliers] @ normal))
            return Plane(normal=normal, offset=offset)
    return None


def plane_holds(
    points: np.ndarray,
    plane: Plane,
    gravity: np.ndarray,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
) -> bool:
    # 上一帧平面还贴着这块点云、法向仍朝上时，直接沿用，不再抽点和 SVD。
    # 格子票数要接近重力先验那一层。只贴住近处一条的平面不能继续沿用。
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return False
    _normal, _offset, angle = _orient_up(np.asarray(plane.normal, dtype=float), float(plane.offset), gravity)
    if angle > max_angle_deg:
        return False
    up = _up_vector(gravity)
    weights = _cell_weights(pts, up)
    offset0 = _prior_offset(pts, up, weights)
    band = _ground_band(pts, up, offset0, max_angle_deg)
    held = np.abs(pts @ plane.normal - plane.offset) < thresh
    if not _accept_inliers(held, band, weights, len(pts)):
        return False
    horizontal = np.abs(pts @ up - offset0) < thresh
    held_mass = float(weights[held & band].sum())
    horizontal_mass = float(weights[horizontal & band].sum())
    if horizontal_mass <= 1e-9:
        return True
    return held_mass >= 0.7 * horizontal_mass


# --- 深度图抽样 ---------------------------------------------------------------
# 按已知内参把像素反投影成相机系点，供地面拟合使用。stride 用来减点。

def _grid_points(depth: np.ndarray, intr, stride: int) -> np.ndarray:
    from geom.transform import deproject_pixels

    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us]
    good = np.isfinite(z)
    if not np.any(good):
        return np.zeros((0, 3))
    pixels = np.column_stack([us[good].ravel(), vs[good].ravel()])
    return deproject_pixels(pixels, z[good].ravel(), intr)


def sample_points(depth: np.ndarray, intr, stride: int = 8, gravity: np.ndarray | None = None) -> np.ndarray:
    # 隔 stride 个像素取一个，地面拟合不需要全分辨率。
    # 有重力时再取一套更密的点。格子计票之后，多出来的近处像素不会多算票，远处空格能被补上。
    coarse = _grid_points(depth, intr, stride)
    if gravity is None:
        return coarse
    fine_stride = max(int(stride) // 2, 1)
    if fine_stride >= int(stride):
        return coarse
    fine = _grid_points(depth, intr, fine_stride)
    if len(fine) == 0:
        return coarse
    if len(coarse) == 0:
        return fine
    return np.vstack([coarse, fine])


def reuse_stable_plane(
    fitted: Plane | None,
    stable: Plane | None,
    frames_left: int,
    turning: bool,
    jump_m: float,
    hold_frames: int,
) -> tuple[Plane | None, Plane | None, int]:
    """这一帧用哪块地面，以及还能沿用几帧。

    相机没有开始俯仰或原地转时，拟合失败或原点跳过 jump_m，继续用上一块合格平面。
    一开始转，就丢掉旧平面，避免把新画面投到转之前的地面上。
    """
    if turning:
        if fitted is None:
            return None, None, 0
        return fitted, fitted, hold_frames
    if fitted is not None and (
        stable is None or float(np.linalg.norm(fitted.origin() - stable.origin())) <= jump_m
    ):
        return fitted, fitted, hold_frames
    if stable is not None and frames_left > 0:
        return stable, stable, frames_left - 1
    if fitted is not None:
        return fitted, fitted, hold_frames
    return None, None, 0


# --- 重力与相机高度给出的地面 ---------------------------------------------------
# 相机离地高度由安装给定，重力方向来自加速度计，两者已经确定一块平面。
# 深度只在这一高度上下 window 以内找地面，用来修正高度和法向。不在整幅里取最低的水平面，
# 白盒顶离这一高度大约 0.35 m，不会被当成地面。
# 只收局部法向接近竖直的点。方块侧面最下面几厘米也落在这一高度附近，但它们是竖直面。
# 40° 仍丢掉竖直面，斜着看到的地板法向有噪声时能留下。

GROUND_NORMAL_MAX_DEG = 40.0


@dataclass
class GroundFix:
    plane: Plane
    # fit：在先验高度附近测到了地面。held：这一帧没测到，沿用上一次量到的离地高度。
    # imu：只用重力和相机高度。ransac：没有先验时的整幅拟合。
    source: str
    # 这一帧的向上方向。held 的法向用它，不用上一帧相机系里的平面。
    up: np.ndarray | None = None
    # 光心沿向上方向到地面的距离。没看到地面时沿用这个高度。
    height: float | None = None


def ground_prior(gravity: np.ndarray, camera_height: float) -> Plane:
    # 相机光心在地面以上 camera_height，所以光心到平面的有符号距离等于这个高度。
    up = -np.asarray(gravity, dtype=float)
    up = up / np.linalg.norm(up)
    return Plane(normal=up, offset=-float(camera_height))


def _tilt_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.asarray(a) @ np.asarray(b)), -1.0, 1.0))))


def _densest_level(heights: np.ndarray, window: float, bin_m: float = 0.01) -> float:
    # 地面在 1 cm 格里挤成一个峰。取最密的一格连同两侧各一格的中值。
    edges = np.arange(-window, window + bin_m, bin_m)
    hist, edges = np.histogram(heights, bins=edges)
    k = int(np.argmax(hist))
    band = heights[(heights >= edges[k] - bin_m) & (heights <= edges[k + 1] + bin_m)]
    return float(np.median(band))


def _refit_spread(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    # 与 _refit 相同，另外返回平面内较窄方向的标准差。地面只剩一条窄带时，法向绕带子方向定不住。
    center = points.mean(axis=0)
    _, s, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return normal, float(normal @ center), float(s[1] / np.sqrt(len(points)))


def _height_along(plane: Plane, up: np.ndarray) -> float:
    # 光心沿向上方向到平面的距离。法向略斜时，-offset 是垂直距离，不是这个高度。
    cosine = float(np.asarray(plane.normal, dtype=float) @ np.asarray(up, dtype=float))
    if cosine < 1e-3:
        return -float(plane.offset)
    return -float(plane.offset) / cosine


def fit_ground_near(
    points: np.ndarray,
    normals: np.ndarray | None,
    prior: Plane,
    window: float = 0.12,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
    min_points: int = 150,
    min_spread: float = 0.05,
) -> Plane | None:
    # 先验平面上下 window 以内、局部法向接近竖直的点才是地面候选。
    # 法向和先验差过 max_angle_deg 说明这些点不是地面，返回空。窄带定不住法向时只修高度。
    pts = np.asarray(points, dtype=float)
    up = np.asarray(prior.normal, dtype=float)
    good = np.all(np.isfinite(pts), axis=1)
    if normals is not None:
        local = np.abs(np.asarray(normals, dtype=float) @ up)
        good &= np.nan_to_num(local, nan=0.0) >= np.cos(np.radians(GROUND_NORMAL_MAX_DEG))
    pts = pts[good]
    if len(pts) < min_points:
        return None
    height = prior.signed_distance(pts)
    slab = np.abs(height) < window
    if int(slab.sum()) < min_points:
        return None
    normal = up.copy()
    offset = float(prior.offset) + _densest_level(height[slab], window)
    # 第一轮放宽，IMU 有一两度偏差时远处地面也能收进来，再逐轮收紧到 thresh。
    for tol in (0.025, 0.015, float(thresh)):
        inliers = np.abs(pts @ normal - offset) < tol
        if int(inliers.sum()) < min_points:
            return None
        sel = pts[inliers]
        fitted, fitted_offset, spread = _refit_spread(sel)
        if fitted @ up < 0.0:
            fitted, fitted_offset = -fitted, -fitted_offset
        if _tilt_deg(fitted, up) > max_angle_deg:
            return None
        if spread >= min_spread:
            normal, offset = fitted, fitted_offset
        else:
            normal, offset = up, float(np.median(sel @ up))
    if int(np.count_nonzero(np.abs(pts @ normal - offset) < float(thresh))) < min_points:
        return None
    if abs(offset - float(prior.offset)) > window:
        return None
    return Plane(normal=normal, offset=float(offset))


def ground_with_prior(
    points: np.ndarray,
    normals: np.ndarray | None,
    gravity: np.ndarray,
    camera_height: float,
    held: GroundFix | None = None,
    window: float = 0.12,
    thresh: float = 0.015,
    max_angle_deg: float = 8.0,
    min_points: int = 150,
) -> GroundFix:
    # 测到地面就用测到的。没测到时，沿用上次的离地高度，法向换成这一帧的重力。
    # 还没测到过才用安装高度。
    prior = ground_prior(gravity, camera_height)
    fitted = fit_ground_near(points, normals, prior, window, thresh, max_angle_deg, min_points)
    if fitted is not None:
        return GroundFix(
            plane=fitted,
            source="fit",
            up=prior.normal,
            height=_height_along(fitted, prior.normal),
        )
    if held is not None and held.source in ("fit", "held") and held.height is not None:
        plane = Plane(normal=prior.normal.copy(), offset=-float(held.height))
        return GroundFix(plane=plane, source="held", up=prior.normal.copy(), height=held.height)
    return GroundFix(plane=prior, source="imu", up=prior.normal)


def sample_grid(depth: np.ndarray, intr, stride: int = 8) -> np.ndarray:
    # 与 sample_points 同样的抽样，但保留网格，没有深度的位置是 NaN。局部法向要用上下左右的邻点。
    from geom.transform import deproject_pixels

    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = np.asarray(depth[vs, us], dtype=float)
    pts = deproject_pixels(np.column_stack([us.ravel(), vs.ravel()]), z.ravel(), intr)
    return pts.reshape(vs.shape + (3,))


def grid_normals(grid: np.ndarray) -> np.ndarray:
    # 左右、上下邻点之差叉乘得到局部法向。边上一圈和缺深度的位置是 NaN。符号不定，用的时候取绝对值。
    normals = np.full(grid.shape, np.nan, dtype=float)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return normals
    dx = grid[1:-1, 2:] - grid[1:-1, :-2]
    dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
    cross = np.cross(dx, dy)
    norm = np.linalg.norm(cross, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals[1:-1, 1:-1] = cross / norm
    return normals
