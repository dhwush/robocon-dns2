"""塔顶红蓝方块的体心。

顶面在相机上方，看不见。用侧面上红蓝贴在一起的那条线当侧面中心的高度，
再沿这面的法向向方块里面挪半个边长。不把白盒中心抬上去，也不把顶面投到地面上。
分界上没有深度时，用下层颜色里已有的深度定侧面，上沿像素的射线交到这个面上。
颜色分界仍失败时，改用塔身上的深度平面，不要求这些点先落在红蓝里。
深度也没有量准时，用黄线里离地更低的那条横边当方块最底端当场推算，不把这一眼写进最终结果。
底面再按黄线最靠下那一段的画面位置对齐：射线交到正面上，偏低就抬高，偏高就降低。
绿框画之前用正面内部的深度核对。贴不住时，旁边另一块成直角的竖直面是侧面，留下；跳开的点才丢掉。
黄线贴着交界时，往红蓝里面取深度。和旁边差得远的点少起作用，不拿它当边的端点。
深度和颜色对不上、盖不住这块颜色、前后差得太大时，改用颜色分界。
每个分界像素的射线交到离地 0.80 米，不再用这些深度。近处一块平整的深度留下。
侧面深度对不成体心时，也改用这套颜色交点，这一眼可以进入那 5 帧。
够 5 帧深度之后，中值只用深度。颜色帧只在深度还不够时补上。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cube.stack_look import CAP_COLOR_SPECS, _cap_faces, cap_color_masks
from cube.color_boundary import boundary_pixels
from fuse.median import fuse_poses
from geom.transform import deproject_pixels, fold_yaw_quarter, project_points, yaw_about_z
from ground.plane import Plane


# 边长 20 厘米，体心在侧面再往里半个边长。
_EDGE_M = 0.20
_HALF_M = 0.10
# 线长和 20 厘米差过这么多，就不当成一整条棱都看见了。
_LENGTH_TOL_M = 0.03
# 分界点离侧面超过 1 厘米就丢掉，棱上的混合深度不参与。
_PLANE_TOL_M = 0.01
# 体心高度用分界自己的高度。两层白上的分界在 0.80 米，允许上下 8 厘米。
# 只坐在一块 0.35 米白盒上时分界约 0.45 米，离这里还有 30 多厘米，进不来。
_CENTER_H_M = 0.80
_HEIGHT_TOL_M = 0.08
# 分界大概落在这一段。太低是地上或一层白上的红蓝，太高不是这块塔顶。
_SEAM_LO_M = 0.68
_SEAM_HI_M = 0.95
_MIN_SEAM = 16
# 端点贴着画面边这么近，就算这条棱出了画面。
_BORDER_PX = 6
# 朝上的颜色看侧面：离地更高的那一截。两色差不到 2 厘米时改看画面上下。
_COLOR_GAP_M = 0.02
_COLOR_MIN = 10
# 只剩一色，并且顶到画面最上这几行，看见的是下半截。
_COLOR_TOP_ROWS = 6
# 下层颜色要在水平和竖直方向都拉开这么多，才用来定侧面。
_FACE_SPAN_M = 0.05
# 大部分点贴住同一个面，就当成一个侧面。否则再拆成两个面。
_ONE_FACE_FRAC = 0.75
# 射线交点可以比已有深度再伸出大约一条边，再远就是飞到别处了。
_FILL_REACH_M = 0.22
# 画面上两色上下差这么多像素，才把上面那截当成顶面颜色。
_COLOR_ROW_GAP = 4.0
# 颜色分界失败后，深度平面收在塔身这一档，不把白盒正面收进来。
_DEPTH_LO_M = 0.62
_DEPTH_HI_M = 0.95
# 深度比彩色分界松。平整的侧面常常比 1 厘米再跳一点。
_DEPTH_TOL_M = 0.02
# 水平跨度太短是碎点，太长是白盒或墙。
_FACE_MIN_M = 0.08
_FACE_MAX_M = 0.30
# 高低中心低于这里的是白盒上沿，不是塔顶侧面。
_DEPTH_MEDIAN_LO = 0.72
# 小车停在地面、还没被推动时，彩色光心在世界系里的位置。向右、向前、向上。
_OPTICAL_WORLD = np.array([-0.211, 0.05, 0.669])
# 近于这么远、深度又对不成平面时，才改用颜色。颜色块比这个距离上的 20 厘米更大，也算近。
_CLOSE_M = 0.40
# 颜色估出的距离比深度中值近过这个比例，就认为深度偏远。
_DEPTH_AGREE = 0.20
# 红蓝上能用的深度少过这个比例，或者左右有一边几乎没有，就不用这块深度。
_DEPTH_COVER = 0.40
_HALF_COVER = 0.15
# 同一块颜色上，深度前后差过大约一条边长，就是鼓包或补出来的斜坡。
_DEPTH_SPAN_M = 0.25
# 散点贴不进这个容差，就不把它们当成一个侧面。
_RGB_PLANE_TOL_M = 0.02
_RGB_PLANE_FRAC = 0.60


@dataclass
class CapMeasure:
    """一帧量到的塔顶。center 在相机系，world 是车右前角为原点的世界系。"""

    center: np.ndarray
    world: np.ndarray
    yaw: float
    corners: np.ndarray
    edge: np.ndarray
    # depth：侧面深度量到的。close：近处深度不能用。side：侧面深度量不成，改用颜色。
    via: str = "depth"


def _level_axes(up: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """相机系里的向右、水平向前、向上。没给朝上方向时，按光轴水平。"""
    if up is None:
        upward = np.array([0.0, -1.0, 0.0])
    else:
        upward = np.asarray(up, dtype=float).reshape(3)
        length = float(np.linalg.norm(upward))
        upward = np.array([0.0, -1.0, 0.0]) if length < 1e-6 else upward / length
    optical_x = np.array([1.0, 0.0, 0.0])
    right = optical_x - float(np.dot(optical_x, upward)) * upward
    if float(np.linalg.norm(right)) < 0.25:
        right = np.cross(np.array([0.0, 0.0, 1.0]), upward)
    right = right / np.linalg.norm(right)
    forward = np.cross(upward, right)
    forward = forward / np.linalg.norm(forward)
    return right, forward, upward


def _level_from_camera(point: np.ndarray, up: np.ndarray | None) -> np.ndarray:
    """相机系的点或方向，改成向右、水平向前、向上。方向不要再加光心位置。"""
    cam = np.asarray(point, dtype=float).reshape(3)
    right, forward, upward = _level_axes(up)
    return np.array(
        [float(np.dot(cam, right)), float(np.dot(cam, forward)), float(np.dot(cam, upward))],
        dtype=float,
    )


def camera_to_car_world(
    point: np.ndarray,
    car_shift: np.ndarray | None = None,
    up: np.ndarray | None = None,
    camera_height: float | None = None,
) -> np.ndarray:
    """相机系向右、向下、向前，换成世界系向右、向前、向上。

    抬头或低头时，先用朝上方向把点放平，不再把光轴距离当成水平距离。
    光心在车右前角的右侧 0.211 米、前方 0.05 米、上方 0.669 米；明确给出 camera_height 时使用该高度。
    car_shift 是雷达给出的小车平移，加在世界系上。没有雷达时是 0。
    """
    origin = _OPTICAL_WORLD.copy()
    if camera_height is not None:
        origin[2] = float(camera_height)
    world = _level_from_camera(point, up) + origin
    if car_shift is not None:
        world = world + np.asarray(car_shift, dtype=float).reshape(3)
    return world


def optical_from_car_world(world: np.ndarray, car_shift: np.ndarray | None = None) -> np.ndarray:
    """相对小车右前角的点，改成相对彩色光心。轴仍是向右、向前、向上。"""
    origin = np.array(_OPTICAL_WORLD, dtype=float)
    if car_shift is not None:
        origin = origin + np.asarray(car_shift, dtype=float).reshape(3)
    return np.asarray(world, dtype=float).reshape(3) - origin


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        return None
    return np.asarray(vector, dtype=float) / length


def _horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(vector, dtype=float) - up * float(np.dot(vector, up))
    return _unit(flat)


def _color_height(mask, depth, intr, plane: Plane, zmin: float, zmax: float, anchor: np.ndarray | None):
    """这一色在塔顶高度里的离地中位数。点太少就不知道。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < _COLOR_MIN:
        return None
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < _COLOR_MIN:
        return None
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]).astype(float), z[good], intr)
    heights = plane.signed_distance(points)
    keep = (heights > _SEAM_LO_M) & (heights < _SEAM_HI_M)
    if anchor is not None:
        flat = plane.project(points)
        center = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
        keep = keep & (np.linalg.norm(flat - center, axis=1) <= 0.45)
    if int(keep.sum()) < _COLOR_MIN:
        return None
    return float(np.median(heights[keep]))


def _image_top_color(red: np.ndarray, blue: np.ndarray) -> str | None:
    """上面那一截没有深度时，看它在画面里是不是更靠上。"""

    def row(mask: np.ndarray) -> float | None:
        ys = np.nonzero(mask)[0]
        if len(ys) < _COLOR_MIN:
            return None
        return float(np.median(ys))

    red_row = row(red)
    blue_row = row(blue)
    if red_row is None or blue_row is None or abs(red_row - blue_row) < _COLOR_ROW_GAP:
        return None
    return "red" if red_row < blue_row else "blue"


def _cropped_top_color(red: np.ndarray, blue: np.ndarray) -> str | None:
    """只剩一色，而且顶着画面最上几行。看见的是下半截，朝上用另一色。"""

    def top(mask: np.ndarray) -> int | None:
        ys = np.nonzero(mask)[0]
        if len(ys) < _COLOR_MIN:
            return None
        return int(ys.min())

    red_top = top(red)
    blue_top = top(blue)
    if red_top is not None and blue_top is None and red_top <= _COLOR_TOP_ROWS:
        return "blue"
    if blue_top is not None and red_top is None and blue_top <= _COLOR_TOP_ROWS:
        return "red"
    return None


def _majority_color(red_n: int, blue_n: int) -> str | None:
    if max(red_n, blue_n) < _COLOR_MIN or red_n == blue_n:
        return None
    return "red" if red_n > blue_n else "blue"


def _other_color(name: str | None) -> str | None:
    if name == "red":
        return "blue"
    if name == "blue":
        return "red"
    return None


def cap_top_color(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    zmin: float,
    zmax: float,
    anchor: np.ndarray | None = None,
) -> str | None:
    """上表面在相机上方，看不见。分界上面、离地更高的那一截就是朝上的颜色。

    上面那一截没有深度，或两色离地差不多，就看哪一色在画面里更靠上。
    只剩一色并且顶着画面上沿时，看见的是下半截，朝上用另一色。
    """
    red, blue = _color_masks(color_bgr)
    colored = _cap_faces(cv2.bitwise_or(red, blue), depth, intr, zmin, zmax)
    if int(np.count_nonzero(colored)) < _COLOR_MIN:
        return None
    red = cv2.bitwise_and(red, colored)
    blue = cv2.bitwise_and(blue, colored)
    red_height = _color_height(red, depth, intr, plane, zmin, zmax, anchor)
    blue_height = _color_height(blue, depth, intr, plane, zmin, zmax, anchor)
    if red_height is not None and blue_height is not None:
        if red_height - blue_height > _COLOR_GAP_M:
            return "red"
        if blue_height - red_height > _COLOR_GAP_M:
            return "blue"
    seen = _image_top_color(red, blue)
    if seen is not None:
        return seen
    return _cropped_top_color(red, blue)


def box_upper_color(color_bgr: np.ndarray, wire: np.ndarray, intr) -> str | None:
    """绿框上半截还在画面里时，用那里更多的一色。

    上半截整段出画时，看得见的是下色，朝上用另一色。两色都少就不定。
    """
    pts = np.asarray(wire, dtype=float).reshape(-1, 3)
    front = pts[np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.05)]
    if len(front) < 3 or intr is None:
        return None
    pix = project_points(front, intr)
    pix = pix[np.isfinite(pix).all(axis=1)]
    if len(pix) < 3:
        return None
    above = float(np.min(pix[:, 1]))
    mid = 0.5 * (above + float(np.max(pix[:, 1])))
    hull = cv2.convexHull(np.round(pix).astype(np.int32))
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if int(np.count_nonzero(mask)) < _COLOR_MIN:
        return None
    red, blue = _color_masks(color_bgr)

    def counts(region: np.ndarray) -> tuple[int, int]:
        return (
            int(np.count_nonzero((red > 0) & (region > 0))),
            int(np.count_nonzero((blue > 0) & (region > 0))),
        )

    if mid < 0.0:
        return _other_color(_majority_color(*counts(mask)))
    upper = mask.copy()
    cut = int(np.ceil(mid))
    if 0 <= cut < upper.shape[0]:
        upper[cut:, :] = 0
    voted = _majority_color(*counts(upper))
    if voted is not None:
        return voted
    if above < 0.0:
        return _other_color(_majority_color(*counts(mask)))
    return None


def _color_masks(color_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cap_color_masks(color_bgr)


def _near_anchor(points: np.ndarray, plane: Plane, anchor: np.ndarray | None) -> np.ndarray:
    if anchor is None or len(points) == 0:
        return np.ones(len(points), dtype=bool)
    flat = plane.project(points)
    center = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
    return np.linalg.norm(flat - center, axis=1) <= 0.45


def _mask_points(mask, depth, intr, plane: Plane, zmin: float, zmax: float, anchor):
    """掩膜里有深度、又落在分界高度里的点。"""
    ys, xs = np.nonzero(mask)
    empty = np.zeros((0, 3)), np.zeros((0, 2), dtype=int), np.zeros(0)
    if len(xs) == 0:
        return empty
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) == 0:
        return empty
    pixels = np.column_stack([xs[good], ys[good]]).astype(int)
    points = deproject_pixels(pixels.astype(float), z[good], intr)
    heights = plane.signed_distance(points)
    keep = (heights > _SEAM_LO_M) & (heights < _SEAM_HI_M) & _near_anchor(points, plane, anchor)
    return points[keep], pixels[keep], heights[keep]


def _lower_mask(red: np.ndarray, blue: np.ndarray) -> np.ndarray | None:
    """画面里更靠下的那一截。它的上沿就是红蓝分界。"""

    def row(mask: np.ndarray) -> float | None:
        ys = np.nonzero(mask)[0]
        if len(ys) < _COLOR_MIN:
            return None
        return float(np.median(ys))

    red_row = row(red)
    blue_row = row(blue)
    if red_row is None and blue_row is None:
        return None
    if red_row is None:
        return blue
    if blue_row is None:
        return red
    return red if red_row > blue_row else blue


@dataclass
class _Side:
    """一块竖直侧面。方向沿着红蓝分界，法向水平。"""

    normal: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    median_z: float
    along_min: float
    along_max: float
    pixels: np.ndarray


def _fit_side(points: np.ndarray, pixels: np.ndarray, up: np.ndarray) -> tuple[_Side | None, float]:
    """用下层颜色的深度定一个竖直平面。贴不住或拉不开 5 厘米就不用。"""
    if len(points) < _MIN_SEAM:
        return None, 0.0
    flat = points - up * (points @ up)[:, None]
    center = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    normal = None if direction is None else _horizontal(np.cross(up, direction), up)
    if direction is None or normal is None:
        return None, 0.0
    offset = np.abs((flat - center) @ normal)
    keep = offset <= _PLANE_TOL_M
    if int(keep.sum()) < _MIN_SEAM:
        return None, 0.0
    kept = points[keep]
    flat = kept - up * (kept @ up)[:, None]
    center = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    normal = None if direction is None else _horizontal(np.cross(up, direction), up)
    if direction is None or normal is None:
        return None, 0.0
    along = (flat - center) @ direction
    vertical = kept @ up
    if float(along.max() - along.min()) < _FACE_SPAN_M or float(vertical.max() - vertical.min()) < _FACE_SPAN_M:
        return None, 0.0
    origin = center + up * float(np.median(vertical))
    side = _Side(
        normal=normal,
        origin=origin,
        direction=direction,
        median_z=float(np.median(kept[:, 2])),
        along_min=float(along.min()),
        along_max=float(along.max()),
        pixels=pixels[keep],
    )
    return side, float(keep.sum()) / float(len(points))


def _line_offset(points: np.ndarray, group: np.ndarray, up: np.ndarray) -> np.ndarray:
    """每个点到这一组所在直线的水平距离。"""
    flat_g = group - up * (group @ up)[:, None]
    center = np.median(flat_g, axis=0)
    _, _, vt = np.linalg.svd(flat_g - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    normal = None if direction is None else _horizontal(np.cross(up, direction), up)
    flat = points - up * (points @ up)[:, None]
    if normal is None:
        return np.full(len(points), np.inf)
    return np.abs((flat - center) @ normal)


def _side_planes(points: np.ndarray, pixels: np.ndarray, up: np.ndarray) -> list[_Side]:
    """一个侧面贴得住就用一个。贴不住再按折线拆成两个，拆不开仍用贴得更稳的那一个。"""
    if len(points) < _MIN_SEAM:
        return []
    side, frac = _fit_side(points, pixels, up)
    if side is not None and frac >= _ONE_FACE_FRAC:
        return [side]
    split = _assign_two_lines(points, up)
    if split is not None:
        group_a, group_b, _, _ = split
        nearer_a = _line_offset(points, group_a, up) <= _line_offset(points, group_b, up)
        sides: list[_Side] = []
        for mark in (nearer_a, ~nearer_a):
            if int(mark.sum()) < _MIN_SEAM:
                continue
            fitted, _frac = _fit_side(points[mark], pixels[mark], up)
            if fitted is not None:
                sides.append(fitted)
        if sides:
            return sides
    return [side] if side is not None else []


def _ray_hits(pixels: np.ndarray, side: _Side, intr) -> np.ndarray:
    """像素只定射线，交到这块竖直侧面上。"""
    dirs = deproject_pixels(pixels.astype(float), np.ones(len(pixels)), intr)
    denom = dirs @ side.normal
    level = float(side.normal @ side.origin)
    t = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    t[ok] = level / denom[ok]
    t[~(t > 0.05)] = np.nan
    return t[:, None] * dirs


def _nearest_side(pixels: np.ndarray, sides: list[_Side], shape: tuple[int, int], reach_px: float) -> np.ndarray:
    """每个分界像素归到画面上离它最近的那块侧面。远过一条边就不归。"""
    chosen = np.full(len(pixels), -1, dtype=int)
    best = np.full(len(pixels), np.inf)
    for index, side in enumerate(sides):
        mask = np.zeros(shape, dtype=np.uint8)
        ys = np.clip(side.pixels[:, 1], 0, shape[0] - 1)
        xs = np.clip(side.pixels[:, 0], 0, shape[1] - 1)
        mask[ys, xs] = 255
        dist = cv2.distanceTransform(np.where(mask > 0, np.uint8(0), np.uint8(255)), cv2.DIST_L2, 3)
        here = dist[np.clip(pixels[:, 1], 0, shape[0] - 1), np.clip(pixels[:, 0], 0, shape[1] - 1)]
        take = here < best
        best[take] = here[take]
        chosen[take] = index
    chosen[best > reach_px] = -1
    return chosen


def _fill_seam(pixels, depth, sides, intr, plane: Plane, zmin: float, zmax: float, anchor, up: np.ndarray):
    """有深度并且贴着侧面的分界留着。缺口用射线交到最近的侧面上。"""
    z = depth[pixels[:, 1], pixels[:, 0]]
    measured = np.zeros((len(pixels), 3))
    has = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(has.sum()):
        measured[has] = deproject_pixels(pixels[has].astype(float), z[has], intr)
    reach = _FILL_REACH_M * float(intr.fx) / max(float(np.median([side.median_z for side in sides])), 0.2)
    owner = _nearest_side(pixels, sides, depth.shape, reach)
    hits = np.full((len(pixels), 3), np.nan)
    for index, side in enumerate(sides):
        part = np.nonzero(owner == index)[0]
        if len(part) == 0:
            continue
        landed = _ray_hits(pixels[part], side, intr)
        flat = landed - up * (landed @ up)[:, None]
        origin = side.origin - up * float(side.origin @ up)
        along = (flat - origin) @ side.direction
        on_line = np.abs((flat - origin) @ side.normal) <= _PLANE_TOL_M
        on_span = (along >= side.along_min - _FILL_REACH_M) & (along <= side.along_max + _FILL_REACH_M)
        finite = np.isfinite(landed).all(axis=1)
        good = finite & on_line & on_span
        hits[part[good]] = landed[good]
    use_measured = has.copy()
    if int(has.sum()):
        # 测到的深度离所有侧面都超过 1 厘米，就是分界上的混合深度，改用射线。
        flat = measured - up * (measured @ up)[:, None]
        near = np.zeros(len(pixels), dtype=bool)
        for side in sides:
            origin = side.origin - up * float(side.origin @ up)
            near |= np.abs((flat - origin) @ side.normal) <= _PLANE_TOL_M
        use_measured &= near
    points = np.where(use_measured[:, None], measured, hits)
    finite = np.isfinite(points).all(axis=1)
    if int(finite.sum()) < _MIN_SEAM:
        return None
    points = points[finite]
    kept_pixels = pixels[finite]
    heights = plane.signed_distance(points)
    keep = (heights > _SEAM_LO_M) & (heights < _SEAM_HI_M) & _near_anchor(points, plane, anchor)
    if int(keep.sum()) < _MIN_SEAM:
        return None
    return points[keep], kept_pixels[keep], heights[keep]


def _seam_cloud(color_bgr, depth, intr, plane: Plane, zmin: float, zmax: float, anchor: np.ndarray | None):
    red, blue = _color_masks(color_bgr)
    colored = cv2.bitwise_or(red, blue)
    colored = _cap_faces(colored, depth, intr, zmin, zmax,
                         red=red, blue=blue, color_bgr=color_bgr)
    if int(np.count_nonzero(colored)) < 10:
        return None
    red = cv2.bitwise_and(red, colored)
    blue = cv2.bitwise_and(blue, colored)
    pixels = boundary_pixels(red, blue, color_bgr)
    if len(pixels) < _MIN_SEAM:
        return None
    up = _unit(np.asarray(plane.normal, dtype=float))
    lower = _lower_mask(red, blue)
    if up is not None and lower is not None:
        face_points, face_pixels, _face_heights = _mask_points(lower, depth, intr, plane, zmin, zmax, anchor)
        sides = _side_planes(face_points, face_pixels, up)
        if sides:
            filled = _fill_seam(pixels, depth, sides, intr, plane, zmin, zmax, anchor, up)
            if filled is not None:
                return filled
    return _measured_seam(pixels, depth, intr, plane, zmin, zmax, anchor)


def _measured_seam(pixels, depth, intr, plane: Plane, zmin: float, zmax: float, anchor):
    """分界面定不住时，仍只用分界上自己的深度。不够就丢掉。"""
    points, kept, heights = _mask_points(
        _pixels_mask(depth.shape, pixels), depth, intr, plane, zmin, zmax, anchor
    )
    if len(points) < _MIN_SEAM:
        return None
    return points, kept, heights


def _pixels_mask(shape, pixels: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    return mask


def _on_border(pixel: np.ndarray, shape: tuple[int, int]) -> bool:
    u, v = int(pixel[0]), int(pixel[1])
    height, width = shape
    return u < _BORDER_PX or v < _BORDER_PX or u >= width - _BORDER_PX or v >= height - _BORDER_PX


def _along_span(points: np.ndarray, up: np.ndarray, direction: np.ndarray) -> float:
    flat = points - up * (points @ up)[:, None]
    coord = flat @ direction
    return float(coord.max() - coord.min())


def _line_fit(points: np.ndarray, up: np.ndarray):
    # 水平面上的一条线。方向取这条点云拉得最长的方向。
    flat = points - up * (points @ up)[:, None]
    center = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    if direction is None:
        return None
    return center, direction, flat


def _line_inliers(points: np.ndarray, up: np.ndarray, tol: float = _PLANE_TOL_M):
    # 离这条线超过容差的点是棱上的混合深度，不参与。
    fitted = _line_fit(points, up)
    if fitted is None:
        return None
    _center, direction, flat = fitted
    normal = _horizontal(np.cross(up, direction), up)
    if normal is None:
        return None
    offset = np.abs((flat - np.median(flat, axis=0)) @ normal)
    keep = offset <= tol
    if int(keep.sum()) < 8:
        return None
    refit = _line_fit(points[keep], up)
    if refit is None:
        return None
    return points[keep], refit


def _assign_two_lines(points: np.ndarray, up: np.ndarray, tol: float = _PLANE_TOL_M):
    # 两个侧面的分界是一条折线。离两端连线最远的点是竖直棱。
    # 一条直线没有这个折点，就不当成两个面。
    flat = points - up * (points @ up)[:, None]
    if len(flat) < 16:
        return None
    # 端点和折点在抽稀后的点上找，再把全部点分到两条边上。
    step = max(1, int(np.ceil(len(flat) / 400)))
    sample = flat[::step]
    span = sample[:, None, :] - sample[None, :, :]
    dist = np.einsum("ijk,ijk->ij", span, span)
    ends = np.unravel_index(int(np.argmax(dist)), dist.shape)
    end_a = sample[ends[0]]
    end_b = sample[ends[1]]
    chord = end_b - end_a
    chord_len = float(np.dot(chord, chord))
    if chord_len < 0.04 ** 2:
        return None
    rel = sample - end_a
    along = (rel @ chord) / chord_len
    perp = np.linalg.norm(rel - along[:, None] * chord, axis=1)
    bend = int(np.argmax(perp))
    if float(perp[bend]) < 0.03:
        return None
    corner = sample[bend]

    def _to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        scale = float(np.dot(ab, ab))
        if scale < 1e-8:
            return np.linalg.norm(pts - a, axis=1)
        t = np.clip((pts - a) @ ab / scale, 0.0, 1.0)
        return np.linalg.norm(pts - a - t[:, None] * ab, axis=1)

    near_a = _to_segment(flat, corner, end_a) <= _to_segment(flat, corner, end_b)
    if int(near_a.sum()) < 8 or int((~near_a).sum()) < 8:
        return None
    first = _line_fit(points[near_a], up)
    second = _line_fit(points[~near_a], up)
    if first is None or second is None:
        return None
    # 接近垂直才是方块的两个侧面。一条直线的两头夹角接近 0，不用。
    if abs(float(np.dot(first[1], second[1]))) > 0.50:
        return None
    kept_a = _line_inliers(points[near_a], up, tol)
    kept_b = _line_inliers(points[~near_a], up, tol)
    if kept_a is None or kept_b is None:
        return None
    return kept_a[0], kept_b[0], kept_a[1], kept_b[1]


def _face_center_from_line(points, pixels, up, shape):
    fitted = _line_fit(points, up)
    if fitted is None:
        return None
    _center, direction, flat = fitted
    origin = np.median(points, axis=0)
    coord = (points - origin) @ direction
    order = np.argsort(coord)
    coord = coord[order]
    pixels = pixels[order]
    points = points[order]
    length = float(coord[-1] - coord[0])
    if length < 0.04:
        return None
    start = points[0]
    stop = points[-1]
    start_edge = _on_border(pixels[0], shape)
    stop_edge = _on_border(pixels[-1], shape)
    if abs(length - _EDGE_M) <= _LENGTH_TOL_M:
        face = 0.5 * (start + stop)
    elif length < _EDGE_M - _LENGTH_TOL_M:
        if start_edge and stop_edge:
            return "border"
        if start_edge == stop_edge:
            return "short"
        # 留在画面里的那一头是真实的棱，朝被切掉的一头补半个边长。
        if start_edge:
            face = stop - direction * _HALF_M
        else:
            face = start + direction * _HALF_M
    else:
        return "long"
    inward = _horizontal(np.cross(up, direction), up)
    if inward is None:
        return None
    if float(np.dot(inward, face)) < 0.0:
        inward = -inward
    return face + inward * _HALF_M, direction


def _corner_center(group_a, group_b, fit_a, fit_b, up):
    center_a, dir_a = fit_a[0], fit_a[1]
    center_b, dir_b = fit_b[0], fit_b[1]
    # 两条水平线的交点是竖直棱。用二维解，不沿视线去推。
    normal_a = _horizontal(np.cross(up, dir_a), up)
    if normal_a is None:
        return None
    denom = float(np.dot(dir_b, normal_a))
    if abs(denom) < 0.2:
        return None
    step = float(np.dot(center_a - center_b, normal_a)) / denom
    corner_flat = center_b + dir_b * step
    into_a = _horizontal(np.cross(up, dir_a), up)
    into_b = _horizontal(np.cross(up, dir_b), up)
    if into_a is None or into_b is None:
        return None
    mid_b = np.median(group_b - up * (group_b @ up)[:, None], axis=0)
    mid_a = np.median(group_a - up * (group_a @ up)[:, None], axis=0)
    if float(np.dot(into_a, mid_b - corner_flat)) < 0.0:
        into_a = -into_a
    if float(np.dot(into_b, mid_a - corner_flat)) < 0.0:
        into_b = -into_b
    height = float(np.median(np.vstack([group_a, group_b]) @ up))
    corner = corner_flat + up * height
    body = corner + into_a * _HALF_M + into_b * _HALF_M
    return body, dir_a


def _bottom_corners(body: np.ndarray, edge: np.ndarray, up: np.ndarray) -> np.ndarray:
    bottom = body - up * _HALF_M
    other = _horizontal(np.cross(up, edge), up)
    if other is None:
        other = _unit(np.cross(up, np.array([1.0, 0.0, 0.0])))
    half_a = edge * _HALF_M
    half_b = other * _HALF_M
    return np.vstack(
        [
            bottom - half_a - half_b,
            bottom + half_a - half_b,
            bottom + half_a + half_b,
            bottom - half_a + half_b,
        ]
    )


def _accept(body: np.ndarray, edge: np.ndarray, plane: Plane, car_shift) -> CapMeasure | str:
    height = float(plane.signed_distance(body.reshape(1, 3))[0])
    if abs(height - _CENTER_H_M) > _HEIGHT_TOL_M:
        return "height"
    up = _unit(plane.normal)
    world_edge = _level_from_camera(edge, up)
    yaw = fold_yaw_quarter(yaw_about_z(world_edge))
    return CapMeasure(
        center=np.asarray(body, dtype=float).reshape(3),
        world=camera_to_car_world(body, car_shift, up, camera_height=-float(plane.offset)),
        yaw=float(yaw),
        corners=_bottom_corners(body, edge, up),
        edge=np.asarray(edge, dtype=float).reshape(3),
    )


def _face_usable(points: np.ndarray, plane: Plane, up: np.ndarray) -> bool:
    """这块竖直深度要像塔顶侧面：不太碎，也不要长到白盒或墙。"""
    if len(points) < _MIN_SEAM:
        return False
    heights = plane.signed_distance(points)
    median = float(np.median(heights))
    if median < _DEPTH_MEDIAN_LO or median > _DEPTH_HI_M:
        return False
    flat = points - up * (points @ up)[:, None]
    center = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    if direction is None:
        return False
    along = (flat - center) @ direction
    vertical = points @ up
    span = float(along.max() - along.min())
    tall = float(vertical.max() - vertical.min())
    return _FACE_MIN_M <= span <= _FACE_MAX_M and _FACE_SPAN_M <= tall <= _FACE_MAX_M


def _depth_points(depth, intr, plane: Plane, zmin: float, zmax: float, anchor: np.ndarray):
    """塔身高度里的深度。不要求落在红蓝上。离白盒太远的不收。"""
    height, width = depth.shape[:2]
    step = max(1, int(np.sqrt((height * width) / 6000.0)))
    vs = np.arange(0, height, step)
    us = np.arange(0, width, step)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < _MIN_SEAM:
        return None
    pixels = np.column_stack([uu[good], vv[good]]).astype(float)
    points = deproject_pixels(pixels, z[good], intr)
    heights = plane.signed_distance(points)
    keep = (heights >= _DEPTH_LO_M) & (heights <= _DEPTH_HI_M) & _near_anchor(points, plane, anchor)
    if int(keep.sum()) < _MIN_SEAM:
        return None
    return points[keep]


def _corner_from_split(split, plane: Plane, up: np.ndarray, car_shift):
    group_a, group_b, fit_a, fit_b = split
    if not _face_usable(group_a, plane, up) or not _face_usable(group_b, plane, up):
        return None
    chosen = _corner_center(group_a, group_b, fit_a, fit_b, up)
    if chosen is None:
        return None
    body, edge = chosen
    result = _accept(body, edge, plane, car_shift)
    return None if isinstance(result, str) else result


def _one_depth_face(points: np.ndarray, up: np.ndarray, tol: float):
    kept = _line_inliers(points, up, tol)
    if kept is None:
        return None
    return kept[0], kept[1]


def _measure_depth(color_bgr, depth, intr, plane: Plane, zmin: float, zmax: float, anchor, up, car_shift):
    """颜色分界失败后，用塔身上的深度平面补侧面。"""
    points = _depth_points(depth, intr, plane, zmin, zmax, anchor)
    if points is None:
        return None
    split = _assign_two_lines(points, up, _DEPTH_TOL_M)
    if split is not None:
        body = _corner_from_split(split, plane, up, car_shift)
        if body is not None:
            return body
    single = _one_depth_face(points, up, _DEPTH_TOL_M)
    if single is None:
        return None
    depth_points, depth_fit = single
    if not _face_usable(depth_points, plane, up):
        return None
    red, blue = _color_masks(color_bgr)
    colored = cv2.bitwise_or(red, blue)
    front_points, _front_pixels, _front_heights = _mask_points(colored, depth, intr, plane, zmin, zmax, anchor)
    if len(front_points) < _MIN_SEAM:
        return None
    # 正面已经圈住的红蓝，和这块深度侧面不是同一个面，才拼到竖直棱上。
    flat = front_points - up * (front_points @ up)[:, None]
    origin = depth_fit[0] - up * float(np.dot(depth_fit[0], up))
    normal = _horizontal(np.cross(up, depth_fit[1]), up)
    if normal is None:
        return None
    off = np.abs((flat - origin) @ normal) > _DEPTH_TOL_M
    if int(off.sum()) < _MIN_SEAM:
        return None
    front = _one_depth_face(front_points[off], up, _DEPTH_TOL_M)
    if front is None or not _face_usable(front[0], plane, up):
        return None
    if abs(float(np.dot(front[1][1], depth_fit[1]))) > 0.50:
        return None
    return _corner_from_split((front[0], depth_points, front[1], depth_fit), plane, up, car_shift)


def _colored_mask(color_bgr: np.ndarray) -> np.ndarray:
    red, blue = _color_masks(color_bgr)
    return cv2.bitwise_or(red, blue)


def color_body_pixels(color_bgr: np.ndarray) -> np.ndarray:
    """红蓝连成一块的外轮廓。里面的颜色和小色点都不画。"""
    mask = _colored_mask(color_bgr)
    empty = np.zeros((0, 2), dtype=int)
    if int(np.count_nonzero(mask)) < 80:
        return empty
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return empty
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    keep = np.zeros(mask.shape, dtype=np.uint8)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < max(80, int(0.15 * largest)):
            continue
        keep[labels == index] = 255
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    parts = [contour[:, 0, :] for contour in contours if len(contour) >= 16]
    if not parts:
        return empty
    return np.vstack(parts).astype(int)


def color_seam_pixels(color_bgr: np.ndarray) -> np.ndarray:
    """红蓝相贴的那一圈像素。不看深度。不够一条分界就返回空。"""
    red, blue = _color_masks(color_bgr)
    if int(np.count_nonzero(red)) < _COLOR_MIN or int(np.count_nonzero(blue)) < _COLOR_MIN:
        return np.zeros((0, 2), dtype=int)
    pixels = boundary_pixels(red, blue, color_bgr)
    return pixels if len(pixels) >= _MIN_SEAM else np.zeros((0, 2), dtype=int)


def _large_color_parts(mask: np.ndarray) -> list[np.ndarray]:
    """红蓝连成的大块。小色点不拿来判断深度可不可靠。"""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return []
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    parts = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < max(80, int(0.15 * largest)):
            continue
        part = np.zeros(mask.shape, dtype=np.uint8)
        part[labels == index] = 255
        parts.append(part)
    return parts


def _implied_distance(mask: np.ndarray, intr) -> float | None:
    """用这块颜色较短的一边当成 20 厘米，估离相机多远。空的、太碎的不算。

    两个侧面并排时，长的一边是两面加起来的。拿它当 20 厘米会把距离估近大约三成，
    平整的深度就会被当成对不上。较短的一边仍接近一面的边长。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 80:
        return None
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    if width < 4.0 or height < 4.0:
        return None
    if float(len(xs)) < 0.20 * width * height:
        return None
    zx = _EDGE_M * float(intr.fx) / width
    zy = _EDGE_M * float(intr.fy) / height
    return float(max(zx, zy))


def _color_looks_close(mask: np.ndarray, intr) -> bool:
    """这块颜色比 0.40 米处的 20 厘米方块更大，就当成近处。"""
    implied = _implied_distance(mask, intr)
    return implied is not None and implied < _CLOSE_M


def _half_has_no_depth(xs: np.ndarray, good: np.ndarray) -> bool:
    """左右各有一块颜色时，有一边几乎没有深度，就不拿剩下的一边去拟合。"""
    if len(xs) < 160:
        return False
    mid = 0.5 * (float(xs.min()) + float(xs.max()))
    for side in (xs <= mid, xs > mid):
        if int(side.sum()) < 80:
            continue
        if float(np.mean(good[side])) < _HALF_COVER:
            return True
    return False


def _part_unreliable(mask: np.ndarray, depth: np.ndarray, intr, zmin: float, zmax: float) -> bool:
    """这一块颜色上的深度有一条对不上，就整块不用。"""
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= float(zmin)) & (z <= float(zmax))
    n_color = int(len(xs))
    n_good = int(good.sum())
    implied = _implied_distance(mask, intr)
    if implied is not None and n_good >= _MIN_SEAM:
        if implied < float(np.median(z[good])) * (1.0 - _DEPTH_AGREE):
            return True
    if n_color >= 80 and float(n_good) / float(n_color) < _DEPTH_COVER:
        return True
    if _half_has_no_depth(xs, good):
        return True
    if n_good >= _MIN_SEAM:
        lo, hi = np.percentile(z[good], [10.0, 90.0])
        if float(hi - lo) > _DEPTH_SPAN_M:
            return True
    return False


def _on_one_plane(points: np.ndarray, up: np.ndarray) -> bool:
    """这些深度点能不能贴成一个竖直侧面。贴不住就是散点或鼓包。"""
    if len(points) < _MIN_SEAM:
        return False
    step = max(1, int(np.ceil(len(points) / 1500)))
    kept = points[::step]
    flat = kept - up * (kept @ up)[:, None]
    center = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - center, full_matrices=False)
    direction = _horizontal(vt[0], up)
    normal = None if direction is None else _horizontal(np.cross(up, direction), up)
    if direction is None or normal is None:
        return False
    offset = np.abs((flat - center) @ normal)
    along = (flat - center) @ direction
    vertical = kept @ up
    if float(along.max() - along.min()) < _FACE_SPAN_M:
        return False
    if float(vertical.max() - vertical.min()) < _FACE_SPAN_M:
        return False
    return bool(float(np.mean(offset <= _RGB_PLANE_TOL_M)) >= _RGB_PLANE_FRAC)


def _depth_forms_planes(points: np.ndarray, up: np.ndarray) -> bool:
    """一个侧面，或两个侧面交成一条棱，都算深度还能用。"""
    if _on_one_plane(points, up):
        return True
    return _assign_two_lines(points, up, _RGB_PLANE_TOL_M) is not None


def close_depth_unusable(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    zmin: float,
    zmax: float,
) -> bool:
    """深度不可靠就改用颜色。和深度差两成以上、盖不住、前后差太大，有一条就丢掉这块深度。

    近处一块平整的深度留下。对不成平面的散点仍然不用。
    两个侧面并排时按较短的一边估距离，不把两面的总宽当成 20 厘米。
    """
    up = _unit(np.asarray(plane.normal, dtype=float))
    if up is None:
        return False
    colored = _colored_mask(color_bgr)
    if any(_part_unreliable(part, depth, intr, zmin, zmax) for part in _large_color_parts(colored)):
        return True
    looks_close = _color_looks_close(colored, intr)
    ys, xs = np.nonzero(colored)
    if len(xs) == 0:
        return False
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    median_z = float(np.median(z[good])) if int(good.sum()) else np.inf
    close = looks_close or (int(good.sum()) >= _MIN_SEAM and median_z < _CLOSE_M)
    if not close:
        return False
    if int(good.sum()) < _MIN_SEAM:
        return True
    pixels = np.column_stack([xs[good], ys[good]]).astype(float)
    points = deproject_pixels(pixels, z[good], intr)
    return not _depth_forms_planes(points, up)


def _hit_height_plane(pixels: np.ndarray, intr, plane: Plane, height: float) -> np.ndarray:
    """像素只定射线，交到离地这个高度的水平面上。平行或交在相机后面的丢掉。"""
    dirs = deproject_pixels(pixels.astype(float), np.ones(len(pixels)), intr)
    denom = dirs @ np.asarray(plane.normal, dtype=float)
    t = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    t[ok] = (float(height) + float(plane.offset)) / denom[ok]
    t[~(t > 0.05)] = np.nan
    return t[:, None] * dirs


def _measure_rgb_seam(
    color_bgr: np.ndarray,
    intr,
    plane: Plane,
    zmin: float,
    zmax: float,
    anchor: np.ndarray | None,
    car_shift: np.ndarray | None,
) -> tuple[CapMeasure | None, str | None]:
    """近处深度不能用时，颜色分界交到离地 0.80 米。短边两头都在画面里仍然丢掉。"""
    pixels = color_seam_pixels(color_bgr)
    up = _unit(np.asarray(plane.normal, dtype=float))
    if len(pixels) < _MIN_SEAM or up is None:
        return None, "seam"
    step = max(1, int(np.ceil(len(pixels) / 2500)))
    pixels = pixels[::step]
    points = _hit_height_plane(pixels, intr, plane, _CENTER_H_M)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    pixels = pixels[finite]
    if len(points) < _MIN_SEAM:
        return None, "seam"
    keep = (points[:, 2] >= zmin) & (points[:, 2] <= zmax) & _near_anchor(points, plane, anchor)
    if int(keep.sum()) < _MIN_SEAM:
        return None, "seam"
    kept_points = points[keep]
    kept_pixels = pixels[keep]
    # 相贴的那一圈有几像素厚。短边只有几厘米时是厚度，收到中线上再量。
    if _seam_is_thick(kept_points, up):
        kept_points = _collapse_to_line(kept_points, up)
    return _measure_seam_points(kept_points, kept_pixels, up, color_bgr.shape[:2], plane, car_shift)


def _seam_is_thick(points: np.ndarray, up: np.ndarray) -> bool:
    """没有第二条棱，或者那条短边只有分界的厚度。"""
    split = _assign_two_lines(points, up)
    if split is None:
        return True
    group_a, group_b, fit_a, fit_b = split
    short = min(
        _along_span(group_a, up, fit_a[1]),
        _along_span(group_b, up, fit_b[1]),
    )
    return short < 0.07


def _collapse_to_line(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """去掉离开这条线的厚度，点都落到中线上。"""
    fitted = _line_fit(points, up)
    if fitted is None:
        return points
    center, _direction, flat = fitted
    normal = _horizontal(np.cross(up, _direction), up)
    if normal is None:
        return points
    offset = (flat - center) @ normal
    return points - offset[:, None] * normal


def _tagged(result: tuple[CapMeasure | None, str | None], via: str) -> tuple[CapMeasure | None, str | None]:
    found, reason = result
    if found is not None:
        found.via = via
    return found, reason


def measure_cap(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    zmin: float,
    zmax: float,
    anchor: np.ndarray | None = None,
    car_shift: np.ndarray | None = None,
) -> tuple[CapMeasure | None, str | None]:
    """量塔顶体心。看不清分界就返回原因，不拿白盒中心代替。

    颜色分界没连成或侧面没看全时，改用塔身上的深度平面再试一次。
    近处深度不可靠时，改用颜色分界交到离地 0.80 米。
    侧面深度仍对不成体心时，再用这套颜色交点。两种都算量准，可以进入那 5 帧。
    """
    up = _unit(np.asarray(plane.normal, dtype=float))
    if up is not None and close_depth_unusable(color_bgr, depth, intr, plane, zmin, zmax):
        return _tagged(_measure_rgb_seam(color_bgr, intr, plane, zmin, zmax, anchor, car_shift), "close")
    cloud = _seam_cloud(color_bgr, depth, intr, plane, zmin, zmax, anchor)
    if cloud is None or up is None:
        reason = "seam"
    else:
        found, reason = _measure_seam_points(cloud[0], cloud[1], up, color_bgr.shape[:2], plane, car_shift)
        if found is not None:
            found.via = "depth"
            return found, None
    if anchor is not None and up is not None and reason in ("seam", "short", "plane"):
        assisted = _measure_depth(color_bgr, depth, intr, plane, zmin, zmax, anchor, up, car_shift)
        if assisted is not None:
            assisted.via = "depth"
            return assisted, None
    if up is None:
        return None, reason
    rgb_found, _rgb_reason = _measure_rgb_seam(color_bgr, intr, plane, zmin, zmax, anchor, car_shift)
    if rgb_found is not None:
        rgb_found.via = "side"
        return rgb_found, None
    return None, reason


def _measure_seam_points(points, pixels, up, shape, plane: Plane, car_shift):
    # 先看是不是两个侧面交成一条棱。一条线更直时，再用这一条。
    split = _assign_two_lines(points, up)
    chosen = None
    if split is not None:
        group_a, group_b, fit_a, fit_b = split
        chosen = _corner_center(group_a, group_b, fit_a, fit_b, up)
    if chosen is None:
        fitted = _line_fit(points, up)
        if fitted is None:
            return None, "seam"
        _center, direction, flat = fitted
        normal = _horizontal(np.cross(up, direction), up)
        if normal is None:
            return None, "seam"
        offset = np.abs((flat - np.median(flat, axis=0)) @ normal)
        keep = offset <= _PLANE_TOL_M
        if int(keep.sum()) < _MIN_SEAM:
            return None, "plane"
        chosen = _face_center_from_line(points[keep], pixels[keep], up, shape)
    if chosen is None:
        return None, "seam"
    if chosen == "border":
        return None, "border"
    if chosen == "short":
        return None, "short"
    if chosen == "long":
        return None, "long"
    body, edge = chosen
    result = _accept(body, edge, plane, car_shift)
    if isinstance(result, str):
        return None, result
    return result, None


class CapFusion:
    """只收量准的帧。够帧数才给出中值，单独一帧失败不把已经得到的结果清掉。

    深度帧够数之后，中值只用深度。颜色帧只在深度还不够时一起算。
    """

    def __init__(self, need: int = 5, window: int = 15):
        self.need = int(need)
        self.window = int(window)
        self.samples: list[dict] = []
        self.latest: CapMeasure | None = None

    def add(self, measure: CapMeasure, color: str | None = None) -> None:
        self.latest = measure
        sample = {
            "x": float(measure.world[0]),
            "y": float(measure.world[1]),
            "z": float(measure.world[2]),
            "yaw": float(measure.yaw),
            "quarter": True,
            "via": measure.via if measure.via in ("depth", "close", "side") else "depth",
        }
        if color in ("red", "blue"):
            sample["color"] = color
        self.samples.append(sample)
        if len(self.samples) > self.window:
            del self.samples[:-self.window]

    def _usable(self) -> list[dict]:
        depth = [sample for sample in self.samples if sample.get("via", "depth") == "depth"]
        if len(depth) >= self.need:
            return depth
        return self.samples

    def result(self):
        chosen = self._usable()
        if len(chosen) < self.need:
            return None
        return fuse_poses(chosen)

    def consensus(self):
        """最近这几帧颜色相同，才把它们的中值当成可以报出的塔顶。深度够数时，这几帧也只看深度。"""
        chosen = self._usable()
        if len(chosen) < self.need:
            return None
        recent = chosen[-self.need :]
        colors = [sample.get("color") for sample in recent]
        if any(color not in ("red", "blue") for color in colors):
            return None
        if len(set(colors)) != 1:
            return None
        fused = fuse_poses(recent)
        if fused is None:
            return None
        return fused, colors[0]

    def count(self) -> int:
        return len(self.samples)


# 黄线边偏离这么多，就不算一条直边。
_OUTLINE_STRAIGHT_M = 0.015
# 两条边的端点离这么近，算连在一起。
_OUTLINE_JOIN_M = 0.03
# 和上一眼体心差过这么多，这一眼的推算不显示。
_OUTLINE_JUMP_M = 0.03
# 轮廓上相邻点跳过这么多，就换成下一条边。和旁边的深度差过这条，这个点也不参与。
_OUTLINE_GAP_M = 0.05
# 和旁边的深度差在这个以内，这个点照常用。再大就往旁边的深度靠。
_OUTLINE_STABLE_M = 0.015
# 黄线往红蓝块里面收这几个像素再取深度，不用交界上那一个像素。
_OUTLINE_INSET_PX = (2, 3, 4)
_VERT_DOT = 0.85
_HORIZ_DOT = 0.30
_RIGHT_DOT = 0.35
# 短于这么多的横边没有方向，不当成方块下沿。
_OUTLINE_MIN_M = 0.04
# 离地和最低那条差在这个以内，都算下沿。更高的不参与。
_BOTTOM_BAND_M = 0.03
# 底面和黄线底端差不到这么多，就当已经齐平。
_FLOOR_GAP_M = 0.01


@dataclass
class _Rim:
    start: np.ndarray
    stop: np.ndarray
    direction: np.ndarray
    length: float
    vertical: bool
    # 这一头是轮廓断掉的地方，那边没有深度。闭合轮廓两头都不是。
    start_open: bool = False
    stop_open: bool = False


def _segment_distance(points: np.ndarray, start: np.ndarray, stop: np.ndarray) -> float:
    chord = stop - start
    scale = float(np.dot(chord, chord))
    if scale < 1e-8:
        return float(np.max(np.linalg.norm(points - start, axis=1))) if len(points) else 0.0
    t = np.clip((points - start) @ chord / scale, 0.0, 1.0)
    return float(np.max(np.linalg.norm(points - start - t[:, None] * chord, axis=1)))


def _douglas(points: np.ndarray, start: int, stop: int) -> list[int]:
    if stop <= start + 1:
        return [start, stop]
    chord = points[stop] - points[start]
    scale = float(np.dot(chord, chord))
    if scale < 1e-8:
        return [start, stop]
    rel = points[start + 1 : stop] - points[start]
    t = np.clip(rel @ chord / scale, 0.0, 1.0)
    dist = np.linalg.norm(rel - t[:, None] * chord, axis=1)
    bend = int(np.argmax(dist))
    if float(dist[bend]) <= _OUTLINE_STRAIGHT_M:
        return [start, stop]
    mid = start + 1 + bend
    left = _douglas(points, start, mid)
    right = _douglas(points, mid, stop)
    return left[:-1] + right


def _thin(chain: np.ndarray, step: float = 0.004) -> np.ndarray:
    kept = [chain[0]]
    for point in chain[1:]:
        if float(np.linalg.norm(point - kept[-1])) >= step:
            kept.append(point)
    if float(np.linalg.norm(chain[-1] - kept[-1])) > 1e-6:
        kept.append(chain[-1])
    return np.vstack(kept)


def _start_at_bend(chain: np.ndarray) -> np.ndarray:
    """闭合轮廓从转角开始，避免一条棱被起点切成两截。首尾不叠成同一个点。"""
    if len(chain) >= 2 and float(np.linalg.norm(chain[0] - chain[-1])) < 1e-6:
        chain = chain[:-1]
    if len(chain) < 8:
        return chain
    prev = chain[:-2]
    mid = chain[1:-1]
    nxt = chain[2:]
    left = mid - prev
    right = nxt - mid
    left_n = np.linalg.norm(left, axis=1)
    right_n = np.linalg.norm(right, axis=1)
    good = (left_n > 1e-6) & (right_n > 1e-6)
    dots = np.ones(len(mid))
    dots[good] = np.sum(left[good] * right[good], axis=1) / (left_n[good] * right_n[good])
    bend = int(np.argmin(dots))
    if float(dots[bend]) > 0.5:
        return chain
    start = bend + 1
    if start <= 0 or start >= len(chain):
        return chain
    return np.vstack([chain[start:], chain[:start]])


def _line_ends(points: np.ndarray):
    """边的两端落在这些点拉成的直线上，不落在单个像素上。"""
    center = np.median(points, axis=0)
    spread = points - center
    if float(np.linalg.norm(spread, axis=1).max()) < 1e-8:
        return None
    _, _, vt = np.linalg.svd(spread, full_matrices=False)
    direction = vt[0]
    coord = spread @ direction
    return center + direction * float(coord.min()), center + direction * float(coord.max())


def _outline_weight(jump: float) -> float:
    """差得越远权重越低。1.5 厘米以内是 1，5 厘米以上是 0。"""
    if jump <= _OUTLINE_STABLE_M:
        return 1.0
    if jump >= _OUTLINE_GAP_M:
        return 0.0
    t = (jump - _OUTLINE_STABLE_M) / (_OUTLINE_GAP_M - _OUTLINE_STABLE_M)
    return float((1.0 - t) ** 2)


def _read_depth(depth: np.ndarray, u: int, v: int, zmin: float, zmax: float) -> float | None:
    if v < 0 or u < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z < zmin or z > zmax:
        return None
    return z


def _inward_normal(pts: np.ndarray, index: int, mask: np.ndarray) -> np.ndarray | None:
    """朝黄线里面的方向。两边都数一数落在红蓝上的像素，多的那一边是里面。"""
    count = len(pts)
    u = int(pts[index, 0])
    v = int(pts[index, 1])
    span = 3 if count > 6 else 1
    prev = pts[(index - span) % count]
    nxt = pts[(index + span) % count]
    tangent = np.array([float(nxt[0] - prev[0]), float(nxt[1] - prev[1])])
    scale = float(np.hypot(tangent[0], tangent[1]))
    if scale < 1e-6:
        return None
    tangent /= scale
    normal = np.array([-tangent[1], tangent[0]])

    def hits(direction: np.ndarray) -> int:
        seen = 0
        for step in _OUTLINE_INSET_PX:
            x = int(round(u + direction[0] * step))
            y = int(round(v + direction[1] * step))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                seen += 1
        return seen

    if hits(normal) < hits(-normal):
        normal = -normal
    return normal


def _borrowed_depth(u: int, v: int, normal: np.ndarray | None, mask: np.ndarray, depth: np.ndarray, zmin: float, zmax: float) -> float | None:
    """往里面 2 到 4 像素取深度，取靠近这几处中位数的那一个。里面没有才用线上这一点。"""
    found: list[float] = []
    if normal is not None:
        for step in _OUTLINE_INSET_PX:
            x = int(round(u + normal[0] * step))
            y = int(round(v + normal[1] * step))
            if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1] or not mask[y, x]:
                continue
            z = _read_depth(depth, x, y, zmin, zmax)
            if z is not None:
                found.append(z)
    if found:
        med = float(np.median(found))
        return min(found, key=lambda z: abs(z - med))
    return _read_depth(depth, u, v, zmin, zmax)


def _neighbor_median(samples: list[tuple[int, int, float] | None], index: int, wrap: bool) -> float:
    count = len(samples)
    chosen = []
    for offset in range(-2, 3):
        other = index + offset
        if wrap:
            other %= count
        elif other < 0 or other >= count:
            continue
        sample = samples[other]
        if sample is not None:
            chosen.append(sample[2])
    return float(np.median(chosen))


def _calm_points(samples: list[tuple[int, int, float] | None], intr, plane: Plane, up: np.ndarray | None) -> list[np.ndarray | None]:
    """跳动的深度往旁边拉。差到 5 厘米的点留空，不拿去定边。"""
    wrap = all(sample is not None for sample in samples) and len(samples) >= 8
    calmed: list[np.ndarray | None] = []
    for index, sample in enumerate(samples):
        if sample is None:
            calmed.append(None)
            continue
        u, v, raw = sample
        med = _neighbor_median(samples, index, wrap)
        weight = _outline_weight(abs(raw - med))
        if weight <= 0.0:
            calmed.append(None)
            continue
        z = weight * raw + (1.0 - weight) * med
        point = deproject_pixels(np.array([[u, v]], dtype=float), np.array([z]), intr)[0]
        if up is not None:
            height = float(plane.signed_distance(point.reshape(1, 3))[0])
            if height < 0.60 or height > 1.02:
                calmed.append(None)
                continue
        calmed.append(point)
    return calmed


def _take_chains(points: list[np.ndarray | None]) -> list[np.ndarray]:
    chains: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for point in points:
        if point is None:
            continue
        if current and float(np.linalg.norm(point - current[-1])) > _OUTLINE_GAP_M:
            if len(current) >= 8:
                chains.append(np.vstack(current))
            current = [point]
        else:
            current.append(point)
    if len(current) >= 8:
        chains.append(np.vstack(current))
    return chains


def _rims(chain: np.ndarray, up: np.ndarray) -> list[_Rim]:
    """一条轮廓上拉直的边。横边短于 20 厘米也留下，下沿靠它补全。"""
    chain = np.asarray(chain, dtype=float)
    if len(chain) < 8:
        return []
    closed = float(np.linalg.norm(chain[0] - chain[-1])) <= _OUTLINE_GAP_M
    if closed:
        chain = _start_at_bend(chain)
    chain = _thin(chain)
    if len(chain) < 8:
        return []
    corners = _douglas(chain, 0, len(chain) - 1)
    found: list[_Rim] = []
    last = len(chain) - 1
    for begin, end in zip(corners, corners[1:]):
        if end - begin < 6:
            continue
        points = chain[begin : end + 1]
        start = chain[begin]
        stop = chain[end]
        if _segment_distance(points, start, stop) > _OUTLINE_STRAIGHT_M:
            continue
        fitted = _line_ends(points)
        if fitted is not None:
            start, stop = fitted
        span = stop - start
        length = float(np.linalg.norm(span))
        direction = _unit(span)
        if direction is None:
            continue
        align = abs(float(np.dot(direction, up)))
        vertical = align >= _VERT_DOT
        horizontal = align <= _HORIZ_DOT
        if vertical and abs(length - _HALF_M) > _LENGTH_TOL_M and abs(length - _EDGE_M) > _LENGTH_TOL_M:
            continue
        if horizontal and (length < _OUTLINE_MIN_M or length > _EDGE_M + _LENGTH_TOL_M):
            continue
        if not vertical and not horizontal:
            continue
        found.append(
            _Rim(
                start=start,
                stop=stop,
                direction=direction,
                length=length,
                vertical=vertical,
                start_open=(not closed) and begin == 0,
                stop_open=(not closed) and end == last,
            )
        )
    return found


def _shared_corner(one: _Rim, other: _Rim) -> np.ndarray | None:
    ends = (one.start, one.stop)
    best = _OUTLINE_JOIN_M
    corner = None
    for left in ends:
        for right in (other.start, other.stop):
            gap = float(np.linalg.norm(left - right))
            if gap <= best:
                best = gap
                corner = 0.5 * (left + right)
    return corner


def _rim_height(rim: _Rim, plane: Plane) -> float:
    mid = 0.5 * (rim.start + rim.stop)
    return float(plane.signed_distance(mid.reshape(1, 3))[0])


def _complete_bottom(rim: _Rim) -> tuple[np.ndarray, np.ndarray]:
    """下沿补成 20 厘米。缺口朝没有深度的那头补，两头都断就从中间往两边补。"""
    start = rim.start
    stop = rim.stop
    if rim.length >= _EDGE_M:
        return start, stop
    missing = _EDGE_M - rim.length
    if rim.start_open and not rim.stop_open:
        start = start - rim.direction * missing
    elif rim.stop_open and not rim.start_open:
        stop = stop + rim.direction * missing
    else:
        start = start - rim.direction * (missing * 0.5)
        stop = stop + rim.direction * (missing * 0.5)
    return start, stop


def _bottom_edge_center(rim: _Rim, up: np.ndarray):
    """这条下沿就是方块最底端。体心在补全后的中点上方 10 厘米，再往里面 10 厘米。"""
    start, stop = _complete_bottom(rim)
    mid = 0.5 * (start + stop)
    face = mid + up * _HALF_M
    inward = _horizontal(np.cross(up, rim.direction), up)
    if inward is None:
        return None
    if float(np.dot(inward, face)) < 0.0:
        inward = -inward
    return face + inward * _HALF_M, rim.direction


def _corner_from_bottoms(front: _Rim, side: _Rim, up: np.ndarray):
    """两条下沿交成底面的角。体心从角往上 10 厘米，再沿两个面各进 10 厘米。"""
    corner = _shared_corner(front, side)
    if corner is None:
        return None
    if abs(float(np.dot(front.direction, side.direction))) > _RIGHT_DOT:
        return None
    into_front = _horizontal(np.cross(up, front.direction), up)
    into_side = _horizontal(np.cross(up, side.direction), up)
    if into_front is None or into_side is None:
        return None
    if float(np.dot(into_front, side.start - corner)) < 0.0 and float(np.dot(into_front, side.stop - corner)) < 0.0:
        into_front = -into_front
    if float(np.dot(into_side, front.start - corner)) < 0.0 and float(np.dot(into_side, front.stop - corner)) < 0.0:
        into_side = -into_side
    edge = front.direction if front.length >= side.length else side.direction
    return corner + up * _HALF_M + into_front * _HALF_M + into_side * _HALF_M, edge


def _center_from_rims(rims: list[_Rim], up: np.ndarray, plane: Plane, car_shift):
    """只用离地最低的横边。它就是方块最底端，上沿不参与。"""
    horizontals = [rim for rim in rims if not rim.vertical]
    if not horizontals:
        return None
    lowest = min(_rim_height(rim, plane) for rim in horizontals)
    bottoms = [rim for rim in horizontals if _rim_height(rim, plane) <= lowest + _BOTTOM_BAND_M]
    chosen = None
    for front in bottoms:
        for side in bottoms:
            if front is side or _shared_corner(front, side) is None:
                continue
            body = _corner_from_bottoms(front, side, up)
            if body is not None:
                chosen = body
                break
        if chosen is not None:
            break
    if chosen is None:
        floor = min(_rim_height(rim, plane) for rim in bottoms)
        candidates = [rim for rim in bottoms if _rim_height(rim, plane) <= floor + 1e-9]
        chosen = _bottom_edge_center(max(candidates, key=lambda rim: rim.length), up)
    if chosen is None:
        return None
    body, edge = chosen
    result = _accept(body, edge, plane, car_shift)
    if isinstance(result, str):
        return None
    return result


def outline_center(chains: list[np.ndarray], plane: Plane, car_shift: np.ndarray | None = None) -> CapMeasure | None:
    """黄线离地最低的横边是方块最底端。上沿不参与，边短了也补。"""
    up = _unit(np.asarray(plane.normal, dtype=float))
    if up is None:
        return None
    rims: list[_Rim] = []
    for chain in chains:
        rims.extend(_rims(np.asarray(chain, dtype=float), up))
    if not rims:
        return None
    return _center_from_rims(rims, up, plane, car_shift)


def _yellow_contour(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if uv is None or len(uv) < 20:
        return np.zeros((0, 2), dtype=int)
    mask = np.zeros(shape, dtype=np.uint8)
    ys = np.clip(uv[:, 1], 0, shape[0] - 1)
    xs = np.clip(uv[:, 0], 0, shape[1] - 1)
    mask[ys, xs] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=int)
    contour = max(contours, key=len)
    return contour[:, 0, :].astype(int)


def _bottom_pixels(contour: np.ndarray) -> np.ndarray | None:
    """黄线最靠下的那一段。用画面位置，不用这些点的深度。"""
    if len(contour) < 8:
        return None
    rows = contour[:, 1]
    cut = float(np.percentile(rows, 90))
    band = contour[rows >= cut]
    if len(band) < 8:
        return None
    return band


def _front_plane(measure: CapMeasure, up: np.ndarray):
    """已经得到的正面：离相机更近、又顺着下沿方向的那条底边。"""
    corners = np.asarray(measure.corners, dtype=float).reshape(4, 3)
    along = _unit(np.asarray(measure.edge, dtype=float))
    if along is None:
        return None
    chosen = None
    nearest = np.inf
    for start, stop in ((0, 1), (1, 2), (2, 3), (3, 0)):
        span = corners[stop] - corners[start]
        direction = _unit(span)
        if direction is None or abs(float(np.dot(direction, along))) < 0.8:
            continue
        mid = 0.5 * (corners[start] + corners[stop])
        dist = float(np.dot(mid, mid))
        if dist < nearest:
            nearest = dist
            chosen = mid, direction
    if chosen is None:
        return None
    mid, direction = chosen
    outward = _horizontal(np.cross(up, direction), up)
    if outward is None:
        return None
    if float(np.dot(outward, mid - measure.center)) < 0.0:
        outward = -outward
    return mid, outward


def _bottom_height_on_face(pixels: np.ndarray, intr, origin: np.ndarray, normal: np.ndarray, plane: Plane) -> float | None:
    """底端像素的射线交到正面上，离地高度取中位数。"""
    dirs = deproject_pixels(pixels.astype(float), np.ones(len(pixels)), intr)
    denom = dirs @ normal
    level = float(normal @ origin)
    scale = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    scale[ok] = level / denom[ok]
    scale[~(scale > 0.05)] = np.nan
    points = scale[:, None] * dirs
    heights = plane.signed_distance(points)
    keep = np.isfinite(heights) & np.isfinite(scale) & (heights > 0.55) & (heights < 1.00)
    if int(keep.sum()) < 8:
        return None
    return float(np.median(heights[keep]))


def _shift_bottom(measure: CapMeasure, target: float, plane: Plane, car_shift) -> CapMeasure:
    """底面低于黄线底端就抬高，高于它就降低。"""
    up = _unit(np.asarray(plane.normal, dtype=float))
    if up is None:
        return measure
    bottom = float(plane.signed_distance(measure.center.reshape(1, 3))[0]) - _HALF_M
    gap = bottom - target
    if abs(gap) <= _FLOOR_GAP_M:
        return measure
    center = np.asarray(measure.center, dtype=float) - up * gap
    shifted = _accept(center, measure.edge, plane, car_shift)
    if isinstance(shifted, str):
        return measure
    return shifted


def align_bottom_to_yellow(measure: CapMeasure, uv: np.ndarray, shape: tuple[int, int], intr, plane: Plane, car_shift=None) -> CapMeasure:
    """绿框底面跟黄线最靠下的那一段齐平。不用这一段像素自己的深度。"""
    up = _unit(np.asarray(plane.normal, dtype=float))
    face = None if up is None else _front_plane(measure, up)
    pixels = _bottom_pixels(_yellow_contour(uv, shape))
    if face is None or pixels is None:
        return measure
    target = _bottom_height_on_face(pixels, intr, face[0], face[1], plane)
    if target is None:
        return measure
    return _shift_bottom(measure, target, plane, car_shift)


def _outline_chains(uv: np.ndarray, depth: np.ndarray, intr, plane: Plane, zmin: float, zmax: float) -> list[np.ndarray]:
    if uv is None or len(uv) < 20:
        return []
    mask = np.zeros(depth.shape, dtype=np.uint8)
    ys = np.clip(uv[:, 1], 0, depth.shape[0] - 1)
    xs = np.clip(uv[:, 0], 0, depth.shape[1] - 1)
    mask[ys, xs] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    chains: list[np.ndarray] = []
    up = _unit(np.asarray(plane.normal, dtype=float))
    for contour in contours:
        pts = contour[:, 0, :]
        samples: list[tuple[int, int, float] | None] = []
        for index, pixel in enumerate(pts):
            u, v = int(pixel[0]), int(pixel[1])
            z = _borrowed_depth(u, v, _inward_normal(pts, index, mask), mask, depth, zmin, zmax)
            samples.append(None if z is None else (u, v, z))
        chains.extend(_take_chains(_calm_points(samples, intr, plane, up)))
    return chains


# 正面内部和这面差在这个以内，算贴住。
_STICK_M = 0.02
_STICK_FRAC = 0.60
# 绿框附近这么远以内的深度才拿来核对，再远是背景。
_CHECK_REACH_M = 0.35
# 比塔顶底面再低大约 2 厘米，就是方块下面那一层，不拿来当侧面。
_CHECK_LO_M = 0.68
# 两个面的法向离直角过了这个，就不算侧面。
_SIDE_DOT = 0.50
# 竖直面要鼓出这么多，才从一整块里拆出侧面。
_CORNER_BULGE_M = 0.04


def _on_face(measure: CapMeasure, up: np.ndarray, depth: np.ndarray, intr, zmin: float, zmax: float) -> bool | None:
    """正面内部贴住这面就是 True。采到的点太少，就还不能判断。"""
    face = _front_plane(measure, up)
    if face is None:
        return None
    mid, outward = face
    along = _horizontal(np.cross(up, outward), up)
    if along is None:
        return None
    grid = [
        mid + along * across + up * rise
        for across in np.linspace(-0.08, 0.08, 7)
        for rise in np.linspace(0.02, 0.18, 7)
    ]
    pts = np.vstack(grid)
    pix = project_points(pts, intr)
    ui = np.rint(pix[:, 0]).astype(int)
    vi = np.rint(pix[:, 1]).astype(int)
    height, width = depth.shape[:2]
    inside = (pts[:, 2] > 0.05) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if int(inside.sum()) < 8:
        return None
    z = depth[vi[inside], ui[inside]]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < 8:
        return None
    measured = deproject_pixels(
        np.column_stack([ui[inside][good], vi[inside][good]]).astype(float),
        z[good],
        intr,
    )
    dist = np.abs((measured - mid) @ outward)
    return bool(float(np.mean(dist <= _STICK_M)) >= _STICK_FRAC)


def _nearby_points(center: np.ndarray, depth: np.ndarray, intr, plane: Plane, up: np.ndarray, zmin: float, zmax: float):
    """绿框附近、塔顶高度里的深度。方块下面更近的那一层不收。"""
    height, width = depth.shape[:2]
    step = 1 if height * width < 80000 else max(1, int(np.sqrt((height * width) / 4000.0)))
    vs = np.arange(0, height, step)
    us = np.arange(0, width, step)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < _MIN_SEAM:
        return None
    points = deproject_pixels(np.column_stack([uu[good], vv[good]]).astype(float), z[good], intr)
    heights = plane.signed_distance(points)
    flat = points - up * (points @ up)[:, None]
    anchor = np.asarray(center, dtype=float) - up * float(np.dot(center, up))
    near = np.linalg.norm(flat - anchor, axis=1) <= _CHECK_REACH_M
    keep = (heights >= _CHECK_LO_M) & (heights <= _DEPTH_HI_M) & near
    if int(keep.sum()) < _MIN_SEAM:
        return None
    return points[keep]


def _plane_group(points: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    """这些点里贴成一块竖直面的那部分。贴不满就当没有。"""
    if len(points) < _MIN_SEAM:
        return None
    flat = points - up * (points @ up)[:, None]
    origin = np.median(flat, axis=0)
    _, _, vt = np.linalg.svd(flat - origin, full_matrices=False)
    direction = _horizontal(vt[0], up)
    normal = None if direction is None else _horizontal(np.cross(up, direction), up)
    if normal is None:
        return None
    offset = np.abs((flat - origin) @ normal)
    keep = offset <= _STICK_M
    if float(np.mean(keep)) < _STICK_FRAC or int(keep.sum()) < _MIN_SEAM:
        return None
    return points[keep]


def _split_faces(points: np.ndarray, up: np.ndarray) -> list[np.ndarray]:
    """一块竖直面，或者两块成直角的竖直面。鼓出来的那块先当侧面，不并进正面。"""
    flat = points - up * (points @ up)[:, None]
    if len(flat) < _MIN_SEAM:
        return []
    step = max(1, int(np.ceil(len(flat) / 400.0)))
    sample = flat[::step]
    span = sample[:, None, :] - sample[None, :, :]
    dist = np.einsum("ijk,ijk->ij", span, span)
    ends = np.unravel_index(int(np.argmax(dist)), dist.shape)
    end_a = sample[ends[0]]
    end_b = sample[ends[1]]
    chord = end_b - end_a
    if float(np.dot(chord, chord)) >= 0.04 ** 2:
        rel = sample - end_a
        along = (rel @ chord) / float(np.dot(chord, chord))
        perp = np.linalg.norm(rel - along[:, None] * chord, axis=1)
        bend = int(np.argmax(perp))
        if float(perp[bend]) >= _CORNER_BULGE_M:
            corner = sample[bend]
            dir_a = _unit(end_a - corner)
            dir_b = _unit(end_b - corner)
            normal_a = None if dir_a is None else _horizontal(np.cross(up, dir_a), up)
            normal_b = None if dir_b is None else _horizontal(np.cross(up, dir_b), up)
            if normal_a is not None and normal_b is not None and abs(float(np.dot(normal_a, normal_b))) <= _SIDE_DOT:
                off_a = np.abs((flat - corner) @ normal_a)
                off_b = np.abs((flat - corner) @ normal_b)
                on_a = off_a <= _STICK_M
                on_b = off_b <= _STICK_M
                both = on_a & on_b
                on_a = (on_a & ~both) | (both & (off_a <= off_b))
                on_b = on_b & ~on_a
                if int(on_a.sum()) >= _MIN_SEAM and int(on_b.sum()) >= _MIN_SEAM:
                    return [points[on_a], points[on_b]]
    single = _plane_group(points, up)
    if single is None:
        return []
    return [single]


def _pose_from_one_face(points: np.ndarray, up: np.ndarray, ground: Plane) -> CapMeasure | None:
    """只剩一块竖直面。体心在下沿再往这个面里面 10 厘米，转角用这面。"""
    fitted = _line_fit(points, up)
    if fitted is None:
        return None
    _center, direction, _flat = fitted
    normal = _horizontal(np.cross(up, direction), up)
    if normal is None:
        return None
    origin = np.median(points, axis=0)
    if float(np.dot(normal, origin)) < 0.0:
        normal = -normal
    heights = ground.signed_distance(points)
    cut = float(np.percentile(heights, 20))
    low = points[heights <= cut + 0.02]
    if len(low) < 8:
        low = points
    bottom = np.median(low, axis=0)
    bottom = bottom - normal * float(np.dot(bottom - origin, normal))
    body = bottom + up * _HALF_M + normal * _HALF_M
    result = _accept(body, direction, ground, None)
    return None if isinstance(result, str) else result


def _pose_from_two_faces(group_a: np.ndarray, group_b: np.ndarray, up: np.ndarray, ground: Plane) -> CapMeasure | None:
    """正面和侧面都在。体心从竖直棱往两个面里各进 10 厘米。"""
    fit_a = _line_fit(group_a, up)
    fit_b = _line_fit(group_b, up)
    if fit_a is None or fit_b is None:
        return None
    chosen = _corner_center(group_a, group_b, fit_a, fit_b, up)
    if chosen is None:
        return None
    body, edge = chosen
    heights = ground.signed_distance(np.vstack([group_a, group_b]))
    target = float(np.percentile(heights, 20)) + _HALF_M
    current = float(ground.signed_distance(np.asarray(body, dtype=float).reshape(1, 3))[0])
    body = np.asarray(body, dtype=float) + up * (target - current)
    result = _accept(body, edge, ground, None)
    return None if isinstance(result, str) else result


def confirm_green_face(measure: CapMeasure | None, depth: np.ndarray, intr, plane: Plane, zmin: float, zmax: float) -> CapMeasure | None:
    """绿框正面贴住深度就留下。贴不住时，成直角的另一面当侧面留下，跳点丢掉。都分不清就不要这一眼。"""
    up = _unit(np.asarray(plane.normal, dtype=float))
    if up is None or measure is None:
        return measure
    stuck = _on_face(measure, up, depth, intr, zmin, zmax)
    if stuck is None or stuck:
        return measure
    cloud = _nearby_points(measure.center, depth, intr, plane, up, zmin, zmax)
    if cloud is None:
        return None
    groups = _split_faces(cloud, up)
    usable = [group for group in groups if _face_usable(group, plane, up)]
    # 两块都像方块的面，才从棱往两边收。只像一块时，用整团里那块竖直面，不用拆坏的半边。
    if len(groups) >= 2 and len(usable) >= 2:
        pose = _pose_from_two_faces(usable[0], usable[1], up, plane)
        if pose is not None:
            return pose
    single = _plane_group(cloud, up)
    if single is not None and _face_usable(single, plane, up):
        return _pose_from_one_face(single, up, plane)
    return None


def measure_outline(uv, depth, intr, plane: Plane, zmin: float, zmax: float) -> CapMeasure | None:
    """从这一眼黄线的下沿推体心。底面再跟黄线齐平，正面再和深度核对。"""
    pose = outline_center(_outline_chains(uv, depth, intr, plane, zmin, zmax), plane)
    if pose is None:
        return None
    pose = align_bottom_to_yellow(pose, uv, depth.shape[:2], intr, plane)
    return confirm_green_face(pose, depth, intr, plane, zmin, zmax)


class OutlineLive:
    """只显示和上一眼接近的推算。差远了这一眼留空，不把旧结果留在画面上。"""

    def __init__(self, jump_m: float = _OUTLINE_JUMP_M):
        self.jump_m = float(jump_m)
        self.last: np.ndarray | None = None

    def reset(self) -> None:
        self.last = None

    def offer(self, measure: CapMeasure | None) -> CapMeasure | None:
        if measure is None:
            self.last = None
            return None
        center = np.asarray(measure.center, dtype=float).reshape(3)
        if self.last is not None and float(np.linalg.norm(center - self.last)) > self.jump_m:
            self.last = center.copy()
            return None
        show = self.last is not None
        self.last = center.copy()
        return measure if show else None
