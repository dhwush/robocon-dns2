"""白盒层数分两眼粗认。

先低头确认贴地的白。白被画面上沿切断，只说明还没看完，不记成两层。
再按下层白的水平距离抬头。远的抬到接近水平，近的还要抬过水平，才能看见顶块侧面。
离地 0.5 米以上的白要连成一块，并且够得上 35 厘米的一面，才是两层。
这块白已经高过 0.5 米时，从 0.43 米起的同一块也算进去。
没有深度时，用当前相机系的塔脚确定竖直面，像素射线与面求交后检查高度。
近处相邻红蓝允许部分出画，只作为候选，不补造三维位置。
第 2 层白只看下层白盒上方、左右再多出大约一块盒子的那一扇。旁边的椅子不算。好几块都够大时，只留离下层中心最近的那一块。
红蓝必须有实测深度关联到塔体才确认塔顶，无深度仅为候选。俯视时也检查第二层白盒顶面。
黄线顶到画面上沿，再抬 5 度，最多两次。顶着上沿、还没抬完的帧不数层。
不量两层白的顶面。层数连续几帧一致才定，定了以后单独一帧对不上不改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from cube.detect import _color_layers, _white_pixels
from cube.color_boundary import adjacent_color_pairs
from geom.transform import deproject_pixels, project_points


# 画面最上 6 行有白，才算上沿被切掉。这段只留给地面切换，不决定层数。
_TOP_ROWS = 6
# 比一个边长再高这么多，才当成上面还有一块。地面切换仍用这个提示。
_ABOVE_M = 0.15
# 连续这么多帧同一种，才把层数定下来。
_QUIET_FRAMES = 3
# 提示的角度可以少转 5 度。不到 8 度的晃动不算已经抬头。
_RISE_SLACK_DEG = 5.0
_MIN_RISE_DEG = 8.0
# 回到认下层时的低头角度附近，就继续更新下层。
_BACK_DEG = 4.0
# 低头不超过这个角度，光轴已经接近水平。视场上沿高于相机，能看到 0.5 米以上那一段。
_LEVEL_DEG = 12.0
# 顶块上沿在相机上方 0.20 米。竖向视场一半 29 度，再留 3 度，所以减 26 度。
_CAP_ABOVE_M = 0.20
_FOV_MARGIN_DEG = 26.0
# 数层时离地 0.62 米那一点还要留在画面里。它在相机下方 0.08 米。
_COVER_BELOW_M = 0.08
# 近处抬头允许少 3 度。再少，顶块上沿还在画面外。
_CAP_SLACK_DEG = 3.0
# 黄线顶到上沿，再抬头这么多，最多两次。
_NUDGE_DEG = 5.0
_MAX_NUDGES = 2
# 水平距离比这还短，多半是坏点，不拿来把相机猛地抬起来。
_MIN_RANGE_M = 0.12
# 贴地的白。一层顶面在 0.35 米，留到 0.42 米，不把接缝收成上一层。
_LOWER_LO_M = 0.05
_LOWER_HI_M = 0.42
# 两层白的正面在这块高度里。不要求顶到 0.70 米。顶面是 0.70 米，再高大约 8 厘米就不再算这块白。
_UPPER_LO_M = 0.50
_UPPER_HI_M = 0.78
# 同一块已经高过 0.50 米时，从这里起的点留在上层，不再被 0.50 米切开。
_UPPER_JOIN_LO_M = 0.43
# 没有深度的红蓝，按下层的距离放上去。落在这块高度里、又像一整面，才算塔顶。
_CAP_COLOR_LO_M = 0.72
_CAP_COLOR_HI_M = 0.88
# 层数已定、接近水平时，顶层白盒正面从 0.35 米就开始。盒顶仍是 0.70 米。
_UPPER_RELAX_LO_M = 0.35
_UPPER_FACE_HI_M = 0.70
# 没有深度的白像素交到这一层。落在顶层白盒正面的中段。
_UPPER_RAY_H_M = 0.60
_MIN_WHITE_RELAX = 15
# 塔顶红蓝的底面约 0.70 米。地上的红蓝体心约 0.1 米，进不了这一档。
_CAP_LO_M = 0.60
_CAP_HI_M = 1.05
# 这块塔顶的红偏品红，色相大约 174，暗的一边大约 164，亮度大约 115。
# 蓝的色相大约 111。饱和度下限放宽到 140，保留受光后饱和度下降的蓝面和红蓝交界。
# 色相仍限于 108–122，亮度下限仍为 30，过滤灰白、偏青和过暗区域。
_CAP_SIZE_M = 0.20
_BOX_SIZE_M = 0.35
# 宽和高都要到这块方块在该距离上应有边长的四成。短边至少是长边的三分之一。面积至少盖住应有一面的两成。
_CAP_SIDE_FRAC = 0.40
_CAP_ASPECT = 0.35
_CAP_AREA_FRAC = 0.22
CAP_COLOR_SPECS = [
    {"name": "red", "hsv_low": [0, 180, 110], "hsv_high": [5, 255, 255], "front": False, "fixed_color": True},
    {"name": "red_wrap", "hsv_low": [162, 180, 110], "hsv_high": [179, 255, 255], "front": False, "fixed_color": True},
    {"name": "blue", "hsv_low": [108, 140, 30], "hsv_high": [122, 255, 255], "front": True, "fixed_color": True},
]
# 高处的白和红蓝都要离已经看见的下层大约一块盒子以内。
_NEAR_M = 0.45
# 用来判断高处这一段有没有落在画面里。取在 0.5 米以上、红蓝底面以下。
_COVER_H_M = 0.62
_MIN_WHITE = 40
_MIN_CAP = 30

_KIND_LINE = {
    "one": "一层白，没有塔顶",
    "two": "两层白，没有塔顶",
    "two_cap": "两层白，有红蓝塔顶",
}


def look_up_degrees(distance: float | None, cube_size: float) -> float:
    """从对着下层顶面抬到接近水平，大约多少度。视场上沿会高过相机。"""
    dist = 0.6 if distance is None or not np.isfinite(distance) or distance < 0.2 else float(distance)
    return float(np.degrees(np.arctan(float(cube_size) / dist)))


def horizontal_range(point, gravity: np.ndarray | None) -> float | None:
    """点到光心的距离去掉竖直分量。这是粗水平距离，不是视线深度。"""
    if point is None or gravity is None:
        return None
    down = np.asarray(gravity, dtype=float).reshape(3)
    norm = float(np.linalg.norm(down))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    down = down / norm
    xyz = np.asarray(point, dtype=float).reshape(3)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        return None
    flat = xyz - down * float(np.dot(xyz, down))
    length = float(np.linalg.norm(flat))
    if not np.isfinite(length) or length < _MIN_RANGE_M:
        return None
    return length


def cap_rise_degrees(horizontal: float | None) -> float:
    """比水平再抬头多少度，顶块上沿才落进视场上沿里面大约 3 度。远了就是 0。"""
    if horizontal is None or not np.isfinite(horizontal) or float(horizontal) < _MIN_RANGE_M:
        return 0.0
    rise = float(np.degrees(np.arctan(_CAP_ABOVE_M / float(horizontal))) - _FOV_MARGIN_DEG)
    return max(0.0, rise)


def vote_rise_degrees(horizontal: float | None) -> float:
    """数层时的抬头。离地 0.62 米那一段还要留在画面下沿里面，所以近处先少抬一点。"""
    full = cap_rise_degrees(horizontal)
    if horizontal is None or not np.isfinite(horizontal) or float(horizontal) < _MIN_RANGE_M:
        return 0.0
    limit = _FOV_MARGIN_DEG - float(np.degrees(np.arctan(_COVER_BELOW_M / float(horizontal))))
    return float(min(full, max(0.0, limit)))


def _at_rise(rise: float, pitch: float | None) -> bool:
    """俯仰已经到这个抬头附近。远距离仍是低头不超过 12 度。"""
    if pitch is None or not np.isfinite(pitch):
        return False
    if rise <= 0.5:
        return float(pitch) <= _LEVEL_DEG
    return float(pitch) <= -float(rise) + _CAP_SLACK_DEG


def cap_touches_top(cap_uv) -> bool:
    """黄线顶到画面最上几行。画面中间的跳动不算。"""
    if cap_uv is None:
        return False
    uv = np.asarray(cap_uv)
    if uv.ndim != 2 or uv.shape[0] < 10 or uv.shape[1] < 2:
        return False
    top = uv[:, 1]
    if not np.any(np.isfinite(top)):
        return False
    return float(np.min(top)) <= float(_TOP_ROWS)


def rise_needed(degrees: float) -> float:
    """真正抬头要达到的角度。比提示少 5 度可以，再小也不低于 8 度。"""
    return max(float(degrees) - _RISE_SLACK_DEG, _MIN_RISE_DEG)


def pitch_down_degrees(gravity: np.ndarray | None) -> float | None:
    """低头为正，单位度。重力在相机系里指向下，向前的分量越大，低头越多。没有重力就不知道俯仰。"""
    if gravity is None:
        return None
    down = np.asarray(gravity, dtype=float)
    norm = float(np.linalg.norm(down))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    down = down / norm
    return float(np.degrees(np.arcsin(np.clip(float(down[2]), -1.0, 1.0))))


def record_distance(record: dict | None) -> float | None:
    if not record or record.get("bottom") is None:
        return None
    bottom = np.asarray(record["bottom"], dtype=float).reshape(3)
    length = float(np.linalg.norm(bottom))
    if not np.isfinite(length) or length < 0.2:
        return None
    return length


def second_layer_cue(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    plane,
    intr,
    colors: list[dict],
    cube_size: float,
    zmin: float,
    zmax: float,
) -> tuple[bool, float | None]:
    """地面切换仍用：白顶到画面最上，或白高出一个边长。这不决定层数。"""
    strict, _loose, _front = _color_layers(color_bgr, colors)
    if int(np.count_nonzero(strict)) < 40:
        return False, None
    touches_top = bool(np.any(strict[:_TOP_ROWS, :] > 0))
    ys, xs = np.nonzero(strict)
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    distance = float(np.median(z[good])) if int(good.sum()) >= 10 else None
    high = False
    if plane is not None and int(good.sum()) >= 10:
        points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
        heights = plane.signed_distance(points)
        high = int(np.sum(np.isfinite(heights) & (heights > float(cube_size) + _ABOVE_M))) >= 30
    # 近处一个边长已经比画面高，不能再拿像素高度和边长比。白一直延伸到画面中下部才算上面还有一块。
    reaches_lower = float(ys.max()) >= 0.45 * float(color_bgr.shape[0])
    return bool(high or (touches_top and reaches_lower)), distance


def cap_color_masks(color_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strict color seeds recover connected, nearby low-saturation reflections.

    Hue and brightness limits stay fixed. Weak colors without reliable seeds,
    gray/white highlights and remote background regions are never filled.
    """
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    seeds = [np.zeros(hsv.shape[:2], np.uint8) for _ in range(2)]
    weak = [np.zeros(hsv.shape[:2], np.uint8) for _ in range(2)]
    for spec in CAP_COLOR_SPECS:
        index = 1 if str(spec["name"]).startswith("blue") else 0
        low = np.array(spec["hsv_low"], np.uint8)
        high = np.array(spec["hsv_high"], np.uint8)
        seeds[index] |= cv2.inRange(hsv, low, high)
        relaxed = low.copy()
        relaxed[1] = min(int(low[1]), 80 if index else 110)
        weak[index] |= cv2.inRange(hsv, relaxed, high)
    result = []
    reach = float(np.clip(min(hsv.shape[:2]) * .25, 12, 180))
    for seed, allowed in zip(seeds, weak):
        keep = seed.copy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(allowed, connectivity=8)
        seed_counts = np.bincount(labels[seed > 0], minlength=count)
        trusted = np.zeros(seed.shape, np.uint8)
        for label in range(1, count):
            # A few accidental strong pixels cannot recruit a large weak region.
            if seed_counts[label] >= max(30, .02 * stats[label, cv2.CC_STAT_AREA]):
                trusted[(labels == label) & (seed > 0)] = 255
        if np.any(trusted):
            distance = cv2.distanceTransform((trusted == 0).astype(np.uint8), cv2.DIST_L2, 3)
            accepted = seed_counts >= np.maximum(30, .02 * stats[:, cv2.CC_STAT_AREA])
            accepted[0] = False
            keep[accepted[labels] & (distance <= reach)] = 255
        result.append(keep)
    # A weak red face can have no strict seeds under glare. Require an adjacent
    # substantial blue face before retaining that red component; no global fill.
    for seam, pair in adjacent_color_pairs(weak[0], result[1], color_bgr):
        red_part = pair & (weak[0] > 0)
        blue_part = pair & (result[1] > 0)
        nr, nb = np.count_nonzero(red_part), np.count_nonzero(blue_part)
        if len(seam) < 16 or min(nr, nb) < max(200, hsv.shape[0] * hsv.shape[1] * .0002):
            continue
        if not .15 <= nr / nb <= 3.5:
            continue
        # Limit recovery to the span actually supported by the red/blue seam.
        lo, hi = int(seam[:, 0].min()), int(seam[:, 0].max())
        margin = max(3, int((hi - lo + 1) * .08))
        red_part[:, :max(0, lo - margin)] = False
        red_part[:, min(hsv.shape[1], hi + margin + 1):] = False
        result[0][red_part] = 255
    return result[0], result[1]


def _cap_mask(color_bgr: np.ndarray) -> np.ndarray:
    red, blue = cap_color_masks(color_bgr)
    return cv2.bitwise_or(red, blue)


def _like_face(
    mask: np.ndarray,
    depth: np.ndarray,
    intr,
    zmin: float,
    zmax: float,
    size_m: float,
    known_dist: float | None = None,
) -> np.ndarray:
    """连成一块，并且在这个距离上够得上这块方块的一面。零星点去掉。

    这块颜色自己没有深度时，用下层白已经量到的距离。
    """
    keep = np.zeros(mask.shape, dtype=np.uint8)
    if int(np.count_nonzero(mask)) < 10:
        return keep
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fx = float(intr.fx)
    fy = float(intr.fy)
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if width < 2 or height < 2:
            continue
        if min(width, height) / float(max(width, height)) < _CAP_ASPECT:
            continue
        component = labels == index
        if known_dist is None:
            z = depth[component]
            good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
            if int(good.sum()) < 10:
                continue
            dist = float(np.median(z[good]))
        else:
            dist = float(known_dist)
        if not np.isfinite(dist) or dist < 0.15:
            continue
        expect_w = float(size_m) * fx / dist
        expect_h = float(size_m) * fy / dist
        if width < _CAP_SIDE_FRAC * expect_w or height < _CAP_SIDE_FRAC * expect_h:
            continue
        if area < _CAP_AREA_FRAC * expect_w * expect_h:
            continue
        keep[component] = 255
    return keep


def _cap_faces(mask: np.ndarray, depth: np.ndarray, intr, zmin: float, zmax: float,
               *, red=None, blue=None, color_bgr=None) -> np.ndarray:
    """完整面走原筛选；近处截断的相邻红蓝只恢复原有颜色像素。"""
    keep = _like_face(mask, depth, intr, zmin, zmax, _CAP_SIZE_M)
    if red is None or blue is None:
        return keep
    height, width = mask.shape
    border = max(2, int(min(height, width) * .005))
    for seam, pair in adjacent_color_pairs(red, blue, color_bgr):
        # 必须有原筛选认可的面作为支撑，不能仅凭两个色点绕过几何筛选。
        supported = pair & (keep > 0)
        if np.count_nonzero(supported) < 30 or len(seam) < 16:
            continue
        z = depth[supported]
        valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
        if valid.sum() < 30:
            continue
        distance = float(np.median(z[valid]))
        if not .15 <= distance <= zmax:
            continue
        ys, xs = np.nonzero(pair)
        clipped = (xs.min() < border or xs.max() >= width - border
                   or ys.min() < border or ys.max() >= height - border)
        if clipped:
            if distance > .65:
                continue
        else:
            # A complete adjacent color may be fragmented or merged with a
            # small background strip. Check each color at the supported depth,
            # instead of judging the combined component as one square face.
            expect = _CAP_SIZE_M * float(intr.fx) / distance
            if np.ptp(seam[:, 0]) + 1 < max(16, .35 * expect):
                continue
            plausible = True
            for color_mask in (red, blue):
                yy, xx = np.nonzero(pair & (color_mask > 0))
                if not len(xx) or xx.max()-xx.min()+1 > 2.2*expect or yy.max()-yy.min()+1 > 1.5*expect:
                    plausible = False
                    break
                if len(xx) < .12 * expect * expect:
                    plausible = False
                    break
            if not plausible:
                continue
        # 两色都必须有面积和横向支持；不填缝。
        if min(np.count_nonzero(pair & (red > 0)),
               np.count_nonzero(pair & (blue > 0))) < max(30, mask.size * .0002):
            continue
        keep[pair & (mask > 0)] = 255
    return keep


def _upper_face(pixels: np.ndarray, high: np.ndarray, depth: np.ndarray, intr, shape, zmin: float, zmax: float) -> np.ndarray:
    """上层白里只留下够得上 35 厘米一面的那一块。一小簇不算。"""
    keep = np.zeros(len(pixels), dtype=bool)
    if int(high.sum()) < 10:
        return keep
    patch = np.zeros(shape, dtype=np.uint8)
    uv = pixels[high]
    patch[uv[:, 1], uv[:, 0]] = 255
    liked = _like_face(patch, depth, intr, zmin, zmax, _BOX_SIZE_M)
    if int(np.count_nonzero(liked)) == 0:
        return keep
    keep[high] = liked[uv[:, 1], uv[:, 0]] > 0
    return keep


def _empty_pixels() -> np.ndarray:
    return np.zeros((0, 2), dtype=int)


def _cloud(mask: np.ndarray, depth: np.ndarray, intr, plane, zmin: float, zmax: float):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        empty = np.zeros((0, 3))
        return empty, np.zeros(0), empty, _empty_pixels()
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) < 3:
        empty = np.zeros((0, 3))
        return empty, np.zeros(0), empty, _empty_pixels()
    pixels = np.column_stack([xs[good], ys[good]]).astype(int)
    points = deproject_pixels(pixels.astype(float), z[good], intr)
    heights = plane.signed_distance(points)
    flat = plane.project(points)
    return points, heights, flat, pixels


def _near(flat: np.ndarray, anchor: np.ndarray | None, plane) -> np.ndarray:
    if anchor is None or len(flat) == 0:
        return np.zeros(len(flat), dtype=bool)
    center = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
    return np.linalg.norm(flat - center, axis=1) <= _NEAR_M


def _covers(anchor: np.ndarray | None, plane, intr) -> bool:
    # 下层正上方 0.62 米那一点落在画面里，才算这一眼覆盖到了要看的高度。
    if anchor is None:
        return False
    point = np.asarray(anchor, dtype=float).reshape(3) + np.asarray(plane.normal, dtype=float).reshape(3) * _COVER_H_M
    if not np.isfinite(point[2]) or float(point[2]) <= 0.05:
        return False
    pixel = project_points(point.reshape(1, 3), intr)[0]
    return bool(0 <= pixel[0] < intr.width and 0 <= pixel[1] < intr.height)


def relax_upper_white(kind: str | None, settled: bool, pitch: float | None) -> bool:
    """层数已经是两层，俯仰又接近水平，才放宽顶层白盒。还没定层时不放。"""
    if not settled or kind not in ("two", "two_cap"):
        return False
    if pitch is None or not np.isfinite(pitch) or float(pitch) > _LEVEL_DEG:
        return False
    return True


def _upper_white_from_color(color_bgr, depth, intr, plane, colors, zmin: float, zmax: float, anchor):
    """红蓝块下面没有深度的白，交到离地 0.60 米。只取大约一块白盒那么高。"""
    mask = _cap_mask(color_bgr)
    if int(np.count_nonzero(mask)) < 10:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[index, cv2.CC_STAT_AREA]) < 10:
        return None
    component = labels == index
    z = depth[component]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if int(good.sum()) >= 10:
        dist = float(np.median(z[good]))
    elif anchor is not None:
        dist = float(np.linalg.norm(np.asarray(anchor, dtype=float).reshape(3)))
    else:
        return None
    if not np.isfinite(dist) or dist < 0.15:
        return None
    left = int(stats[index, cv2.CC_STAT_LEFT])
    top = int(stats[index, cv2.CC_STAT_TOP])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    cap_bottom = top + int(stats[index, cv2.CC_STAT_HEIGHT])
    span = int(np.ceil((_UPPER_FACE_HI_M - _UPPER_RELAX_LO_M) * float(intr.fy) / dist))
    if span < 2 or cap_bottom >= color_bgr.shape[0]:
        return None
    white = _white_pixels(color_bgr, colors) > 0
    usable = np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)
    band = np.zeros(color_bgr.shape[:2], dtype=bool)
    band[cap_bottom : min(color_bgr.shape[0], cap_bottom + span), left : left + width] = True
    chosen = white & band & ~usable
    ys, xs = np.nonzero(chosen)
    if len(xs) < _MIN_WHITE_RELAX:
        return None
    step = max(1, int(np.ceil(len(xs) / 2000)))
    pixels = np.column_stack([xs[::step], ys[::step]]).astype(float)
    dirs = deproject_pixels(pixels, np.ones(len(pixels)), intr)
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    denom = dirs @ normal
    scale = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    scale[ok] = (_UPPER_RAY_H_M + float(plane.offset)) / denom[ok]
    scale[~(scale > 0.05)] = np.nan
    points = scale[:, None] * dirs
    heights = plane.signed_distance(points)
    flat = plane.project(points)
    keep = np.isfinite(scale) & (heights > _UPPER_RELAX_LO_M) & (heights < _UPPER_FACE_HI_M)
    keep = keep & (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    keep = keep & _near(flat, anchor, plane)
    if int(keep.sum()) < _MIN_WHITE_RELAX:
        return None
    return np.median(points[keep], axis=0), pixels[keep].astype(int)


def _above_lower_window(shape, lower_uv: np.ndarray, dist: float | None, intr):
    """下层白盒上方、左右再多出大约一块盒子。没有下层像素就不定这扇窗。"""
    if lower_uv is None or len(lower_uv) < 10 or dist is None or not np.isfinite(dist) or float(dist) < 0.15:
        return None
    rows = lower_uv[:, 1]
    cols = lower_uv[:, 0]
    # 靠上的一截当盒顶。一个飞点不会把窗口抬走。
    top = int(np.percentile(rows, 10))
    left = int(np.min(cols))
    right = int(np.max(cols))
    pad = int(np.ceil(_BOX_SIZE_M * float(intr.fx) / float(dist)))
    mask = np.zeros(shape, dtype=bool)
    row1 = min(int(shape[0]), top + 4)
    c0 = max(0, left - pad)
    c1 = min(int(shape[1]), right + 1 + pad)
    if row1 > 0 and c1 > c0:
        mask[:row1, c0:c1] = True
    return mask


def _closest_mask(mask: np.ndarray, ys, xs, flat: np.ndarray, anchor, plane) -> np.ndarray:
    """好几块都留着时，只留地面位置离下层中心最近的那一块。"""
    if anchor is None or int(np.count_nonzero(mask)) < 10:
        return mask
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 3:
        return mask
    center = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
    on = mask[ys, xs] > 0
    if int(on.sum()) < 10:
        return mask
    labels_on = labels[ys[on], xs[on]]
    flats = flat[on]
    best = None
    best_dist = None
    for index in np.unique(labels_on):
        if int(index) == 0:
            continue
        part = labels_on == index
        if int(part.sum()) < 10:
            continue
        dist = float(np.linalg.norm(np.median(flats[part], axis=0) - center))
        if best_dist is None or dist < best_dist:
            best = int(index)
            best_dist = dist
    if best is None:
        return mask
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def _closest_high(high: np.ndarray, pixels: np.ndarray, flat: np.ndarray, anchor, plane, shape) -> np.ndarray:
    if anchor is None or int(high.sum()) < 10:
        return high
    uv = pixels[high]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[uv[:, 1], uv[:, 0]] = 255
    kept = _closest_mask(mask, uv[:, 1], uv[:, 0], flat[high], anchor, plane)
    on = kept[uv[:, 1], uv[:, 0]] > 0
    out = np.zeros(len(high), dtype=bool)
    out[np.flatnonzero(high)[on]] = True
    return out


def _tower_distance(distance: float | None, anchor: np.ndarray | None) -> float | None:
    """下层白的视线距离。这一眼没有，就用已经记下的锚点向前的分量。"""
    if distance is not None and np.isfinite(distance) and float(distance) >= 0.15:
        return float(distance)
    if anchor is None:
        return None
    forward = float(np.asarray(anchor, dtype=float).reshape(3)[2])
    if np.isfinite(forward) and forward >= 0.15:
        return forward
    return None


def _join_upper_seam(pixels: np.ndarray, heights: np.ndarray, near: np.ndarray, shape) -> np.ndarray:
    """已经高过 0.50 米的那一块，把同一块从 0.43 米起的点留在上层。

    只停在 0.43 到 0.50 米、上面没有更高的白，这里什么都不留。
    """
    core = (heights > _UPPER_LO_M) & (heights < _UPPER_HI_M) & near
    if int(core.sum()) < 10:
        return core
    band = (heights > _UPPER_JOIN_LO_M) & (heights < _UPPER_HI_M) & near
    if int(band.sum()) == int(core.sum()):
        return core
    patch = np.zeros(shape, dtype=np.uint8)
    uv = pixels[band]
    patch[uv[:, 1], uv[:, 0]] = 255
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    if count < 2:
        return core
    core_uv = pixels[core]
    keep_ids = np.unique(labels[core_uv[:, 1], core_uv[:, 0]])
    keep_ids = keep_ids[keep_ids > 0]
    if len(keep_ids) == 0:
        return core
    wanted = np.zeros(count, dtype=bool)
    wanted[keep_ids] = True
    chosen = wanted[labels[uv[:, 1], uv[:, 0]]]
    out = np.zeros(len(pixels), dtype=bool)
    out[np.flatnonzero(band)[chosen]] = True
    return out


def _placed_at(xs: np.ndarray, ys: np.ndarray, dist: float, intr, plane):
    """没有深度的像素，按下层白的距离放回射线。高度用这块地面。"""
    pixels = np.column_stack([xs, ys]).astype(float)
    points = deproject_pixels(pixels, np.full(len(xs), dist, dtype=float), intr)
    heights = plane.signed_distance(points)
    flat = plane.project(points)
    return points, heights, flat, pixels


def _at_tower_face(pixels, intr, plane, anchor):
    """射线交到经过塔脚的竖直面，俯仰后不再给所有像素同一个 z。"""
    foot = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
    horizontal = foot - plane.origin()
    length = float(np.linalg.norm(horizontal))
    if length < 0.12:
        return np.full((len(pixels), 3), np.nan)
    normal = horizontal / length
    rays = deproject_pixels(np.asarray(pixels, dtype=float), np.ones(len(pixels)), intr)
    denom = rays @ normal
    scale = np.full(len(pixels), np.nan)
    good = denom > 1e-6
    scale[good] = float(foot @ normal) / denom[good]
    scale[scale <= 0.05] = np.nan
    return rays * scale[:, None]


def _color_face_at(
    chosen: np.ndarray,
    depth: np.ndarray,
    intr,
    plane,
    anchor,
    dist: float,
    zmin: float,
    zmax: float,
    low: float,
    high: float,
    size_m: float,
    nearest: bool = False,
    diagnostics: dict | None = None,
):
    """射线交到塔所在竖直面，检查高度、位置和完整面尺寸。"""
    ys, xs = np.nonzero(chosen)
    if diagnostics is not None:
        diagnostics.update(input=int(len(xs)), distance=float(dist), stage="color")
    if len(xs) < 10:
        return None
    pixels = np.column_stack([xs, ys]).astype(int)
    points = _at_tower_face(pixels, intr, plane, anchor)
    heights = plane.signed_distance(points)
    flat = plane.project(points)
    height_ok = (heights > low) & (heights < high)
    keep = height_ok & _near(flat, anchor, plane)
    keep = keep & (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
    if diagnostics is not None:
        diagnostics.update(height=int(height_ok.sum()), near=int(keep.sum()), stage="height/position")
    if int(keep.sum()) < 10:
        return None
    mask = np.zeros(chosen.shape, dtype=np.uint8)
    mask[ys[keep], xs[keep]] = 255
    # 面尺寸使用交点的当前相机深度，不能使用塔脚 z 或水平距离。
    face_distance = float(np.median(points[keep, 2]))
    liked = _like_face(mask, depth, intr, zmin, zmax, size_m, known_dist=face_distance)
    if diagnostics is not None:
        diagnostics.update(face=int(np.count_nonzero(liked)), stage="face")
    if nearest:
        liked = _closest_mask(liked, ys[keep], xs[keep], flat[keep], anchor, plane)
    if int(np.count_nonzero(liked)) == 0:
        return None
    on_face = liked[ys[keep], xs[keep]] > 0
    if int(on_face.sum()) < 10:
        return None
    if diagnostics is not None:
        diagnostics["stage"] = "accepted"
    return np.median(points[keep][on_face], axis=0), pixels[keep][on_face].astype(int)


def _cap_depth_mask(color_bgr, depth, intr, plane, anchor, diagnostics):
    """只有实测深度支持的红蓝可确认塔顶；超量程也不作为缺失补回。"""
    mask = _cap_mask(color_bgr)
    if anchor is None:
        return np.zeros(mask.shape, np.uint8)
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    accepted = np.zeros(len(xs), dtype=bool)
    if valid.any():
        points = deproject_pixels(np.column_stack([xs[valid], ys[valid]]), z[valid], intr)
        heights = plane.signed_distance(points)
        accepted[valid] = (_near(plane.project(points), anchor, plane)
                           & (heights > _CAP_LO_M) & (heights < _CAP_HI_M))
    out = np.zeros(mask.shape, np.uint8)
    out[ys[accepted], xs[accepted]] = 255
    diagnostics.update(depth_valid=int(valid.sum()), depth_supported=int(accepted.sum()))
    return out


def _upper_top_support(points, pixels, high, plane, shape):
    """俯视时按顶面在地面上的二维跨度认第二层，不要求顶面像正面一样高。"""
    keep = np.zeros(len(points), dtype=bool)
    heights = plane.signed_distance(points)
    top = high & (heights > .64) & (heights < .76)
    patch = np.zeros(shape, np.uint8)
    uv = pixels[top]
    patch[uv[:, 1], uv[:, 0]] = 255
    count, labels = cv2.connectedComponents(patch)
    axes = plane.basis()
    for label in range(1, count):
        selected = top.copy()
        selected[top] = labels[uv[:, 1], uv[:, 0]] == label
        if selected.sum() < _MIN_WHITE:
            continue
        flat = plane.project(points[selected])
        spans = [np.diff(np.percentile(flat @ axis, [5, 95]))[0] for axis in axes]
        if min(spans) >= .14 and max(spans) <= .50:
            keep |= selected
    return keep


def _rgb_cap_candidate(color_bgr, intr, plane, anchor, diagnostics):
    """只确认相邻红蓝候选；允许截断，不产生体心或角度。"""
    masks = cap_color_masks(color_bgr)
    diagnostics.update(red=int(np.count_nonzero(masks[0])), blue=int(np.count_nonzero(masks[1])))
    if anchor is None:
        diagnostics["candidate"] = "no_anchor"
        return _empty_pixels()
    # 先关联原始可见红蓝，再检查它们是否位于当前塔附近。
    # 不能先把掩膜裁成窄高度带，再要求裁剩的碎片构成完整面。
    max_gap = max(8, int(color_bgr.shape[0] * .12))
    for seam, component in adjacent_color_pairs(masks[0], masks[1], color_bgr,
                                                gap_fraction=.12, max_gap=max_gap):
        minimum = max(30, int(intr.width * intr.height * .0002))
        if np.count_nonzero((masks[0] > 0) & component) < minimum or np.count_nonzero((masks[1] > 0) & component) < minimum:
            continue
        # 射线交点仅用于候选关联，不作为测得的体心、距离或拟合输入。
        points = _at_tower_face(seam, intr, plane, anchor)
        heights = plane.signed_distance(points)
        supported = ((heights > .62) & (heights < 1.02)
                     & _near(plane.project(points), anchor, plane)
                     & np.isfinite(points).all(axis=1))
        if int(supported.sum()) < max(8, int(len(seam) * .35)):
            continue
        yy, xx = np.nonzero(component)
        border = max(2, int(min(color_bgr.shape[:2]) * .005))
        edges = []
        if int(xx.min()) < border:
            edges.append("left")
        if int(xx.max()) >= color_bgr.shape[1] - border:
            edges.append("right")
        if int(yy.min()) < border:
            edges.append("top")
        if int(yy.max()) >= color_bgr.shape[0] - border:
            edges.append("bottom")
        diagnostics.update(candidate="visible", boundary_points=len(seam),
                           boundary_supported=int(supported.sum()), clipped_edges=edges,
                           partial=bool(edges) or int(supported.sum()) < .8 * len(seam))
        return np.column_stack([xx, yy]).astype(int)
    diagnostics["candidate"] = "no_adjacent_pair"
    return _empty_pixels()


def _upper_white_by_color(color_bgr, depth, intr, plane, colors, zmin, zmax, anchor, distance, lower_uv):
    """下层还在、上层没有深度时，只在下层白盒上方找白，用下层的距离放回 0.50 米以上。"""
    dist = _tower_distance(distance, anchor)
    window = _above_lower_window(color_bgr.shape[:2], lower_uv, dist, intr)
    if anchor is None or dist is None or window is None:
        return None
    white = _white_pixels(color_bgr, colors) > 0
    usable = np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)
    return _color_face_at(
        white & ~usable & window,
        depth,
        intr,
        plane,
        anchor,
        dist,
        zmin,
        zmax,
        _UPPER_LO_M,
        _UPPER_HI_M,
        _BOX_SIZE_M,
        nearest=True,
    )


def _cap_by_color(color_bgr, depth, intr, plane, colors, zmin, zmax, anchor, distance, diagnostics=None):
    """红蓝没有深度时，按当前塔脚的竖直面求交，再检查高度和尺寸。"""
    dist = None
    if anchor is not None:
        foot = plane.project(np.asarray(anchor, dtype=float).reshape(1, 3))[0]
        dist = float(np.linalg.norm(foot - plane.origin()))
        if not np.isfinite(dist) or dist < _MIN_RANGE_M:
            dist = None
    if anchor is None or dist is None:
        if diagnostics is not None:
            diagnostics["stage"] = "no_anchor_or_distance"
        return None
    colored = _cap_mask(color_bgr)
    colored[_white_pixels(color_bgr, colors) > 0] = 0
    usable = np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)
    return _color_face_at(
        (colored > 0) & ~usable,
        depth,
        intr,
        plane,
        anchor,
        dist,
        zmin,
        zmax,
        _CAP_COLOR_LO_M,
        _CAP_COLOR_HI_M,
        _CAP_SIZE_M,
        diagnostics=diagnostics,
    )


def coarse_bands(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    plane,
    intr,
    colors: list[dict],
    zmin: float,
    zmax: float,
    anchor: np.ndarray | None = None,
    relax_upper: bool = False,
    pitch: float | None = None,
) -> dict:
    """这一眼的高度带。下层白给出位置。高处的白和红蓝都靠这个位置，不量顶面。

    有深度、并且已经高过 0.50 米时，同一块从 0.43 米起留在上层。上层白收到 0.78 米。
    下层还在画面里时，第 2 层只看它上方、左右再多出大约一块盒子的那一扇。
    没有深度时，用下层的距离把这扇里的白放回 0.50 米以上，把红蓝放回 0.72 到 0.88 米。
    层数已定并且接近水平时，顶层白盒的正面可以从 0.35 米算起。
    还在低头、贴地的白还在画面里，深度这条的下限仍是 0.50 米。
    """
    found = anchor
    lower = False
    upper = False
    cap = False
    distance = None
    lower_point = None
    upper_point = None
    cap_point = None
    lower_uv = _empty_pixels()
    upper_uv = _empty_pixels()
    cap_uv = _empty_pixels()
    rgb_debug = {}
    rgb_candidate = False
    blank = {
        "lower": False,
        "upper_white": False,
        "cap": False,
        "distance": None,
        "anchor": anchor,
        "covers": False,
        "lower_point": None,
        "upper_point": None,
        "cap_point": None,
        "lower_uv": lower_uv,
        "upper_uv": upper_uv,
        "cap_uv": cap_uv,
    }
    if plane is None:
        return blank
    white = _white_pixels(color_bgr, colors)
    points, heights, flat, pixels = _cloud(white, depth, intr, plane, zmin, zmax)
    upper_lo = _UPPER_LO_M
    need = _MIN_WHITE
    if len(points):
        body = (heights > _LOWER_LO_M) & (heights < _LOWER_HI_M)
        if int(body.sum()) >= _MIN_WHITE:
            lower = True
            found = np.median(flat[body], axis=0)
            lower_point = np.median(points[body], axis=0)
            lower_uv = pixels[body]
            distance = float(np.median(points[body, 2]))
    looking_down = pitch is not None and np.isfinite(pitch) and float(pitch) > 0.0
    if relax_upper and not (looking_down and lower):
        upper_lo = _UPPER_RELAX_LO_M
        need = _MIN_WHITE_RELAX
    if len(points):
        near = _near(flat, found, plane)
        window = _above_lower_window(color_bgr.shape[:2], lower_uv, distance, intr)
        if window is not None:
            near = near & window[pixels[:, 1], pixels[:, 0]]
        # 还没放宽时，一小簇白不算上层。已经高过 0.50 米的，从 0.43 米起并进同一块。
        # 层数已定、接近水平，只剩一条棱，仍按点数留。
        if need == _MIN_WHITE:
            high = _join_upper_seam(pixels, heights, near, color_bgr.shape[:2])
            top_support = (_upper_top_support(points, pixels, high, plane, color_bgr.shape[:2])
                           if looking_down else np.zeros(len(points), dtype=bool))
            high = _upper_face(pixels, high, depth, intr, color_bgr.shape[:2], zmin, zmax) | top_support
        else:
            high = (heights > upper_lo) & (heights < _UPPER_HI_M) & near
        high = _closest_high(high, pixels, flat, found, plane, color_bgr.shape[:2])
        upper = int(high.sum()) >= need
        if upper:
            upper_point = np.median(points[high], axis=0)
            upper_uv = pixels[high]
    if not upper and found is not None:
        filled = _upper_white_by_color(
            color_bgr, depth, intr, plane, colors, zmin, zmax, found, distance, lower_uv
        )
        if filled is not None:
            upper = True
            upper_point, upper_uv = filled
    if relax_upper and not lower and not upper:
        filled = _upper_white_from_color(color_bgr, depth, intr, plane, colors, zmin, zmax, found)
        if filled is not None:
            upper = True
            upper_point, upper_uv = filled
    colored = _cap_mask(color_bgr)
    colored[white > 0] = 0
    colored &= _cap_depth_mask(color_bgr, depth, intr, plane, found, rgb_debug)
    colored = _cap_faces(colored, depth, intr, zmin, zmax)
    cap_points, cap_heights, cap_flat, cap_pixels = _cloud(colored, depth, intr, plane, zmin, zmax)
    if len(cap_points):
        band = (cap_heights > _CAP_LO_M) & (cap_heights < _CAP_HI_M) & _near(cap_flat, found, plane)
        cap = int(band.sum()) >= _MIN_CAP
        if cap:
            cap_point = np.median(cap_points[band], axis=0)
            cap_uv = cap_pixels[band]
    # 无深度红蓝只作观察候选，不允许候选把层数改成 two_cap。
    candidate_uv = _rgb_cap_candidate(color_bgr, intr, plane, found, rgb_debug)
    if not cap and len(candidate_uv):
        rgb_candidate = True
        cap_uv = candidate_uv
        rgb_debug["stage"] = "depth_unconfirmed"
    return {
        "lower": lower,
        "upper_white": bool(upper),
        "cap": bool(cap),
        "distance": distance,
        "anchor": found,
        "covers": _covers(found, plane, intr),
        "lower_point": lower_point,
        "upper_point": upper_point,
        "cap_point": cap_point,
        "lower_uv": lower_uv,
        "upper_uv": upper_uv,
        "cap_uv": cap_uv,
        "cap_rgb_candidate": rgb_candidate,
        "cap_partial": bool(rgb_debug.get("partial", False)),
        "rgb_debug": rgb_debug,
    }


def _vote(bands: dict | None) -> str | None:
    if not bands:
        return None
    if bands.get("cap"):
        return "two_cap"
    if bands.get("cap_rgb_candidate"):
        return None
    if bands.get("upper_white"):
        return "two"
    if bands.get("covers"):
        return "one"
    return None


@dataclass
class StackSearch:
    """先确认贴地的白，再抬头分三种塔。近处要抬过水平。没定之前不报层数。"""

    stage: str = "ground"
    kind: str | None = None
    lower: dict | None = None
    upper: dict | None = None
    anchor: np.ndarray | None = None
    quiet: int = 0
    pending: str | None = None
    degrees: float = 0.0
    cue: bool = False
    looked_up: bool = False
    settled: bool = False
    settled_one: bool = False
    base_pitch: float | None = None
    remembered: list[dict] = field(default_factory=list)
    range_m: float | None = None
    vote_rise: float = 0.0
    nudge_deg: float = 0.0
    nudges: int = 0
    clip_run: int = 0
    pitch: float | None = None
    high_view: bool = False
    anchor_basis: np.ndarray | None = None

    def reorient(self, gravity: np.ndarray | None) -> np.ndarray:
        """把重力对齐系里保存的锚点转换到当前相机系；只补偿俯仰/横滚。"""
        identity = np.eye(3)
        if gravity is None:
            # 姿态中断后不能把未补偿的旧锚点用于 RGB 定位。
            if self.anchor_basis is not None:
                self.anchor = None
                self.anchor_basis = None
            return identity
        up = -np.asarray(gravity, dtype=float).reshape(3)
        norm = float(np.linalg.norm(up))
        if not np.isfinite(norm) or norm < 1e-6:
            self.anchor = None
            self.anchor_basis = None
            return identity
        up /= norm
        right = np.array([1.0, 0.0, 0.0])
        right -= up * float(right @ up)
        if np.linalg.norm(right) < 0.25:
            self.anchor = None
            self.anchor_basis = None
            return identity
        right /= np.linalg.norm(right)
        basis = np.column_stack([right, np.cross(up, right), up])
        rotation = identity if self.anchor_basis is None else basis @ self.anchor_basis.T
        if self.anchor is not None:
            self.anchor = rotation @ self.anchor
        self.anchor_basis = basis
        return rotation

    def update(
        self,
        found: list[dict],
        bands: dict | None,
        distance: float | None,
        cube_size: float,
        pitch: float | None = None,
        gravity: np.ndarray | None = None,
    ) -> str:
        # 层数只看高度带。白盒位姿可以留着画框，不拿来数层。
        lower_now = next((item for item in found if int(item.get("layer", 1)) == 1), None)
        info = bands or {}
        self.reorient(gravity)
        pitch_now = None if pitch is None or not np.isfinite(pitch) else float(pitch)
        self.pitch = pitch_now
        # 接近水平以后，下层位置不再跟着这一眼的白带跑。数层还要再等到近处该抬的角度。
        self.high_view = pitch_now is not None and pitch_now <= _LEVEL_DEG
        if info.get("anchor") is not None and (self.anchor is None or not self.high_view):
            self.anchor = np.asarray(info["anchor"], dtype=float).reshape(3)
        if not self.high_view and pitch_now is not None and self.anchor is not None:
            if self.base_pitch is None or pitch_now > self.base_pitch:
                self.base_pitch = pitch_now
        fresh = horizontal_range(info.get("lower_point"), gravity)
        if fresh is None:
            fresh = horizontal_range(self.anchor if self.anchor is not None else info.get("anchor"), gravity)
        if fresh is not None and (self.range_m is None or not self.high_view):
            self.range_m = fresh
        self.vote_rise = vote_rise_degrees(self.range_m) + self.nudge_deg
        self.looked_up = _at_rise(self.vote_rise, pitch_now)
        used = record_distance(lower_now if lower_now is not None else self.lower)
        if used is None:
            used = info.get("distance") if info.get("distance") is not None else distance
        back_down = (
            pitch_now is None
            or self.base_pitch is None
            or float(self.base_pitch - pitch_now) < _BACK_DEG
        )
        if self.anchor is not None and self.lower is None:
            self.lower = {"layer": 1}
        if not self.looked_up and lower_now is not None and back_down and not self.settled:
            self.lower = lower_now
        self.cue = self.anchor is not None and not self.settled_one

        # 已经到了数层的角度，黄线还顶着上沿，就先不数。连续三帧再抬 5 度，最多两次。
        clipped = cap_touches_top(info.get("cap_uv"))
        hold_vote = False
        if self.looked_up and clipped and not self.settled and self.nudges < _MAX_NUDGES:
            self.clip_run += 1
            self.quiet = 0
            self.pending = None
            hold_vote = True
            if self.clip_run >= _QUIET_FRAMES:
                self.nudges += 1
                self.nudge_deg += _NUDGE_DEG
                self.clip_run = 0
                self.vote_rise = vote_rise_degrees(self.range_m) + self.nudge_deg
                self.looked_up = _at_rise(self.vote_rise, pitch_now)
        elif not clipped:
            self.clip_run = 0

        if self.settled:
            self.stage = "one" if self.kind == "one" else "both"
        elif not self.looked_up:
            self.pending = None
            self.quiet = 0
            self.stage = "one" if self.anchor is not None else "ground"
        elif hold_vote:
            self.stage = "upstairs"
        else:
            self.stage = "upstairs"
            kind = _vote(info)
            if kind is not None and self.anchor is not None:
                if kind == self.pending:
                    self.quiet += 1
                else:
                    self.pending = kind
                    self.quiet = 1
                if self.quiet >= _QUIET_FRAMES:
                    self.kind = kind
                    self.settled = True
                    self.settled_one = kind == "one"
                    self.upper = None if kind == "one" else {"layer": 2}
                    self.stage = "one" if kind == "one" else "both"
                    self.cue = not self.settled_one

        self._set_degrees(pitch_now, used, cube_size)
        kept = self.lower if self.lower is not None and "x" in self.lower else None
        self.remembered = [kept] if kept is not None else []
        return self.line()

    def _set_degrees(self, pitch_now: float | None, used: float | None, cube_size: float) -> None:
        # 还差多少度才到这一眼要的抬头。层数定了以后画面不再用这个数。
        aim = self.vote_rise
        target = -aim if aim > 0.5 else 0.0
        if pitch_now is not None:
            self.degrees = max(0.0, float(pitch_now) - target)
        elif self.base_pitch is not None:
            self.degrees = max(0.0, float(self.base_pitch) - target)
        else:
            self.degrees = look_up_degrees(used, cube_size) + max(aim, 0.0)

    def line(self) -> str:
        if self.settled and self.kind in _KIND_LINE:
            return _KIND_LINE[self.kind]
        if self.looked_up:
            return "正在看上面"
        if self.anchor is not None and self.nudges:
            return f"黄线顶到上沿。再抬头大约 {self.degrees:.0f} 度"
        if self.anchor is not None:
            return f"正在看下面。抬头大约 {self.degrees:.0f} 度"
        return "先看地面和下层"

    def asking_to_look_up(self) -> bool:
        return bool(self.anchor is not None and not self.looked_up and not self.settled)
