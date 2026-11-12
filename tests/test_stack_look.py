"""两层分两眼看：先记下层，再按距离抬头看上层。画面用白话。"""

from __future__ import annotations

import cv2
import numpy as np

from cube.detect import _blank_lower_view
from cube.stack_look import (
    StackSearch,
    cap_rise_degrees,
    coarse_bands,
    horizontal_range,
    look_up_degrees,
    pitch_down_degrees,
    relax_upper_white,
    rise_needed,
    second_layer_cue,
    vote_rise_degrees,
)
from ground.plane import Plane
from main import load_config
from sim.scene import intrinsics_for
from view_cube_center import _lines


def _cfg():
    return load_config("config/default.yaml")


def _box(layer: int, bottom: list[float]) -> dict:
    return {
        "layer": layer,
        "bottom": np.array(bottom, dtype=float),
        "x": bottom[0],
        "y": bottom[1],
        "z": bottom[2],
        "yaw": 0.2,
        "long_side": 0.35,
        "short_side": 0.34,
    }


def _bands(**kwargs) -> dict:
    base = {
        "lower": False,
        "upper_white": False,
        "cap": False,
        "distance": 0.8,
        "anchor": np.array([0.1, 0.2, 0.8]),
        "covers": False,
    }
    base.update(kwargs)
    return base


def test_look_up_is_about_one_edge_at_that_distance():
    # 0.6 m 外从下层顶面抬到水平，大约 30 度。视场上沿再高过相机。
    degrees = look_up_degrees(0.6, 0.35)
    assert 28.0 < degrees < 33.0


def test_low_look_asks_to_rise_and_does_not_count_layers():
    search = StackSearch()
    line = search.update([_box(1, [0.1, 0.2, 0.8])], _bands(lower=True, upper_white=True), 0.8, 0.35, pitch=35.0)
    assert line.startswith("正在看下面")
    assert "抬头大约 35 度" in line
    assert search.kind is None
    assert search.asking_to_look_up()
    for _ in range(4):
        line = search.update([_box(1, [0.1, 0.2, 0.8])], _bands(lower=True, upper_white=True), 0.8, 0.35, pitch=35.0)
    assert search.kind is None
    assert line.startswith("正在看下面")


def test_pitch_down_reads_the_forward_part_of_gravity():
    assert pitch_down_degrees(None) is None
    assert abs(pitch_down_degrees(np.array([0.0, 1.0, 0.0]))) < 1e-6
    alpha = np.deg2rad(35.0)
    down = np.array([0.0, np.cos(alpha), np.sin(alpha)])
    assert abs(pitch_down_degrees(down) - 35.0) < 1e-6
    # 0.8 m 外抬高一块是 24 度左右，少 5 度大约 19 度，不能被 8 度的下限抬得更高。
    assert 18.0 < rise_needed(look_up_degrees(0.8, 0.35)) < 20.0


def test_high_band_is_ignored_until_the_camera_is_near_level():
    search = StackSearch()
    lower = _box(1, [0.10, 0.20, 0.80])
    # 低头看着地面。高处已经有白，也不数成两层。
    line = search.update([lower], _bands(lower=True, upper_white=True), 0.8, 0.35, pitch=35.0)
    assert line.startswith("正在看下面")
    assert search.kind is None
    assert search.stage == "one"
    nudged = _box(1, [0.12, 0.20, 0.80])
    search.update([nudged], _bands(lower=True, upper_white=True), 0.8, 0.35, pitch=32.0)
    assert np.allclose(search.lower["bottom"], nudged["bottom"])
    # 离开低头角度、还没到接近水平时，下层先留着，层数仍不报。
    search.update([_box(1, [9.0, 9.0, 9.0])], _bands(lower=True, upper_white=True), 0.8, 0.35, pitch=29.0)
    assert np.allclose(search.lower["bottom"], nudged["bottom"])
    assert search.kind is None
    line = ""
    high = _bands(upper_white=True, covers=True)
    for _ in range(3):
        line = search.update([_box(1, [8.0, 8.0, 8.0])], high, 0.7, 0.35, pitch=10.0)
    assert line == "两层白，没有塔顶"
    assert np.allclose(search.lower["bottom"], nudged["bottom"])
    assert search.kind == "two"
    # 定下来之后，单独一帧改口不改层数。
    again = search.update([], _bands(covers=True), 0.7, 0.35, pitch=10.0)
    assert again == "两层白，没有塔顶"
    assert search.kind == "two"


def test_level_view_without_high_white_or_cap_is_one_layer():
    search = StackSearch()
    lower = _box(1, [0.0, 0.2, 0.8])
    search.update([lower], _bands(lower=True), 0.8, 0.35, pitch=35.0)
    line = ""
    for _ in range(3):
        line = search.update([], _bands(covers=True), 0.8, 0.35, pitch=0.0)
    assert line == "一层白，没有塔顶"
    assert search.kind == "one"
    assert search.upper is None
    held = search.update([], _bands(upper_white=True), 0.8, 0.35, pitch=0.0)
    assert held == "一层白，没有塔顶"


def test_cap_near_camera_height_marks_two_layers_with_a_top():
    search = StackSearch()
    search.update([_box(1, [0.0, 0.2, 0.8])], _bands(lower=True), 0.8, 0.35, pitch=30.0)
    line = ""
    for _ in range(3):
        line = search.update([], _bands(cap=True, covers=True), 0.8, 0.35, pitch=2.0)
    assert line == "两层白，有红蓝塔顶"
    assert search.kind == "two_cap"
    assert search.settled_one is False


def _level_plane() -> Plane:
    # 相机水平，Y 向下。地面在相机下方 0.70 米，高度就是 0.70 减相机 Y。
    return Plane(np.array([0.0, -1.0, 0.0]), -0.70)


def _height_rows(intr, z: float, low: float, high: float) -> np.ndarray:
    fy = float(intr.fy)
    cy = float(intr.cy)
    rows = np.arange(intr.height, dtype=float)
    height = 0.70 - (rows - cy) / fy * z
    return (height >= low) & (height <= high)


def _fill_height(color, depth, intr, z: float, low: float, high: float, u0: int, u1: int, bgr) -> None:
    fy = float(intr.fy)
    cy = float(intr.cy)
    for v in range(intr.height):
        height = 0.70 - (v - cy) / fy * z
        if low <= height <= high:
            color[v, u0:u1] = bgr
            depth[v, u0:u1] = z


def test_height_bands_separate_the_three_towers():
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60

    one = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    one_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(one, one_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    # 地上的红蓝停在低处，不算塔顶。
    _fill_height(one, one_depth, intr, z, 0.05, 0.18, 60, 100, (0, 0, 255))
    seen = coarse_bands(one, one_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert seen["lower_point"] is not None
    assert abs(float(seen["lower_point"][2]) - z) < 0.05
    assert not seen["upper_white"]
    assert not seen["cap"]
    assert seen["covers"]

    far = one.copy()
    far_depth = one_depth.copy()
    _fill_height(far, far_depth, intr, z, 0.52, 0.68, 148, 160, (255, 255, 255))
    seen = coarse_bands(far, far_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert not seen["upper_white"]

    two = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    two_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(two, two_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(two, two_depth, intr, z, 0.52, 0.68, 60, 100, (255, 255, 255))
    # 远处的白墙不算第二层。
    _fill_height(two, two_depth, intr, z, 0.52, 0.68, 148, 160, (255, 255, 255))
    seen = coarse_bands(two, two_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert seen["upper_white"]
    assert not seen["cap"]

    capped = two.copy()
    capped_depth = two_depth.copy()
    _fill_height(capped, capped_depth, intr, z, 0.74, 0.90, 60, 100, (255, 0, 0))
    seen = coarse_bands(capped, capped_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["upper_white"]
    assert seen["cap"]


def _bgr(hue: int, sat: int, val: int) -> tuple[int, int, int]:
    patch = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
    blue, green, red = (int(v) for v in patch[0, 0])
    return blue, green, red


def test_upper_white_rejects_a_small_cluster():
    # 点数够四十，但没有 35 厘米一面那么大。椅子上的一小簇不算两层。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(color, depth, intr, z, 0.55, 0.60, 70, 82, (255, 255, 255))
    seen = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert not seen["upper_white"]


def test_upper_face_without_depth_counts_from_the_lower_distance():
    # 上层正面没有深度。用下层的距离放回 0.50 米以上，够 35 厘米的一面就算。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.20, 0.40, 60, 110, (255, 255, 255))
    _fill_height(color, depth, intr, z, 0.50, 0.72, 60, 110, (255, 255, 255))
    # 只擦掉高处那一块的深度。贴地的白还留着。
    depth[np.broadcast_to(_height_rows(intr, z, 0.50, 0.72)[:, None], color.shape[:2])] = np.nan
    seen = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert seen["upper_white"]
    assert seen["upper_point"] is not None
    assert not seen["cap"]

    small = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    small_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(small, small_depth, intr, z, 0.20, 0.40, 60, 110, (255, 255, 255))
    _fill_height(small, small_depth, intr, z, 0.55, 0.62, 70, 82, (255, 255, 255))
    small_depth[np.broadcast_to(_height_rows(intr, z, 0.55, 0.62)[:, None], small.shape[:2])] = np.nan
    missed = coarse_bands(small, small_depth, plane, intr, colors, 0.05, 1.2)
    assert missed["lower"]
    assert not missed["upper_white"]


def test_upper_seam_joins_when_the_face_continues_above_half_a_meter():
    # 只留 0.50 米以上，这一面不够高。连上 0.43 米起的同一块，才够 35 厘米的一面。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    short = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    short_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(short, short_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(short, short_depth, intr, z, 0.50, 0.62, 60, 100, (255, 255, 255))
    missed = coarse_bands(short, short_depth, plane, intr, colors, 0.05, 1.2)
    assert missed["lower"]
    assert not missed["upper_white"]

    joined = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    joined_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(joined, joined_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(joined, joined_depth, intr, z, 0.43, 0.62, 60, 100, (255, 255, 255))
    seen = coarse_bands(joined, joined_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert seen["upper_white"]


def test_upper_white_stays_above_the_lower_box():
    # 旁边的椅子、以及高过 0.78 米的白，都不算第 2 层。正上方那一块留下。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    both = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    both_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(both, both_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(both, both_depth, intr, z, 0.50, 0.70, 60, 100, (255, 255, 255))
    _fill_height(both, both_depth, intr, z, 0.55, 0.75, 0, 8, (255, 255, 255))
    seen = coarse_bands(both, both_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert seen["upper_white"]
    assert abs(float(seen["upper_point"][0])) < 0.15

    chair = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    chair_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(chair, chair_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(chair, chair_depth, intr, z, 0.55, 0.75, 0, 8, (255, 255, 255))
    only_chair = coarse_bands(chair, chair_depth, plane, intr, colors, 0.05, 1.2)
    assert only_chair["lower"]
    assert not only_chair["upper_white"]

    tall = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    tall_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(tall, tall_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(tall, tall_depth, intr, z, 0.82, 0.92, 60, 100, (255, 255, 255))
    too_high = coarse_bands(tall, tall_depth, plane, intr, colors, 0.05, 1.2)
    assert too_high["lower"]
    assert not too_high["upper_white"]


def test_cap_without_depth_cannot_confirm_at_the_lower_distance():
    # 红蓝没有深度时，不能按下层距离补造塔顶。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(color, depth, intr, z, 0.74, 0.90, 60, 100, (0, 0, 255))
    depth[np.broadcast_to(_height_rows(intr, z, 0.74, 0.90)[:, None], color.shape[:2])] = np.nan
    seen = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2)
    assert seen["lower"]
    assert not seen["cap"]

    low = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    low_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(low, low_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(low, low_depth, intr, z, 0.05, 0.18, 60, 100, (0, 0, 255))
    low_depth[np.broadcast_to(_height_rows(intr, z, 0.05, 0.18)[:, None], low.shape[:2])] = np.nan
    grounded = coarse_bands(low, low_depth, plane, intr, colors, 0.05, 1.2)
    assert grounded["lower"]
    assert not grounded["cap"]

    far = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    far_depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(far, far_depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(far, far_depth, intr, z, 0.74, 0.90, 145, 159, (0, 0, 255))
    far_depth[np.broadcast_to(_height_rows(intr, z, 0.74, 0.90)[:, None], far.shape[:2])] = np.nan
    missed = coarse_bands(far, far_depth, plane, intr, colors, 0.05, 1.2)
    assert missed["lower"]
    assert not missed["cap"]


def test_level_two_layer_keeps_the_lower_part_of_the_upper_box():
    # 层数已定、接近水平。0.43 到 0.48 米原先进不了上层白，放宽后留下。
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.43, 0.48, 60, 100, (255, 255, 255))
    anchor = np.array([0.0, 0.70, z])
    held = coarse_bands(
        color, depth, plane, intr, colors, 0.05, 1.2, anchor, relax_upper=True, pitch=-5.0
    )
    assert held["upper_white"]
    assert not held["lower"]
    plain = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2, anchor)
    assert not plain["upper_white"]


def test_looking_down_does_not_count_the_lower_box_as_upper():
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(color, depth, intr, z, 0.43, 0.48, 60, 100, (255, 255, 255))
    seen = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2, relax_upper=True, pitch=8.0)
    assert seen["lower"]
    assert not seen["upper_white"]


def test_white_under_the_cap_without_depth_still_counts_when_level():
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(color, depth, intr, z, 0.74, 0.90, 70, 90, (0, 0, 255))
    _fill_height(color, depth, intr, z, 0.45, 0.62, 70, 90, (255, 255, 255))
    white_rows = np.where(np.all(color == (255, 255, 255), axis=2))
    depth[white_rows] = np.nan
    anchor = np.array([0.0, 0.70, 0.35])
    seen = coarse_bands(
        color, depth, plane, intr, colors, 0.05, 1.2, anchor, relax_upper=True, pitch=-5.0
    )
    assert seen["upper_white"]
    assert seen["cap"]
    plain = coarse_bands(color, depth, plane, intr, colors, 0.05, 1.2, anchor)
    assert not plain["upper_white"]


def test_relax_upper_white_waits_until_two_layers_are_settled():
    assert not relax_upper_white(None, False, 0.0)
    assert not relax_upper_white("one", True, 0.0)
    assert not relax_upper_white("two", True, 20.0)
    assert relax_upper_white("two", True, 0.0)
    assert relax_upper_white("two_cap", True, -5.0)


def test_cap_rejects_a_thin_rim_and_washed_or_cyan_color():
    intr = intrinsics_for(160, 120)
    plane = _level_plane()
    colors = _cfg()["colors"]
    z = 0.60
    base = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _fill_height(base, depth, intr, z, 0.20, 0.40, 60, 100, (255, 255, 255))
    _fill_height(base, depth, intr, z, 0.52, 0.68, 60, 100, (255, 255, 255))

    rim = base.copy()
    rim_depth = depth.copy()
    _fill_height(rim, rim_depth, intr, z, 0.78, 0.82, 20, 140, (255, 0, 0))
    seen = coarse_bands(rim, rim_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]

    washed = base.copy()
    washed_depth = depth.copy()
    _fill_height(washed, washed_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(120, 90, 200))
    seen = coarse_bands(washed, washed_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]

    cyan = base.copy()
    cyan_depth = depth.copy()
    _fill_height(cyan, cyan_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(100, 200, 220))
    seen = coarse_bands(cyan, cyan_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]

    # 实拍的蓝面色相大约 111、饱和度很高，亮度只有大约 70。
    dark = base.copy()
    dark_depth = depth.copy()
    _fill_height(dark, dark_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(111, 250, 73))
    seen = coarse_bands(dark, dark_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["cap"]

    # 实拍的红面偏品红。色相大约 166，亮度大约 115，原先从 172、亮度 120 起收不进。
    dim_red = base.copy()
    dim_red_depth = depth.copy()
    _fill_height(dim_red, dim_red_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(166, 190, 115))
    seen = coarse_bands(dim_red, dim_red_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["cap"]

    # 发灰的红饱和度不够，仍不算塔顶。
    dull = base.copy()
    dull_depth = depth.copy()
    _fill_height(dull, dull_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(166, 80, 160))
    seen = coarse_bands(dull, dull_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]

    # 色相偏出实拍的红蓝，或饱和度不够，仍不算塔顶。
    loose = base.copy()
    loose_depth = depth.copy()
    _fill_height(loose, loose_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(125, 220, 200))
    seen = coarse_bands(loose, loose_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]
    # 蓝面受光后饱和度会下降，160 和录屏中观察到的 176 仍应识别。
    for saturation in (160, 176):
        pale = base.copy()
        pale_depth = depth.copy()
        _fill_height(pale, pale_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(111, saturation, 200))
        seen = coarse_bands(pale, pale_depth, plane, intr, colors, 0.05, 1.2)
        assert seen["cap"]
    # 更低饱和度的灰蓝仍拒绝。
    pale = base.copy()
    pale_depth = depth.copy()
    _fill_height(pale, pale_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(111, 120, 200))
    seen = coarse_bands(pale, pale_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]
    # 亮度 55 仍是这块偏暗的蓝。再暗到 20 才丢掉。
    dim_blue = base.copy()
    dim_blue_depth = depth.copy()
    _fill_height(dim_blue, dim_blue_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(111, 250, 55))
    seen = coarse_bands(dim_blue, dim_blue_depth, plane, intr, colors, 0.05, 1.2)
    assert seen["cap"]
    black_blue = base.copy()
    black_blue_depth = depth.copy()
    _fill_height(black_blue, black_blue_depth, intr, z, 0.74, 0.90, 60, 100, _bgr(111, 250, 20))
    seen = coarse_bands(black_blue, black_blue_depth, plane, intr, colors, 0.05, 1.2)
    assert not seen["cap"]


def test_tall_white_touching_the_top_is_a_second_layer():
    intr = intrinsics_for(320, 240)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    color[:200, 100:180] = (255, 255, 255)
    depth[:200, 100:180] = 0.8
    cue, distance = second_layer_cue(
        color, depth, None, intr, _cfg()["colors"], 0.35, 0.2, 3.0
    )
    assert cue
    assert distance is not None and abs(distance - 0.8) < 0.05


def test_close_stack_touching_the_top_still_asks_to_look_up():
    # 0.25 m 时一个边长已经比 240 像素的画面更高。白从最上沿延伸到中下部，仍要抬头。
    intr = intrinsics_for(320, 240)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    color[:180, 40:280] = (255, 255, 255)
    depth[:180, 40:280] = 0.25
    cue, distance = second_layer_cue(
        color, depth, None, intr, _cfg()["colors"], 0.35, 0.2, 3.0
    )
    assert cue
    assert distance is not None and abs(distance - 0.25) < 0.05


def test_small_patch_on_the_top_edge_is_not_a_second_layer():
    intr = intrinsics_for(320, 240)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    color[:24, 130:190] = (255, 255, 255)
    depth[:24, 130:190] = 0.8
    cue, _distance = second_layer_cue(
        color, depth, None, intr, _cfg()["colors"], 0.35, 0.2, 3.0
    )
    assert not cue


def test_lower_view_drops_white_above_one_edge():
    # 图像向下是重力。离地 0.50 m 的白要涂掉；朝上、没有深度的白也涂掉。朝下的白留下。
    intr = intrinsics_for(80, 60)
    color = np.full((intr.height, intr.width, 3), (255, 255, 255), dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    depth[40:55, 20:60] = 0.8
    plane = Plane(np.array([0.0, -1.0, 0.0]), -0.7)
    ys, xs = np.nonzero(np.ones(color.shape[:2], dtype=bool))
    z = depth[ys, xs]
    good = np.isfinite(z)
    heights = np.full(len(xs), np.nan)
    # 深度 0.8 m、像素在画面下半，按这个平面大约离地 0.5 m 以上才涂。这里直接标高度。
    heights[good] = 0.50
    view, cleared, hid = _blank_lower_view(color, depth, ys, xs, good, heights, plane, intr, 0.37)
    assert hid
    assert int(view[40:55, 20:60].sum()) < int(color[40:55, 20:60].sum())
    assert not np.isfinite(cleared[45, 40])
    # 画面上半没有深度，射线朝上，也要涂掉。画面下半朝下的、本来就没有深度的白还在。
    assert int(view[0, 0].sum()) < 255 * 3
    assert int(view[58, 10].sum()) == 255 * 3


def test_window_text_is_plain_chinese():
    record = _box(1, [0.12, 0.08, 0.90])
    lines = _lines(
        10.0,
        {"name": "indoor", "exposure": 150, "gain": 28},
        "地面测到了，相机离地 0.70 米",
        [record],
        [record],
        None,
        "只有一层，不用抬头",
    )
    text = "\n".join(lines)
    assert "每秒 10 帧" in text
    assert "室内" in text
    assert "向右 0.12 米" in text
    assert "向下 0.08 米" in text
    assert "向前 0.90 米" in text
    assert "转角 正 11 度" in text
    assert "长边 0.35 米" in text
    assert "m" not in text
    assert "yaw" not in text
    assert "fps" not in text


def test_close_rise_follows_horizontal_distance():
    # 视线深度里的竖直分量要去掉。水平 0.30 米才是用来抬头的距离。
    level = np.array([0.0, 1.0, 0.0])
    assert abs(horizontal_range(np.array([0.0, 0.40, 0.30]), level) - 0.30) < 1e-6
    assert cap_rise_degrees(0.80) == 0.0
    assert 7.0 < cap_rise_degrees(0.30) < 8.5
    assert abs(vote_rise_degrees(0.30) - cap_rise_degrees(0.30)) < 0.2
    assert 12.0 < cap_rise_degrees(0.25) < 13.5
    assert 7.5 < vote_rise_degrees(0.25) < 9.0
    assert vote_rise_degrees(0.25) < cap_rise_degrees(0.25) - 3.0
    assert 18.0 < cap_rise_degrees(0.20) < 20.0
    assert 3.5 < vote_rise_degrees(0.20) < 5.0


def _close(distance: float, **kwargs) -> dict:
    bands = _bands(
        lower=True,
        distance=distance,
        lower_point=np.array([0.0, 0.35, distance]),
        anchor=np.array([0.0, 0.70, distance]),
    )
    bands.update(kwargs)
    return bands


def _pixels(row: int) -> np.ndarray:
    return np.array([[80 + i, row] for i in range(12)], dtype=int)


def test_close_tower_waits_until_the_extra_rise():
    level = np.array([0.0, 1.0, 0.0])
    search = StackSearch()
    low = _close(0.30)
    search.update([], low, 0.30, 0.35, pitch=28.0, gravity=level)
    high = _close(0.30, cap=True, covers=True, upper_white=True)
    for _ in range(4):
        line = search.update([], high, 0.30, 0.35, pitch=0.0, gravity=level)
    assert search.kind is None
    assert search.asking_to_look_up()
    assert search.vote_rise > 7.0
    rise = vote_rise_degrees(0.30)
    for _ in range(3):
        line = search.update([], high, 0.30, 0.35, pitch=-rise, gravity=level)
    assert line == "两层白，有红蓝塔顶"


def test_top_edge_cap_adds_one_look_and_skips_those_frames():
    level = np.array([0.0, 1.0, 0.0])
    search = StackSearch()
    search.update([], _close(0.30), 0.30, 0.35, pitch=28.0, gravity=level)
    rise = vote_rise_degrees(0.30)
    clipped = _close(0.30, cap=True, covers=True, upper_white=True, cap_uv=_pixels(2))
    for _ in range(2):
        search.update([], clipped, 0.30, 0.35, pitch=-rise, gravity=level)
    assert search.kind is None
    assert search.nudges == 0
    search.update([], clipped, 0.30, 0.35, pitch=-rise, gravity=level)
    assert search.nudges == 1
    assert search.kind is None
    assert 4.0 < search.degrees < 6.0
    assert "黄线顶到上沿" in search.line()
    # 抬到新角度后，中间再顶一下上沿，已经看过的帧不算进这三帧，也不再加第二次。
    steady = _close(0.30, cap=True, covers=True, upper_white=True, cap_uv=_pixels(40))
    arrived = -(rise + 5.0)
    search.update([], steady, 0.30, 0.35, pitch=arrived, gravity=level)
    search.update([], steady, 0.30, 0.35, pitch=arrived, gravity=level)
    search.update([], clipped, 0.30, 0.35, pitch=arrived, gravity=level)
    assert search.kind is None
    assert search.quiet == 0
    assert search.nudges == 1
    for _ in range(3):
        line = search.update([], steady, 0.30, 0.35, pitch=arrived, gravity=level)
    assert line == "两层白，有红蓝塔顶"
    assert search.nudges == 1
