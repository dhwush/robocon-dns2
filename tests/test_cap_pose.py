"""塔顶体心从红蓝分界量，再换成世界坐标。不打开相机。"""

from __future__ import annotations

import cv2
import numpy as np

from cube.cap_pose import (
    CapFusion,
    CapMeasure,
    _accept,
    _color_masks,
    box_upper_color,
    camera_to_car_world,
    cap_top_color,
    _implied_distance,
    close_depth_unusable,
    measure_cap,
)
from ground.plane import Plane
from sim.scene import intrinsics_for


def _plane() -> Plane:
    # 相机水平，Y 向下。地面在相机下方 0.70 米。
    return Plane(np.array([0.0, -1.0, 0.0]), -0.70)


def _fill_front(color, depth, intr, x0, x1, z, y0, y1, bgr) -> None:
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - intr.cx) / intr.fx * z
    y = (vv - intr.cy) / intr.fy * z
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    color[inside] = bgr
    depth[inside] = z


def _fill_side(color, depth, intr, z0, z1, x, y0, y1, bgr) -> None:
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    xd = (uu - intr.cx) / intr.fx
    yd = (vv - intr.cy) / intr.fy
    facing = np.abs(xd) > 1e-4
    z = np.full(xd.shape, np.nan)
    z[facing] = x / xd[facing]
    y = yd * z
    inside = facing & (z >= z0) & (z <= z1) & (y >= y0) & (y <= y1)
    color[inside] = bgr
    depth[inside] = z[inside]


def _blank():
    intr = intrinsics_for(160, 120)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    return color, depth, intr


def test_front_face_steps_inward_half_an_edge_and_reaches_world():
    # 正面在前方 0.55 米，分界在离地 0.80 米。体心要再往里 0.10 米，不能停在表面上。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.02
    assert abs(float(pose.center[1]) - (-0.10)) < 0.02
    assert abs(float(pose.center[2]) - 0.65) < 0.02
    assert abs(float(pose.world[0]) - (-0.211)) < 0.02
    assert abs(float(pose.world[1]) - 0.70) < 0.02
    assert abs(float(pose.world[2]) - 0.80) < 0.02
    assert abs(float(pose.yaw)) < 0.15


def test_off_center_face_does_not_slide_along_the_viewing_ray():
    # 方块在右边。沿视线往里推会把向右的位置改掉，沿侧面法向则不会。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, 0.05, 0.25, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, 0.05, 0.25, 0.55, -0.10, -0.01, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0]) - 0.15) < 0.025
    assert abs(float(pose.center[2]) - 0.65) < 0.025


def test_seam_a_few_centimeters_off_eighty_is_kept_and_one_layer_is_not():
    # 0.74 米比 0.80 米低 6 厘米，在放宽后的 8 厘米里。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.13, -0.04, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.04, 0.05, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(_plane().signed_distance(pose.center.reshape(1, 3))[0]) - 0.74) < 0.03

    # 一块 0.35 米白盒上的交界约 0.45 米，不能当成两层白上的塔顶。
    low_color, low_depth, low_intr = _blank()
    _fill_front(low_color, low_depth, low_intr, -0.10, 0.10, 0.55, 0.16, 0.25, (0, 0, 255))
    _fill_front(low_color, low_depth, low_intr, -0.10, 0.10, 0.55, 0.25, 0.34, (255, 0, 0))
    pose, reason = measure_cap(low_color, low_depth, low_intr, _plane(), 0.05, 1.2)
    assert pose is None
    assert reason in ("seam", "height")


def test_seam_far_from_eighty_centimeters_is_dropped():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.30, -0.21, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.21, -0.12, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert pose is None
    assert reason == "height"


def test_wide_color_keeps_planar_depth():
    # 颜色比一面更宽，像两个侧面并排。按较短的一边估距，才和 0.60 米的平面对得上。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.14, 0.14, 0.60, -0.20, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.14, 0.14, 0.60, -0.10, 0.00, (255, 0, 0))
    mask = np.zeros(color.shape[:2], dtype=np.uint8)
    mask[np.any(color != 40, axis=2)] = 255
    implied = _implied_distance(mask, intr)
    assert implied is not None and abs(implied - 0.60) < 0.08
    assert not close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)


def test_fusion_keeps_color_until_depth_frames_are_enough():
    fusion = CapFusion(need=5, window=15)

    def sample(x: float, via: str) -> CapMeasure:
        return CapMeasure(
            center=np.zeros(3),
            world=np.array([x, 0.70, 0.80]),
            yaw=0.0,
            corners=np.zeros((4, 3)),
            edge=np.array([1.0, 0.0, 0.0]),
            via=via,
        )

    for _ in range(3):
        fusion.add(sample(-0.20, "close"), color="blue")
    for _ in range(3):
        fusion.add(sample(-0.50, "depth"), color="blue")
    mixed = fusion.result()
    assert mixed is not None
    assert abs(mixed.x - (-0.35)) < 0.02
    for _ in range(2):
        fusion.add(sample(-0.50, "depth"), color="blue")
    depth_only = fusion.result()
    assert depth_only is not None
    assert abs(depth_only.x - (-0.50)) < 0.01
    agreed = fusion.consensus()
    assert agreed is not None
    pose, tint = agreed
    assert tint == "blue"
    assert abs(pose.x - (-0.50)) < 0.01


def test_two_faces_meet_at_the_vertical_edge():
    # 正面和右侧面在 x=0.10、z=0.55 相交。体心从这条棱沿两个面各进 10 厘米。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    _fill_side(color, depth, intr, 0.55, 0.65, 0.10, -0.19, -0.10, (0, 0, 255))
    _fill_side(color, depth, intr, 0.55, 0.65, 0.10, -0.10, -0.01, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0]) - 0.00) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_world_shift_adds_the_car_motion_from_radar():
    moved = camera_to_car_world(np.array([0.0, -0.10, 0.65]), np.array([1.0, 2.0, 0.0]))
    assert abs(float(moved[0]) - 0.789) < 1e-6
    assert abs(float(moved[1]) - 2.70) < 1e-6
    assert abs(float(moved[2]) - 0.769) < 1e-6


def test_pitch_is_removed_before_the_world_position_and_yaw():
    # 抬头 19 度，塔顶在水平前方 0.20 米、高出光心 0.10 米。棱在水平面上偏 30 度。
    pitch = np.deg2rad(19.0)
    up = np.array([0.0, -np.cos(pitch), np.sin(pitch)])
    right = np.array([1.0, 0.0, 0.0])
    forward = np.cross(up, right)
    body = 0.20 * forward + 0.10 * up
    yaw = np.deg2rad(30.0)
    edge = np.cos(yaw) * right + np.sin(yaw) * forward
    pose = _accept(body, edge, Plane(up, -0.70), None)
    assert not isinstance(pose, str)
    assert abs(float(pose.world[0]) - (-0.211)) < 1e-6
    assert abs(float(pose.world[1]) - 0.25) < 1e-6
    assert abs(float(pose.world[2]) - 0.80) < 1e-6
    assert abs(float(np.degrees(pose.yaw)) - 30.0) < 0.05


def test_fusion_waits_for_several_frames_and_drops_a_jump():
    fusion = CapFusion(need=5, window=15)
    assert fusion.result() is None
    for index in range(5):
        world = np.array([-0.20, 0.70, 0.80]) + np.array([0.002 * index, 0.0, 0.0])
        fusion.add(
            CapMeasure(
                center=np.zeros(3),
                world=world,
                yaw=0.02 * index,
                corners=np.zeros((4, 3)),
                edge=np.array([1.0, 0.0, 0.0]),
            )
        )
    fused = fusion.result()
    assert fused is not None
    assert abs(fused.x - (-0.196)) < 0.01
    fusion.add(
        CapMeasure(
            center=np.zeros(3),
            world=np.array([2.0, 0.70, 0.80]),
            yaw=0.0,
            corners=np.zeros((4, 3)),
            edge=np.array([1.0, 0.0, 0.0]),
        )
    )
    again = fusion.result()
    assert again is not None
    assert abs(again.x - (-0.196)) < 0.02
    assert fusion.consensus() is None


def _sample(color: str, x: float = -0.20) -> CapMeasure:
    return CapMeasure(
        center=np.zeros(3),
        world=np.array([x, 0.70, 0.80]),
        yaw=0.1,
        corners=np.zeros((4, 3)),
        edge=np.array([1.0, 0.0, 0.0]),
    )


def test_top_color_follows_the_higher_band():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "red"

    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (255, 0, 0))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (0, 0, 255))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "blue"


def test_top_color_stays_unknown_when_the_two_bands_are_level():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.00, 0.55, -0.19, -0.01, (0, 0, 255))
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.19, -0.01, (255, 0, 0))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) is None


def test_close_heights_follow_the_higher_image_row():
    # 离地只差 1.5 厘米，不到 2 厘米。画面里更靠上的那一色仍是朝上的。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.12, -0.02, 0.30, -0.16, -0.06, (0, 0, 255))
    _fill_front(color, depth, intr, 0.02, 0.12, 0.30, -0.145, -0.045, (255, 0, 0))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "red"


def test_one_color_touching_the_top_is_the_lower_band():
    # 上半截出画，只剩一色顶着上沿。看见的是下半截，朝上用另一色。没贴上沿就不定。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.32, -0.12, (255, 0, 0))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "red"

    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.32, -0.12, (0, 0, 255))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "blue"

    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.15, 0.00, (255, 0, 0))
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) is None


def _drop_depth(depth, intr, keep) -> None:
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    known = np.isfinite(depth) & (depth > 0)
    z = np.where(known, depth, np.nan)
    x = (uu - intr.cx) / intr.fx * z
    y = (vv - intr.cy) / intr.fy * z
    depth[known & keep(x, y)] = np.nan


def test_partial_depth_uses_the_color_seam_and_not_the_right_half():
    # 深度只剩右半边。体心仍在整条颜色分界的中点再往里 10 厘米，不能偏到右边。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    _drop_depth(depth, intr, lambda x, _y: x < 0.0)
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.02
    assert abs(float(pose.center[2]) - 0.65) < 0.025
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "red"


def test_top_color_without_depth_on_the_upper_band():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    _drop_depth(depth, intr, lambda _x, y: y < -0.10)
    assert cap_top_color(color, depth, intr, _plane(), 0.05, 1.2) == "red"
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[2]) - 0.65) < 0.025


def test_short_color_seam_is_still_dropped():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.10, -0.01, (255, 0, 0))
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert pose is None
    assert reason == "short"


def test_two_faces_fill_the_seam_from_the_lower_color():
    # 分界上没有深度。下层两个侧面还在，体心仍从竖直棱沿两个面各进 10 厘米。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    _fill_side(color, depth, intr, 0.55, 0.65, 0.10, -0.19, -0.10, (0, 0, 255))
    _fill_side(color, depth, intr, 0.55, 0.65, 0.10, -0.10, -0.01, (255, 0, 0))
    red, blue = _color_masks(color)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    touch = cv2.bitwise_or(
        cv2.bitwise_and(cv2.dilate(red, kernel), blue),
        cv2.bitwise_and(cv2.dilate(blue, kernel), red),
    )
    depth[touch > 0] = np.nan
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0]) - 0.00) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_depth_plane_supplies_the_side_the_color_missed():
    # 正面仍是红蓝。右边侧面在深度里是一整块，颜色却不是红蓝。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, 0.00, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    _fill_side(color, depth, intr, 0.55, 0.65, 0.10, -0.19, -0.01, (40, 40, 40))
    anchor = np.array([0.05, 0.0, 0.55])
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2, anchor)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0]) - 0.00) < 0.03
    assert abs(float(pose.center[1]) - (-0.10)) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_short_front_without_a_second_plane_stays_short():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.10, -0.01, (255, 0, 0))
    anchor = np.array([0.0, 0.0, 0.55])
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2, anchor)
    assert pose is None
    assert reason == "short"


def test_a_wide_wall_does_not_become_the_side_face():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.40, 0.40, 0.90, -0.19, -0.01, (40, 40, 40))
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.05, 0.05, 0.55, -0.10, -0.01, (255, 0, 0))
    anchor = np.array([0.0, 0.0, 0.55])
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2, anchor)
    assert pose is None
    assert reason == "short"


def test_close_face_without_depth_uses_the_color_seam():
    # 0.25 米近处整面没有深度。分界交到离地 0.80 米，体心再往里 10 厘米。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.10, -0.01, (255, 0, 0))
    depth[:] = np.nan
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.025
    assert abs(float(pose.center[2]) - 0.35) < 0.025


def test_close_bulge_is_not_used_as_a_plane():
    # 深度中间近、四周远，对不成平面。体心仍按颜色分界，不跟着鼓包走。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.10, -0.01, (255, 0, 0))
    known = np.isfinite(depth)
    uu, vv = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    xd = (uu - intr.cx) / intr.fx
    yd = (vv - intr.cy) / intr.fy
    # 脸的中间近，往四边变远。投到水平面上不是一条直线。
    bulge = 0.16 + 4.0 * (xd ** 2 + (yd + 0.40) ** 2)
    depth[known] = bulge[known]
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[2]) - 0.35) < 0.025
    assert abs(float(pose.center[0])) < 0.025


def test_close_planar_depth_stays_on_the_measured_face():
    # 0.25 米的正面深度是平的，就用这块深度。体心在这面前 10 厘米，不交到 0.80 米那一层。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.25, -0.10, -0.01, (255, 0, 0))
    assert not close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert pose.via == "depth"
    assert abs(float(pose.center[2]) - 0.35) < 0.025


def test_close_short_seam_is_still_dropped():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.05, 0.05, 0.25, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.05, 0.05, 0.25, -0.10, -0.01, (255, 0, 0))
    depth[:] = np.nan
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert pose is None
    assert reason == "short"


def test_face_without_depth_uses_the_color_seam():
    # 整面没有深度，覆盖不够，不论远近都改用颜色分界。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    depth[:] = np.nan
    assert close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.025
    assert abs(float(pose.center[2]) - 0.65) < 0.025


def test_depth_farther_than_the_face_uses_the_color_seam():
    # 方块看起来在 0.45 米，深度却整面读成 0.90 米。体心按颜色交到 0.80 米。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.45, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.45, -0.10, -0.01, (255, 0, 0))
    depth[np.isfinite(depth)] = 0.90
    assert close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.025
    assert abs(float(pose.center[2]) - 0.55) < 0.025


def test_depth_ramp_on_one_face_uses_the_color_seam():
    # 同一面前后差超过大约 25 厘米。不跟着斜坡走，仍交到离地 0.80 米。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    known = np.isfinite(depth)
    cols = np.nonzero(known)[1].astype(float)
    span = max(1.0, float(cols.max() - cols.min()))
    depth[known] = 0.40 + 0.50 * (cols - float(cols.min())) / span
    assert close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.025
    assert abs(float(pose.center[2]) - 0.65) < 0.025


def test_side_depth_falls_back_to_the_color_seam():
    # 深度还在，但每个点前后跳 4 厘米，侧面贴不成。颜色分界仍交到 0.80 米，这一眼算量准。
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    known = np.isfinite(depth)
    uu, vv = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
    depth[known] = 0.55 + np.where((uu + vv) % 2 == 0, 0.04, -0.04)[known]
    assert not close_depth_unusable(color, depth, intr, _plane(), 0.05, 1.2)
    anchor = np.array([0.0, -0.10, 0.55])
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2, anchor)
    assert reason is None and pose is not None
    assert pose.via == "side"
    assert abs(float(pose.center[0])) < 0.025
    assert abs(float(pose.center[1]) - (-0.10)) < 0.025
    assert abs(float(pose.center[2]) - 0.65) < 0.025


def test_close_color_seam_is_marked_separately_from_the_side_fallback():
    color, depth, intr = _blank()
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.19, -0.10, (0, 0, 255))
    _fill_front(color, depth, intr, -0.10, 0.10, 0.55, -0.10, -0.01, (255, 0, 0))
    depth[:] = np.nan
    pose, reason = measure_cap(color, depth, intr, _plane(), 0.05, 1.2)
    assert reason is None and pose is not None
    assert pose.via == "close"


def test_consensus_needs_the_same_color_on_the_latest_frames():
    fusion = CapFusion(need=5, window=15)
    for _ in range(4):
        fusion.add(_sample("red"), "red")
    assert fusion.consensus() is None
    fusion.add(_sample("red"), "red")
    pose, color = fusion.consensus()
    assert color == "red"
    assert abs(pose.y - 0.70) < 1e-6
    fusion.add(_sample("blue"), "blue")
    assert fusion.consensus() is None
    for _ in range(5):
        fusion.add(_sample("red"), "red")
    pose, color = fusion.consensus()
    assert color == "red"


def test_box_upper_half_uses_the_color_above_the_middle():
    intr = intrinsics_for(160, 120)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    color[: intr.height // 2] = (0, 0, 255)
    color[intr.height // 2 :] = (255, 0, 0)
    wire = np.array(
        [[x, y, z] for z in (0.75, 0.85) for y in (-0.08, 0.08) for x in (-0.08, 0.08)],
        dtype=float,
    )
    assert box_upper_color(color, wire, intr) == "red"
    color[: intr.height // 2] = (255, 0, 0)
    color[intr.height // 2 :] = (0, 0, 255)
    assert box_upper_color(color, wire, intr) == "blue"
    color[:] = 40
    assert box_upper_color(color, wire, intr) is None


def test_box_upper_half_outside_the_frame_uses_the_other_color():
    # 绿框上半截整段在画面上方。画面里这一截是下色，朝上用另一色。
    intr = intrinsics_for(160, 120)
    color = np.full((intr.height, intr.width, 3), (255, 0, 0), dtype=np.uint8)
    wire = np.array(
        [[x, y, 0.50] for y in (-0.50, -0.20) for x in (-0.08, 0.08)],
        dtype=float,
    )
    assert box_upper_color(color, wire, intr) == "red"
    color[:] = (0, 0, 255)
    assert box_upper_color(color, wire, intr) == "blue"
