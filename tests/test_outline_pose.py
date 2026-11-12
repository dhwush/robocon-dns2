"""黄线的下沿是方块最底端。不进 5 帧，和上一眼差远了这一眼不显示。"""

from __future__ import annotations

import cv2
import numpy as np

from cube.cap_pose import (
    CapFusion,
    OutlineLive,
    align_bottom_to_yellow,
    confirm_green_face,
    measure_outline,
    outline_center,
)
from cube.stack_look import StackSearch
from geom.transform import project_points
from ground.plane import Plane
from sim.scene import intrinsics_for
from view_tower import describe, live_cap


def _plane() -> Plane:
    return Plane(np.array([0.0, -1.0, 0.0]), -0.70)


def _edge(start, stop, count=24) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    t = np.linspace(0.0, 1.0, count)
    return (1.0 - t)[:, None] * start + t[:, None] * stop


def test_bottom_and_vertical_edge_step_inward_to_the_body_center():
    # 下沿 20 厘米，左竖边 10 厘米。体心在正面中点再往里 10 厘米。
    chain = np.vstack(
        [
            _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, -0.10, 0.55))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.02
    assert abs(float(pose.center[1]) - (-0.10)) < 0.02
    assert abs(float(pose.center[2]) - 0.65) < 0.02


def test_two_bottom_edges_meet_at_the_corner():
    # 正面和右侧面各一条 20 厘米下沿，在 x=0.10、z=0.55 相交。
    chain = np.vstack(
        [
            _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, 0.0, 0.75))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.03
    assert abs(float(pose.center[1]) - (-0.10)) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_bottom_edge_alone_is_the_cube_bottom():
    # 只有下沿也算。体心在它上方 10 厘米，再往里 10 厘米。
    alone = _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55))
    pose = outline_center([alone], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.02
    assert abs(float(pose.center[1]) - (-0.10)) < 0.02
    assert abs(float(pose.center[2]) - 0.65) < 0.02


def test_short_bottom_extends_toward_the_open_end():
    # 只看见右半段下沿，右端接竖边。朝左边没有深度的那头补到 20 厘米。
    chain = np.vstack(
        [
            _edge((0.0, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, -0.10, 0.55))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.03
    assert abs(float(pose.center[1]) - (-0.10)) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_high_edge_does_not_become_the_bottom_when_the_lower_edge_is_there():
    # 上沿离地 0.84 米，旧逻辑会把它当成 0.80 的分界。下沿在，体心仍从下沿往上抬。
    chain = np.vstack(
        [
            _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, -0.14, 0.55))[1:],
            _edge((0.10, -0.14, 0.55), (-0.10, -0.14, 0.55))[1:],
            _edge((-0.10, -0.14, 0.55), (-0.10, 0.0, 0.55))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.03
    assert abs(float(pose.center[1]) - (-0.10)) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_jagged_low_points_do_not_pull_the_bottom_down():
    # 弯下去的点不再把底面拽低。直边仍在离地 0.78 米。
    high = _edge((-0.10, -0.08, 0.55), (0.10, -0.08, 0.55))
    xs = np.linspace(0.10, -0.10, 25)
    zig = np.array([[x, 0.0, 0.55 + (0.025 if index % 2 else 0.0)] for index, x in enumerate(xs)])
    link_r = _edge((0.10, -0.08, 0.55), (0.10, 0.0, 0.55))[1:]
    link_l = _edge((-0.10, 0.0, 0.55), (-0.10, -0.08, 0.55))[1:]
    chain = np.vstack([high, link_r, zig[1:], link_l])
    pose = outline_center([chain], _plane())
    assert pose is not None
    assert abs(float(pose.center[1]) - (-0.18)) < 0.03


def _face_uv(intr, z: float, y0: float, y1: float) -> np.ndarray:
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - intr.cx) / intr.fx * z
    y = (vv - intr.cy) / intr.fy * z
    inside = (x >= -0.10) & (x <= 0.10) & (y >= y0) & (y <= y1)
    return np.column_stack(np.nonzero(inside)[::-1]).astype(int)


def test_yellow_bottom_in_the_image_sets_the_face_height():
    # 直边偏高。黄线底端在画面上对应离地 0.70 米，底面降到那里。不读这些像素的深度。
    intr = intrinsics_for(160, 120)
    high = _edge((-0.10, -0.08, 0.55), (0.10, -0.08, 0.55))
    pose = outline_center([high], _plane())
    uv = _face_uv(intr, 0.55, -0.10, 0.0)
    aligned = align_bottom_to_yellow(pose, uv, (intr.height, intr.width), intr, _plane())
    assert abs(float(aligned.center[1]) - (-0.10)) < 0.03
    assert abs(float(aligned.center[2]) - 0.65) < 0.03


def test_low_green_box_rises_to_the_yellow_bottom():
    # 绿框底面已经低于黄线底端，就抬到和它对齐。
    intr = intrinsics_for(160, 120)
    low = _edge((-0.10, 0.06, 0.55), (0.10, 0.06, 0.55))
    pose = outline_center([low], _plane())
    uv = _face_uv(intr, 0.55, -0.10, 0.0)
    aligned = align_bottom_to_yellow(pose, uv, (intr.height, intr.width), intr, _plane())
    assert abs(float(aligned.center[1]) - (-0.10)) < 0.03


def test_closed_outline_that_starts_mid_edge_still_finds_the_center():
    # 闭合的下半截，起点在下沿中间。不能因为起点把 20 厘米切成两截就丢掉。
    corners = [
        np.array([0.0, 0.0, 0.55]),
        np.array([0.10, 0.0, 0.55]),
        np.array([0.10, -0.10, 0.55]),
        np.array([-0.10, -0.10, 0.55]),
        np.array([-0.10, 0.0, 0.55]),
    ]
    parts = [_edge(corners[i], corners[i + 1], 20)[:-1] for i in range(4)]
    parts.append(_edge(corners[4], corners[0], 20)[:-1])
    pose = outline_center([np.vstack(parts)], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def _front_face(intr, z: float = 0.55):
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - intr.cx) / intr.fx * z
    y = (vv - intr.cy) / intr.fy * z
    inside = (x >= -0.10) & (x <= 0.10) & (y >= -0.10) & (y <= 0.0)
    depth[inside] = z
    uv = np.column_stack(np.nonzero(inside)[::-1]).astype(int)
    return depth, inside, uv


def _assert_front_center(pose) -> None:
    assert pose is not None
    assert abs(float(pose.center[0])) < 0.03
    assert abs(float(pose.center[1]) - (-0.10)) < 0.03
    assert abs(float(pose.center[2]) - 0.65) < 0.03


def test_outline_from_depth_pixels_on_the_yellow_contour():
    intr = intrinsics_for(160, 120)
    depth, _inside, uv = _front_face(intr)
    pose = measure_outline(uv, depth, intr, _plane(), 0.05, 1.2)
    _assert_front_center(pose)


def test_jumping_depth_on_the_color_edge_does_not_move_the_center():
    # 交界上整圈深度都远 4 厘米。框应跟着里面的深度，不跟着这一圈。
    intr = intrinsics_for(160, 120)
    depth, inside, uv = _front_face(intr)
    eroded = inside.copy()
    eroded[1:-1, 1:-1] = (
        inside[1:-1, 1:-1]
        & inside[:-2, 1:-1]
        & inside[2:, 1:-1]
        & inside[1:-1, :-2]
        & inside[1:-1, 2:]
    )
    depth[inside & ~eroded] = 0.59
    pose = measure_outline(uv, depth, intr, _plane(), 0.05, 1.2)
    _assert_front_center(pose)


def test_sparse_depth_spikes_along_the_outline_are_pulled_back():
    # 里面没有深度，黄线上每四个点有一个远 4 厘米。这个点要被拉回来，不能把边拆掉。
    intr = intrinsics_for(160, 120)
    depth, inside, uv = _front_face(intr)
    depth[inside] = np.nan
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[inside] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for index, pixel in enumerate(contours[0][:, 0, :]):
        u, v = int(pixel[0]), int(pixel[1])
        depth[v, u] = 0.59 if index % 4 == 0 else 0.55
    pose = measure_outline(uv, depth, intr, _plane(), 0.05, 1.2)
    _assert_front_center(pose)


def test_live_outline_hides_the_first_frame_and_a_jump():
    chain = np.vstack(
        [
            _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, -0.10, 0.55))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    assert pose is not None
    live = OutlineLive()
    assert live.offer(pose) is None
    assert live.offer(pose) is pose
    jumped = outline_center(
        [
            np.vstack(
                [
                    _edge((-0.10, 0.0, 0.70), (0.10, 0.0, 0.70)),
                    _edge((0.10, 0.0, 0.70), (0.10, -0.10, 0.70))[1:],
                ]
            )
        ],
        _plane(),
    )
    assert jumped is not None
    assert float(np.linalg.norm(jumped.center - pose.center)) > 0.03
    assert live.offer(jumped) is None
    assert live.offer(jumped) is jumped
    assert live.offer(None) is None
    assert live.offer(pose) is None


def test_outline_does_not_enter_fusion_or_replace_a_measured_seam():
    chain = np.vstack(
        [
            _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.55)),
            _edge((0.10, 0.0, 0.55), (0.10, -0.10, 0.55))[1:],
        ]
    )
    pose = outline_center([chain], _plane())
    caps = CapFusion()
    live = OutlineLive()
    assert live_cap(None, None, pose, live) is None
    shown = live_cap(None, None, pose, live)
    assert shown is pose
    assert caps.count() == 0
    assert live_cap(pose, None, pose, live) is None
    assert live_cap(None, None, pose, live) is None


def _splat(depth: np.ndarray, intr, points: np.ndarray) -> None:
    pix = project_points(points, intr)
    ui = np.rint(pix[:, 0]).astype(int)
    vi = np.rint(pix[:, 1]).astype(int)
    z = points[:, 2]
    inside = (z > 0.05) & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
    for u, v, dist in zip(ui[inside], vi[inside], z[inside]):
        old = depth[v, u]
        if not np.isfinite(old) or dist < old:
            depth[v, u] = dist


def _grid(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray, steps: int = 36) -> np.ndarray:
    uu, vv = np.meshgrid(np.linspace(0.0, 1.0, steps), np.linspace(0.0, 1.0, steps))
    return origin + uu.ravel()[:, None] * axis_u + vv.ravel()[:, None] * axis_v


def test_shifted_box_moves_onto_the_front_plane():
    # 下沿右端被拉近，绿框斜着伸出正面。正面内部对不上，就改到这块竖直面里面。
    intr = intrinsics_for(160, 120)
    depth, _inside, _uv = _front_face(intr)
    tilted = _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.45))
    pose = outline_center([tilted], _plane())
    assert pose is not None
    assert abs(float(pose.center[0])) > 0.03 or abs(float(pose.center[2]) - 0.65) > 0.03
    fixed = confirm_green_face(pose, depth, intr, _plane(), 0.05, 1.2)
    _assert_front_center(fixed)


def test_side_face_is_kept_when_half_the_depth_disagrees():
    # 正面只有 10 厘米，右边连着一整块侧面。斜边把两面捏成一条时，侧面留下，体心从棱往两面里收。
    intr = intrinsics_for(160, 120)
    yaw = np.deg2rad(40.0)
    front_n = np.array([np.sin(yaw), 0.0, -np.cos(yaw)])
    side_n = np.array([-np.cos(yaw), 0.0, -np.sin(yaw)])
    across_front = -side_n
    across_side = -front_n
    corner = np.array([0.0, 0.0, 0.62])
    up = np.array([0.0, -0.12, 0.0])
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    _splat(depth, intr, _grid(corner, across_front * 0.10, up))
    _splat(depth, intr, _grid(corner, across_side * 0.20, up))
    front_far = corner + across_front * 0.10
    side_far = corner + across_side * 0.20
    pose = outline_center([_edge(front_far, side_far)], _plane())
    fixed = confirm_green_face(pose, depth, intr, _plane(), 0.05, 1.2)
    assert fixed is not None
    corner_center = corner + across_front * 0.10 + across_side * 0.10
    front_only = corner + across_front * 0.05 + across_side * 0.10
    assert float(np.linalg.norm(fixed.center[[0, 2]] - corner_center[[0, 2]])) < 0.04
    assert float(np.linalg.norm(fixed.center[[0, 2]] - front_only[[0, 2]])) > 0.04


def test_lower_near_surface_is_not_a_side_face():
    # 正面下面还有一块更近的表面。它不是侧面，框仍落在正面上。
    intr = intrinsics_for(160, 120)
    depth, _inside, _uv = _front_face(intr)
    us = np.arange(intr.width)
    vs = np.arange(intr.height)
    uu, vv = np.meshgrid(us, vs)
    z = 0.40
    x = (uu - intr.cx) / intr.fx * z
    y = (vv - intr.cy) / intr.fy * z
    ledge = (x >= -0.10) & (x <= 0.10) & (y >= 0.05) & (y <= 0.16)
    empty = ~np.isfinite(depth)
    depth[ledge & empty] = z
    tilted = _edge((-0.10, 0.0, 0.55), (0.10, 0.0, 0.45))
    pose = outline_center([tilted], _plane())
    fixed = confirm_green_face(pose, depth, intr, _plane(), 0.05, 1.2)
    _assert_front_center(fixed)


def test_outline_line_names_the_top_color_and_is_not_a_finished_result():
    lines = describe(
        StackSearch(),
        {"cap": True},
        0.0,
        cap_world=np.array([-0.13, 0.70, 0.80]),
        cap_yaw=0.0,
        cap_outline=True,
        cap_outline_color="red",
    )
    text = "\n".join(lines)
    assert "按黄线推算 相对小车右前" in text
    assert "相对光心" in text
    assert "朝上是红色" in text
    assert "识别完成" not in text
    assert "绿框是这一眼按黄线推算" in text
