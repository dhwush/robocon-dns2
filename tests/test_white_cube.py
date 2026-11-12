"""地面 0.35 m 白盒：体心、yaw，以及测试窗口上的棱线和地面轮廓。"""

from __future__ import annotations

import cv2
import numpy as np

from cube.detect import (
    CubePose,
    _coarse_ring,
    _color_layers,
    _complete_short,
    _ground_under_cube,
    _keep_far_span,
    _ring_px,
    _pull_lower_onto_white,
    _stack_lower_from_face,
    detect_white_layers,
)
from geom.transform import fold_yaw_quarter, project_points
from ground.plane import GroundFix, Plane, ground_prior, ground_with_prior
from cube.stack_look import StackSearch
from main import (
    GroundMemory,
    acquire_plane,
    clean_frame,
    estimate_frame,
    estimate_layers,
    fit_visible_ground,
    ground_cloud,
    ground_for_layers,
    load_config,
    lower_top_on_ground,
)
from sim.scene import _face, _paint, intrinsics_for, render_cube
from view_cube_center import cube_wire_corners, draw_cube_edges, draw_ground, draw_labels, overlay, reject_text


def _cfg() -> dict:
    return load_config("config/default.yaml")


def _scene(center: np.ndarray, yaw: float, size: float, top_bgr: tuple[int, int, int]):
    intr = intrinsics_for(640, 360)
    color, depth = render_cube(
        intr,
        center,
        yaw,
        size=size,
        ground_z=float(center[2]),
        top_bgr=top_bgr,
        noise_m=0.0,
    )
    gravity = np.array([0.0, 0.0, 1.0])
    frame = clean_frame(color, depth, gravity, _cfg())
    return intr, frame


def _measure(center, yaw, size, top_bgr):
    intr, frame = _scene(center, yaw, size, top_bgr)
    record = estimate_frame(frame, intr, _cfg(), camera_height=0.7)
    return intr, frame, record


def test_white_cube_center_and_yaw():
    center = np.array([0.12, -0.08, 0.70])
    yaw = 0.20
    _intr, _frame, record = _measure(center, yaw, 0.35, (255, 255, 255))
    assert record is not None
    assert abs(record["x"] - center[0]) < 0.03
    assert abs(record["y"] - center[1]) < 0.03
    assert abs(record["z"] - (0.70 - 0.175)) < 0.03
    assert abs(fold_yaw_quarter(record["yaw"] - yaw)) < np.deg2rad(8.0)
    assert abs(record["long_side"] - 0.35) < 0.06
    assert abs(record["short_side"] - 0.35) < 0.06


def test_red_cube_and_wrong_size_are_not_the_white_box():
    center = np.array([0.10, 0.05, 0.70])
    _intr, _frame, red = _measure(center, 0.1, 0.35, (40, 40, 255))
    _intr, _frame, small = _measure(center, 0.1, 0.15, (255, 255, 255))
    assert red is None
    assert small is None


def test_washed_red_is_not_counted_as_white():
    image = np.full((80, 120, 3), 40, dtype=np.uint8)
    image[20:50, 20:40] = (0, 0, 255)
    image[20:50, 40:48] = (250, 250, 250)
    image[20:50, 70:90] = (255, 255, 255)
    strict, _loose, _front = _color_layers(image, _cfg()["colors"])
    assert strict[35, 44] == 0
    assert strict[35, 80] > 0


def test_overlay_shows_ground_cube_and_readout():
    center = np.array([0.12, -0.08, 0.70])
    intr, frame, record = _measure(center, 0.2, 0.35, (255, 255, 255))
    assert record is not None
    vis, drawn = overlay(
        frame.color,
        frame.depth,
        intr,
        record["plane"],
        record,
        0.35,
        [
            "30 fps  daylight  exp 50  gain 16",
            "地面 测到  相机离地 0.700 m",
            "认出白盒",
            f"x {record['x']:+.3f}  y {record['y']:+.3f}  z {record['z']:+.3f} m",
            f"yaw {np.degrees(record['yaw']):+.1f} deg",
            f"long {record['long_side']:.3f}  short {record['short_side']:.3f} m",
        ],
    )
    assert drawn == 12
    assert vis.shape == frame.color.shape
    cyan = (vis[:, :, 0] > 180) & (vis[:, :, 1] > 180) & (vis[:, :, 2] < 90)
    orange = (vis[:, :, 2] > 180) & (vis[:, :, 1] > 80) & (vis[:, :, 0] < 90)
    assert int(cyan.sum()) > 80
    assert int(orange.sum()) > 200
    # 地面线是外轮廓，画面内部的地面仍是原来的灰色，不是涂满的橙块。
    assert int(orange[40, 40]) == 0
    text = vis[:36, :220]
    white = (text[:, :, 0] > 200) & (text[:, :, 1] > 200) & (text[:, :, 2] > 200)
    assert int(white.sum()) > 20


def test_forward_box_draws_separated_top_and_bottom():
    intr = intrinsics_for(640, 360)
    half = 0.175
    center = np.array([0.04, 0.10, 1.10])
    bottom = np.array(
        [
            center + np.array([-half, half, -half]),
            center + np.array([half, half, -half]),
            center + np.array([half, half, half]),
            center + np.array([-half, half, half]),
        ]
    )
    normal = np.array([0.0, -1.0, 0.0])
    image = np.full((intr.height, intr.width, 3), 30, dtype=np.uint8)
    wire = cube_wire_corners(bottom, normal, 0.35)
    drawn = draw_cube_edges(image, wire, intr)
    assert drawn == 12
    cyan = (image[:, :, 0] > 180) & (image[:, :, 1] > 180) & (image[:, :, 2] < 90)
    rows = np.nonzero(cyan)[0]
    assert rows.max() - rows.min() > 40


def test_ground_strip_is_a_solid_outline():
    intr = intrinsics_for(320, 180)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    depth[70:110, 40:280] = 1.2
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-1.2)
    image = np.full((intr.height, intr.width, 3), 30, dtype=np.uint8)
    draw_ground(image, depth, intr, plane)
    orange = (image[:, :, 2] > 180) & (image[:, :, 1] > 80) & (image[:, :, 0] < 90)
    assert int(orange.sum()) > 80
    # 带子中部不被涂满，说明画的是轮廓。
    assert int(orange[90, 160]) == 0


def test_split_floor_draws_the_largest_piece():
    intr = intrinsics_for(320, 180)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    depth[40:140, 10:50] = 1.2
    depth[40:140, 200:300] = 1.2
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-1.2)
    image = np.full((intr.height, intr.width, 3), 30, dtype=np.uint8)
    draw_ground(image, depth, intr, plane)
    orange = (image[:, :, 2] > 180) & (image[:, :, 1] > 80) & (image[:, :, 0] < 90)
    # 只描最大的一块地面。小的一截和中间空档不连成一条跨过去的线。
    assert int(orange[:, 190:].sum()) > 20
    assert int(orange[:, :60].sum()) == 0
    assert int(orange[90, 120]) == 0


def test_white_cube_is_left_out_of_ground():
    intr = intrinsics_for(320, 180)
    depth = np.full((intr.height, intr.width), 0.70, dtype=np.float32)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    color[60:120, 120:200] = (255, 255, 255)
    depth[60:120, 120:200] = 0.60
    cfg = _cfg()
    cfg["ground_stride"] = 4
    points, normals = ground_cloud(depth, intr, cfg, color)
    assert len(points) > 50
    assert int(np.sum(points[:, 2] < 0.65)) == 0
    gravity = np.array([0.0, 0.0, 1.0])
    fix = acquire_plane(points, normals, cfg, gravity, None, 0.70)
    assert fix is not None
    assert abs(-float(fix.plane.offset) - 0.70) < 0.03


def test_visible_ground_refits_when_the_old_height_misses_the_floor():
    intr = intrinsics_for(160, 120)
    depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
    depth[40:110, 10:150] = 0.90
    cfg = _cfg()
    cfg["ground_stride"] = 4
    gravity = np.array([0.0, 0.0, 1.0])
    stale = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.66)
    fix = fit_visible_ground(depth, intr, cfg, gravity, stale)
    assert fix is not None
    assert fix.source == "ransac"
    assert abs(-float(fix.plane.offset) - 0.90) < 0.03


def test_one_layer_keeps_the_installed_height_and_two_layers_use_the_lowest_floor():
    # 地面在 0.90 米，超出安装高度上下 12 厘米。一块白在一层高，两层的白更高。
    intr = intrinsics_for(160, 120)
    cfg = _cfg()
    cfg["ground_stride"] = 4
    gravity = np.array([0.0, 0.0, 1.0])
    depth = np.full((intr.height, intr.width), 0.90, dtype=np.float32)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    color[40:80, 60:110] = (255, 255, 255)
    depth[40:80, 60:110] = 0.55
    memory = GroundMemory()
    fix, stacked = ground_for_layers(depth, color, intr, cfg, gravity, 0.70, memory, StackSearch())
    assert not stacked
    assert fix is not None
    assert fix.source in ("fit", "held", "imu")
    assert abs(-float(fix.plane.offset) - 0.70) < 0.05

    depth[40:80, 60:110] = 0.15
    fix, stacked = ground_for_layers(depth, color, intr, cfg, gravity, 0.70, GroundMemory(), StackSearch())
    assert stacked
    assert fix is not None
    assert fix.source == "ransac"
    assert abs(-float(fix.plane.offset) - 0.90) < 0.05

    # 已经在看上面那块时，不再先套 0.70 米。确认还是一层之后，回到安装高度。
    looking_up = StackSearch()
    looking_up.stage = "upstairs"
    looking_up.upper = {"layer": 2}
    fix, stacked = ground_for_layers(depth, color, intr, cfg, gravity, 0.70, GroundMemory(), looking_up)
    assert stacked
    assert fix is not None
    assert abs(-float(fix.plane.offset) - 0.90) < 0.05
    done = StackSearch()
    done.settled_one = True
    done.cue = True
    done.lower = {"layer": 1}
    fix, stacked = ground_for_layers(depth, color, intr, cfg, gravity, 0.70, GroundMemory(), done)
    assert not stacked
    assert fix is not None
    assert abs(-float(fix.plane.offset) - 0.70) < 0.05


def test_missing_ground_keeps_last_height():
    gravity = np.array([0.0, 0.0, 1.0])
    empty = np.zeros((0, 3))
    first = ground_with_prior(empty, None, gravity, 0.70, None)
    assert first.source == "imu"
    held = GroundFix(plane=ground_prior(gravity, 0.66), source="fit", height=0.66)
    again = ground_with_prior(empty, None, gravity, 0.70, held)
    assert again.source == "held"
    assert abs(float(again.height) - 0.66) < 1e-6


def _close_ring(floor_z: float, edge_z: float | None = None):
    # 俯视，地面在 0.70 m。方块深度 0.60 m，外圈按这个距离算像素。
    intr = intrinsics_for(320, 240)
    depth = np.full((intr.height, intr.width), floor_z, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[80:160, 110:210] = 255
    depth[mask > 0] = 0.60
    if edge_z is not None:
        near = cv2.dilate(mask, np.ones((9, 9), np.uint8))
        depth[(near > 0) & (mask == 0)] = edge_z
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.70)
    return intr, depth, mask, plane


def test_close_ring_allows_two_centimetres():
    # 脚下地面高出 1.2 cm。旧的 8 mm 会丢掉，现在留下。贴边的 4 cm 跳变不参与。
    intr, depth, mask, plane = _close_ring(0.712, edge_z=0.66)
    assert _ground_under_cube(depth, mask, plane, intr)
    far, depth_far, mask_far, plane_far = _close_ring(0.66)
    assert not _ground_under_cube(depth_far, mask_far, plane_far, intr)
    wrong = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.50)
    intr, depth, mask, _plane = _close_ring(0.70)
    assert not _ground_under_cube(depth, mask, wrong, intr)
    _ = cv2


def _rings(mask, intr):
    z_med = 0.60
    outer = _ring_px(0.06, z_med, intr.fx, 8, 100)
    inner = _ring_px(0.02, z_med, intr.fx, 3, max(outer - 2, 3))
    far_in = _ring_px(0.08, z_med, intr.fx, outer + 1, 480)
    far_out = _ring_px(0.14, z_med, intr.fx, far_in + 1, 480)
    return _coarse_ring(mask, inner, outer), _coarse_ring(mask, far_in, far_out)


def test_far_floor_saves_a_bad_near_ring_and_near_ring_still_wins():
    # 贴边对不上、8 cm 以外的红地板对得上：正面留下。贴边对得上、远处不对：侧面仍留下。
    intr, depth, mask, plane = _close_ring(0.70)
    near, far = _rings(mask, intr)
    depth[near > 0] = 0.50
    assert _ground_under_cube(depth, mask, plane, intr)
    depth[near > 0] = 0.70
    depth[far > 0] = 0.50
    assert _ground_under_cube(depth, mask, plane, intr)
    depth[near > 0] = 0.50
    outside = (mask == 0) & (near == 0) & (far == 0)
    depth[outside] = 0.50
    assert not _ground_under_cube(depth, mask, plane, intr)
    colors = [{"name": "white", "hsv_low": [0, 0, 170], "hsv_high": [180, 110, 255]}]
    color = np.full((*depth.shape, 3), 40, dtype=np.uint8)
    color[far > 0] = (255, 255, 255)
    keep = (far > 0) & (np.arange(depth.shape[1])[None, :] < 110)
    color[keep] = 40
    depth[keep] = 0.70
    # 远处白盒子底下露出一条贴地的红地板，画面里就能对上，不再要求整圈都贴地。
    assert _ground_under_cube(depth, mask, plane, intr)
    assert _ground_under_cube(depth, mask, plane, intr, color, colors)


def test_filled_frame_uses_visible_floor_or_keeps_the_plane():
    # 白块顶到画面上边。两圈是深度空洞，角上的红地板对得上就留下。
    intr = intrinsics_for(320, 240)
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.70)
    depth = np.full((intr.height, intr.width), 0.50, dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[0:80, 100:220] = 255
    depth[mask > 0] = 0.60
    depth[180:, :] = 0.70
    assert _ground_under_cube(depth, mask, plane, intr)
    depth[180:, :] = 0.66
    assert not _ground_under_cube(depth, mask, plane, intr)
    depth[:, :] = np.nan
    depth[mask > 0] = 0.60
    assert _ground_under_cube(depth, mask, plane, intr)
    inset, inset_depth, inset_mask, _plane = _close_ring(0.70)
    near, far = _rings(inset_mask, inset)
    inset_depth[near > 0] = 0.50
    inset_depth[far > 0] = 0.50
    # 方块留在画面里，贴边两圈被挡住。其余红地板仍在 0.70 m，就留下。
    assert _ground_under_cube(inset_depth, inset_mask, plane, inset)
    inset_depth[inset_mask == 0] = 0.50
    assert not _ground_under_cube(inset_depth, inset_mask, plane, inset)


def _front_face_cloud(width: float = 0.35, tall: bool = True):
    # 相机朝下看，地面在 y = 0.70。正面在 z = 0.80，向右是 x。
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset=-0.70)
    xs = np.linspace(-0.5 * width, 0.5 * width, 16)
    ys = np.linspace(0.36, 0.60, 12) if tall else np.linspace(0.34, 0.36, 4)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.80)])
    heights = plane.signed_distance(points)
    count = len(points)
    return plane, points, heights, np.arange(count) // 20, np.arange(count) % 20, (count // 20 + 1, 20)


def test_stacked_lower_cube_completes_a_thin_front_edge():
    # 顶面只剩正面这条边。盒身上下成面，中心补到棱后半个边长。
    plane, points, heights, ys, xs, shape = _front_face_cloud()
    pose = _stack_lower_from_face(
        ys, xs, np.ones(len(points), dtype=bool), points, heights, plane, shape, 0.35, 0.06, None, None
    )
    assert pose is not None
    assert abs(float(pose.bottom_center[0])) < 0.03
    assert abs(float(pose.bottom_center[2]) - 0.975) < 0.04
    assert abs(pose.long_side - 0.35) < 1e-6
    assert abs(pose.short_side - 0.35) < 1e-6


def test_two_visible_faces_keep_the_front_edge():
    # 角朝向镜头。两条边都在，转角用更正对镜头的那条，不用合成的外沿。
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset=-0.70)
    corner = np.array([0.0, 0.60, 0.75])
    front_dir = np.array([0.99, 0.0, 0.14])
    front_dir = front_dir / np.linalg.norm(front_dir)
    side_dir = np.array([0.14, 0.0, 0.99])
    side_dir = side_dir / np.linalg.norm(side_dir)
    ys = np.linspace(0.40, 0.62, 8)
    ts = np.linspace(0.0, 0.35, 12)
    faces = []
    for direction in (front_dir, side_dir):
        grid_y, grid_t = np.meshgrid(ys, ts)
        faces.append(corner + grid_t.ravel()[:, None] * direction + np.column_stack([
            np.zeros(grid_y.size),
            grid_y.ravel() - corner[1],
            np.zeros(grid_y.size),
        ]))
    points = np.vstack(faces)
    heights = plane.signed_distance(points)
    count = len(points)
    pose = _stack_lower_from_face(
        np.arange(count) // 20,
        np.arange(count) % 20,
        np.ones(count, dtype=bool),
        points,
        heights,
        plane,
        (count // 20 + 1, 20),
        0.35,
        0.06,
        None,
        None,
    )
    assert pose is not None
    assert abs(float(pose.edge_direction @ front_dir)) > 0.9


def test_thin_line_without_a_face_is_not_a_cube():
    plane, points, heights, ys, xs, shape = _front_face_cloud(tall=False)
    assert (
        _stack_lower_from_face(
            ys, xs, np.ones(len(points), dtype=bool), points, heights, plane, shape, 0.35, 0.06, None, None
        )
        is None
    )
    assert _complete_short(points, plane, 0.35, 0.06) is None


def _pose_at(bottom: np.ndarray, layer: int = 1) -> CubePose:
    return CubePose(bottom_center=bottom, yaw=0.0, edge_direction=np.array([1.0, 0.0, 0.0]), footprint_fraction=1.0, front_color=False, layer=layer)


def test_lower_cube_must_sit_one_edge_below_the_camera():
    # 相机朝下，地面在 z = 0.70。下层顶面在 z = 0.35，也就是相机正下方 0.35 米。
    up = np.array([0.0, 0.0, -1.0])
    plane = Plane(normal=up, offset=-0.70)
    on_ground = _pose_at(np.array([0.10, 0.0, 0.70]))
    floating = _pose_at(np.array([0.10, 0.0, 0.35]))
    upper = _pose_at(np.array([0.10, 0.0, 0.35]), layer=2)
    assert lower_top_on_ground(on_ground, plane, up, 0.70, 0.35)
    assert not lower_top_on_ground(floating, plane, up, 0.70, 0.35)
    assert lower_top_on_ground(upper, plane, up, 0.70, 0.35)


def test_reject_reason_is_plain_chinese():
    text = reject_text("ground_ring")
    assert text == "方块旁边的地面对不上"
    assert "ground_ring" not in text
    assert reject_text("mask") == "没找到成块的白"
    assert reject_text(None) == "没认出"


def _level_to_camera(points: np.ndarray, alpha: float) -> np.ndarray:
    # 平视坐标：x 向右，y 向前，z 向上。alpha 是低头角。
    pts = np.asarray(points, dtype=float)
    c, s = np.cos(alpha), np.sin(alpha)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 1] * s - pts[:, 2] * c
    out[:, 2] = pts[:, 1] * c - pts[:, 2] * s
    return out


def test_missing_top_depth_uses_lifted_plane():
    # 低头看近处白盒。下半截正面有深度，顶面和上半截没有。体心仍应落在盒子上。
    alpha = np.deg2rad(35.0)
    intr = intrinsics_for(640, 360)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    step = 0.02
    xs = np.arange(-1.0, 1.0, step)
    ys = np.arange(0.40, 2.0, step)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.70)])
    _paint(
        _level_to_camera(ground, alpha),
        np.full((len(ground), 3), 40, dtype=np.uint8),
        intr,
        depth,
        color,
    )
    white = (255, 255, 255)
    lower = _face(
        np.array([0.05, 0.675, -0.6125]),
        np.array([0.35, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.175]),
        white,
        160,
    )
    _paint(_level_to_camera(lower[0], alpha), lower[1], intr, depth, color)
    before = depth.copy()
    upper = _face(
        np.array([0.05, 0.675, -0.4375]),
        np.array([0.35, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.175]),
        white,
        160,
    )
    top = _face(
        np.array([0.05, 0.85, -0.35]),
        np.array([0.35, 0.0, 0.0]),
        np.array([0.0, 0.35, 0.0]),
        white,
        160,
    )
    for pts, cols in (upper, top):
        _paint(_level_to_camera(pts, alpha), cols, intr, depth, color)
    depth[~np.isclose(depth, before, equal_nan=True)] = np.nan
    gravity = np.array([0.0, np.cos(alpha), np.sin(alpha)])
    frame = clean_frame(color, depth, gravity, _cfg())
    rejects: dict = {}
    record = estimate_frame(frame, intr, _cfg(), rejects=rejects, camera_height=0.7)
    assert record is not None, rejects
    body = _level_to_camera(np.array([[0.05, 0.85, -0.525]]), alpha)[0]
    assert abs(record["x"] - body[0]) < 0.05
    assert abs(record["y"] - body[1]) < 0.05
    assert abs(record["z"] - body[2]) < 0.05
    assert abs(record["long_side"] - 0.35) < 0.06
    assert abs(record["short_side"] - 0.35) < 0.06


def test_close_front_face_does_not_pull_the_far_cluster():
    # 正面交点又多又近，还偏到一侧。方向应跟着最远的那一团，中心留在顶面上。
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset=-0.70)
    xs = np.linspace(-0.175, 0.175, 12)
    zs = np.linspace(0.80, 1.15, 12)
    gx, gz = np.meshgrid(xs, zs)
    top = np.column_stack([gx.ravel(), np.full(gx.size, 0.35), gz.ravel()])
    fx = np.linspace(-0.35, 0.45, 40)
    fz = np.linspace(0.35, 0.75, 30)
    sx, sz = np.meshgrid(fx, fz)
    front = np.column_stack([sx.ravel(), np.full(sx.size, 0.35), sz.ravel()])
    kept = _keep_far_span(np.vstack([top, front]), plane, 0.35)
    assert kept is not None
    center = kept.mean(axis=0)
    assert abs(center[0]) < 0.08
    assert abs(center[2] - 0.975) < 0.08
    assert float(kept[:, 2].min()) > 0.70


def _paint_level_cube(color, depth, intr, alpha, center, size, yaw, bgr, samples=140):
    # center 是平视坐标里的体心：x 向右，y 向前，z 向上。
    c, s = np.cos(yaw), np.sin(yaw)
    ax = np.array([c, s, 0.0]) * size
    ay = np.array([-s, c, 0.0]) * size
    az = np.array([0.0, 0.0, size])
    faces = (
        _face(center + az * 0.5, ax, ay, bgr, samples),
        _face(center - ay * 0.5, ax, az, bgr, samples),
        _face(center + ay * 0.5, ax, az, bgr, samples),
        _face(center + ax * 0.5, ay, az, bgr, samples),
        _face(center - ax * 0.5, ay, az, bgr, samples),
    )
    for pts, cols in faces:
        _paint(_level_to_camera(pts, alpha), cols, intr, depth, color)


def _pitched_stack(lower_center, upper_center, upper_yaw=0.0, white_depth=True):
    alpha = np.deg2rad(35.0)
    intr = intrinsics_for(640, 360)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    step = 0.025
    xs = np.arange(-1.2, 1.2, step)
    ys = np.arange(0.35, 2.2, step)
    gx, gy = np.meshgrid(xs, ys)
    ground = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.70)])
    _paint(
        _level_to_camera(ground, alpha),
        np.full((len(ground), 3), 40, dtype=np.uint8),
        intr,
        depth,
        color,
    )
    white = (255, 255, 255)
    _paint_level_cube(color, depth, intr, alpha, lower_center, 0.35, 0.0, white)
    _paint_level_cube(color, depth, intr, alpha, upper_center, 0.35, upper_yaw, white)
    if not white_depth:
        depth[np.all(color == 255, axis=2)] = np.nan
    gravity = np.array([0.0, np.cos(alpha), np.sin(alpha)])
    frame = clean_frame(color, depth, gravity, _cfg())
    return intr, frame, alpha


def _near(record, level_point, alpha, tol=0.06):
    cam = _level_to_camera(np.asarray(level_point, dtype=float).reshape(1, 3), alpha)[0]
    return (
        abs(record["x"] - cam[0]) < tol
        and abs(record["y"] - cam[1]) < tol
        and abs(record["z"] - cam[2]) < tol
    )


def test_one_white_cube_stays_one_layer():
    center = np.array([0.12, -0.08, 0.70])
    _intr, frame, record = _measure(center, 0.2, 0.35, (255, 255, 255))
    assert record is not None
    layers = estimate_layers(frame, _intr, _cfg(), camera_height=0.7)
    assert len(layers) == 1
    assert layers[0]["layer"] == 1
    assert abs(layers[0]["x"] - record["x"]) < 1e-6
    assert abs(layers[0]["z"] - record["z"]) < 1e-6


def test_two_stacked_white_cubes_keep_their_own_centers():
    # 上面那块往右错开，两层顶面都能看见。中心不能是下层再抬高 0.35 m。
    alpha = np.deg2rad(35.0)
    lower = np.array([-0.20, 0.80, -0.525])
    upper = np.array([0.22, 0.80, -0.175])
    intr, frame, alpha = _pitched_stack(lower, upper, upper_yaw=0.2)
    rejects: dict = {}
    layers = estimate_layers(frame, intr, _cfg(), rejects=rejects, camera_height=0.7)
    assert len(layers) == 2, rejects
    by_layer = {int(item["layer"]): item for item in layers}
    assert _near(by_layer[1], lower, alpha)
    assert _near(by_layer[2], upper, alpha)
    lifted = lower + np.array([0.0, 0.0, 0.35])
    assert not _near(by_layer[2], lifted, alpha, tol=0.04)
    assert abs(fold_yaw_quarter(by_layer[2]["yaw"] - 0.2)) < np.deg2rad(12.0)
    assert abs(by_layer[1]["long_side"] - 0.35) < 0.06
    assert abs(by_layer[2]["long_side"] - 0.35) < 0.06
    vis, drawn = overlay(frame.color, frame.depth, intr, by_layer[1]["plane"], layers, 0.35, ["两层"])
    assert drawn >= 20
    cyan = (vis[:, :, 0] > 180) & (vis[:, :, 1] > 180) & (vis[:, :, 2] < 90)
    magenta = (vis[:, :, 0] > 180) & (vis[:, :, 1] < 80) & (vis[:, :, 2] > 180)
    assert int(cyan.sum()) > 80
    assert int(magenta.sum()) > 80


def test_stacked_front_without_depth_still_places_the_lower_cube():
    # 两层正面都是白，白像素上没有深度。下层仍按贴地的下沿补到盒子中心。
    alpha = np.deg2rad(35.0)
    lower = np.array([0.02, 0.90, -0.525])
    upper = np.array([0.02, 0.90, -0.175])
    intr, frame, alpha = _pitched_stack(lower, upper, white_depth=False)
    rejects: dict = {}
    layers = estimate_layers(frame, intr, _cfg(), rejects=rejects, camera_height=0.7)
    lower_rec = next((item for item in layers if int(item["layer"]) == 1), None)
    assert lower_rec is not None, rejects
    assert _near(lower_rec, lower, alpha, tol=0.08)
    assert abs(lower_rec["long_side"] - 0.35) < 1e-6
    lifted = lower + np.array([0.0, 0.0, 0.35])
    upper_rec = next((item for item in layers if int(item["layer"]) == 2), None)
    if upper_rec is not None:
        assert not _near(upper_rec, lifted, alpha, tol=0.04)


def test_one_cube_without_depth_is_not_completed_from_its_face():
    # 没有更高的白。正面缺深度时，不能把单独一块补成方块。
    alpha = np.deg2rad(35.0)
    lower = np.array([0.02, 0.90, -0.525])
    intr, frame, alpha = _pitched_stack(lower, lower, white_depth=False)
    assert estimate_layers(frame, intr, _cfg(), camera_height=0.7) == []


def test_far_white_above_the_ground_is_not_the_upper_cube():
    # 下层旁边一块盒子以外的高处白，不当成品红框。
    alpha = np.deg2rad(35.0)
    lower = np.array([0.02, 0.90, -0.525])
    far = np.array([1.30, 0.90, -0.175])
    intr, frame, alpha = _pitched_stack(lower, far)
    layers = estimate_layers(frame, intr, _cfg(), camera_height=0.7)
    assert not any(int(item["layer"]) == 2 for item in layers)
    lower_rec = next((item for item in layers if int(item["layer"]) == 1), None)
    assert lower_rec is not None
    assert _near(lower_rec, lower, alpha, tol=0.08)


def test_aligned_stack_still_reports_the_upper_cube():
    alpha = np.deg2rad(35.0)
    lower = np.array([0.02, 0.90, -0.525])
    upper = np.array([0.02, 0.90, -0.175])
    intr, frame, alpha = _pitched_stack(lower, upper)
    layers = estimate_layers(frame, intr, _cfg(), camera_height=0.7)
    assert any(int(item["layer"]) == 2 for item in layers)
    upper_rec = next(item for item in layers if int(item["layer"]) == 2)
    assert _near(upper_rec, upper, alpha, tol=0.08)
    lower_rec = next((item for item in layers if int(item["layer"]) == 1), None)
    if lower_rec is not None:
        assert _near(lower_rec, lower, alpha, tol=0.08)


def test_dim_front_counts_as_white_and_red_floor_does_not():
    image = np.full((40, 60, 3), 40, dtype=np.uint8)
    image[5:15, 5:15] = (145, 145, 145)
    image[5:15, 25:35] = (100, 100, 100)
    image[5:15, 45:55] = (30, 30, 200)
    strict, _loose, _front = _color_layers(image, _cfg()["colors"])
    assert int(strict[10, 10]) > 0
    assert int(strict[10, 30]) == 0
    assert int(strict[10, 50]) == 0


def test_ground_corner_places_the_cube_when_white_is_scattered():
    intr = intrinsics_for(320, 240)
    depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    xs = np.arange(-0.55, 0.85, 0.012)
    ys = np.arange(-0.45, 0.45, 0.012)
    gx, gy = np.meshgrid(xs, ys)
    hole = (gx > 0.15) & (gx < 0.50) & (np.abs(gy) < 0.175)
    floor = np.column_stack([gx[~hole], gy[~hole], np.full(int((~hole).sum()), 0.80)])
    _paint(floor, np.full((len(floor), 3), 40, np.uint8), intr, depth, color)
    near = np.array([[0.20, -0.02, 0.80]])
    pix = project_points(near, intr)[0]
    u = int(round(float(pix[0])))
    v = int(round(float(pix[1])))
    color[v - 4 : v + 4, u - 5 : u + 5] = (220, 220, 220)
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.80)
    poses = detect_white_layers(color, depth, intr, plane, _cfg()["colors"], cube_size=0.35, size_tolerance=0.06)
    assert len(poses) == 1
    bottom = poses[0].bottom_center
    assert abs(float(bottom[0]) - 0.325) < 0.06
    assert abs(float(bottom[1])) < 0.06
    assert abs(float(bottom[2]) - 0.80) < 0.03


def test_box_on_the_floor_moves_onto_the_near_white():
    intr = intrinsics_for(320, 240)
    depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
    color = np.full((intr.height, intr.width, 3), (30, 30, 180), dtype=np.uint8)
    xs = np.arange(-0.8, 0.8, 0.02)
    zs = np.arange(1.15, 2.5, 0.02)
    gx, gz = np.meshgrid(xs, zs)
    cube = (np.abs(gx) < 0.175) & (gz > 1.60) & (gz < 1.95)
    floor = np.column_stack([gx[~cube], np.full(int((~cube).sum()), 0.70), gz[~cube]])
    _paint(floor, np.full((len(floor), 3), (30, 30, 180), np.uint8), intr, depth, color)
    face_x = np.arange(-0.175, 0.176, 0.015)
    face_y = np.arange(0.40, 0.71, 0.015)
    fx, fy = np.meshgrid(face_x, face_y)
    face = np.column_stack([fx.ravel(), fy.ravel(), np.full(fx.size, 1.60)])
    _paint(face, np.full((len(face), 3), (240, 240, 240), np.uint8), intr, depth, color)
    # 正面可以没有深度，位置仍由白的下沿和地面决定。
    pix = project_points(face, intr)
    ui = np.rint(pix[:, 0]).astype(int)
    vi = np.rint(pix[:, 1]).astype(int)
    keep = (ui >= 0) & (vi >= 0) & (ui < intr.width) & (vi < intr.height)
    depth[vi[keep], ui[keep]] = np.nan
    far = np.array(
        [
            [0.45, 0.70, 1.80],
            [0.80, 0.70, 1.80],
            [0.80, 0.70, 2.15],
            [0.45, 0.70, 2.15],
        ]
    )
    pose = CubePose(
        bottom_center=np.array([0.625, 0.70, 1.975]),
        yaw=0.0,
        edge_direction=np.array([1.0, 0.0, 0.0]),
        footprint_fraction=0.5,
        front_color=False,
        corners=far,
    )
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), offset=-0.70)
    moved = _pull_lower_onto_white(
        pose, color, depth, plane, intr, _cfg()["colors"], 0.35, 0.06, None, None
    )
    assert moved is not None
    assert abs(float(moved.bottom_center[2]) - 1.775) < 0.08
    assert abs(float(moved.bottom_center[0])) < 0.08


def test_ground_corner_without_white_is_not_a_cube():
    intr = intrinsics_for(320, 240)
    depth = np.full((intr.height, intr.width), np.nan, dtype=np.float32)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    xs = np.arange(-0.55, 0.85, 0.012)
    ys = np.arange(-0.45, 0.45, 0.012)
    gx, gy = np.meshgrid(xs, ys)
    hole = (gx > 0.15) & (gx < 0.50) & (np.abs(gy) < 0.175)
    floor = np.column_stack([gx[~hole], gy[~hole], np.full(int((~hole).sum()), 0.80)])
    _paint(floor, np.full((len(floor), 3), 40, np.uint8), intr, depth, color)
    plane = Plane(normal=np.array([0.0, 0.0, -1.0]), offset=-0.80)
    poses = detect_white_layers(color, depth, intr, plane, _cfg()["colors"], cube_size=0.35, size_tolerance=0.06)
    assert poses == []


def test_labels_stay_in_the_corner():
    image = np.zeros((80, 200, 3), dtype=np.uint8)
    draw_labels(image, ["认出白盒"])
    assert int(image[:40, :160].sum()) > 0
    assert image[70, 190].sum() == 0
