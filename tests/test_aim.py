"""白盒贴到画面边时，给出还要转的俯仰和旋转。"""

from __future__ import annotations

import numpy as np

from cube.aim import plan_view
from main import clean_frame, load_config
from sim.scene import intrinsics_for
from view_cube_center import aim_line


def _cfg():
    return load_config("config/default.yaml")


def _frame(paint) -> tuple:
    intr = intrinsics_for(320, 240)
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), np.nan, dtype=float)
    paint(color, depth)
    return intr, color, depth


def test_cube_inside_the_frame_does_not_ask_to_turn():
    def paint(color, depth):
        color[80:160, 120:200] = (255, 255, 255)
        depth[80:160, 120:200] = 0.8

    intr, color, depth = _frame(paint)
    view = plan_view(color, depth, intr, _cfg()["colors"], gravity=np.array([0.0, 1.0, 0.0]))
    assert view.ready
    assert view.pitch.direction == "none"
    assert view.yaw.direction == "none"
    assert aim_line(view) == "俯仰 可以了，停止（±3°）  旋转 可以了，停止（±3°）"


def test_cube_cut_by_the_right_edge_turns_right():
    def paint(color, depth):
        color[80:160, 280:] = (255, 255, 255)
        depth[80:160, 280:] = 0.8

    intr, color, depth = _frame(paint)
    view = plan_view(color, depth, intr, _cfg()["colors"], gravity=np.array([0.0, 1.0, 0.0]))
    assert view.clipped["right"]
    assert not view.ready
    assert view.yaw.direction == "right"
    assert view.yaw.degrees > 5.0
    assert "右转" in aim_line(view)


def test_close_cube_that_cannot_fit_holds_both_axes():
    # 0.30 m 时一个边长已经比 240 像素的画面更高。贴着右边也不再左转或低头。
    def paint(color, depth):
        color[4:, 40:] = (255, 255, 255)
        depth[4:, 40:] = 0.30

    intr, color, depth = _frame(paint)
    view = plan_view(color, depth, intr, _cfg()["colors"], gravity=np.array([0.0, 1.0, 0.0]))
    assert view.reason == "close"
    assert view.ready
    assert view.pitch.direction == "none"
    assert view.yaw.direction == "none"
    assert aim_line(view) == "俯仰 可以了，停止（±3°）  旋转 可以了，停止（±3°）"


def test_cube_cut_by_the_top_edge_looks_up():
    def paint(color, depth):
        color[:24, 130:190] = (255, 255, 255)
        depth[:24, 130:190] = 0.8

    intr, color, depth = _frame(paint)
    view = plan_view(color, depth, intr, _cfg()["colors"], gravity=np.array([0.0, 1.0, 0.0]))
    assert view.clipped["top"]
    assert view.pitch.direction == "up"
    assert view.pitch.degrees > 5.0
    assert "抬头" in aim_line(view)


def test_cap_on_the_axis_holds_both_axes():
    from cube.aim import aim_at

    view = aim_at(np.array([0.0, 0.0, 0.8]), gravity=np.array([0.0, 1.0, 0.0]))
    assert view.ready
    assert view.pitch.direction == "none"
    assert view.yaw.direction == "none"


def test_cap_off_center_but_inside_the_frame_holds():
    from cube.aim import keep_in_frame

    intr = intrinsics_for(320, 240)
    level = np.array([0.0, 1.0, 0.0])
    pixels = np.array([[220, 70], [260, 70], [260, 110], [220, 110]], dtype=float)
    view = keep_in_frame(pixels, intr, level)
    assert view.ready
    assert view.reason == "inside"
    assert view.pitch.direction == "none"
    assert view.yaw.direction == "none"


def test_cap_touching_the_right_and_top_only_tucks_inside():
    from cube.aim import keep_in_frame

    intr = intrinsics_for(320, 240)
    level = np.array([0.0, 1.0, 0.0])
    pixels = np.array([[300, -20], [340, -20], [340, 40], [300, 40]], dtype=float)
    view = keep_in_frame(pixels, intr, level)
    assert view.clipped["right"]
    assert view.clipped["top"]
    assert not view.clipped["left"]
    assert not view.clipped["bottom"]
    assert view.yaw.direction == "right"
    assert view.pitch.direction == "up"
    assert abs(view.yaw.degrees - 6.0) < 1e-6
    assert abs(view.pitch.degrees - 6.0) < 1e-6


def test_cap_barely_inside_the_edge_holds():
    from cube.aim import keep_in_frame

    intr = intrinsics_for(320, 240)
    level = np.array([0.0, 1.0, 0.0])
    # 离右缘大约 1.5 度，还要补的不到 2 度，写成保持。
    pixels = np.array([[312, 120]], dtype=float)
    view = keep_in_frame(pixels, intr, level)
    assert view.clipped["right"]
    assert view.yaw.direction == "none"
    assert view.pitch.direction == "none"


def test_cap_touching_both_sides_holds_that_axis():
    from cube.aim import keep_in_frame

    intr = intrinsics_for(320, 240)
    level = np.array([0.0, 1.0, 0.0])
    pixels = np.array([[0, 100], [319, 120]], dtype=float)
    view = keep_in_frame(pixels, intr, level)
    assert view.clipped["left"] and view.clipped["right"]
    assert view.yaw.direction == "none"
    assert view.pitch.direction == "none"


def test_cap_right_of_center_turns_right_and_above_center_looks_up():
    from cube.aim import aim_at

    level = np.array([0.0, 1.0, 0.0])
    right = aim_at(np.array([0.30, 0.0, 1.0]), gravity=level)
    assert right.yaw.direction == "right"
    assert right.yaw.degrees > 10.0
    assert right.pitch.direction == "none"
    above = aim_at(np.array([0.0, -0.20, 1.0]), gravity=level)
    assert above.pitch.direction == "up"
    assert above.pitch.degrees > 8.0
    assert above.yaw.direction == "none"


def test_white_beyond_the_range_is_not_aimed_at():
    def paint(color, depth):
        color[80:160, 280:] = (255, 255, 255)
        depth[80:160, 280:] = 3.0

    intr, color, depth = _frame(paint)
    cfg = _cfg()
    view = plan_view(
        color,
        depth,
        intr,
        cfg["colors"],
        gravity=np.array([0.0, 1.0, 0.0]),
        zmin=float(cfg["depth_min"]),
        zmax=float(cfg["depth_max"]),
    )
    assert view.reason == "no_cube"
    assert aim_line(view) == "俯仰停止，旋转停止，等待有效目标"


def test_frame_drops_depth_outside_the_working_range():
    cfg = _cfg()
    depth = np.array([[0.04, 0.40, 3.0]], dtype=np.float32)
    color = np.zeros((1, 3, 3), dtype=np.uint8)
    cleaned = clean_frame(color, depth, None, cfg)
    assert not np.isfinite(cleaned.depth[0, 0])
    assert abs(float(cleaned.depth[0, 1]) - 0.40) < 1e-4
    assert not np.isfinite(cleaned.depth[0, 2])
