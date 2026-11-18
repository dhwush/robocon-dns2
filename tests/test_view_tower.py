"""粗层数画面上的白话。不打开相机。"""

from __future__ import annotations

import numpy as np

from cube.cap_pose import color_body_pixels, color_seam_pixels
from cube.stack_look import StackSearch
from main import GroundMemory, load_config
from sim.scene import intrinsics_for
from view_tower import (
    BoxColorHold,
    TowerWatch,
    color_view_held,
    box_leans,
    cap_aim_text,
    describe,
    face_leans,
    logic_line,
    paint,
    park_result,
    pitch_need,
    result_line,
    tower_ground,
)


def _bands(**kwargs) -> dict:
    base = {
        "lower": False,
        "upper_white": False,
        "cap": False,
        "distance": 0.8,
        "anchor": np.array([0.1, 0.35, 0.8]),
        "covers": False,
        "lower_point": np.array([0.10, 0.40, 0.80]),
        "upper_point": None,
        "cap_point": None,
        "lower_uv": np.zeros((0, 2), dtype=int),
        "upper_uv": np.zeros((0, 2), dtype=int),
        "cap_uv": np.zeros((0, 2), dtype=int),
    }
    base.update(kwargs)
    return base


def test_low_look_says_to_rise_and_does_not_give_a_layer():
    search = StackSearch()
    bands = _bands(lower=True)
    search.update([], bands, 0.8, 0.35, pitch=35.0)
    lines = describe(search, bands, 35.0)
    text = "\n".join(lines)
    assert "层数还没定" in text
    assert "还在低头" in text
    assert "还要抬头大约 35 度" in text
    assert lines[-2] == "层数还没定"
    assert lines[-1].startswith("还要抬头大约 35 度")
    assert "当前低头 35 度" in text
    assert "下层白 缺少有效姿态，车身坐标待确认" in text
    assert "上层白 这眼还没看到" in text
    assert "塔顶 这眼没有红蓝" in text


def test_close_pitch_says_how_far_above_level():
    level = np.array([0.0, 1.0, 0.0])
    search = StackSearch()
    low = _bands(
        lower=True,
        distance=0.30,
        lower_point=np.array([0.0, 0.35, 0.30]),
        anchor=np.array([0.0, 0.70, 0.30]),
    )
    search.update([], low, 0.30, 0.35, pitch=28.0, gravity=level)
    search.update([], low, 0.30, 0.35, pitch=0.0, gravity=level)
    assert "比水平再高 8 度" in pitch_need(search, 0.0)
    assert "再抬一点才看全顶块" in logic_line(search, low)


def test_level_view_settles_one_layer_in_plain_words():
    search = StackSearch()
    search.update([], _bands(lower=True), 0.8, 0.35, pitch=35.0)
    high = _bands(covers=True, lower_point=None)
    line = ""
    for _ in range(3):
        search.update([], high, 0.8, 0.35, pitch=2.0)
        line = logic_line(search, high)
    lines = describe(search, high, 2.0)
    text = "\n".join(lines)
    assert "层数 1，没有塔顶" in text
    assert "高处没有成片的白" in text
    assert "俯仰保持" in pitch_need(search, 2.0)
    assert "定了" in line


def test_overlay_text_is_small_and_does_not_cover_with_a_black_panel():
    from view_tower import _write

    image = np.full((180, 480, 3), 90, dtype=np.uint8)
    _write(image, ["为什么这么判", "当前低头 10 度"])
    top = image[:70, :220]
    still_open = np.all(top == 90, axis=2)
    assert int(still_open.sum()) > top.shape[0] * top.shape[1] // 2


def test_layer_color_and_pitch_are_drawn_at_the_bottom_edge():
    from view_tower import _write

    image = np.full((240, 320, 3), 40, dtype=np.uint8)
    _write(
        image,
        ["为什么这么判", "当前低头 10 度"],
        ["层数 2，有红蓝塔顶", "朝上是红色", "还要抬头大约 8 度"],
    )
    changed = np.any(image != 40, axis=2).any(axis=1)
    rows = np.flatnonzero(changed)
    assert int(rows[0]) < 30
    assert int(rows[-1]) > 200
    assert not bool(changed[80:160].all())


def test_settled_layer_without_a_cap_keeps_pitch_hold_at_the_bottom():
    search, high = _settled("one")
    lines = describe(search, high, 2.0)
    assert lines[-2] == "层数 1，没有塔顶"
    assert lines[-1] == "层数已经定了，俯仰保持"
    assert "朝上" not in lines[-2] and "朝上" not in lines[-1]


def test_light_line_lists_the_five_levels():
    from main import load_config
    from view_tower import light_line, light_names

    cfg = load_config("config/default.yaml")
    names = light_names(cfg)
    assert names == ["bright", "daylight", "indoor", "dim", "dark"]
    text = light_line({"name": "indoor", "exposure": 150}, names)
    assert "室内" in text
    assert "曝光 150" in text
    assert "1很亮" in text and "5很暗" in text


def test_view_starts_on_the_third_light_level():
    from view_tower import starting_light

    names = ["bright", "daylight", "indoor", "dim", "dark"]
    assert starting_light(names, None) == "indoor"
    assert starting_light(names, "dark") == "dark"


def test_tilted_ground_is_not_a_level_box():
    down = np.array([0.0, 1.0, 0.0])
    assert not box_leans(np.array([0.0, -1.0, 0.0]), down)
    tilted = np.array([np.sin(np.deg2rad(10.0)), -np.cos(np.deg2rad(10.0)), 0.0])
    assert box_leans(tilted, down)
    flat = np.array(
        [
            [-0.1, 0.1, 0.6],
            [0.1, 0.1, 0.6],
            [0.1, 0.1, 0.8],
            [-0.1, 0.1, 0.8],
        ]
    )
    assert not face_leans(flat, down)
    flat[:, 1] = flat[:, 0] * np.tan(np.deg2rad(10.0))
    assert face_leans(flat, down)
    search = StackSearch()
    search.kind = "two_cap"
    search.settled = True
    text = "\n".join(describe(search, _bands(cap=True), 0.0, cap_reason="tilt"))
    assert "方块和地面不平行" in text


def test_cap_settles_two_layers_and_can_show_a_measured_center():
    search = StackSearch()
    search.update([], _bands(lower=True), 0.8, 0.35, pitch=30.0)
    high = _bands(cap=True, covers=True, upper_white=True, upper_point=np.array([0.12, 0.05, 0.70]))
    for _ in range(3):
        search.update([], high, 0.8, 0.35, pitch=0.0)
    center = np.array([0.08, -0.10, 0.65])
    lines = describe(search, high, 0.0, cap_center=center, cap_yaw=0.2)
    text = "\n".join(lines)
    assert "层数 2，有红蓝塔顶" in text
    assert "上层白 缺少有效姿态，车身坐标待确认" in text
    assert "相对小车右前 向左 0.13 米" in text
    assert "相对光心" not in text
    assert "转角正" in text
    missed = describe(search, high, 0.0, cap_reason="depth")
    assert "上面的深度太少" in "\n".join(missed)
    world = describe(
        search,
        high,
        0.0,
        cap_world=np.array([-0.13, 0.70, 0.80]),
        cap_yaw=0.0,
    )
    assert (
        "塔顶 相对小车右前 向左 0.13 米，向前 0.70 米，向上 0.80 米，转角基本摆正"
    ) in "\n".join(world)
    waiting = describe(search, high, 0.0, cap_waiting=2)
    assert "塔顶还在对齐，已有 2 帧" in "\n".join(waiting)
    assert "俯仰保持" not in pitch_need(search, 0.0)
    assert pitch_need(search, 0.0) == "层数已经定了"


def test_cap_inside_the_frame_holds_even_off_center():
    search, high = _settled("two_cap")
    level = np.array([0.0, 1.0, 0.0])
    intr = intrinsics_for(320, 240)
    inside = np.array([[220, 70], [260, 70], [260, 110], [220, 110]], dtype=float)
    held = cap_aim_text(inside, intr, level)
    assert held == "顶块入画 俯仰保持，可以了，停止，旋转保持，可以了，停止"
    edge = np.array([[300, -20], [340, -20], [340, 40], [300, 40]], dtype=float)
    turned = cap_aim_text(edge, intr, level)
    assert "右转" in turned
    assert "抬头" in turned
    lines = describe(search, high, 0.0, cap_aim=turned, cap_color="blue")
    text = "\n".join(lines)
    assert lines[-3] == "层数 2，有红蓝塔顶"
    assert lines[-2] == "朝上是蓝色"
    assert lines[-1] == turned
    parked = park_result(lines, "识别完成 层数 2 顶块红色", 35.0)
    assert parked[-3] == "层数 2，有红蓝塔顶"
    assert parked[-2] == "朝上是蓝色"
    assert parked[-1] == "低头 13.8 度，到低头 48.8 度；旋转停止"
    assert turned not in parked
    assert "识别完成，换下一个" not in parked
    same = park_result(describe(search, high, 35.0, cap_aim="俯仰低头 6.0 度", cap_color="red"), "识别完成", 35.0)
    assert same[-1] == parked[-1]
    missing = cap_aim_text(None, intr, level)
    assert missing == "顶块这一眼看不出在不在画面里，停止俯仰和旋转，等待有效目标"
    plain = "\n".join(describe(search, high, 0.0))
    assert "顶块入画" not in plain
    assert "顶块对准" not in text


def _settled(kind: str) -> tuple[StackSearch, dict]:
    search = StackSearch()
    search.update([], _bands(lower=True), 0.8, 0.35, pitch=35.0)
    if kind == "two":
        high = _bands(upper_white=True, covers=True)
    elif kind == "two_cap":
        high = _bands(cap=True, covers=True, upper_white=True)
    else:
        high = _bands(covers=True, lower_point=None)
    for _ in range(3):
        search.update([], high, 0.8, 0.35, pitch=2.0)
    return search, high


def test_plain_towers_announce_once_without_a_cap():
    search, _high = _settled("one")
    watch = TowerWatch()
    line = watch.offer(search, None, None, None)
    assert line == "识别完成 层数 1 没有顶块"
    assert watch.offer(search, None, None, None) is None

    search, _high = _settled("two")
    assert result_line(search.kind, None, None, None) == "识别完成 层数 2 没有顶块"


def test_cap_announcement_waits_for_a_pose_and_a_color():
    search, _high = _settled("two_cap")
    watch = TowerWatch()
    assert watch.offer(search, None, None, None) is None
    assert result_line("two_cap", np.array([0.12, 0.55, 0.80]), None, "red") is None
    yaw = np.deg2rad(15.0)
    line = watch.offer(search, np.array([0.12, 0.55, 0.80]), yaw, "red")
    assert line == (
        "识别完成 层数 2 顶块红色 "
        "相对小车右前 向右 0.12 米，向前 0.55 米，向上 0.80 米，转角正 15 度"
    )
    assert watch.offer(search, np.array([9.0, 9.0, 9.0]), yaw, "blue") is None
    assert watch.line == line


def test_finished_tower_clears_when_the_base_leaves_or_jumps():
    search, high = _settled("one")
    watch = TowerWatch(hold_s=0.5)
    watch.offer(search, None, None, None)
    assert watch.depart(0.0, search, high) is False
    assert watch.depart(0.4, search, high) is False
    assert watch.depart(0.6, search, high) is True

    search, _high = _settled("one")
    watch = TowerWatch(hold_s=0.5)
    watch.offer(search, None, None, None)
    watch.depart(0.0, search, _bands(lower=False))
    assert watch.depart(0.2, search, _bands(lower=True)) is False
    watch.depart(0.3, search, _bands(lower=False))
    assert watch.depart(0.7, search, _bands(lower=False)) is False

    search, _high = _settled("two_cap")
    watch = TowerWatch()
    moved = _bands(lower=True, anchor=np.array([0.40, 0.35, 0.80]))
    assert watch.depart(0.0, search, moved.copy()) is False
    assert watch.depart(0.2, search, moved.copy()) is False
    assert watch.depart(0.4, search, moved.copy()) is True
    fresh = StackSearch()
    assert watch.depart(5.0, fresh, _bands(lower=False)) is False


def test_tower_ground_stays_near_the_camera_height_when_the_floor_leaves_the_frame():
    # 抬头后画面被近处表面占满，或地板远在先验窗外。层数仍用 0.70 米，不跟最低平面走。
    intr = intrinsics_for(160, 120)
    cfg = load_config("config/default.yaml")
    cfg["ground_stride"] = 4
    gravity = np.array([0.0, 0.0, 1.0])
    color = np.full((intr.height, intr.width, 3), 40, dtype=np.uint8)
    depth = np.full((intr.height, intr.width), 0.25, dtype=np.float32)
    fix = tower_ground(depth, color, intr, cfg, gravity, 0.70, GroundMemory())
    assert fix is not None
    assert fix.source == "imu"
    assert abs(-float(fix.plane.offset) - 0.70) < 0.05

    depth[:] = 0.90
    color[40:80, 60:110] = (255, 255, 255)
    depth[40:80, 60:110] = 0.15
    fix = tower_ground(depth, color, intr, cfg, gravity, 0.70, GroundMemory())
    assert fix is not None
    assert abs(-float(fix.plane.offset) - 0.70) < 0.05


def test_close_rgb_uses_a_white_seam_and_an_orange_outline():
    # 外轮廓贴着方块边，交界在红蓝相接的地方。画面上这两条不走黄线。
    color = np.full((120, 160, 3), 40, dtype=np.uint8)
    color[20:80, 30:120] = (255, 0, 0)
    color[20:50, 30:120] = (0, 0, 255)
    body = color_body_pixels(color)
    seam = color_seam_pixels(color)
    assert len(body) >= 16
    on_edge = (body[:, 0] <= 32) | (body[:, 0] >= 118) | (body[:, 1] <= 22) | (body[:, 1] >= 78)
    assert float(on_edge.mean()) > 0.85
    assert len(seam) >= 16
    assert abs(float(np.median(seam[:, 1])) - 49) < 8
    search, high = _settled("two_cap")
    text = "\n".join(describe(search, high, 0.0, cap_rgb=True))
    assert "白线是红蓝交界" in text
    assert "橙线是方块轮廓" in text
    assert "黄线塔顶轮廓" not in text
    high["cap_uv"] = np.column_stack([np.full(30, 70), np.arange(30, 60)])
    high["rgb_body_uv"] = body
    high["rgb_seam_uv"] = seam
    vis = paint(color, high, ["层数还没定"], None, None, None)
    yellow = np.all(vis == np.array([0, 220, 255]), axis=2)
    assert int(yellow.sum()) == 0
    assert int(np.all(vis == np.array([255, 255, 255]), axis=2).sum()) > 10
    assert int(np.all(vis == np.array([0, 140, 255]), axis=2).sum()) > 10


def test_side_rgb_uses_a_blue_seam_and_a_red_outline():
    # 侧面深度量不成时的颜色，不跟近处的白线和橙线混。
    color = np.full((120, 160, 3), 40, dtype=np.uint8)
    color[20:80, 30:120] = (255, 0, 0)
    color[20:50, 30:120] = (0, 0, 255)
    body = color_body_pixels(color)
    seam = color_seam_pixels(color)
    search, high = _settled("two_cap")
    text = "\n".join(describe(search, high, 0.0, cap_rgb="side"))
    assert "浅蓝是红蓝交界" in text
    assert "浅红是方块轮廓" in text
    assert "侧面深度量不成，改用颜色" in text
    assert "白线是红蓝交界" not in text
    assert "黄线塔顶轮廓" not in text
    high["cap_uv"] = np.column_stack([np.full(30, 70), np.arange(30, 60)])
    high["rgb_style"] = "side"
    high["rgb_body_uv"] = body
    high["rgb_seam_uv"] = seam
    vis = paint(color, high, ["层数还没定"], None, None, None)
    yellow = np.all(vis == np.array([0, 220, 255]), axis=2)
    assert int(yellow.sum()) == 0
    assert int(np.all(vis == np.array([0, 140, 255]), axis=2).sum()) == 0
    assert int(np.all(vis == np.array([255, 128, 0]), axis=2).sum()) > 10
    assert int(np.all(vis == np.array([128, 128, 255]), axis=2).sum()) > 10


def test_looking_up_still_counts_as_waiting_for_the_top_color():
    from cube.aim import keep_in_frame

    intr = intrinsics_for(320, 240)
    level = np.array([0.0, 1.0, 0.0])
    touching = keep_in_frame(np.array([[160, -20], [160, 40]], dtype=float), intr, level)
    assert touching.pitch.direction == "up"
    assert touching.yaw.direction == "none"
    assert color_view_held(touching)

    turning = keep_in_frame(np.array([[340, -20], [300, 40]], dtype=float), intr, level)
    assert turning.yaw.direction == "right"
    assert not color_view_held(turning)


def test_upper_color_waits_out_ten_seconds_then_keeps_the_choice():
    hold = BoxColorHold()
    world = np.array([0.10, 0.70, 0.80])
    assert not hold.consider(0.0, world, True, False)
    assert not hold.consider(9.9, world, True, False)
    assert hold.consider(10.0, world, True, False)
    hold.latch("red", world, 0.2)
    assert hold.latched
    assert hold.consider(12.0, world + 1.0, False, False)
    assert hold.color == "red"
    assert np.allclose(hold.world, world)


def test_upper_color_wait_restarts_when_the_box_or_the_view_moves():
    hold = BoxColorHold()
    world = np.array([0.10, 0.70, 0.80])
    hold.consider(0.0, world, True, False)
    jumped = world + np.array([0.05, 0.0, 0.0])
    assert not hold.consider(9.0, jumped, True, False)
    assert not hold.consider(18.0, jumped, True, False)
    assert hold.consider(19.0, jumped, True, False)

    hold = BoxColorHold()
    hold.consider(0.0, world, True, False)
    assert not hold.consider(9.0, world, False, False)
    assert not hold.consider(20.0, world, True, False)
    assert hold.consider(30.0, world, True, False)

    hold = BoxColorHold()
    hold.consider(0.0, world, True, False)
    assert not hold.consider(6.0, world, True, True)
    assert not hold.consider(20.0, world, True, False)
