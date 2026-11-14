"""实时看塔的层数。不走地面白盒那套精测。

左上角写为什么这么判、当前俯仰、白盒大概在哪。
窗口最下面写层数、红蓝朝上的颜色、俯仰还要怎么变。俯仰保持也在这下面。
在抬头位置保持 15 秒未看到顶块，改为提示低头检查第一层顶面露出面积和上层遮挡；连续 5 帧确认白盒层数后，保持俯仰并自动开启下一轮识别。
近处的抬头按下层白的水平距离，黄线顶到画面上沿再抬一点。
有顶块并且层数已经定了，再写把顶块留在画面里还要转的俯仰和旋转。已经在画面里就保持。贴边超过 2 度写成 6 度。
青线是下层白，品红线是上层白，黄线是塔顶红蓝的轮廓。
绿框是这一帧量准的塔顶方块。分界没量准时，用黄线离地更低的那条边当方块最底端，只在这一眼推算。正面内部对不上深度时，成直角的另一面留下当侧面，对不上的跳点才丢掉；两面都分不清这一眼不画。
绿框的上下面必须和重力垂直。相机只在水平面里转，地面不会跟着斜。差过大约 3 度的这一帧不画，也不拿去平均。
近处深度对不成平面时，白线是红蓝交界，橙线是方块外轮廓。黄线这时不画。体心仍交到离地 0.80 米。
字直接写在画面上，不铺黑底。
这座塔认完之后，窗口最下面按当前俯仰提示转到低头 48.8 度，误差不超过 2 度时提示停止。这是相对水平的角度，不按距离改。终端仍报层数、顶块颜色和最终方位。
白盒层数只在先检查塔顶、再返回低头完成顶面检查后发布。有红蓝时要等体心和朝上的颜色一起稳定。
绿框稳住满 10 秒仍分不清朝上的颜色，就用绿框上半区里更多的那一色。上沿出画还在抬头，这 10 秒照样计。绿框上半截已经出画时，看得见的是下色，朝上用另一色。
有顶块时，报完返回低头 48.8 度后自动开启下一轮；无顶块时，返回低头确认白盒顶面和层数，保持俯仰自动开启下一轮。
一打开用第 3 档室内。按 1 到 5 可换：很亮、白天、室内、偏暗、很暗。第 1 档是原来的白天，第 5 档是原来的偏暗，第 2 档和第 4 档的亮度在相邻两档正中。按 q 或 Esc 退出。

    python view_tower.py
    python view_tower.py --competition
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tower_debug_window import DebugWindow
from camera.source import RealSenseSource, color_lock_from_config
from cube.aim import keep_in_frame
from cube.pose_projection import pose_matches_color
from cube.cap_pose import (
    CapFusion,
    _colored_mask,
    OutlineLive,
    box_upper_color,
    camera_to_car_world,
    cap_top_color,
    close_depth_unusable,
    color_body_pixels,
    color_seam_pixels,
    measure_cap as measure_cap_pose,
    measure_outline,
    optical_from_car_world,
)
from cube.stack_look import (
    StackSearch,
    _QUIET_FRAMES,
    coarse_bands,
    pitch_down_degrees,
    relax_upper_white,
)
from geom.transform import load_intrinsics
from cube.top_fallback import TopFallback
from ground.plane import Plane
from main import GroundMemory, _prior_ground, clean_frame, load_config, selected_camera_height
from view_cube_center import cube_wire_corners, draw_cube_edges, draw_center


# 青：下层白。品红：上层白。黄：塔顶颜色轮廓。绿：量准的塔顶方块。
LOWER_BGR = (255, 255, 0)
UPPER_BGR = (255, 0, 255)
CAP_BGR = (0, 220, 255)
# 近处改用颜色时：白线是红蓝交界，橙线是方块外轮廓。不跟黄线混用。
SEAM_BGR = (255, 255, 255)
BODY_BGR = (0, 140, 255)
# 侧面深度量不成、改用颜色时：浅蓝是交界，浅红是外轮廓。和近处那一套分开。
SIDE_SEAM_BGR = (255, 128, 0)
SIDE_BODY_BGR = (128, 128, 255)
POSE_BGR = (0, 220, 0)
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_SIZE = 15
_LIGHT_ORDER = ("bright", "daylight", "indoor", "dim", "dark")
_LIGHT_TEXT = {
    "bright": "很亮",
    "daylight": "白天",
    "indoor": "室内",
    "dim": "偏暗",
    "dark": "很暗",
}
_CAP_SIZE = 0.20
_CAP_REASON = {
    "mask": "没圈成一块",
    "depth": "上面的深度太少",
    "seam": "红蓝分界没连成一条",
    "plane": "侧面深度对不齐",
    "short": "侧面没看全",
    "border": "两端都在画面边上",
    "long": "分界比一块方块还长",
    "height": "高度离 0.80 米太远",
    "align": "还在对齐",
    "color": "分界上下的颜色分不清",
    "tilt": "方块和地面不平行",
    "projection": "测量框与当前红蓝目标不一致",
}
# 绿框朝上的方向和重力差过这么多，就算斜了。左右转不改变这个方向。
_BOX_LEAN_DEG = 3.0
# 下层白挪开这么远，就当成另一座塔。
_JUMP_M = 0.25
# 朝上颜色一直分不清时，绿框稳住这么久才用上半区定色。上沿出画还在抬头，这段照样计。
_COLOR_HOLD_S = 10.0
_BOX_STABLE_M = 0.03
# 认完以后要转到的低头角，相对水平。固定俯仰停靠角保持原值；安装高度按配置读取。
# 水平 20 到 80 厘米里，最远的白盒上沿大约 24 度，最近的地面大约 74 度，正中是 48.8 度。不按距离改。
_SETTLED_LOOK_DOWN_DEG = 48.8
_PARK_TOLERANCE_DEG = 2.0


def _side(value: float, positive: str, negative: str) -> str:
    if abs(value) < 0.005:
        return "正中"
    word = positive if value > 0.0 else negative
    return f"{word} {abs(value):.2f} 米"


def _spot(point, ground: Plane | None) -> str:
    if point is None:
        return "这眼没看到"
    xyz = np.asarray(point, dtype=float).reshape(3)
    text = f"{_side(float(xyz[0]), '向右', '向左')}，{_side(float(xyz[1]), '向下', '向上')}，{_side(float(xyz[2]), '向前', '向后')}"
    if ground is not None:
        height = float(ground.signed_distance(xyz.reshape(1, 3))[0])
        text += f"，离地大约 {height:.2f} 米"
    return text


def _world_spot(point) -> str:
    xyz = np.asarray(point, dtype=float).reshape(3)
    return (
        f"{_side(float(xyz[0]), '向右', '向左')}，"
        f"{_side(float(xyz[1]), '向前', '向后')}，"
        f"{_side(float(xyz[2]), '向上', '向下')}"
    )


def _both_spots(world) -> str:
    """同一个点写两遍：小车右前角，以及彩色光心。轴都是向右、向前、向上。"""
    optical = optical_from_car_world(world)
    return f"相对小车右前 {_world_spot(world)}；相对光心 {_world_spot(optical)}"


def _turn(yaw: float) -> str:
    deg = float(np.degrees(yaw))
    if abs(deg) < 2.0:
        return "转角基本摆正"
    if deg > 0.0:
        return f"转角正 {deg:.0f} 度"
    return f"转角负 {abs(deg):.0f} 度"


def pitch_now(pitch: float | None) -> str:
    if pitch is None:
        return "当前俯仰读不出来"
    if abs(pitch) <= 12.0:
        if pitch >= 0.0:
            return f"当前接近水平，还低头 {pitch:.0f} 度"
        return f"当前接近水平，还抬头 {abs(pitch):.0f} 度"
    if pitch > 0.0:
        return f"当前低头 {pitch:.0f} 度"
    return f"当前抬头 {abs(pitch):.0f} 度"


def pitch_need(search: StackSearch, pitch: float | None) -> str:
    if pitch is None:
        return "没有俯仰读数，看不出要不要再抬头"
    if search.settled:
        if search.kind == "two_cap":
            return "层数已经定了"
        return "层数已经定了，俯仰保持"
    if search.anchor is None:
        return "先低头，直到看见贴在地上的白盒"
    if not search.looked_up:
        if search.nudges:
            return f"黄线顶到画面上沿，再抬头大约 {max(0.0, search.degrees):.0f} 度"
        if search.vote_rise > 0.5:
            return (
                f"还要抬头大约 {max(0.0, search.degrees):.0f} 度，"
                f"抬到比水平再高 {search.vote_rise:.0f} 度"
            )
        return f"还要抬头大约 {max(0.0, search.degrees):.0f} 度，抬到接近水平。要看到离地半米以上，直到略高于相机"
    if search.clip_run:
        return "黄线顶着画面上沿，这几帧先不改俯仰"
    if search.vote_rise > 0.5:
        return "已经抬到看顶块的角度，这一眼看高处，俯仰先保持"
    return "已经接近水平，这一眼看高处，俯仰先保持"


def layer_line(search: StackSearch) -> str:
    if not search.settled:
        return "层数还没定"
    if search.kind == "one":
        return "层数 1，没有塔顶"
    if search.kind == "two_cap":
        return "层数 2，有红蓝塔顶"
    return "层数 2，没有塔顶"


def logic_line(search: StackSearch, bands: dict | None) -> str:
    if search.settled and search.kind == "one":
        return "定了：高处没有成片的白，也没有红蓝，所以是一层白"
    if search.settled and search.kind == "two":
        return "定了：离地半米以上有成片的白，上面没有红蓝，所以是两层白"
    if search.settled and search.kind == "two_cap":
        return "定了：红蓝在相机高度附近，所以是两层白，并且有塔顶"
    info = bands or {}
    if search.anchor is None and not info.get("lower"):
        return "还没看到贴地的白，先不数层"
    if not search.looked_up:
        if search.nudges:
            text = "看到贴地的白了。黄线顶到画面上沿，再抬头才看全，先不数层"
        elif search.pitch is not None and search.pitch <= 12.0:
            text = "看到贴地的白了。还要再抬一点才看全顶块，先不数层"
        else:
            text = "看到贴地的白了。还在低头，高处没看全，先不数层"
        if info.get("cap"):
            return text + "。这一眼扫到高处的红蓝，等抬够再确认"
        if info.get("upper_white"):
            return text + "。这一眼扫到高处的白，等抬够再确认"
        return text
    if search.clip_run:
        return "黄线顶着画面上沿，这几帧先不数层"
    if info.get("cap"):
        guess = "这一眼相机高度附近有红蓝，像是两层白，并且有塔顶"
    elif info.get("upper_white"):
        guess = "这一眼离地半米以上有成片的白，没有红蓝，像是两层白"
    elif info.get("covers"):
        guess = "这一眼高处没有成片的白，也没有红蓝，像是一层白"
    else:
        return "已经接近水平，但这座塔的高处还没进画面，先不数层"
    left = max(0, _QUIET_FRAMES - int(search.quiet))
    if left:
        return f"{guess}。还要连续再看 {left} 帧"
    return guess


def describe(
    search: StackSearch,
    bands: dict | None,
    pitch: float | None,
    ground: Plane | None = None,
    cap_center: np.ndarray | None = None,
    cap_yaw: float | None = None,
    cap_reason: str | None = None,
    cap_world: np.ndarray | None = None,
    cap_waiting: int | None = None,
    cap_aim: str | None = None,
    cap_outline: bool = False,
    cap_outline_color: str | None = None,
    cap_color: str | None = None,
    cap_rgb: bool | str = False,
    cap_unmeasured: bool = False,
) -> list[str]:
    """这一帧要写在画面上的白话。白盒位置相对光心。塔顶同时写小车右前角和光心。

    层数、朝上颜色和俯仰变化留在最后，画的时候贴在窗口下沿。
    """
    info = bands or {}
    lines = [
        logic_line(search, info),
        pitch_now(pitch),
        "下层白 " + _spot(info.get("lower_point"), ground),
    ]
    if info.get("cap_rgb_candidate"):
        visibility = "局部入画" if info.get("cap_partial") else "候选可见"
        lines.append(f"RGB 塔顶{visibility}，深度未确认，不计为塔顶")
    debug = info.get("rgb_debug")
    if debug:
        stage = {"color": "颜色点不足", "height/position": "高度或位置", "face": "面尺寸", "no_anchor_or_distance": "缺少位置参照", "accepted": "投影与面筛通过", "depth_unconfirmed": "深度未确认"}.get(debug.get("stage"), "深度候选已通过")
        candidate = {"visible": "相邻红蓝可见", "no_anchor": "缺少塔位置参照", "no_adjacent_pair": "未找到相邻红蓝"}.get(debug.get("candidate"), "未检查")
        lines.append(
            f"RGB诊断 红{debug.get('red', 0)} 蓝{debug.get('blue', 0)} "
            f"水平距离{debug.get('distance', 0):.2f}米 "
            f"高度后{debug.get('height', 0)} 位置后{debug.get('near', 0)} "
            f"面筛后{debug.get('face', 0)} "
            f"阶段：{stage} 候选：{candidate}"
        )
    if info.get("upper_white"):
        lines.append("上层白 " + _spot(info.get("upper_point"), ground))
    elif search.looked_up and info.get("covers"):
        lines.append("上层白 高处没有成片的白")
    else:
        lines.append("上层白 这眼还没看到")
    if cap_unmeasured:
        why = _CAP_REASON.get(cap_reason or "", "几何依据不足")
        lines.append(f"目标可见，本帧无法精测：{why}")
    elif cap_outline and cap_world is not None and cap_yaw is not None:
        lines.append(f"按黄线推算 {_both_spots(cap_world)}，{_turn(cap_yaw)}")
    elif cap_world is not None and cap_yaw is not None:
        lines.append(f"塔顶 {_both_spots(cap_world)}，{_turn(cap_yaw)}")
    elif cap_center is not None and cap_yaw is not None:
        lines.append(
            f"塔顶 {_both_spots(camera_to_car_world(cap_center, up=None if ground is None else ground.normal))}，{_turn(cap_yaw)}"
        )
    elif cap_waiting:
        lines.append(f"塔顶还在对齐，已有 {cap_waiting} 帧")
    elif info.get("cap") or search.kind == "two_cap":
        why = _CAP_REASON.get(cap_reason or "", "这一帧体心还没量准")
        lines.append(f"塔顶红蓝看到了，{why}")
    else:
        lines.append("塔顶 这眼没有红蓝")
    if cap_rgb == "side":
        lines.append("青线下层白，品红线上层白，浅蓝是红蓝交界，浅红是方块轮廓，侧面深度量不成，改用颜色，绿框是量准的塔顶")
    elif cap_rgb:
        lines.append("青线下层白，品红线上层白，白线是红蓝交界，橙线是方块轮廓，绿框是量准的塔顶")
    elif info.get("cap_rgb_candidate"):
        lines.append("浅蓝轮廓仅表示可见候选；有可靠位姿时才显示绿框")
    elif cap_outline:
        lines.append("青线下层白，品红线上层白，黄线塔顶轮廓，绿框是这一眼按黄线推算")
    else:
        lines.append("青线下层白，品红线上层白，黄线塔顶轮廓，绿框是量准的塔顶")
    return lines + status_lines(search, info, pitch, cap_aim, cap_color or cap_outline_color)


def upward_color_line(color: str | None, search: StackSearch, bands: dict | None) -> str | None:
    """红蓝朝上的那一色。没有这块方块就不写。看见了但分不清，就写还没看清。"""
    if color == "red":
        return "朝上是红色"
    if color == "blue":
        return "朝上是蓝色"
    info = bands or {}
    if info.get("cap") or search.kind == "two_cap":
        return "朝上的颜色还没看清"
    return None


def pitch_change_line(search: StackSearch, pitch: float | None, cap_aim: str | None) -> str:
    """这一眼俯仰要怎么变。有顶块时以顶块那句为准，没有就用找层数的那句。"""
    if cap_aim:
        return cap_aim
    return pitch_need(search, pitch)


def status_lines(
    search: StackSearch,
    bands: dict | None,
    pitch: float | None,
    cap_aim: str | None,
    color: str | None,
) -> list[str]:
    """贴在窗口下沿的三句：层数、朝上颜色、俯仰变化。没有红蓝就少一句颜色。"""
    lines = [layer_line(search)]
    tint = upward_color_line(color, search, bands)
    if tint:
        lines.append(tint)
    lines.append(pitch_change_line(search, pitch, cap_aim))
    return lines


def _status_start(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if str(line).startswith("层数"):
            return index
    return None


def split_status(lines: list[str]) -> tuple[list[str], list[str]]:
    """层数开头的那一句往后，都是要贴在窗口下沿的。"""
    start = _status_start(lines)
    if start is None:
        return lines, []
    return lines[:start], lines[start:]


def settled_look_down_line(pitch: float | None = None) -> str:
    """相对水平停靠；到位后明确停止，过头则要求抬头。"""
    if pitch is None or not np.isfinite(pitch):
        return "俯仰读数无效，停止俯仰和旋转，等待姿态恢复"
    delta = _SETTLED_LOOK_DOWN_DEG - pitch
    if abs(delta) <= _PARK_TOLERANCE_DEG:
        return "已到位，可以停止俯仰和旋转；保持当前俯仰，开启下一轮识别"
    direction = "低头" if delta > 0 else "抬头"
    return f"{direction} {abs(delta):.1f} 度，到低头 {_SETTLED_LOOK_DOWN_DEG:.1f} 度；旋转停止"


def park_result(lines: list[str], finished: str | None, pitch: float | None = None) -> list[str]:
    """识别完成后，最下面提示停靠动作或到位停止。层数和颜色保留。"""
    if not finished:
        return lines
    body, status = split_status(lines)
    kept = status[:-1] if status else []
    return body + kept + [settled_look_down_line(pitch)]


def competition_lines(search, pitch, cap_aim, finished, fallback, valid):
    """比赛只展示当前角度、动作要求和可确认的结果。"""
    current = (f"当前俯仰：低头为正 {pitch:+.1f} 度"
               if valid and pitch is not None and np.isfinite(pitch) else "当前俯仰：无有效读数")
    if not valid:
        return [current, "停止俯仰和旋转，等待姿态恢复", finished or "识别结果：待确认"]
    if fallback.active:
        action = fallback.instruction + "；旋转停止"
        result = (f"识别完成 白盒层数 {fallback.layers}，没有顶块"
                  if fallback.layers is not None else "识别结果：待确认")
    elif finished:
        action, result = settled_look_down_line(pitch), finished
    else:
        guidance = (fallback.instruction if search.anchor is not None and fallback.instruction
                    else pitch_need(search, pitch))
        action = cap_aim or (guidance + "；旋转停止")
        result = "识别结果：待确认"
    return [current, action, result]


class RecognitionCycle:
    """本轮完成后返回低头；无顶块则在顶面层数确认后直接换轮。"""

    def __init__(self):
        self.cap_reported = False
        self.restart = False

    def update(self, finished, layers, pitch, valid, moving):
        self.cap_reported = self.cap_reported or bool(finished)
        if not valid or moving or pitch is None or not np.isfinite(pitch):
            return False
        self.restart = bool(layers is not None or (
            self.cap_reported and abs(pitch - _SETTLED_LOOK_DOWN_DEG) <= _PARK_TOLERANCE_DEG))
        return self.restart


def color_view_held(adjustment) -> bool:
    """朝上颜色的等待。已经保持，或者上沿出画还在抬头，都算。左右还要转、还在低头就不算。"""
    return bool(
        adjustment.visible
        and adjustment.yaw.direction == "none"
        and adjustment.pitch.direction in ("none", "up")
    )


def cap_aim_line(adjustment) -> str:
    """顶块留在画面里，还要再转多少。保持表示这个轴不用动。"""
    words = {"down": "低头", "up": "抬头", "right": "右转", "left": "左转"}

    def one(name: str, move) -> str:
        if move.direction == "none":
            return f"{name}保持"
        return f"{name} {words[move.direction]} {move.degrees:.1f} 度"

    if not adjustment.visible:
        return "顶块这一眼看不出在不在画面里"
    return f"顶块入画 {one('俯仰', adjustment.pitch)}，{one('旋转', adjustment.yaw)}"


def cap_aim_text(pixels, intr, gravity) -> str:
    """按看见的红蓝算还要不要转。没有红蓝，就说明这一眼看不出。"""
    if intr is None:
        return "顶块这一眼看不出在不在画面里"
    return cap_aim_line(keep_in_frame(pixels, intr, gravity))


def result_line(kind, world, yaw, color) -> str | None:
    """这座塔可以报出时的那一行。还没齐就什么都不写。"""
    if kind == "one":
        return "识别完成 层数 1 没有顶块"
    if kind == "two":
        return "识别完成 层数 2 没有顶块"
    if kind != "two_cap" or world is None or yaw is None or color not in ("red", "blue"):
        return None
    name = "红色" if color == "red" else "蓝色"
    return f"识别完成 层数 2 顶块{name} {_both_spots(world)}，{_turn(yaw)}"


def camera_turning_fast(frame, threshold_deg_s=25.0):
    samples = []
    for stamp, omega in (frame.gyro or []):
        value = np.asarray(omega, dtype=float)
        if value.shape == (3,) and np.isfinite(value).all() and np.isfinite(stamp):
            samples.append((float(stamp), float(np.linalg.norm(value))))
    if not samples:
        return False
    newest = max(t for t, _ in samples)
    return max(v for t, v in samples if newest - t <= .15) > np.deg2rad(threshold_deg_s)


class TowerWatch:
    """一座塔只报一次。整体失踪超时或下层挪走才认下一座。"""

    def __init__(self, hold_s: float = 0.5, jump_m: float = _JUMP_M):
        self.hold_s = float(hold_s)
        self.jump_m = float(jump_m)
        self.line: str | None = None
        self.anchor: np.ndarray | None = None
        self.miss_since: float | None = None
        self.reset()

    def reset(self) -> None:
        self.line = None
        self.anchor = None
        self.miss_since = None
        self.pending_anchor = None
        self.pending_count = 0
        self.pending_at = None
        self.status = "等待定位参照"

    def depart(self, now: float, search: StackSearch, bands: dict,
               moving: bool = False, tracking: bool = False) -> bool:
        if self.anchor is None and search.anchor is not None:
            self.anchor = np.array(search.anchor, dtype=float, copy=True).reshape(3)
        self.status = "参照保持" if self.anchor is not None else "等待定位参照"
        if moving:
            self.pending_anchor = None
            self.pending_count = 0
            self.pending_at = None
            self.miss_since = None
            self.status = "快速转动：暂停参照切换"
            self._hold_bands(search, bands)
            return False
        if self.anchor is None:
            return False
        if bands.get("lower") and bands.get("anchor") is not None:
            self.miss_since = None
            candidate = np.asarray(bands["anchor"], dtype=float).reshape(3)
            moved = float(np.linalg.norm(candidate - self.anchor))
            if moved > self.jump_m:
                consistent = (self.pending_anchor is not None and self.pending_at is not None
                              and now - self.pending_at <= 1.0
                              and np.linalg.norm(candidate - self.pending_anchor) <= .08)
                self.pending_count = self.pending_count + 1 if consistent else 1
                self.pending_anchor = candidate.copy()
                self.pending_at = now
                self.status = f"候选跳变待确认 {self.pending_count}/3（偏移 {moved:.2f} 米）"
                if self.pending_count >= 3:
                    self.status = "位置跳变连续确认：重新定位"
                    return True
                self._hold_bands(search, bands)
                return False
            self.pending_anchor = None
            self.pending_count = 0
            self.pending_at = None
            return False
        self.pending_anchor = None
        self.pending_count = 0
        self.pending_at = None
        if bands.get("cap") or bands.get("upper_white"):
            self.miss_since = None
            return False
        if self.line is None and not tracking:
            return False
        if self.miss_since is None:
            self.miss_since = now
        expired = now - self.miss_since > self.hold_s
        self.status = "超时清除参照" if expired else "目标暂时丢失"
        return expired

    @staticmethod
    def _hold_bands(search, bands):
        # Neither an unconfirmed base nor its associated upper layer may replace
        # the old search anchor through StackSearch.update on the same frame.
        bands["rejected_lower_uv"] = bands.get("lower_uv")
        bands["lower"] = False
        bands["upper_white"] = False
        bands["cap"] = False
        bands["lower_point"] = None
        bands["upper_point"] = None
        bands["cap_point"] = None
        bands["anchor"] = search.anchor
        bands["distance"] = search.range_m
        for key in ("lower_uv", "upper_uv", "cap_uv"):
            bands[key] = np.zeros((0, 2), dtype=int)

    def offer(self, search: StackSearch, world, yaw, color) -> str | None:
        if self.line is not None or not search.settled:
            return None
        text = result_line(search.kind, world, yaw, color)
        if text is None:
            return None
        self.line = text
        if self.anchor is None and search.anchor is not None:
            self.anchor = np.array(search.anchor, dtype=float, copy=True).reshape(3)
        return text


class BoxColorHold:
    """绿框稳住满 10 秒仍没有朝上颜色时，允许用上半区定色。上沿出画还在抬头也算稳住。定了就留着。"""

    def __init__(self, hold_s: float = _COLOR_HOLD_S, jump_m: float = _BOX_STABLE_M):
        self.hold_s = float(hold_s)
        self.jump_m = float(jump_m)
        self.since: float | None = None
        self.center: np.ndarray | None = None
        self.color: str | None = None
        self.world: np.ndarray | None = None
        self.yaw: float | None = None

    def reset(self) -> None:
        self.since = None
        self.center = None
        self.color = None
        self.world = None
        self.yaw = None

    @property
    def latched(self) -> bool:
        return self.color in ("red", "blue") and self.world is not None and self.yaw is not None

    def consider(self, now: float, world, view_held: bool, color_known: bool) -> bool:
        """到了可以用绿框上半区的时候。已经定过色就一直保持，直到这座塔清掉。"""
        if self.latched:
            return True
        if color_known or not view_held or world is None:
            self.since = None
            self.center = None
            return False
        center = np.asarray(world, dtype=float).reshape(3)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            self.since = None
            self.center = None
            return False
        if self.center is None or float(np.linalg.norm(center - self.center)) > self.jump_m:
            self.center = center.copy()
            self.since = now
            return False
        return self.since is not None and now - self.since >= self.hold_s

    def latch(self, color: str, world, yaw: float) -> None:
        self.color = color
        self.world = np.asarray(world, dtype=float).reshape(3).copy()
        self.yaw = float(yaw)


def box_leans(normal, gravity, limit_deg: float = _BOX_LEAN_DEG) -> bool:
    """朝上的方向和重力差过限度，绿框就和水平面不平行。"""
    up = -np.asarray(gravity, dtype=float).reshape(3)
    direction = np.asarray(normal, dtype=float).reshape(3)
    up_len = float(np.linalg.norm(up))
    direction_len = float(np.linalg.norm(direction))
    if up_len < 1e-8 or direction_len < 1e-8:
        return False
    cos = abs(float((direction / direction_len) @ (up / up_len)))
    return float(np.degrees(np.arccos(min(cos, 1.0)))) > float(limit_deg)


def face_leans(corners, gravity) -> bool:
    """底面四个角不落在水平面上，这块绿框就是斜的。"""
    if corners is None:
        return False
    base = np.asarray(corners, dtype=float).reshape(-1, 3)
    if len(base) < 4:
        return False
    normal = np.cross(base[1] - base[0], base[3] - base[0])
    return box_leans(normal, gravity)


def tower_ground(depth, color, intr, cfg, gravity, camera_height, memory: GroundMemory):
    """已知安装高度时固定高度；深度可修正法向，不能改写标定的光心高度。"""
    fix = _prior_ground(depth, color, intr, cfg, gravity, camera_height, memory)
    if fix is not None and camera_height is not None:
        fix = replace(fix, plane=Plane(fix.plane.normal.copy(), -float(camera_height)),
                      height=float(camera_height))
        memory.prior = fix
    return fix


def live_cap(measured, fused, outline, outlines: OutlineLive):
    """分界这一眼没量到体心，才用黄线推算。推算不进那 5 帧。"""
    if measured is not None or fused is not None:
        outlines.reset()
        return None
    return outlines.offer(outline)


def measure_cap(color: np.ndarray, depth: np.ndarray, intr, ground: Plane, cfg: dict, anchor=None):
    """塔顶按侧面的红蓝分界量。顶面在相机上方，不把顶面投到地面上，也不把白盒中心抬上去。"""
    pose, reason = measure_cap_pose(
        color,
        depth,
        intr,
        ground,
        float(cfg["depth_min"]),
        float(cfg["depth_max"]),
        anchor=anchor,
    )
    if pose is not None and not pose_matches_color(
        pose, color, intr, ground, _colored_mask(color), depth=depth,
        zmin=float(cfg["depth_min"]), zmax=float(cfg["depth_max"])
    ):
        return None, "projection"
    return pose, reason


def _blob(
    image: np.ndarray,
    uv: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dilate: int = 5,
) -> None:
    if uv is None or len(uv) < 10:
        return
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    ys = np.clip(uv[:, 1], 0, image.shape[0] - 1)
    xs = np.clip(uv[:, 0], 0, image.shape[1] - 1)
    mask[ys, xs] = 255
    if dilate > 1:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if draw.textlength(text, font=font) <= width:
        return [text]
    rows: list[str] = []
    buf = ""
    for char in text:
        trial = buf + char
        if buf and draw.textlength(trial, font=font) > width:
            rows.append(buf)
            buf = char
        else:
            buf = trial
    if buf:
        rows.append(buf)
    return rows


def light_names(cfg: dict) -> list[str]:
    levels = cfg.get("light_levels") or {}
    names = [name for name in _LIGHT_ORDER if name in levels]
    names.extend(str(name) for name in levels if str(name) not in names)
    return names


def starting_light(names: list[str], chosen: str | None) -> str | None:
    """测试画面没指定档位时用第 3 档。指定了就用指定的。"""
    if chosen:
        return chosen
    if len(names) >= 3:
        return names[2]
    return names[0] if names else None


def light_line(lock: dict, names: list[str]) -> str:
    # 画面上写当前档，并提示按数字换档。数字和档位顺序一致。
    current = _LIGHT_TEXT.get(str(lock.get("name")), str(lock.get("name")))
    choices = "  ".join(f"{index}{_LIGHT_TEXT.get(name, name)}" for index, name in enumerate(names, start=1))
    return f"光线 {current}，曝光 {float(lock['exposure']):.0f}。按 {choices}"


def _write(image: np.ndarray, lines: list[str], bottom: list[str] | None = None) -> None:
    # 小字直接压在画面上，黑边让白字还能看清，不再铺一整块黑底。
    # 上面是说明。层数、朝上颜色和俯仰变化贴在窗口下沿，认完后提示停靠和停止。
    font = ImageFont.truetype(_FONT, _FONT_SIZE)
    panel = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    limit = max(image.shape[1] - 16, 40)
    step = _FONT_SIZE + 4

    def rows(text: list[str]) -> list[str]:
        wrapped: list[str] = []
        for line in text:
            wrapped.extend(_wrap(draw, line, font, limit))
        return wrapped

    y = 6
    for line in rows(lines):
        draw.text((8, y), line, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        y += step
    parked = rows(bottom or [])
    if parked:
        y = image.shape[0] - 6 - step * len(parked)
        for line in parked:
            draw.text((8, y), line, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
            y += step
    image[:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def paint(
    color: np.ndarray,
    bands: dict,
    lines: list[str],
    intr,
    ground: Plane | None,
    cap_pose,
) -> np.ndarray:
    vis = color.copy()
    _blob(vis, bands.get("lower_uv"), LOWER_BGR)
    _blob(vis, bands.get("upper_uv"), UPPER_BGR)
    # 改用颜色时不画黄线。交界压在外轮廓上面才看得清。近处是白和橙，侧面量不成是浅蓝和浅红。
    if bands.get("rgb_body_uv") is None and bands.get("rgb_seam_uv") is None:
        # 候选轮廓不是物理边缘，单独用浅蓝显示，不能作为精测绿框的证明。
        _blob(vis, bands.get("cap_uv"), (255, 160, 0) if bands.get("cap_rgb_candidate") else CAP_BGR)
    if bands.get("rgb_style") == "side":
        body_bgr, seam_bgr = SIDE_BODY_BGR, SIDE_SEAM_BGR
    else:
        body_bgr, seam_bgr = BODY_BGR, SEAM_BGR
    _blob(vis, bands.get("rgb_body_uv"), body_bgr, thickness=3, dilate=3)
    _blob(vis, bands.get("rgb_seam_uv"), seam_bgr, thickness=2, dilate=3)
    if ground is not None and cap_pose is not None:
        pose, center = cap_pose
        if pose.corners is not None:
            wire = cube_wire_corners(pose.corners, ground.normal, _CAP_SIZE)
            draw_cube_edges(vis, wire, intr, POSE_BGR)
            draw_center(vis, center, intr, POSE_BGR)
    body, status = split_status(lines)
    _write(vis, body, status)
    return vis


def frame_attitude_valid(frame) -> bool:
    gravity = frame.gravity
    return (gravity is not None and np.asarray(gravity).shape == (3,)
            and np.isfinite(gravity).all() and np.linalg.norm(gravity) > 1e-6
            and getattr(frame, "gravity_status", "unspecified") not in ("missing", "stale"))


def attitude_line(frame) -> str:
    age = getattr(frame, "gravity_age_s", None)
    if frame_attitude_valid(frame):
        return "姿态有效" + ("" if age is None else f"，加速度样本 {age * 1000:.0f} 毫秒前")
    if getattr(frame, "gravity_status", None) == "stale":
        elapsed = "" if age is None else f" {age:.2f} 秒"
        return f"姿态失效：加速度已过期{elapsed}，暂停高度筛选和位姿输出"
    return "姿态失效：未收到有效加速度，暂停高度筛选和位姿输出"


def main() -> None:
    parser = argparse.ArgumentParser(description="实时看塔的层数")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--light", default=None)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--competition", action="store_true", help="比赛简洁窗口：只显示俯仰、动作要求和识别结果")
    parser.add_argument("--diagnostics", nargs="?", const="auto", default=None,
                        metavar="DIR", help="保存原始帧和诊断信息；省略 DIR 时保存到 /tmp")
    parser.add_argument("--diagnostic-seconds", type=float, default=20.0,
                        help="诊断采集时长，默认 20 秒；结束后识别继续")
    args = parser.parse_args()
    if not np.isfinite(args.diagnostic_seconds) or args.diagnostic_seconds <= 0:
        parser.error("--diagnostic-seconds 必须是大于 0 的有限数")
    cfg = load_config(args.config)
    names = light_names(cfg)
    lock = color_lock_from_config(cfg, starting_light(names, args.light))
    intr = load_intrinsics("config/intrinsics.yaml")
    camera_height = selected_camera_height(args, cfg)
    diagnostics = None
    if args.diagnostics is not None:
        from tower_diagnostics import DiagnosticRecorder
        directory = args.diagnostics
        if directory == "auto":
            directory = "/tmp/robocon-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        diagnostics = DiagnosticRecorder(directory, args.diagnostic_seconds, cfg, intr,
                                         camera_height, Path(__file__).resolve().parent)
    source = RealSenseSource(
        intr.width,
        intr.height,
        fps=int(cfg.get("fps", 30)),
        color_lock=lock,
        depth_width=int(cfg.get("depth_width", 848)),
        depth_height=int(cfg.get("depth_height", 480)),
    )
    memory = GroundMemory()
    search = StackSearch()
    fallback = TopFallback(timeout=float(cfg.get("top_fallback_seconds", 15.0)))
    need = int(cfg.get("min_samples", 5))
    window = int(cfg.get("frames_to_fuse", 15))
    caps = CapFusion(need=need, window=window)
    outlines = OutlineLive()
    watch = TowerWatch(hold_s=float(cfg.get("hold_s", 0.5)))
    color_hold = BoxColorHold()
    last = time.monotonic()
    fps = 0.0
    attitude_was_valid = False
    cycle = RecognitionCycle()
    debug_window = DebugWindow()
    debug_window.visible = not args.competition
    debug_frame = 0
    debug_start = time.monotonic()
    try:
        while True:
            if cycle.restart:
                search = StackSearch()
                fallback = TopFallback(timeout=float(cfg.get("top_fallback_seconds", 15.0)))
                caps = CapFusion(need=need, window=window)
                outlines.reset()
                watch.reset()
                color_hold.reset()
                memory = GroundMemory()
                cycle = RecognitionCycle()
            frame = source.read()
            receive_time = time.monotonic()
            capturing = diagnostics is not None and diagnostics.begin()
            if capturing:
                diagnostics.read_at = receive_time
                search_before = copy.deepcopy(search)
            reset_reason = None
            attitude_valid = frame_attitude_valid(frame)
            if not attitude_valid and attitude_was_valid:
                caps = CapFusion(need=need, window=window)
                outlines.reset()
                color_hold.reset()
                memory = GroundMemory()
                fallback = TopFallback(timeout=float(cfg.get("top_fallback_seconds", 15.0)))
                cycle = RecognitionCycle()
                reset_reason = "attitude_invalid"
            attitude_was_valid = attitude_valid
            cleaned = clean_frame(frame.color, frame.depth, frame.gravity if attitude_valid else None, cfg)
            gravity = cleaned.gravity if cleaned.gravity is not None else np.array([0.0, 1.0, 0.0])
            height = camera_height if attitude_valid else None
            fix = None if not attitude_valid else tower_ground(
                cleaned.depth,
                cleaned.color,
                intr,
                cfg,
                gravity,
                height,
                memory,
            )
            plane = None if fix is None else fix.plane
            pitch = pitch_down_degrees(cleaned.gravity)
            rotation = search.reorient(cleaned.gravity)
            if watch.anchor is not None:
                watch.anchor = rotation @ watch.anchor
            if watch.pending_anchor is not None:
                watch.pending_anchor = rotation @ watch.pending_anchor
            relax = relax_upper_white(search.kind, search.settled, pitch)
            bands = coarse_bands(
                cleaned.color,
                cleaned.depth,
                plane,
                intr,
                cfg["colors"],
                float(cfg["depth_min"]),
                float(cfg["depth_max"]),
                search.anchor,
                relax_upper=relax,
                pitch=pitch,
            )
            now = time.monotonic()
            if attitude_valid and not cycle.cap_reported and not fallback.active and watch.depart(now, search, bands, moving=camera_turning_fast(frame), tracking=fallback.active):
                cycle = RecognitionCycle()
                reset_reason = "lower_anchor_moved" if bands.get("lower") and bands.get("anchor") is not None else "target_missing_timeout"
                confirmed_status = watch.status
                search = StackSearch()
                fallback = TopFallback(timeout=float(cfg.get("top_fallback_seconds", 15.0)))
                caps = CapFusion(need=need, window=window)
                outlines.reset()
                watch.reset()
                watch.status = confirmed_status
                color_hold.reset()
                bands = coarse_bands(
                    cleaned.color,
                    cleaned.depth,
                    plane,
                    intr,
                    cfg["colors"],
                    float(cfg["depth_min"]),
                    float(cfg["depth_max"]),
                    None,
                )
            if attitude_valid:
                # 普通高度带只供定位和找塔顶，白盒层数留到返回低头后的顶面检查。
                cap_search = dict(bands, upper_white=False, covers=False)
                search.update([], cap_search, bands.get("distance"), float(cfg["cube_size"]), pitch, cleaned.gravity)
            if not fallback.cap_confirmed and (bands.get("cap") or bands.get("cap_rgb_candidate")):
                search.kind = "two_cap"
            cap_draw = None
            cap_reason = None
            cap_yaw = None
            cap_world = None
            cap_waiting = None
            cap_aim = None
            measured = None
            ready = None
            fused = None
            cap_tint = None
            leans = plane is not None and box_leans(plane.normal, gravity)
            if plane is not None and search.kind == "two_cap" and not leans:
                measured, cap_reason = measure_cap(
                    cleaned.color, cleaned.depth, intr, plane, cfg, search.anchor
                )
                if measured is not None:
                    tint = cap_top_color(
                        cleaned.color,
                        cleaned.depth,
                        intr,
                        plane,
                        float(cfg["depth_min"]),
                        float(cfg["depth_max"]),
                        search.anchor,
                    )
                    if tint is None:
                        caps.latest = measured
                        if caps.result() is None:
                            cap_reason = "color"
                    else:
                        cap_tint = tint
                        caps.add(measured, tint)
            if plane is not None and search.kind == "two_cap":
                fused = caps.result()
                ready = caps.consensus()
                if fused is not None:
                    cap_world = np.array([fused.x, fused.y, fused.z])
                    cap_yaw = float(fused.yaw)
                    cap_reason = None
                elif caps.count():
                    cap_waiting = caps.count()
                if not leans and caps.latest is not None:
                    cap_draw = (caps.latest, caps.latest.center)
                elif leans and fused is None:
                    cap_reason = "tilt"
            outline = None
            if (
                plane is not None
                and search.kind == "two_cap"
                and measured is None
                and fused is None
                and not bands.get("cap_rgb_candidate", False)
                and not leans
            ):
                outline = measure_outline(
                    bands.get("cap_uv"),
                    cleaned.depth,
                    intr,
                    plane,
                    float(cfg["depth_min"]),
                    float(cfg["depth_max"]),
                )
            if outline is not None and not pose_matches_color(
                outline, cleaned.color, intr, plane, _colored_mask(cleaned.color),
                depth=cleaned.depth, zmin=float(cfg["depth_min"]), zmax=float(cfg["depth_max"])
            ):
                outline = None
                cap_reason = "projection"
            live = live_cap(measured, fused, outline, outlines)
            if live is not None:
                cap_draw = (live, live.center)
                cap_world = np.asarray(live.world, dtype=float)
                cap_yaw = float(live.yaw)
                cap_reason = None
                cap_waiting = None
                cap_tint = cap_top_color(
                    cleaned.color,
                    cleaned.depth,
                    intr,
                    plane,
                    float(cfg["depth_min"]),
                    float(cfg["depth_max"]),
                    search.anchor,
                )
            draw_bands = bands
            cap_rgb: bool | str = False
            aim_uv = bands.get("cap_uv")
            rgb_style = None
            if plane is not None and search.kind == "two_cap" and close_depth_unusable(
                cleaned.color,
                cleaned.depth,
                intr,
                plane,
                float(cfg["depth_min"]),
                float(cfg["depth_max"]),
            ):
                rgb_style = "close"
            elif measured is not None and measured.via == "side":
                rgb_style = "side"
            if rgb_style is not None:
                # 近处深度失效用白线和橙线。侧面深度量不成用浅蓝和浅红。黄线都不画。
                seam = color_seam_pixels(cleaned.color)
                body = color_body_pixels(cleaned.color)
                if len(seam) >= 16 or len(body) >= 16:
                    cap_rgb = "side" if rgb_style == "side" else True
                    draw_bands = dict(bands)
                    draw_bands["rgb_style"] = rgb_style
                    draw_bands["rgb_seam_uv"] = seam
                    draw_bands["rgb_body_uv"] = body
                    aim_uv = body if len(body) >= 16 else seam
            view_held = False
            if search.kind == "two_cap":
                # 看看见的红蓝贴没贴到画面边。已经在画面里就保持，不转到中心。
                adjustment = keep_in_frame(aim_uv, intr, gravity)
                cap_aim = cap_aim_line(adjustment)
                view_held = color_view_held(adjustment)
            elif not color_hold.latched:
                color_hold.reset()
            if cap_draw is not None and (
                leans or face_leans(cap_draw[0].corners, gravity)
            ):
                cap_draw = None
                if cap_world is None:
                    cap_reason = "tilt"
            # 历史 latest 也要与当前目标核对，避免失败帧继续画出错位旧框。
            if (
                cap_draw is not None
                and cap_draw[0] is not measured
                and cap_draw[0] is not outline
                and not pose_matches_color(
                    cap_draw[0], cleaned.color, intr, plane, _colored_mask(cleaned.color),
                    depth=cleaned.depth, zmin=float(cfg["depth_min"]), zmax=float(cfg["depth_max"])
                )
            ):
                cap_draw = None
                cap_reason = "projection"
            candidate_unmeasured = bool(bands.get("cap_rgb_candidate")) and measured is None
            if candidate_unmeasured:
                # 纯可见候选不补造位姿，也不把历史 latest 当成本帧精测画出。
                cap_draw = None
            box_pose = None if cap_draw is None else cap_draw[0]
            box_world = None if box_pose is None else np.asarray(box_pose.world, dtype=float)
            box_yaw = None if box_pose is None else float(box_pose.yaw)
            final_world = None
            final_yaw = None
            final_color = None
            if ready is not None:
                pose, final_color = ready
                final_world = np.array([pose.x, pose.y, pose.z])
                final_yaw = float(pose.yaw)
            elif color_hold.consider(now, box_world, view_held, cap_tint is not None):
                tint = color_hold.color
                if (
                    tint is None
                    and plane is not None
                    and box_pose is not None
                    and box_pose.corners is not None
                    and box_world is not None
                    and box_yaw is not None
                ):
                    wire = cube_wire_corners(box_pose.corners, plane.normal, _CAP_SIZE)
                    tint = box_upper_color(cleaned.color, wire, intr)
                    if tint is not None:
                        color_hold.latch(tint, box_world, box_yaw)
                if color_hold.latched:
                    cap_tint = color_hold.color
                    final_color = color_hold.color
                    final_world = np.array(color_hold.world, dtype=float)
                    final_yaw = float(color_hold.yaw)
                    cap_reason = None
                    cap_world = final_world
                    cap_yaw = final_yaw
            was_fallback = fallback.active
            previous_layers = fallback.layers
            if not watch.line:
                fallback.update(now, search.anchor, final_world is not None, attitude_valid,
                                camera_turning_fast(frame), pitch, cleaned.color, cleaned.depth,
                                intr, plane, cfg,
                                cap_visible=bool(bands.get("cap") or bands.get("cap_rgb_candidate")))
            if fallback.active:
                if not was_fallback:
                    watch.line = None
                if fallback.layers is not None and fallback.layers != previous_layers:
                    print(f"识别完成 白盒层数 {fallback.layers}，没有顶块", flush=True)
                # 白盒数与塔顶独立；重新看到红蓝时恢复原有精测路径。
                if fallback.layers == 2 and (bands.get("cap") or bands.get("cap_rgb_candidate")):
                    search.kind = "two_cap"
            spoken = watch.offer(search, final_world, final_yaw, final_color) if attitude_valid and not candidate_unmeasured and not fallback.active and final_world is not None else None
            if spoken:
                print(spoken, flush=True)
            cycle.update(watch.line, fallback.layers, pitch, attitude_valid, camera_turning_fast(frame))
            lines = describe(
                search,
                bands,
                pitch,
                plane,
                None,
                cap_yaw,
                cap_reason,
                cap_world,
                cap_waiting,
                cap_aim,
                cap_outline=live is not None,
                cap_color=final_color or cap_tint,
                cap_rgb=cap_rgb,
                cap_unmeasured=candidate_unmeasured,
            )
            lines = fallback.display(lines)
            lines = park_result(lines, watch.line if attitude_valid and not fallback.active else None, pitch)
            if fix is None:
                lines.insert(1, "还没看到地面")
            else:
                above = float(fix.height) if fix.height is not None else -float(fix.plane.offset)
                lines.insert(1, f"相机离地大约 {above:.2f} 米")
            lines.insert(1, f"每秒 {fps:.0f} 帧")
            lines.insert(2, light_line(lock, names))
            lines.insert(0, "定位状态：" + watch.status)
            lines.insert(0, attitude_line(frame))
            if diagnostics is not None:
                diagnostics.stop_trace()
                if capturing:
                    draw_source = "none"
                    if cap_draw is not None:
                        draw_source = "outline" if live is not None else (
                            "measured_latest" if measured is not None and cap_draw[0] is measured
                            else "historical_latest")
                    diagnostic_state = {
                        "top_fallback": {"active": fallback.active, "layers": fallback.layers,
                                         "target_pitch": fallback.target_pitch, "evidence": fallback.evidence,
                                         "instruction": fallback.instruction},
                        "search_before": search_before, "search_after": search,
                        "bands": bands, "ground": plane, "pitch": pitch,
                        "reset_reason": reset_reason, "anchor_status": watch.status,
                        "anchor_pending_count": watch.pending_count, "light": lock,
                        "attitude_valid": attitude_valid,
                        "candidate_unmeasured": candidate_unmeasured,
                        "measured": measured, "measured_via": None if measured is None else measured.via,
                        "cap_reason": cap_reason, "outline": outline, "live": live,
                        "fused": fused, "fusion_count": caps.count(),
                        "draw_source": draw_source, "drawn_pose": box_pose,
                        "final_world": final_world, "final_yaw": final_yaw,
                        "final_color": final_color, "reported_line": watch.line,
                        "screen_lines": lines.copy(),
                    }
                    diagnostics.save(frame, cleaned, diagnostic_state)
                    lines[0:0] = diagnostics.lines(diagnostic_state)
                elif diagnostics.finished or diagnostics.error:
                    lines.insert(0, f"诊断采集{'失败' if diagnostics.error else '结束'}，已保存 {diagnostics.index} 帧")
            if args.competition:
                lines = competition_lines(search, pitch, cap_aim, watch.line, fallback, attitude_valid)
                vis = frame.color.copy()
                _write(vis, lines)
            else:
                vis = paint(frame.color, draw_bands, lines, intr, plane, cap_draw)
            rejected = bands.get("rejected_lower_uv")
            if not args.competition and rejected is not None and len(rejected):
                mask = np.zeros(vis.shape[:2], np.uint8)
                mask[rejected[:, 1], rejected[:, 0]] = 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (0, 140, 255), 2)
            draw_source = "none"
            if cap_draw is not None:
                draw_source = "outline" if live is not None else (
                    "measured_latest" if measured is not None and cap_draw[0] is measured
                    else "historical_latest")
            debug_window.update(cleaned.color, cleaned.depth, intr,
                                float(cfg["depth_min"]), float(cfg["depth_max"]),
                                bands, vis, _write,
                                {"frame": debug_frame, "elapsed_s": time.monotonic() - debug_start,
                                 "draw_source": draw_source,
                                 "via": None if measured is None else measured.via,
                                 "reason": cap_reason, "reset_reason": reset_reason,
                                 "anchor_status": watch.status,
                                 "light": light_line(lock, names)})
            debug_frame += 1
            cv2.imshow("塔的层数", vis)
            key = cv2.waitKey(1) & 0xFF
            if not args.competition:
                debug_window.key(key)
            if key in (ord("q"), 27):
                break
            if ord("1") <= key < ord("1") + len(names):
                lock = color_lock_from_config(cfg, names[key - ord("1")])
                source.set_color_lock(lock)
            fps = 1.0 / max(now - last, 1e-3)
            last = now
    finally:
        if diagnostics is not None:
            diagnostics.stop_trace()
            print(f"诊断已保存 {diagnostics.index} 帧：{diagnostics.path.resolve()}", flush=True)
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
