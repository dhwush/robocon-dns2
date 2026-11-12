"""实时查看地面白盒。彩色画面保持原分辨率。

下层用青色棱线，上层用品红棱线，框旁边标下层或上层。一层从地面抬起。上面那块用自己的位置，底边抬高一个边长。
橙色线是这块地面在画面里的一圈范围。白盒把地面隔开时，左右和远处仍算同一块。左上角白字写这一帧的参数。
按 q 或 Esc 退出。

    python view_cube_center.py
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from camera.source import RealSenseSource, color_lock_from_config
from cube.aim import plan_view
from cube.outline import OutlineTrack
from cube.stack_look import StackSearch, coarse_bands, pitch_down_degrees, record_distance
from geom.transform import deproject_pixels, load_intrinsics, project_points
from ground.plane import Plane
from main import (
    GroundMemory,
    clean_frame,
    estimate_layers,
    ground_caption,
    ground_for_layers,
    load_config,
    selected_camera_height,
)


# 地面橙。下层青，上层品红，两层叠在一起也能分开。
GROUND_BGR = (0, 140, 255)
CUBE_BGR = (255, 255, 0)
UPPER_BGR = (255, 0, 255)
_CUBE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _pixel(point: np.ndarray, intr) -> tuple[int, int] | None:
    if float(point[2]) <= 1e-4:
        return None
    uv = project_points(np.asarray(point, dtype=float), intr)[0]
    if not np.isfinite(uv).all():
        return None
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def draw_ground(image: np.ndarray, depth: np.ndarray, intr, plane: Plane, thresh: float = 0.018) -> None:
    # 只画贴地的最大一块。方块高出地面，不进这条线。左右被挡住时不把方块下沿连进来。
    stride = 8
    draw_thresh = max(float(thresh), 0.018)
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[vs, us]
    good = np.isfinite(z)
    if int(good.sum()) < 30:
        return
    points = deproject_pixels(
        np.column_stack([us[good].ravel(), vs[good].ravel()]),
        z[good].ravel(),
        intr,
    )
    on = np.abs(plane.signed_distance(points)) < draw_thresh
    if int(on.sum()) < 30:
        return
    mask = np.zeros(depth.shape[:2], dtype=np.uint8)
    mask[vs[good][on], us[good][on]] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stride * 2 + 1, stride * 2 + 1))
    mask = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest], -1, GROUND_BGR, 2, cv2.LINE_AA)


def cube_wire_corners(bottom: np.ndarray, normal: np.ndarray, cube_size: float) -> np.ndarray:
    # 前四个是地面上的足迹，后四个是沿法向抬起一个边长的顶面。
    base = np.asarray(bottom, dtype=float)
    if base.shape != (4, 3):
        base = base.reshape(4, 3)
    top = base + np.asarray(normal, dtype=float) * float(cube_size)
    return np.vstack([base, top])


def _in_front(a: np.ndarray, b: np.ndarray, near: float = 0.05) -> tuple[np.ndarray, np.ndarray] | None:
    # 一头伸到相机背后时，只留下画面这一侧，棱线才不会整条消失。
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a[2] < near and b[2] < near:
        return None
    if a[2] >= near and b[2] >= near:
        return a, b
    far = a if a[2] >= near else b
    close = b if a[2] >= near else a
    span = float(far[2] - close[2])
    if abs(span) < 1e-6:
        return None
    close = close + ((near - close[2]) / span) * (far - close)
    if a[2] < near:
        return close, far
    return far, close


def draw_cube_edges(image: np.ndarray, corners: np.ndarray, intr, color: tuple[int, int, int] = CUBE_BGR) -> int:
    # 十二条棱都画。完全在相机背后的棱跳过。返回实际画上的条数。
    height, width = image.shape[:2]
    drawn = 0
    for i, j in _CUBE_EDGES:
        segment = _in_front(corners[i], corners[j])
        if segment is None:
            continue
        pix = [_pixel(point, intr) for point in segment]
        if pix[0] is None or pix[1] is None:
            continue
        ok, start, end = cv2.clipLine((0, 0, width, height), pix[0], pix[1])
        if not ok:
            continue
        cv2.line(image, start, end, color, 3, cv2.LINE_AA)
        drawn += 1
    return drawn


def draw_center(image: np.ndarray, point: np.ndarray, intr, color: tuple[int, int, int] = CUBE_BGR) -> None:
    pix = _pixel(point, intr)
    if pix is None:
        return
    height, width = image.shape[:2]
    if not (0 <= pix[0] < width and 0 <= pix[1] < height):
        return
    cv2.drawMarker(image, pix, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)


def draw_cube_name(image: np.ndarray, point: np.ndarray, intr, text: str, color: tuple[int, int, int]) -> None:
    # 名字标在这块自己的框旁边，不和左上角的读数混在一起。
    pix = _pixel(point, intr)
    if pix is None:
        return
    height, width = image.shape[:2]
    x = int(np.clip(pix[0] + 10, 8, max(width - 72, 8)))
    y = int(np.clip(pix[1] - 10, 22, max(height - 8, 22)))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_labels(image: np.ndarray, lines: list[str]) -> None:
    # 白字黑边，放在左上角。白盒本身很亮，黑边让字还能看清。
    y = 28
    for line in lines:
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


_LIGHT_TEXT = {
    "bright": "偏亮",
    "daylight": "白天",
    "indoor": "室内",
    "dim": "偏暗",
    "dark": "很暗",
}


def _place(value: float | None, positive: str, negative: str) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 0.005:
        return "正中"
    word = positive if value > 0.0 else negative
    return f"{word} {abs(value):.2f} 米"


def _turn_words(yaw: float) -> str:
    deg = float(np.degrees(yaw))
    if abs(deg) < 2.0:
        return "转角基本摆正"
    if deg > 0.0:
        return f"转角 正 {deg:.0f} 度"
    return f"转角 负 {abs(deg):.0f} 度"


def _as_records(record: dict | list | None) -> list[dict]:
    if record is None:
        return []
    if isinstance(record, dict):
        return [record]
    return [item for item in record if item is not None]


def overlay(
    color: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane | None,
    record: dict | list | None,
    cube_size: float,
    lines: list[str],
    ground_thresh: float = 0.018,
) -> tuple[np.ndarray, int]:
    vis = color.copy()
    if plane is not None:
        draw_ground(vis, depth, intr, plane, ground_thresh)
    drawn = 0
    items = [item for item in _as_records(record) if item.get("corners") is not None]
    several = sum(int(item.get("layer", 1)) > 1 for item in items) > 0 and len(items) > 1
    if plane is not None:
        for item in sorted(items, key=lambda one: int(one.get("layer", 1))):
            level = int(item.get("layer", 1))
            ink = UPPER_BGR if level > 1 else CUBE_BGR
            base = np.asarray(item["corners"], dtype=float)
            if level > 1:
                base = base + plane.normal * ((level - 1) * float(cube_size))
            wire = cube_wire_corners(base, plane.normal, cube_size)
            drawn += draw_cube_edges(vis, wire, intr, ink)
            center = item["bottom"] + plane.normal * ((level - 0.5) * float(cube_size))
            draw_center(vis, center, intr, ink)
            if several:
                draw_cube_name(vis, center, intr, "上层" if level > 1 else "下层", ink)
    draw_labels(vis, lines)
    return vis, drawn


# 测试画面上的未认出原因。一句说清这一帧卡在哪。
_REJECT_TEXT = {
    "ground": "没有地面",
    "world_up": "地面不是朝上的",
    "mask": "没找到成块的白",
    "depth": "白块上有深度的点太少",
    "height_band": "白点的高度不像 35 厘米的方块",
    "body": "看不见盒身",
    "peak": "最高处对不上 35 厘米",
    "cube_height": "下层没有贴在地面上",
    "top": "顶面的点太少",
    "ground_ring": "方块旁边的地面对不上",
    "footprint_split": "两种位置差超过 2 厘米",
    "footprint_long": "顶面边长不像 35 厘米的方块",
    "miss": "没认出",
}


def reject_text(reason: str | None) -> str:
    return _REJECT_TEXT.get(reason or "miss", "没认出")


def _cube_lines(name: str, record: dict) -> list[str]:
    return [
        name,
        f"{_place(record['x'], '向右', '向左')}，{_place(record['y'], '向下', '向上')}，{_place(record['z'], '向前', '向后')}",
        _turn_words(record["yaw"]),
        f"长边 {record['long_side']:.2f} 米，短边 {record['short_side']:.2f} 米",
    ]


def aim_line(adjustment) -> str:
    # 这一帧还要转多少。保持表示这个轴不用动。
    words = {"down": "低头", "up": "抬头", "right": "右转", "left": "左转"}

    def one(name: str, move) -> str:
        if move.direction == "none":
            return f"{name} 保持"
        return f"{name} {words[move.direction]} {move.degrees:.1f}°"

    return f"{one('俯仰', adjustment.pitch)}  {one('旋转', adjustment.yaw)}"


def yaw_only_line(adjustment) -> str | None:
    # 抬头看上层时，俯仰另说。左右偏了仍用白话写要转多少。
    words = {"right": "右转", "left": "左转"}
    move = adjustment.yaw
    if move.direction == "none":
        return None
    return f"左右还要{words[move.direction]} {move.degrees:.0f} 度"


def _lines(
    fps: float,
    lock: dict,
    ground: str,
    shown: list[dict],
    live: list[dict],
    reason: str | None,
    watch: str,
    pending: bool = False,
) -> list[str]:
    light = _LIGHT_TEXT.get(lock["name"], lock["name"])
    rows = [
        f"每秒 {fps:.0f} 帧，{light}，曝光 {lock['exposure']:.0f}，增益 {lock['gain']:.0f}",
        ground,
        watch,
    ]
    # 层数还没定时只留正在看下面或正在看上面，不写白盒体心。
    if pending or not shown:
        settled = watch.startswith("一层") or watch.startswith("两层")
        if not pending and not shown and not settled:
            rows.append(f"还没认出：{reject_text(reason)}")
        return rows
    live_layers = {int(item.get("layer", 1)) for item in live}
    several = len(shown) > 1
    for item in shown:
        level = int(item.get("layer", 1))
        if level == 1:
            name = "下层" if several else "这一层"
            if several and level not in live_layers:
                name = "下层，低头那一眼"
        else:
            name = "上层" if level in live_layers else "上层，抬头那一眼"
        rows.extend(_cube_lines(name, item))
    return rows


def measure_view(frame, depth: np.ndarray, intr, cfg: dict, memory: GroundMemory, search: StackSearch, camera_height: float | None):
    gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 1.0, 0.0])
    height = camera_height if frame.gravity is not None else None
    fix, stacked = ground_for_layers(depth, frame.color, intr, cfg, gravity, height, memory, search)
    plane = None if fix is None else fix.plane
    rejects: dict[str, int] = {}
    records: list[dict] = []
    up_ok = plane is not None and float(plane.normal @ (-gravity / np.linalg.norm(gravity))) >= 0.95
    if plane is None:
        rejects["ground"] = 1
    elif not up_ok:
        rejects["world_up"] = 1
    else:
        records = estimate_layers(
            frame,
            intr,
            cfg,
            rejects=rejects,
            fix=fix,
            camera_height=height,
        )
    reason = None if records else next(iter(rejects), "miss")
    return fix, records, ground_caption(fix, stacked), reason


def main() -> None:
    parser = argparse.ArgumentParser(description="查看地面白盒")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--light", default=None)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--on-platform", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    lock = color_lock_from_config(cfg, args.light)
    intr = load_intrinsics("config/intrinsics.yaml")
    source = RealSenseSource(
        intr.width,
        intr.height,
        fps=int(cfg.get("fps", 30)),
        color_lock=lock,
        depth_width=int(cfg.get("depth_width", 848)),
        depth_height=int(cfg.get("depth_height", 480)),
    )
    camera_height = selected_camera_height(args, cfg)
    memory = GroundMemory()
    search = StackSearch()
    outlines = OutlineTrack(
        window=int(cfg.get("frames_to_fuse", 15)),
        need=int(cfg.get("min_samples", 5)),
        jump=float(cfg.get("jump_m", 0.03)),
    )
    last = time.monotonic()
    fps = 0.0
    cube_size = float(cfg["cube_size"])
    try:
        while True:
            frame = source.read()
            cleaned = clean_frame(frame.color, frame.depth, frame.gravity, cfg)
            fix, records, ground, reason = measure_view(
                cleaned, cleaned.depth, intr, cfg, memory, search, camera_height
            )
            search.reorient(cleaned.gravity)
            bands = coarse_bands(
                cleaned.color,
                cleaned.depth,
                None if fix is None else fix.plane,
                intr,
                cfg["colors"],
                float(cfg["depth_min"]),
                float(cfg["depth_max"]),
                search.anchor,
            )
            lower_now = next((item for item in records if int(item.get("layer", 1)) == 1), None)
            near = record_distance(lower_now)
            pitch = pitch_down_degrees(cleaned.gravity)
            watch = search.update(
                records,
                bands,
                near if near is not None else bands.get("distance"),
                cube_size,
                pitch,
                cleaned.gravity,
            )
            drawn_records = []
            shown = []
            if search.settled:
                drawn_records = outlines.update(
                    records,
                    search.stage,
                    None if fix is None else fix.plane,
                    cube_size,
                )
                for item in search.remembered:
                    match = next(
                        (one for one in drawn_records if int(one.get("layer", 1)) == int(item.get("layer", 1))),
                        None,
                    )
                    shown.append(match if match is not None else item)
                if not shown:
                    shown = drawn_records
            adjustment = plan_view(
                cleaned.color,
                cleaned.depth,
                intr,
                cfg["colors"],
                cube_size=cube_size,
                gravity=cleaned.gravity,
                zmin=float(cfg["depth_min"]),
                zmax=float(cfg["depth_max"]),
            )
            lines = _lines(
                fps,
                lock,
                ground,
                shown,
                records,
                reason,
                watch,
                pending=not search.settled,
            )
            if search.settled and records and not drawn_records:
                lines.append("方块框还在对齐")
            # 还在抬头时，俯仰由层数这句说，左右偏了仍写出来。
            if search.asking_to_look_up():
                turn = yaw_only_line(adjustment)
                if turn is not None:
                    lines.append(turn)
            else:
                lines.append(aim_line(adjustment))
            vis, _drawn = overlay(
                frame.color,
                cleaned.depth,
                intr,
                None if fix is None else fix.plane,
                drawn_records,
                cube_size,
                lines,
                float(cfg["ground_ransac_thresh"]),
            )
            cv2.imshow("white cube", vis)
            now = time.monotonic()
            fps = 1.0 / max(now - last, 1e-3)
            last = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
