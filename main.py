"""地面白盒位姿。外参是单位阵，体心相对光心：向右 x、向下 y、向前 z。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from camera.source import Frame, RealSenseSource, color_lock_from_config, filter_depth
from cube.aim import plan_view
from cube.detect import detect_white_layers, white_keepout
from cube.stack_look import StackSearch, coarse_bands, pitch_down_degrees, record_distance, second_layer_cue
from fuse.median import fuse_poses
from geom.transform import fold_yaw_quarter, load_intrinsics
from ground.plane import (
    GroundFix,
    Plane,
    fit_ground,
    grid_normals,
    ground_with_prior,
    plane_holds,
    sample_grid,
    sample_points,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    hsv_path = Path(cfg.get("hsv_path", "config/hsv.yaml"))
    if hsv_path.exists():
        with open(hsv_path, "r", encoding="utf-8") as handle:
            saved = yaml.safe_load(handle) or {}
        if saved.get("colors"):
            cfg["colors"] = saved["colors"]
    return cfg


def clean_frame(color: np.ndarray, depth: np.ndarray, gravity: np.ndarray | None, cfg: dict) -> Frame:
    cleaned = filter_depth(
        depth,
        zmin=float(cfg["depth_min"]),
        zmax=float(cfg["depth_max"]),
        flying_pixel_m=float(cfg["flying_pixel_m"]),
    )
    return Frame(color, cleaned, gravity)


def ground_cloud(
    depth: np.ndarray,
    intr,
    cfg: dict,
    color: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # 有彩色时先挖掉白盒和周围一圈。盒子下沿不再进地面点。
    # 剩下的地面可以在画面里是分开的，仍拿去拟合同一块平面。够宽才修法向，窄条只修高度。
    stride = int(cfg.get("ground_stride", 8))
    grid = sample_grid(depth, intr, stride)
    points = grid.reshape(-1, 3)
    normals = grid_normals(grid).reshape(-1, 3)
    keep = np.all(np.isfinite(points), axis=1)
    if color is not None:
        drop = white_keepout(color, cfg["colors"], depth, intr)
        vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        keep &= ~drop[vs, us].reshape(-1)
    return points[keep], normals[keep]


@dataclass
class GroundMemory:
    """一层和两层各记自己的上一块地面，互不覆盖。"""

    prior: GroundFix | None = None
    floor: Plane | None = None


def _known_stack(search: StackSearch) -> bool:
    # 已经看见上面那块，或正在抬头看它。确认上面没有第二块之后，回到一层。
    if search.settled_one:
        return False
    if search.upper is not None or search.stage in ("upstairs", "both"):
        return True
    return bool(search.cue and search.lower is not None)


def _second_layer(color: np.ndarray, depth: np.ndarray, plane: Plane | None, intr, cfg: dict) -> bool:
    cue, _distance = second_layer_cue(
        color,
        depth,
        plane,
        intr,
        cfg["colors"],
        float(cfg["cube_size"]),
        float(cfg["depth_min"]),
        float(cfg["depth_max"]),
    )
    return bool(cue)


def _prior_ground(
    depth: np.ndarray,
    color: np.ndarray,
    intr,
    cfg: dict,
    gravity: np.ndarray,
    camera_height: float | None,
    memory: GroundMemory,
) -> GroundFix | None:
    # 单独一块：地面锁在安装高度附近，白盒先挖掉。没看到就沿用上次的离地高度。
    points, normals = ground_cloud(depth, intr, cfg, color)
    fix = acquire_plane(points, normals, cfg, gravity, memory.prior, camera_height)
    if fix is not None:
        memory.prior = fix
    return fix


def _floor_ground(
    depth: np.ndarray,
    intr,
    cfg: dict,
    gravity: np.ndarray,
    memory: GroundMemory,
) -> GroundFix | None:
    # 两层：这一眼的深度自己找最低的平地。上一块平面还贴着才沿用。
    fix = fit_visible_ground(depth, intr, cfg, gravity, memory.floor)
    if fix is not None:
        memory.floor = fix.plane
    return fix


def ground_for_layers(
    depth: np.ndarray,
    color: np.ndarray,
    intr,
    cfg: dict,
    gravity: np.ndarray,
    camera_height: float | None,
    memory: GroundMemory,
    search: StackSearch,
) -> tuple[GroundFix | None, bool]:
    """一层用安装高度附近的地面。两层整座塔用最低的平地，低头和抬头都一样。"""
    if search.settled_one:
        return _prior_ground(depth, color, intr, cfg, gravity, camera_height, memory), False
    if _known_stack(search):
        return _floor_ground(depth, intr, cfg, gravity, memory), True
    fix = _prior_ground(depth, color, intr, cfg, gravity, camera_height, memory)
    plane = None if fix is None else fix.plane
    if _second_layer(color, depth, plane, intr, cfg):
        return _floor_ground(depth, intr, cfg, gravity, memory), True
    return fix, False


def ground_caption(fix: GroundFix | None, stacked: bool) -> str:
    if fix is None:
        return "还没看到地面"
    above = -float(fix.plane.offset)
    names = {"fit": "测到了", "held": "沿用上次的", "imu": "按安装高度", "ransac": "整幅对上了"}
    kind = "两层地面" if stacked else "一层地面"
    return f"{kind}{names.get(fix.source, fix.source)}，相机离地 {above:.2f} 米"


def fit_visible_ground(
    depth: np.ndarray,
    intr,
    cfg: dict,
    gravity: np.ndarray,
    held: Plane | None,
) -> GroundFix | None:
    # 和 robocon1 一样：这一眼的深度自己找最低的一层平地。
    # 上一帧的平面还贴着这些点才沿用。对不上就不用上次的离地高度，避免橙线贴到方块下沿。
    thresh = float(cfg["ground_ransac_thresh"])
    max_angle = float(cfg["ground_max_angle_deg"])
    cloud = sample_points(depth, intr, stride=int(cfg.get("ground_stride", 8)), gravity=gravity)
    if held is not None and plane_holds(cloud, held, gravity, thresh, max_angle):
        return GroundFix(plane=held, source="held")
    plane = fit_ground(
        cloud,
        gravity,
        thresh=thresh,
        max_angle_deg=max_angle,
        iterations=int(cfg.get("ground_ransac_iterations", 50)),
    )
    return None if plane is None else GroundFix(plane=plane, source="ransac")


def acquire_plane(
    points: np.ndarray,
    normals: np.ndarray,
    cfg: dict,
    gravity: np.ndarray,
    held: GroundFix | None,
    camera_height: float | None,
) -> GroundFix | None:
    # 有相机高度时，地面由重力和这个高度给出，深度只在附近修正。
    # 没有高度时退回整幅拟合。上一帧平面仍贴着点云就沿用。
    thresh = float(cfg["ground_ransac_thresh"])
    max_angle = float(cfg["ground_max_angle_deg"])
    if camera_height is not None:
        return ground_with_prior(
            points,
            normals,
            gravity,
            float(camera_height),
            held,
            window=float(cfg.get("ground_prior_window", 0.12)),
            thresh=thresh,
            max_angle_deg=max_angle,
            min_points=int(cfg.get("ground_min_points", 150)),
        )
    if held is not None and held.source == "ransac" and plane_holds(points, held.plane, gravity, thresh, max_angle):
        return held
    plane = fit_ground(
        points,
        gravity,
        thresh=thresh,
        max_angle_deg=max_angle,
        iterations=int(cfg.get("ground_ransac_iterations", 50)),
    )
    return None if plane is None else GroundFix(plane=plane, source="ransac")


def _up_from(gravity: np.ndarray) -> np.ndarray:
    down = np.asarray(gravity, dtype=float)
    return -down / np.linalg.norm(down)


# 下层顶面相对「相机下方一个边长」可以偏这么多。地面差几厘米仍留下，差出一块盒子就丢掉。
_LOWER_TOP_TOL_M = 0.08


def lower_top_on_ground(pose, plane, up, camera_height, cube_size, tolerance: float = _LOWER_TOP_TOL_M) -> bool:
    # 底下那块贴在地面上。相机离地 0.70 米时，顶面在相机正下方 0.35 米。上层不按这个筛。
    if camera_height is None or int(getattr(pose, "layer", 1)) != 1:
        return True
    top = np.asarray(pose.bottom_center, dtype=float).reshape(3)
    top = top + np.asarray(plane.normal, dtype=float).reshape(3) * float(cube_size)
    below = -float(np.dot(top, np.asarray(up, dtype=float).reshape(3)))
    return abs(below - (float(camera_height) - float(cube_size))) <= float(tolerance)


def ground_yaw(edge: np.ndarray, plane, fallback: float) -> float:
    # 棱在地面上的转角，从相机右方量到前方。相机正对地面时前方投不出来，改用检测给出的角。
    normal = np.asarray(plane.normal, dtype=float)
    forward = np.array([0.0, 0.0, 1.0]) - float(np.dot(np.array([0.0, 0.0, 1.0]), normal)) * normal
    if float(np.linalg.norm(forward)) < 1e-3:
        return float(fallback)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, normal)
    right = right / np.linalg.norm(right)
    flat = np.asarray(edge, dtype=float) - float(np.dot(edge, normal)) * normal
    if float(np.linalg.norm(flat)) < 1e-6:
        return float(fallback)
    flat = flat / np.linalg.norm(flat)
    return fold_yaw_quarter(float(np.arctan2(flat @ forward, flat @ right)))


def pose_to_record(pose, plane, rotation: np.ndarray, translation: np.ndarray, height: float, ground_source: str) -> dict:
    # 一层的体心是地面足迹再抬半个边长。上面那块用自己的足迹，再抬一层半，不借用下面的中心。
    level = int(getattr(pose, "layer", 1))
    center = pose.bottom_center + plane.normal * ((level - 0.5) * height)
    world = rotation.T @ (center - translation)
    yaw = ground_yaw(pose.edge_direction, plane, pose.yaw)
    return {
        "x": float(world[0]),
        "y": float(world[1]),
        "z": float(world[2]),
        "yaw": float(yaw),
        "quarter": not pose.front_color,
        "inliers": int(pose.inliers),
        "long_side": float(pose.long_side),
        "short_side": float(pose.short_side),
        "bottom": np.asarray(pose.bottom_center, dtype=float),
        "corners": None if pose.corners is None else np.asarray(pose.corners, dtype=float),
        "edge": np.asarray(pose.edge_direction, dtype=float),
        "ground": ground_source,
        "plane": plane,
        "layer": level,
    }


def estimate_layers(
    frame: Frame,
    intr,
    cfg: dict,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    rejects: dict | None = None,
    fix: GroundFix | None = None,
    camera_height: float | None = None,
) -> list[dict]:
    # 下面一层和上面一层各给一条记录。没有第二层时列表里只有原来的那一块。
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    gravity = frame.gravity if frame.gravity is not None else np.array([0.0, 0.0, 1.0])
    if fix is None:
        points, normals = ground_cloud(frame.depth, intr, cfg, frame.color)
        fix = acquire_plane(points, normals, cfg, gravity, None, camera_height)
    if fix is None:
        if rejects is not None:
            rejects["ground"] = int(rejects.get("ground", 0)) + 1
        return []
    plane = fix.plane
    if float(plane.normal @ _up_from(gravity)) < 0.95:
        if rejects is not None:
            rejects["world_up"] = int(rejects.get("world_up", 0)) + 1
        return []
    poses = detect_white_layers(
        frame.color,
        frame.depth,
        intr,
        plane,
        cfg["colors"],
        cube_size=float(cfg["cube_size"]),
        min_pixels=int(cfg["min_face_pixels"]),
        min_footprint_fraction=float(cfg["min_footprint_fraction"]),
        size_tolerance=float(cfg["size_tolerance"]),
        rotation=rotation,
        translation=translation,
        zmin=float(cfg["depth_min"]),
        zmax=float(cfg["depth_max"]),
        rejects=rejects,
    )
    height = float(cfg["cube_size"])
    up = _up_from(gravity)
    kept = []
    lost_lower = False
    for pose in poses:
        if lower_top_on_ground(pose, plane, up, camera_height, height):
            kept.append(pose)
        elif int(getattr(pose, "layer", 1)) == 1:
            lost_lower = True
    if lost_lower and rejects is not None and not kept:
        rejects["cube_height"] = int(rejects.get("cube_height", 0)) + 1
    return [pose_to_record(pose, plane, rotation, translation, height, fix.source) for pose in kept]


def estimate_frame(
    frame: Frame,
    intr,
    cfg: dict,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    rejects: dict | None = None,
    fix: GroundFix | None = None,
    camera_height: float | None = None,
) -> dict | None:
    # 原来的单块接口留下下面那一层。上面那块只在 estimate_layers 里。
    found = estimate_layers(
        frame,
        intr,
        cfg,
        rotation=rotation,
        translation=translation,
        rejects=rejects,
        fix=fix,
        camera_height=camera_height,
    )
    for record in found:
        if int(record.get("layer", 1)) == 1:
            return record
    return None


def selected_camera_height(args, cfg: dict) -> float | None:
    if getattr(args, "camera_height", None) is not None:
        height = float(args.camera_height)
    else:
        base = cfg.get("camera_height")
        height = None if base is None else float(base)
    if getattr(args, "on_platform", False):
        if height is None:
            raise SystemExit("配置里 camera_height 为空，--on-platform 没有可加的基准")
        height += float(cfg.get("platform_lift", 0.6))
    return height


def run_live(cfg: dict, intr, camera_height: float | None, color_lock: dict | None) -> None:
    # 至少 5 帧、最多 15 帧取中值。地面原点跳过 30 mm 就清空。失败后 0.5 s 内沿用上一结果。
    source = RealSenseSource(
        intr.width,
        intr.height,
        fps=int(cfg.get("fps", 30)),
        color_lock=color_lock,
        depth_width=int(cfg.get("depth_width", 848)),
        depth_height=int(cfg.get("depth_height", 480)),
    )
    memory = GroundMemory()
    search = StackSearch()
    buckets: dict[int, list[dict]] = {}
    seen: dict[int, float] = {}
    last_origin: np.ndarray | None = None
    last_public: dict | None = None
    fail_since: float | None = None
    next_out = 0.0
    period = 1.0 / float(cfg.get("output_hz", 10))
    window = int(cfg.get("frames_to_fuse", 15))
    need = int(cfg.get("min_samples", 5))
    jump = float(cfg.get("jump_m", 0.03))
    hold_s = float(cfg.get("hold_s", 0.5))
    try:
        while True:
            frame = source.read()
            cleaned = clean_frame(frame.color, frame.depth, frame.gravity, cfg)
            gravity = cleaned.gravity if cleaned.gravity is not None else np.array([0.0, 1.0, 0.0])
            height = camera_height if cleaned.gravity is not None else None
            fix, _stacked = ground_for_layers(
                cleaned.depth,
                cleaned.color,
                intr,
                cfg,
                gravity,
                height,
                memory,
                search,
            )
            if fix is not None:
                origin = fix.plane.origin()
                if last_origin is not None and float(np.linalg.norm(origin - last_origin)) > jump:
                    buckets.clear()
                    seen.clear()
                    last_public = None
                last_origin = origin
            found = estimate_layers(
                cleaned,
                intr,
                cfg,
                fix=fix,
                camera_height=height,
            )
            # 层数看高度带，不看白盒顶面。低头时只积累贴地的白。远的接近水平才投票，近的要再抬过水平。
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
            lower_now = next((item for item in found if int(item.get("layer", 1)) == 1), None)
            near = record_distance(lower_now)
            search.update(
                found,
                bands,
                near if near is not None else bands.get("distance"),
                float(cfg["cube_size"]),
                pitch_down_degrees(frame.gravity),
                cleaned.gravity,
            )
            now = time.monotonic()
            if search.settled_one:
                buckets.pop(2, None)
                seen.pop(2, None)
            kept = []
            for sample in found:
                level = int(sample["layer"])
                if level == 2 and search.stage not in ("upstairs", "both"):
                    continue
                if level == 1 and search.stage in ("upstairs", "both"):
                    continue
                kept.append(sample)
            present = {int(sample["layer"]) for sample in kept}
            for sample in kept:
                level = int(sample["layer"])
                bucket = buckets.setdefault(level, [])
                bucket.append(sample)
                if len(bucket) > window:
                    del bucket[:-window]
                seen[level] = now
            if search.stage in ("upstairs", "both") and 1 in seen:
                seen[1] = now
                present.add(1)
            if search.upper is not None and search.stage == "one" and 2 in seen:
                seen[2] = now
                present.add(2)
            for level in list(buckets):
                if level not in present and now - seen.get(level, 0.0) > hold_s:
                    buckets.pop(level, None)
                    seen.pop(level, None)
            cubes = []
            for level in sorted(buckets):
                bucket = buckets[level]
                fused = fuse_poses(bucket) if len(bucket) >= need else None
                if fused is None:
                    continue
                latest = bucket[-1]
                cubes.append(
                    {
                        "layer": level,
                        "x": fused.x,
                        "y": fused.y,
                        "z": fused.z,
                        "yaw": fused.yaw,
                        "long_side": latest["long_side"],
                        "short_side": latest["short_side"],
                        "ground": latest["ground"],
                        "quality": {"inliers": latest["inliers"]},
                    }
                )
            if not search.settled:
                # 层数还没定，不把白盒体心当成结果。
                public = {
                    "frame": "camera",
                    "looking": "up" if search.looked_up else ("down" if search.anchor is not None else "ground"),
                }
            elif not cubes and search.kind == "one":
                fail_since = None
                public = {"frame": "camera", "layers": 1, "kind": "one", "cap": False}
                last_public = public
            elif not cubes:
                if fail_since is None:
                    fail_since = now
                public = last_public if last_public is not None and now - fail_since <= hold_s else {
                    "frame": "camera",
                    "layers": 1 if search.kind == "one" else 2,
                    "kind": search.kind,
                    "cap": search.kind == "two_cap",
                }
            else:
                fail_since = None
                primary = next((cube for cube in cubes if cube["layer"] == 1), cubes[0])
                public = {
                    "frame": "camera",
                    "layers": 1 if search.kind == "one" else 2,
                    "kind": search.kind,
                    "cap": search.kind == "two_cap",
                    "cubes": cubes,
                    "x": primary["x"],
                    "y": primary["y"],
                    "z": primary["z"],
                    "yaw": primary["yaw"],
                    "long_side": primary["long_side"],
                    "short_side": primary["short_side"],
                    "ground": primary["ground"],
                    "quality": primary["quality"],
                }
                last_public = public
            adjustment = plan_view(
                cleaned.color,
                cleaned.depth,
                intr,
                cfg["colors"],
                cube_size=float(cfg["cube_size"]),
                gravity=cleaned.gravity,
                zmin=float(cfg["depth_min"]),
                zmax=float(cfg["depth_max"]),
            )
            if now >= next_out:
                printed = dict(public)
                printed["pitch"] = adjustment.pitch.as_dict()
                printed["yaw"] = adjustment.yaw.as_dict()
                print(json.dumps(printed, ensure_ascii=False), flush=True)
                next_out = now + period
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="测量地面上的 0.35 m 白盒")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--light", default=None)
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--on-platform", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    intr = load_intrinsics("config/intrinsics.yaml")
    run_live(cfg, intr, selected_camera_height(args, cfg), color_lock_from_config(cfg, args.light))


if __name__ == "__main__":
    main()
