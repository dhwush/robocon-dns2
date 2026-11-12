"""测试画面上的方块框。

同一层攒够多帧后取中值再画。单帧脚印跳开就不换已经画出的框。
框是边长正好一块的正方，立在这一眼的地面上。镜头从看地面改成抬头，或从抬头回到地面，上一眼的框清掉。
"""

from __future__ import annotations

import numpy as np

from fuse.median import fuse_poses
from ground.plane import Plane


def _flat_edge(edge: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(edge, dtype=float) - normal * float(np.dot(edge, normal))
    length = float(np.linalg.norm(flat))
    if length < 1e-8:
        return None
    return flat / length


def _align_edge(edge: np.ndarray, ref: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # 正方形转 90 度还是同一块。挑和参考边最接近的那一条，再拿去取中值。
    side = np.cross(normal, edge)
    side = side / np.linalg.norm(side)
    return max((edge, -edge, side, -side), key=lambda item: float(np.dot(item, ref)))


def square_corners(bottom: np.ndarray, edge: np.ndarray, normal: np.ndarray, cube_size: float) -> np.ndarray:
    half = 0.5 * float(cube_size)
    side = np.cross(normal, edge)
    side = side / np.linalg.norm(side)
    return np.vstack(
        [
            bottom - half * edge - half * side,
            bottom + half * edge - half * side,
            bottom + half * edge + half * side,
            bottom - half * edge + half * side,
        ]
    )


class OutlineTrack:
    """只给出当前这一眼、已经对齐的那一层方块框。"""

    def __init__(self, window: int = 15, need: int = 5, jump: float = 0.03):
        self.window = int(window)
        self.need = int(need)
        self.jump = float(jump)
        self.buckets: dict[int, list[dict]] = {1: [], 2: []}
        self.glance = "down"
        self.origin: np.ndarray | None = None

    def update(self, records: list[dict], stage: str, plane: Plane | None, cube_size: float) -> list[dict]:
        glance = "up" if stage in ("upstairs", "both") else "down"
        if glance != self.glance:
            self.buckets = {1: [], 2: []}
            self.glance = glance
        if plane is not None:
            origin = np.asarray(plane.origin(), dtype=float)
            if self.origin is not None and float(np.linalg.norm(origin - self.origin)) > self.jump:
                self.buckets = {1: [], 2: []}
            self.origin = origin
        level = 2 if glance == "up" else 1
        sample = next(
            (
                item
                for item in records
                if int(item.get("layer", 1)) == level and item.get("bottom") is not None and item.get("edge") is not None
            ),
            None,
        )
        if sample is not None and plane is not None:
            self._push(level, sample, plane)
        fused = self._fuse(level, plane, cube_size)
        return [] if fused is None else [fused]

    def _push(self, level: int, sample: dict, plane: Plane) -> None:
        bucket = self.buckets[level]
        if len(bucket) >= self.need:
            bottoms = np.stack([np.asarray(item["bottom"], dtype=float) for item in bucket])
            center = np.asarray(plane.project(np.median(bottoms, axis=0)), dtype=float).reshape(3)
            moved = float(np.linalg.norm(np.asarray(sample["bottom"], dtype=float) - center))
            if moved > self.jump:
                return
        bucket.append(sample)
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def _fuse(self, level: int, plane: Plane | None, cube_size: float) -> dict | None:
        samples = self.buckets[level]
        if plane is None or len(samples) < self.need:
            return None
        normal = np.asarray(plane.normal, dtype=float)
        bottoms = np.stack([np.asarray(item["bottom"], dtype=float) for item in samples])
        bottom = np.asarray(plane.project(np.median(bottoms, axis=0)), dtype=float).reshape(3)
        ref = None
        edges = []
        for sample in samples:
            edge = _flat_edge(sample["edge"], normal)
            if edge is None:
                continue
            if ref is None:
                ref = edge
            edges.append(_align_edge(edge, ref, normal))
        if not edges:
            return None
        edge = _flat_edge(np.median(np.stack(edges), axis=0), normal)
        if edge is None:
            return None
        fused = fuse_poses(samples)
        yaw = float(fused.yaw) if fused is not None else float(samples[-1]["yaw"])
        body = bottom + normal * ((int(level) - 0.5) * float(cube_size))
        drawn = dict(samples[-1])
        drawn.update(
            {
                "bottom": bottom,
                "corners": square_corners(bottom, edge, normal, cube_size),
                "edge": edge,
                "yaw": yaw,
                "x": float(body[0]),
                "y": float(body[1]),
                "z": float(body[2]),
                "long_side": float(np.median([item["long_side"] for item in samples])),
                "short_side": float(np.median([item["short_side"] for item in samples])),
                "layer": int(level),
                "plane": plane,
            }
        )
        return drawn
