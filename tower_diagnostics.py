"""Optional lossless capture and return-stage tracing; never changes recognition results."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


# Capture existing return values/locals instead of rerunning any recognition step.
TRACE_FUNCTIONS = {
    "_color_masks", "_cap_mask", "_like_face", "coarse_bands",
    "_seam_cloud", "_fill_seam", "_measured_seam", "_measure_rgb_seam",
    "_measure_depth", "_measure_seam_points", "_assign_two_lines",
    "_face_center_from_line", "_corner_center", "_accept", "measure_cap",
    "close_depth_unusable", "_center_from_rims", "_complete_bottom",
    "_bottom_edge_center", "_corner_from_bottoms", "_outline_chains",
    "align_bottom_to_yellow", "_on_face", "confirm_green_face", "measure_outline",
    "offer", "depart", "adjacent_color_pairs", "boundary_pixels", "pose_matches_color",
}
LOCAL_KEYS = {
    "points", "pixels", "kept_points", "kept_pixels", "keep", "band", "high",
    "colored", "red", "blue", "touch", "has", "finite", "use_measured",
    "body", "edge", "direction", "length", "start", "stop", "chosen",
    "rims", "bottoms", "front", "side", "rim", "corner", "pose", "stuck",
    "moved", "center", "show", "lowest", "distance", "cap", "upper", "lower", "split", "diagnostics",
}


def encode(value, arrays, prefix):
    """JSON metadata + numeric NPZ arrays; no pickle or lossy depth conversion."""
    if isinstance(value, np.ndarray):
        key = f"a{len(arrays):05d}_{prefix}".replace("/", "_")
        arrays[key] = value.copy()
        return {"array": key, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return encode(value.item(), arrays, prefix)
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name), arrays, prefix + "_" + f.name)
                for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): encode(v, arrays, prefix + "_" + str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v, arrays, prefix + "_" + str(i)) for i, v in enumerate(value)]
    return {"type": type(value).__name__}


class DiagnosticRecorder:
    def __init__(self, directory, seconds, cfg, intr, camera_height, root):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.seconds = float(seconds)
        self.start = None
        self.index = 0
        self.finished = False
        self.error = None
        self.previous_profile = None
        self.tracing = False
        self.events = []
        self.arrays = {}
        self.calls = {}
        self.read_at = None
        self.capture_at = None
        self.root = Path(root).resolve()
        self.allowed = {str((self.root / p).resolve()) for p in
                        ("cube/cap_pose.py", "cube/stack_look.py", "cube/color_boundary.py",
                         "cube/pose_projection.py", "view_tower.py")}
        sources = {}
        snapshot = self.path / "source"
        # Preserve the exact code/config used, including local uncommitted changes.
        for file in sorted(self.root.rglob("*")):
            if file.is_relative_to(self.path.resolve()):
                continue
            relative = file.relative_to(self.root)
            if any(part.startswith(".") or part in ("__pycache__", "diagnostics")
                   for part in relative.parts):
                continue
            if file.is_file() and file.suffix in (".py", ".yaml"):
                payload = file.read_bytes()
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                sources[str(relative)] = hashlib.sha256(payload).hexdigest()
        arrays = {}
        manifest = encode({
            "schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
            "duration_limit_s": self.seconds, "config": cfg, "intrinsics": intr,
            "camera_height": camera_height, "source_sha256": sources,
            "depth_unit": "meter", "color_order": "BGR",
            "input": "RealSenseSource output: depth aligned to color, before clean_frame",
            "timestamps": "receive_monotonic_s is host time; gravity_timestamp_s is accel device time (seconds); gravity_age_s is host sample age at image consumption",
            "scope": "every processed frame during capture; not every sensor frame",
            "storage": "uncompressed lossless NPZ; synchronous writes affect observed FPS",
            "trace": "return-site lines refer to source snapshot; failed child calls can be recovered by later fallbacks",
        }, arrays, "manifest")
        (self.path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        np.savez(self.path / "manifest_arrays.npz", **arrays)
        print(f"诊断采集目录：{self.path.resolve()}（{self.seconds:g} 秒）", flush=True)

    def begin(self):
        now = time.monotonic()
        if self.start is None:
            self.start = now
        if self.finished or self.error:
            return False
        if now - self.start >= self.seconds:
            self.finished = True
            print(f"诊断采集结束：{self.index} 帧，{self.path.resolve()}", flush=True)
            return False
        self.events, self.arrays, self.calls = [], {}, {}
        self.capture_at = now
        self.previous_profile = sys.getprofile()
        self.tracing = True
        sys.setprofile(self._profile)
        return True

    def _profile(self, frame, event, result):
        code = frame.f_code
        if code.co_filename not in self.allowed or code.co_name not in TRACE_FUNCTIONS:
            return
        if event == "call":
            self.calls[id(frame)] = time.perf_counter()
        elif event == "return":
            started = self.calls.pop(id(frame), time.perf_counter())
            try:
                name = code.co_name
                local = {k: v for k, v in frame.f_locals.items() if k in LOCAL_KEYS}
                owner = frame.f_locals.get("self")
                entry = {"function": name, "owner": None if owner is None else type(owner).__name__,
                         "file": str(Path(code.co_filename).relative_to(self.root)),
                         "return_line": frame.f_lineno,
                         "elapsed_ms": (time.perf_counter() - started) * 1000,
                         "result": result, "locals": local}
                self.events.append(encode(entry, self.arrays, f"trace{len(self.events)}"))
            except Exception as exc:
                self.error = f"trace: {type(exc).__name__}: {exc}"

    def stop_trace(self):
        if self.tracing:
            sys.setprofile(self.previous_profile)
            self.tracing = False

    def save(self, frame, cleaned, state):
        self.stop_trace()
        if self.error:
            print(f"诊断记录失败，识别继续：{self.error}", flush=True)
            return
        try:
            name = f"frame_{self.index:06d}"
            data = encode({
                "frame_id": self.index, "receive_monotonic_s": self.read_at,
                "elapsed_s": self.capture_at - self.start,
                "processing_before_save_ms": (time.monotonic() - self.read_at) * 1000,
                "raw_color": frame.color, "raw_depth": frame.depth,
                "raw_gravity": frame.gravity, "gyro": frame.gyro,
                "gravity_timestamp_s": getattr(frame, "gravity_timestamp_s", None),
                "gravity_age_s": getattr(frame, "gravity_age_s", None),
                "gravity_status": getattr(frame, "gravity_status", "unspecified"),
                "clean_color": cleaned.color, "clean_depth": cleaned.depth,
                "clean_gravity": cleaned.gravity, "state": state,
            }, self.arrays, "frame")
            data["trace"] = self.events
            # JSON is the completion marker; partial writes never look like complete frames.
            np.savez(self.path / (name + ".npz"), **self.arrays)
            (self.path / (name + ".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2))
            self.index += 1
        except Exception as exc:
            self.error = f"save: {type(exc).__name__}: {exc}"
            print(f"诊断保存失败，识别继续：{self.error}", flush=True)

    def lines(self, state):
        if self.error:
            return [f"诊断保存失败：{self.error}"]
        if self.finished:
            return [f"诊断采集结束，已保存 {self.index} 帧"]
        attempts = [e["result"] for e in self.events if e["function"] == "measure_cap"]
        failure = next((v[1] for v in reversed(attempts)
                        if isinstance(v, list) and len(v) == 2 and v[0] is None), None)
        models = [e for e in self.events if e["function"] in
                  ("_corner_center", "_face_center_from_line", "_center_from_rims")]
        model = " / ".join(e["function"] + (":失败" if e["result"] is None or isinstance(e["result"], str)
                                               else ":有解") for e in models) or "未拟合"
        return [
            f"诊断帧 {self.index - 1} 来源 {state['draw_source']} 单帧 {state['measured_via'] or '无'}",
            f"候选点 {len(state['bands'].get('cap_uv', []))} 测量原因 {failure or state['cap_reason'] or '无'} 重置 {state['reset_reason'] or '无'}",
            f"拟合尝试 {model}",
            "各步骤拒绝位置与使用点已写入同帧 JSON/NPZ",
        ]


def load_frame(directory, index):
    """Read a saved frame and all diagnostics without pickle; suitable for offline replay."""
    path = Path(directory) / f"frame_{index:06d}"
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        def restore(value):
            if isinstance(value, dict):
                if "array" in value and "shape" in value and "dtype" in value:
                    return arrays[value["array"]].copy()
                if set(value) == {"nonfinite"}:
                    return float(value["nonfinite"])
                return {k: restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v) for v in value]
            return value
        return restore(metadata)
