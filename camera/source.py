"""深度预处理与 D435i 取流。

几何计算使用实际启用的彩色流内参；深度已对齐到彩色。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import sys
import time

import numpy as np

from geom.transform import Intrinsics


# --- 一帧数据 -----------------------------------------------------------------
# color 为 BGR，depth 单位是米，gravity 是相机系中的重力方向（静止时由加速度计得到）。

@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray
    gravity: np.ndarray | None = None
    # 彩色相机系角速度 (时间戳秒, rad/s)。没有陀螺仪时为空，跟踪只靠视觉和场地边。
    gyro: list | None = None
    # 加速度设备时间戳（秒）及主机收到该样本到取图时的间隔。
    gravity_timestamp_s: float | None = None
    gravity_age_s: float | None = None
    gravity_status: str = "unspecified"


# --- 深度清洗 -----------------------------------------------------------------
# 先丢掉 0.3–3 m 以外的值，再去掉边缘飞点，最后做中值平滑。
# 稀疏深度不做中值：用 0 填空洞再模糊会把有效点抹掉。
# 缓冲用 float32。米级深度用不上双精度，720p 双精度每帧大约 7 MB。

def clip_range(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # D435i 有效距离大约 0.3–3 m。更近的视差不可靠，更远的深度噪声按距离平方变大。
    # 非法值统一写成 NaN，后面的拟合会跳过它们。
    out = np.array(depth, dtype=np.float32, copy=True)
    invalid = ~np.isfinite(out) | (out < zmin) | (out > zmax) | (out <= 0.0)
    out[invalid] = np.nan
    return out


def remove_flying_pixels(depth: np.ndarray, thresh_m: float) -> np.ndarray:
    # 飞点出现在深度不连续的边缘：一个像素跳到前景或背景。
    # 看 3×3 邻域的最大深度差，超过阈值就丢掉该像素。邻域有效点少于 3 个时不判，避免把稀疏点误删。
    # 膨胀和腐蚀一次扫完整幅图，不再对每个邻域各拷贝一次。
    import cv2

    src = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(src)
    count = cv2.boxFilter(
        finite.astype(np.float32),
        -1,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    hi = np.where(finite, src, np.float32(-1.0e6))
    lo = np.where(finite, src, np.float32(1.0e6))
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(hi, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=-1.0e6)
    local_min = cv2.erode(lo, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1.0e6)
    jump = ((local_max - local_min) > thresh_m) & (count >= 3)
    out = np.array(depth, dtype=np.float32, copy=True)
    out[jump] = np.nan
    return out


def median_smooth(depth: np.ndarray, ksize: int = 3) -> np.ndarray:
    # 中值能压散粒噪声，又比均值更能保住深度台阶。
    import cv2

    finite = np.isfinite(depth)
    # 有效像素不到七成时，空洞会被 0 填上，中值会把真深度拉向 0。这种情况直接返回。
    if float(finite.mean()) < 0.7:
        return np.array(depth, dtype=np.float32, copy=True)
    filled = np.where(finite, depth, 0.0).astype(np.float32)
    smooth = cv2.medianBlur(filled, ksize)
    # 窗口里有效像素不足一半的位置仍视为空洞。
    mask = cv2.blur(finite.astype(np.float32), (ksize, ksize))
    out = smooth.astype(np.float32)
    out[mask < 0.5] = np.nan
    return out


# --- 多帧深度 -----------------------------------------------------------------
# 没有 RealSense 时间滤波器时，用指数滑动平均补洞。

class TemporalDepthFilter:
    def __init__(self, alpha: float = 0.4):
        # alpha 越大越跟当前帧。还没有上一帧时，apply 先把这一帧当作状态。
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        # 两帧都有效时做指数平均；当前帧是洞、上一帧有值时沿用上一帧补闪烁。
        cur = np.array(depth, dtype=np.float32, copy=True)
        if self._state is None or self._state.shape != cur.shape:
            self._state = cur
            return cur.copy()
        both = np.isfinite(cur) & np.isfinite(self._state)
        blended = self.alpha * cur + (1.0 - self.alpha) * self._state
        out = cur.copy()
        out[both] = blended[both]
        only_old = ~np.isfinite(cur) & np.isfinite(self._state)
        out[only_old] = self._state[only_old]
        self._state = out
        return out


def filter_depth(
    depth: np.ndarray,
    zmin: float = 0.3,
    zmax: float = 3.0,
    flying_pixel_m: float = 0.08,
    temporal: TemporalDepthFilter | None = None,
) -> np.ndarray:
    # 顺序不能颠倒：先截距离，再删飞点，否则飞点的大跳变会污染邻域判断。
    out = clip_range(depth, zmin, zmax)
    out = remove_flying_pixels(out, flying_pixel_m)
    out = median_smooth(out, 3)
    if temporal is not None:
        out = temporal.apply(out)
    return out


# --- 实机取流 -----------------------------------------------------------------
# 彩色与深度对齐。深度只做时间滤波。视差域空间滤波在 720p 上每帧要几十毫秒以上，
# 飞点和 3×3 中值已经在 filter_depth 里做，这里不再做一遍边缘保持。
# 彩色曝光和白平衡锁定。
# 加速度静止时指向“上”，重力取反方向，并旋到彩色相机坐标系。

def color_lock_from_config(cfg: dict, level: str | None = None) -> dict:
    # 档位只改曝光、增益和亮度偏移。自动曝光仍关闭，避免平台一动颜色阈值跟着漂。
    levels = cfg.get("light_levels") or {}
    name = str(level or cfg.get("light") or "daylight")
    spec = levels.get(name)
    if spec is None:
        known = ", ".join(str(key) for key in levels) or "无"
        raise SystemExit(f"没有亮度档 {name}。可选: {known}")
    return {
        "name": name,
        "exposure": float(spec["exposure"]),
        "gain": float(spec["gain"]),
        "brightness": float(spec["brightness"]),
    }


def list_camera_serials() -> list[str]:
    # 只问当前接上的设备号码，不打开彩色和深度流。没接相机时返回空列表。
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    try:
        devices = rs.context().query_devices()
    except Exception:
        return []
    serials: list[str] = []
    for device in devices:
        try:
            serials.append(str(device.get_info(rs.camera_info.serial_number)))
        except Exception:
            continue
    return serials


def device_color_intrinsics(profile, rs) -> Intrinsics:
    """读取实际启用的彩色流，转换成几何模块支持的内参模型。"""
    raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    coefficients = np.asarray(raw.coeffs, dtype=float)
    values = np.asarray([raw.fx, raw.fy, raw.ppx, raw.ppy], dtype=float)
    if (raw.width <= 0 or raw.height <= 0 or coefficients.shape != (5,)
            or not np.isfinite(coefficients).all() or not np.isfinite(values).all()
            or raw.fx <= 0 or raw.fy <= 0):
        raise ValueError("设备彩色内参无效")
    model = str(raw.model).split(".")[-1]
    if not np.any(coefficients):
        model = "none"
    elif model in ("brown_conrady", "brown"):
        model = "brown"
    elif model != "inverse_brown_conrady":
        raise ValueError(f"不支持设备彩色畸变模型 {model}，不能输出可靠坐标")
    return Intrinsics(width=int(raw.width), height=int(raw.height),
                      fx=float(raw.fx), fy=float(raw.fy), cx=float(raw.ppx), cy=float(raw.ppy),
                      k1=float(coefficients[0]), k2=float(coefficients[1]),
                      p1=float(coefficients[2]), p2=float(coefficients[3]),
                      k3=float(coefficients[4]), distortion_model=model)


class RealSenseSource:
    """彩色与深度对齐，内参从实际启用的彩色流读取。"""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        serial: str | None = None,
        color_lock: dict | None = None,
        depth_width: int = 848,
        depth_height: int = 480,
        gravity_timeout_s: float = 0.5,
    ):
        # 彩色、深度和惯性一起开。对齐和滤波要在回调之前建好，否则第一帧还没有这些对象。
        if not np.isfinite(gravity_timeout_s) or gravity_timeout_s <= 0:
            raise ValueError("gravity_timeout_s must be finite and positive")
        import pyrealsense2 as rs

        self._rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            # 接了多台时只开号码相符的那台，避免内参和画面不是同一台。
            config.enable_device(str(serial))
        # 彩色保持 1280×720。深度用 848×480，最近大约 0.17 米才有数；1280×720 要到大约 0.28 米。
        # 深度对齐到彩色之后，几何仍用彩色内参。
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, int(depth_width), int(depth_height), rs.format.z16, fps)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._pending: tuple | None = None
        self._gyro: deque = deque(maxlen=400)
        self._accel: np.ndarray | None = None
        self._accel_timestamp_s: float | None = None
        self._accel_received_at: float | None = None
        self._gyro_timestamp_s: float | None = None
        self._gravity_timeout_s = float(gravity_timeout_s)
        self._gyro_to_color = np.eye(3)
        self._accel_to_color = np.eye(3)
        self._gyro_ready = False
        self._video_ready = False
        self.depth_scale = np.float32(0.001)
        self._color_lock = color_lock or {"name": "daylight", "exposure": 100, "gain": 22, "brightness": 0}
        # 对齐和时间滤波要在启动回调之前建好，否则第一帧进来时还没有这些对象。
        self.align = rs.align(rs.stream.color)
        self.temporal_rs = rs.temporal_filter()
        self._config = config
        self._gyro_on = True
        self._open_pipeline()
        # 后续重连仍打开这台设备，保持内参与图像来自同一台相机。
        self._config.enable_device(self.serial)
        if serial is not None and self.serial != str(serial):
            self.close()
            raise RuntimeError(f"打开的相机是 {self.serial}，不是 {serial}")
        # 刚打开时常常只有惯性。先等彩色和深度来一帧，再锁曝光，避免把还没出图的流打断。
        if not self._event.wait(timeout=3.0):
            self._reopen()
            self._event.wait(timeout=3.0)
        self._discard_pending()
        self._lock_color()

    def _open_pipeline(self) -> None:
        # 没有陀螺仪时关掉该流再启动。视觉跟踪和场地边仍然可用。
        rs = self._rs
        self._video_ready = False
        self._gyro_ready = False
        self._event.clear()
        with self._lock:
            self._pending = None
            self._gyro.clear()
            self._accel = None
            self._accel_timestamp_s = None
            self._accel_received_at = None
            self._gyro_timestamp_s = None
        try:
            self.profile = self.pipeline.start(self._config, self._on_frame)
            self._gyro_on = True
        except Exception as exc:
            if "No device connected" in str(exc):
                raise
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = rs.pipeline()
            self._config.disable_stream(rs.stream.gyro)
            self.profile = self.pipeline.start(self._config, self._on_frame)
            self._gyro_on = False
        device = self.profile.get_device()
        self.serial = str(device.get_info(rs.camera_info.serial_number))
        try:
            self.intrinsics = device_color_intrinsics(self.profile, rs)
        except Exception:
            self.pipeline.stop()
            raise
        intr = self.intrinsics
        print(f"相机 {self.serial}：设备彩色内参 {intr.width}x{intr.height}，"
              f"fx={intr.fx:.4f}, fy={intr.fy:.4f}, cx={intr.cx:.4f}, cy={intr.cy:.4f}，"
              f"model={intr.distortion_model}, D={intr.D.tolist()}", file=sys.stderr, flush=True)
        self.depth_scale = np.float32(device.first_depth_sensor().get_depth_scale())
        self._accel_to_color = self._accel_rotation()
        if self._gyro_on:
            self._gyro_to_color = self._motion_rotation(rs.stream.gyro)
        self._video_ready = True
        self._gyro_ready = True

    def _reopen(self) -> None:
        # 彩色和深度没来时，关掉再开。惯性还在走，不代表画面也在走。
        print("彩色画面还没来，重新打开相机。", flush=True)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = self._rs.pipeline()
        self._open_pipeline()

    def _discard_pending(self) -> None:
        # 锁曝光之前的那一帧是自动曝光，颜色不能拿去认方块。
        with self._lock:
            self._event.clear()
            self._pending = None

    def set_color_lock(self, color_lock: dict) -> None:
        # 测试时换环境光档位。只改曝光、增益和亮度，不重开取流。
        self._color_lock = color_lock
        if getattr(self, "profile", None) is not None:
            self._lock_color()

    def _accel_rotation(self) -> np.ndarray:
        # 加速度计到彩色相机的旋转。重力要先变到彩色系，再交给地面拟合。
        return self._motion_rotation(self._rs.stream.accel)

    def _motion_rotation(self, stream) -> np.ndarray:
        # 读设备给出的惯性系到彩色相机的外参。读失败时退回单位阵，避免整路取流起不来。
        try:
            # 惯性系和彩色相机不一致，要用设备外参旋过来。失败时退回单位阵。
            motion = self.profile.get_stream(stream).as_motion_stream_profile()
            color = self.profile.get_stream(self._rs.stream.color)
            ext = motion.get_extrinsics_to(color)
            return np.array(ext.rotation, dtype=float).reshape(3, 3, order="F")
        except Exception:
            return np.eye(3)

    def _gravity_from_specific(self, specific: np.ndarray) -> np.ndarray | None:
        # 加速度计静止时指向“上”。旋到彩色系后取反，得到相机系里的重力方向。
        up = self._accel_to_color @ specific
        norm = np.linalg.norm(up)
        if not np.isfinite(norm) or norm <= 1.0:
            return None
        return -up / norm

    def _on_frame(self, frame) -> None:
        # 回调里出错不能把后面的彩色和深度一起停掉。
        try:
            self._handle_frame(frame)
        except Exception as exc:
            if not getattr(self, "_frame_error", False):
                self._frame_error = True
                print(f"相机这一帧没收下：{exc}", flush=True)

    def _handle_motion(self, frame) -> None:
        if not self._gyro_ready or not frame or not frame.is_motion_frame():
            return
        motion = frame.as_motion_frame()
        kind = motion.profile.stream_type()
        timestamp = float(motion.timestamp) * 1e-3
        data = motion.get_motion_data()
        sample = np.array([data.x, data.y, data.z], dtype=float)
        if not np.isfinite(timestamp) or not np.isfinite(sample).all():
            return
        if kind == self._rs.stream.accel:
            gravity = self._gravity_from_specific(sample)
            if gravity is not None:
                with self._lock:
                    # 同一加速度可能被多个图像帧带来，重复样本不能刷新有效期。
                    if self._accel_timestamp_s is None or timestamp > self._accel_timestamp_s:
                        self._accel = gravity
                        self._accel_timestamp_s = timestamp
                        self._accel_received_at = time.monotonic()
            return
        if kind == self._rs.stream.gyro:
            omega = self._gyro_to_color @ sample
            with self._lock:
                if self._gyro_timestamp_s is None or timestamp > self._gyro_timestamp_s:
                    self._gyro.append((timestamp, omega))
                    self._gyro_timestamp_s = timestamp

    def _handle_frame(self, frame) -> None:
        rs = self._rs
        if not frame.is_frameset():
            self._handle_motion(frame)
            return
        original = frame.as_frameset()
        # 对齐会丢掉运动流；必须先从原始集合更新惯性，即使集合没有图像。
        for stream in (rs.stream.accel, rs.stream.gyro):
            motion = original.first_or_default(stream)
            if motion:
                self._handle_motion(motion)
        if not self._video_ready or not original.get_depth_frame() or not original.get_color_frame():
            return
        aligned = self.align.process(original)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return
        filtered = self.temporal_rs.process(depth_frame)
        depth = np.asanyarray(filtered.get_data()).astype(np.float32) * self.depth_scale
        color = np.asanyarray(color_frame.get_data()).copy()
        with self._lock:
            gravity = None if self._accel is None else self._accel.copy()
            # 将图像与当时的惯性时间配对；取图时还会重新检查是否过期。
            self._pending = (color, depth, gravity, self._accel_timestamp_s,
                             self._accel_received_at)
            self._event.set()

    def _set_clamped(self, sensor, option, value: float) -> None:
        # 不同固件的上下限不一样。超出范围就夹到设备允许的值，避免设置失败把整路取流打断。
        if not sensor.supports(option):
            return
        try:
            span = sensor.get_option_range(option)
            sensor.set_option(option, float(np.clip(value, span.min, span.max)))
        except Exception:
            return

    def _lock_color(self) -> None:
        # 自动曝光在平台移动时会连跳，颜色阈值跟着漂。锁定曝光和白平衡。
        # 关掉自动后若留着默认曝光和增益，白天会把红蓝冲成接近白色。
        rs = self._rs
        try:
            sensor = self.profile.get_device().first_color_sensor()
        except Exception:
            return
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
        lock = self._color_lock
        self._set_clamped(sensor, rs.option.exposure, lock["exposure"])
        self._set_clamped(sensor, rs.option.gain, lock["gain"])
        self._set_clamped(sensor, rs.option.brightness, lock["brightness"])
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 0)

    def read(self) -> Frame:
        # 等到回调写好一帧再取出。第一次经常只有惯性，关掉再开一次。
        if not self._event.wait(timeout=5.0):
            self._reopen()
            self._lock_color()
            if not self._event.wait(timeout=5.0):
                raise RuntimeError("missing color or depth frame")
        return self._take()

    def _take(self) -> Frame:
        with self._lock:
            self._event.clear()
            pending = self._pending
            self._pending = None
            # 在消费者取图时清空，不能在每个图像回调清空，否则慢速识别会漏掉中间陀螺数据。
            gyro = list(self._gyro)
            self._gyro.clear()
        if pending is None:
            raise RuntimeError("missing color or depth frame")
        color, depth, gravity, timestamp, received_at = pending
        age = None if received_at is None else max(0.0, time.monotonic() - received_at)
        status = "missing" if gravity is None or age is None else (
            "fresh" if age <= self._gravity_timeout_s else "stale")
        if status != "fresh":
            gravity = None
        return Frame(color=color, depth=depth, gravity=gravity, gyro=gyro,
                     gravity_timestamp_s=timestamp, gravity_age_s=age, gravity_status=status)

    def close(self) -> None:
        # 停掉管线，释放相机，避免下次启动报设备忙。
        self.pipeline.stop()
