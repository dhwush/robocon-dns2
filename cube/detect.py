"""面能看清时的方块位姿。

位置取地面足迹，不拟合上表面。上表面 RMS、12° 法向和 6 mm 正方形残差在这里没有平面可算。
丢帧只看足迹：面积比例、长边相对边长 12%、短边是否还像一条棱。
看不清一整块面时返回 None，不在这里做贴地模型补救。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from geom.transform import deproject_pixels, fold_yaw_quarter, project_points, yaw_about_z
from ground.plane import Plane


# --- 输出 ---------------------------------------------------------------------
# bottom_center 在相机系，yaw 已变到世界系并只保留绕 Z 的角。

@dataclass
class CubePose:
    bottom_center: np.ndarray
    yaw: float
    edge_direction: np.ndarray
    footprint_fraction: float
    front_color: bool
    inliers: int = 0
    corners: np.ndarray | None = None
    source: str = "depth"
    long_side: float = 0.0
    short_side: float = 0.0
    peak: float = 0.0
    mask: np.ndarray | None = None
    layer: int = 1


# --- 颜色分割 -----------------------------------------------------------------
# 红色在色相环两端，配置里写成两段再并起来。方块上下两半是红和蓝，哪一面在上都可以。
# 绿线只描成片的严格红蓝。放宽颜色只补大约 10 px 的接缝，搜索不超过一个边长。
# 没有深度的点不拿来撑外沿，凸包也不把方块和旁边木板之间的地面填进掩膜。
# 地面漆成红或蓝时，贴地 3.5 cm 以内的颜色点按高度丢掉，不靠把色相收窄来区分。

# 贴地同色时，离地不足这条的点视为地面。侧面最底下贴着方块的一条留下。
GROUND_CLEARANCE_M = 0.035

# 白盒是低饱和、高明度。饱和度上限 110，明度下限不随暗光下调，否则灰地面整片进来。
# 饱和度低于 40、明度高于 245、并且挨着红蓝的点是被光冲掉的杂物，不算白。
_WASH_S_MAX = 40
_WASH_V_MIN = 245
_WASH_DILATE_PX = 15
_RED_BLUE_SPECS = (
    ((0, 80, 60), (10, 255, 255)),
    ((170, 80, 60), (180, 255, 255)),
    ((100, 80, 60), (130, 255, 255)),
)

# 足迹只收实测顶面往下这一条以内的点。再宽就会带上正面上沿，矩形被拉向相机。
TOP_LAYER_BELOW_PEAK_M = 0.010

# 方块四周大约 6 cm 的地面要贴着平面。贴着轮廓的 2 cm 是深度跳变，不看。
# 识别距离 0.2–0.8 m，这一圈像素最多约 100，高度可以偏 2 cm，散布可以到 3 cm。
# 这一圈对不上时，再看挖掉的大约 8 cm 以外、再往外 6 cm 的红地板。侧面原来能过的不改。
_GROUND_RING_M = 0.06
_GROUND_RING_SKIP_M = 0.02
_GROUND_RING_PX_MAX = 100
_GROUND_KEEPOUT_M = 0.08
_GROUND_OUTER_PX_MAX = 480
_GROUND_BIAS_M = 0.020
_GROUND_SPREAD_M = 0.030

# 射线交点与测到的深度差过这条，像素就在侧面而不是顶面。位置仍用交点，不用测到的深度。
_RAY_DEPTH_AGREE_M = 0.02
# 顶面掩膜只剥掉最外 1 个像素，避开棱上的混合深度。再往里改在平面坐标里裁，避免斜视时把矩形裁歪。

# 两套足迹中心差过这条就丢帧，不在深度投影和射线之间切换。
_SCHEME_SPLIT_M = 0.02


def _scale_for_dim_light(hsv: np.ndarray) -> float:
    # 整幅中值亮度低于约 110 时，按比例放宽饱和度和亮度下限，暗处的红蓝仍能进掩膜。
    # 不再低于颜色函数里的保底，灰地面饱和度接近 0，不会因此被收进来。
    sample = hsv[::8, ::8, 2]
    if sample.size == 0:
        return 1.0
    median_v = float(np.median(sample))
    return float(np.clip(median_v / 110.0, 0.45, 1.0))


def _hue_range(hsv: np.ndarray, spec: dict, s_low: int, v_low: int, v_high: int = 255) -> np.ndarray:
    # 色相沿用配置。饱和度和亮度单独放宽，用来补被光冲淡或压暗的那一半。
    low = np.array([spec["hsv_low"][0], s_low, v_low], dtype=np.uint8)
    high = np.array([spec["hsv_high"][0], 255, v_high], dtype=np.uint8)
    return cv2.inRange(hsv, low, high)


def _scaled_odd(full_px: int, scale: float) -> int:
    # 半分辨率上核按比例缩小，并保持奇数。5→3，15→7。再小就固定 3，1 像素核不起作用。
    if scale >= 0.99:
        return full_px
    px = max(3, int(full_px * scale))
    if px % 2 == 0:
        px += 1
    return px


def _is_white_spec(spec: dict) -> bool:
    return str(spec.get("name", "")).startswith("white")


def white_keepout(color_bgr: np.ndarray, colors: list[dict], depth: np.ndarray, intr, margin_m: float = 0.08) -> np.ndarray:
    # 白盒和周围大约 8 cm 不参与地面拟合。下沿和棱上的混合点离地面很近，会把平面拧斜。
    # 半径按这块白的距离来算，近处盒子占得多，圈也更大。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for spec in colors:
        if not _is_white_spec(spec):
            continue
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        white = cv2.bitwise_or(white, cv2.inRange(hsv, low, high))
    if int(np.count_nonzero(white)) == 0:
        return np.zeros(color_bgr.shape[:2], dtype=bool)
    z = depth[white > 0]
    z = z[np.isfinite(z) & (z > 0.05)]
    z_med = float(np.median(z)) if len(z) else 0.8
    radius = int(np.clip(round(float(margin_m) * float(intr.fx) / z_med), 6, 96))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(white, kernel) > 0


def _red_blue_mask(hsv: np.ndarray) -> np.ndarray:
    # 只用来认出旁边的红蓝杂物。白盒自己的颜色不走这段。
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in _RED_BLUE_SPECS:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)),
        )
    return mask


def _drop_washed_white(hsv: np.ndarray, white: np.ndarray, pixel_scale: float) -> np.ndarray:
    # 白掩膜会把冲掉的红蓝也收进来。这些点饱和度很低、明度很高，并且紧挨着还没冲掉的红蓝。
    colored = _red_blue_mask(hsv)
    if int(np.count_nonzero(colored)) == 0:
        return white
    px = _scaled_odd(_WASH_DILATE_PX, pixel_scale)
    near = cv2.dilate(colored, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px)))
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    drop = (white > 0) & (near > 0) & (sat < _WASH_S_MAX) & (val > _WASH_V_MIN)
    out = white.copy()
    out[drop] = 0
    return out


def _color_layers(
    color_bgr: np.ndarray, colors: list[dict], pixel_scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 白盒用饱和度和明度，不放宽到 25，也不按暗光把明度下限缩小。接缝只靠后面的闭运算。
    # 红蓝仍按色相。放宽的只放松饱和度和亮度。塔顶标明固定颜色时不放宽。不在这里做大范围闭运算。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    dim = _scale_for_dim_light(hsv)
    strict = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    loose = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    front = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_scaled_odd(3, pixel_scale),) * 2)
    for spec in colors:
        if _is_white_spec(spec):
            low = np.array(spec["hsv_low"], dtype=np.uint8)
            high = np.array(spec["hsv_high"], dtype=np.uint8)
            part = cv2.inRange(hsv, low, high)
            part = cv2.morphologyEx(part, cv2.MORPH_OPEN, speck)
            part = _drop_washed_white(hsv, part, pixel_scale)
            strict = cv2.bitwise_or(strict, part)
            continue
        if spec.get("fixed_color"):
            low = np.array(spec["hsv_low"], dtype=np.uint8)
            high = np.array(spec["hsv_high"], dtype=np.uint8)
        else:
            low = np.array(spec["hsv_low"], dtype=float)
            low[1] = np.clip(low[1] * dim, 30, 255)
            low[2] = np.clip(low[2] * dim, 20, 255)
            low = low.astype(np.uint8)
            high = np.array(spec["hsv_high"], dtype=np.uint8)
        part = cv2.inRange(hsv, low, high)
        part = cv2.morphologyEx(part, cv2.MORPH_OPEN, speck)
        strict = cv2.bitwise_or(strict, part)
        if not spec.get("fixed_color"):
            loose = cv2.bitwise_or(loose, _hue_range(hsv, spec, 25, 25))
        if spec.get("front"):
            front = cv2.bitwise_or(front, part)
    loose = cv2.bitwise_or(loose, strict)
    return strict, loose, front


def color_mask(color_bgr: np.ndarray, colors: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    strict, _loose, front = _color_layers(color_bgr, colors)
    return strict, cv2.bitwise_and(front, strict)


def _spec_is_red(spec: dict) -> bool:
    # 红在色相环两端。蓝落在 100–130。名字优先，没有名字时用色相段判断。
    name = str(spec.get("name", ""))
    if name.startswith("red"):
        return True
    if name.startswith("blue"):
        return False
    low = int(spec["hsv_low"][0])
    high = int(spec["hsv_high"][0])
    return low <= 15 or high >= 165


def _hsv_to_bgr(hue: int, sat: int, val: int) -> tuple[int, int, int]:
    patch = np.array([[[hue, sat, val]]], dtype=np.uint8)
    bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def standard_color_chips(colors: list[dict]) -> list[dict]:
    # 图像识别里的标准红、标准蓝是满饱和、满明度的原色。
    # 色相取配置门槛里包住的那一个：红是 0，蓝是 120。门槛里没有时用该段中点。
    grouped: dict[str, list[dict]] = {"red": [], "blue": []}
    for spec in colors:
        grouped["red" if _spec_is_red(spec) else "blue"].append(spec)
    chips = []
    for name, preferred in (("red", 0), ("blue", 120)):
        specs = grouped[name]
        if not specs:
            continue
        covered = any(int(spec["hsv_low"][0]) <= preferred <= int(spec["hsv_high"][0]) for spec in specs)
        if covered:
            hue = preferred
        else:
            spec = specs[0]
            hue = int(round(0.5 * (int(spec["hsv_low"][0]) + int(spec["hsv_high"][0]))))
        sat_low = min(int(spec["hsv_low"][1]) for spec in specs)
        val_low = min(int(spec["hsv_low"][2]) for spec in specs)
        chips.append(
            {
                "name": name,
                "hsv": (hue, 255, 255),
                "bgr": _hsv_to_bgr(hue, 255, 255),
                "gate_bgr": _hsv_to_bgr(hue, sat_low, val_low),
                "sat_low": sat_low,
                "val_low": val_low,
                "ranges": [(int(spec["hsv_low"][0]), int(spec["hsv_high"][0])) for spec in specs],
            }
        )
    return chips


def measure_scene_hsv(color_bgr: np.ndarray, colors: list[dict]) -> dict[str, tuple[int, int, int] | None]:
    # 只按色相圈出红和蓝，饱和度和明度留着跟门槛比。进不了严格掩膜时仍能看出偏暗还是发白。
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    found: dict[str, tuple[int, int, int] | None] = {}
    for chip in standard_color_chips(colors):
        band = np.zeros(hue.shape, dtype=bool)
        for low, high in chip["ranges"]:
            if low <= high:
                band |= (hue >= low) & (hue <= high)
            else:
                band |= (hue >= low) | (hue <= high)
        count = int(band.sum())
        if count < 30:
            found[chip["name"]] = None
            continue
        pix = hsv[band]
        values = pix[:, 0].astype(np.int16)
        if chip["name"] == "red":
            values = np.where(values > 90, values - 180, values)
        median_h = int(np.median(values)) % 180
        found[chip["name"]] = (median_h, int(np.median(pix[:, 1])), int(np.median(pix[:, 2])))
    return found


def _light_step(light_name: str, levels: dict | None, measured: tuple[int, int, int] | None, chip: dict) -> str:
    # 曝光从 bright 到 dark 递增。发白就往曝光更低的一档走，偏暗就往更高的一档走。
    names = list(levels) if levels else []
    index = names.index(light_name) if light_name in names else -1
    if measured is None:
        return "no pixels"
    _hue, sat, val = measured
    if val < chip["val_low"]:
        if 0 <= index < len(names) - 1:
            return f"too dark -> --light {names[index + 1]}"
        return "too dark"
    if sat < chip["sat_low"]:
        if index > 0:
            return f"washed out -> --light {names[index - 1]}"
        return "washed out"
    return "in range"


def render_color_standard(
    colors: list[dict],
    light_name: str = "daylight",
    light_levels: dict | None = None,
    measured: dict | None = None,
) -> np.ndarray:
    # 左块是标准色，右块是当前画面里同色相的中值。底下一条是还能进掩膜的最暗颜色。
    chips = standard_color_chips(colors)
    width, height = 880, 300
    image = np.full((height, width, 3), 32, dtype=np.uint8)
    if not chips:
        return image
    slot = width // len(chips)

    def _text(origin: tuple[int, int], line: str) -> None:
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for index, chip in enumerate(chips):
        x0 = index * slot + 24
        hue, sat, val = chip["hsv"]
        cv2.rectangle(image, (x0, 48), (x0 + 150, 148), chip["bgr"], -1)
        cv2.rectangle(image, (x0, 156), (x0 + 150, 184), chip["gate_bgr"], -1)
        _text((x0, 32), f"{chip['name']}  H{hue} S{sat} V{val}")
        _text((x0, 206), f"gate S>={chip['sat_low']} V>={chip['val_low']}")
        scene = None if measured is None else measured.get(chip["name"])
        if scene is None:
            _text((x0, 240), "scene: none")
        else:
            swatch = _hsv_to_bgr(*scene)
            cv2.rectangle(image, (x0 + 168, 48), (x0 + 250, 148), swatch, -1)
            _text((x0, 240), f"scene H{scene[0]} S{scene[1]} V{scene[2]}")
        _text((x0, 272), _light_step(light_name, light_levels, scene, chip))
    return image


def show_color_standard(image: np.ndarray, pump: bool = True) -> bool:
    # 没有显示器时不打断取流。调用方继续跑，只是看不到这块色板。
    # pump 为假时不取键盘，留给已经在等按键的那个窗口。
    try:
        cv2.imshow("color standard", image)
        if pump:
            cv2.waitKey(1)
    except cv2.error:
        return False
    return True


def _valid_depth(depth: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    # 工作距离内的有限深度才参与选块。NaN 和超距点不能拿来判断这块是不是方块。
    return np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)


def _cube_silhouettes(
    strict: np.ndarray,
    loose: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float = 1.0,
    seam_full_px: int = 11,
) -> list[np.ndarray]:
    # 绿线沿严格红蓝走。种子先要成片，再比地面长边谁更接近边长。
    # 长边超过 2.5 倍边长的块丢掉之后，稀疏的线材补不上这个门槛，不能接替方块。
    # 放宽颜色只填红蓝接缝那大约 10 px。再往外一个边长是上限，木板和机器人进不来。
    valid = _valid_depth(depth, zmin, zmax)
    ys, xs = np.nonzero(strict)
    if len(xs) < 3:
        return []
    z = depth[ys, xs]
    good = valid[ys, xs]
    if int(good.sum()) < 3:
        return []
    points = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
    height = plane.signed_distance(points)
    elevated = (height > GROUND_CLEARANCE_M) & (height < cube_size + 0.08)
    if int(elevated.sum()) < 3:
        return []
    elev = np.zeros(strict.shape, dtype=np.uint8)
    elev[ys[good][elevated], xs[good][elevated]] = 255
    height_img = np.full(strict.shape, np.nan, dtype=float)
    height_img[ys[good], xs[good]] = height
    # 定位用大约 11 px 把红蓝接上。全分辨率重描时用 3 px，避免把棱向外胀开。
    seam_px = _scaled_odd(seam_full_px, pixel_scale)
    seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seam_px, seam_px))
    blob = cv2.dilate(elev, seam_kernel)
    seeds = _ranked_seeds(blob, elev, depth, intr, plane, cube_size, limit=3)
    masks: list[np.ndarray] = []
    for seed in seeds:
        mask = _mask_from_seed(
            seed,
            strict,
            loose,
            depth,
            intr,
            plane,
            cube_size,
            height_img,
            pixel_scale,
            seam_kernel,
            zmin,
            zmax,
        )
        if mask is not None and int(np.count_nonzero(mask)) >= 10:
            masks.append(mask)
    return masks


def _cube_silhouette(
    strict: np.ndarray,
    loose: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float = 1.0,
    seam_full_px: int = 11,
) -> np.ndarray | None:
    masks = _cube_silhouettes(
        strict, loose, depth, intr, plane, cube_size, zmin, zmax, pixel_scale, seam_full_px
    )
    if not masks:
        return None
    return masks[0]


def _ranked_seeds(
    blob: np.ndarray,
    elev: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    limit: int = 3,
) -> list[np.ndarray]:
    # 通过成片、外接框和 2.5 倍边长的块都留下，最多 3 块。长边更接近边长的排在前面。
    count, labels = cv2.connectedComponents(blob, connectivity=8)
    ranked: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for lab in range(1, count):
        seed = (labels == lab) & (elev > 0)
        n_seed = int(seed.sum())
        if n_seed < 3:
            continue
        z_med = float(np.median(depth[seed]))
        if not np.isfinite(z_med) or z_med <= 1e-3:
            continue
        edge_px = cube_size * float(intr.fx) / z_med
        # 门槛按这一距离上一个面的 2% 走。远处半分辨率只剩三十来个像素时，不再用近处的 8%。
        min_solid = max(12, int(0.02 * edge_px * edge_px))
        if n_seed < min_solid:
            continue
        sy, sx = np.nonzero(seed)
        box = (int(sx.max()) - int(sx.min()) + 1) * (int(sy.max()) - int(sy.min()) + 1)
        # 线材铺开时长边也能接近边长，像素占不满外接框，先丢掉。
        if n_seed < 0.12 * box:
            continue
        long_side = _span_long_side(seed.astype(np.uint8) * 255, depth, intr, plane)
        if long_side is not None and long_side > cube_size * 2.5:
            continue
        # 量不出长边的成片块排在后面。远处深度稀时仍能留下，不让它盖过已经量准的方块。
        if long_side is None:
            rank = (1.0, 0.0, -float(n_seed))
        else:
            rank = (0.0, abs(long_side - cube_size), -float(n_seed))
        ranked.append((rank, seed))
    ranked.sort(key=lambda item: item[0])
    return [seed for _rank, seed in ranked[: max(int(limit), 1)]]


def _mask_from_seed(
    seed: np.ndarray,
    strict: np.ndarray,
    loose: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    cube_size: float,
    height_img: np.ndarray,
    pixel_scale: float,
    seam_kernel: np.ndarray,
    zmin: float,
    zmax: float,
) -> np.ndarray | None:
    valid = _valid_depth(depth, zmin, zmax)
    z_med = float(np.median(depth[seed]))
    if not np.isfinite(z_med) or z_med <= 1e-3:
        return None
    # 搜索半径是一个边长，不再乘 2.2。接缝核比这个更紧，实际只往外补大约 10 px。
    reach = float(np.clip(cube_size * float(intr.fx) / z_med, 12.0 * pixel_scale, 420.0 * pixel_scale))
    inv = np.where(seed, np.uint8(0), np.uint8(255))
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    window = dist <= reach
    radius = int(np.ceil(GROUND_CLEARANCE_M * float(intr.fx) / z_med))
    radius = int(np.clip(radius, max(2, int(round(4 * pixel_scale))), max(4, int(round(20 * pixel_scale)))))
    near = cv2.dilate(
        seed.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    contact = (
        (strict > 0)
        & np.isfinite(height_img)
        & (height_img > -0.02)
        & (height_img <= GROUND_CLEARANCE_M)
        & (near > 0)
        & window
    )
    seam_band = cv2.dilate(seed.astype(np.uint8) * 255, seam_kernel)
    bridge_px = (loose > 0) & (seam_band > 0) & window & valid & ~seed
    bridge = np.zeros(strict.shape, dtype=bool)
    by, bx = np.nonzero(bridge_px)
    if len(bx) > 0:
        bz = depth[by, bx]
        bpts = deproject_pixels(np.column_stack([bx, by]), bz, intr)
        bheight = plane.signed_distance(bpts)
        keep = (bheight > GROUND_CLEARANCE_M) & (bheight < cube_size + 0.08)
        bridge[by[keep], bx[keep]] = True
    mask = np.zeros(strict.shape, dtype=np.uint8)
    mask[seed | contact | bridge] = 255
    # 接缝上没进放宽颜色的缝用闭运算补上。补上的像素仍限制在一个边长以内。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, seam_kernel)
    mask = cv2.bitwise_and(mask, window.astype(np.uint8) * 255)
    # 上半段常常没有深度，凸包只围住下沿。同一块白、大约两个边长以内、没有深度的像素补进轮廓。
    if int(np.count_nonzero(seed)) > 0:
        near_color = (strict > 0) & (dist <= reach * 2.0) & ~valid
        if int(near_color.sum()) > 0:
            color = np.zeros(strict.shape, dtype=np.uint8)
            color[seed] = 255
            color[near_color] = 255
            _count, labels = cv2.connectedComponents(color, connectivity=8)
            touched = np.unique(labels[seed])
            touched = touched[touched > 0]
            if len(touched) > 0:
                mask[np.isin(labels, touched)] = 255
    # 凸包的顶点只来自这块方块。斜视时顶面内部被补上，旁边的木板不是顶点，进不了掩膜。
    yy, xx = np.nonzero(mask)
    if len(xx) >= 3:
        hull = cv2.convexHull(np.column_stack([xx, yy]).astype(np.int32))
        filled = np.zeros(strict.shape, dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull, 255)
        mask = filled
    return mask


def _span_long_side(component: np.ndarray, depth: np.ndarray, intr, plane: Plane) -> float | None:
    # 抽样量这块颜色在地面上的长边，用来丢掉比方块大一圈的背景。
    ys, xs = np.nonzero(component)
    if len(xs) < 12:
        return None
    step = max(len(xs) // 600, 1)
    z = depth[ys[::step], xs[::step]]
    good = np.isfinite(z)
    if int(good.sum()) < 8:
        return None
    points = deproject_pixels(
        np.column_stack([xs[::step][good], ys[::step][good]]),
        z[good],
        intr,
    )
    _coords, _projected, rect = _footprint(points, plane)
    (_center, (width, height_rect), _angle) = rect
    return float(max(width, height_rect))


def _select_component(
    mask: np.ndarray,
    front: np.ndarray,
    depth: np.ndarray,
    zmin: float,
    zmax: float,
    plane: Plane | None = None,
    intr=None,
    cube_size: float = 0.2,
    link_px: int = 15,
) -> tuple[np.ndarray, np.ndarray] | None:
    """只留一块轮廓：面积在画面的 0.4%–40%，且深度中值落在工作距离内。

    斜视时顶面和正面之间会空出大约十个像素。先膨胀再取轮廓，把同一方块的面连起来。
    轮廓内部仍只用原来的颜色像素。有地面时，在长边没有大过方块太多的轮廓里，
    选长边最接近边长的一块，避免背景里另一块红蓝占满掩膜。
    """
    valid = _valid_depth(depth, zmin, zmax)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link_px, link_px))
    linked = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_err = np.inf
    for contour in ranked:
        area = float(cv2.contourArea(contour))
        if area < 0.004 * image_area or area > 0.40 * image_area:
            continue
        filled = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
        component = cv2.bitwise_and(mask, filled)
        # 深度只用来判断这块是不是方块。面上的深度洞不从掩膜里挖掉，轮廓才留在外沿。
        measured = cv2.bitwise_and(component, valid.astype(np.uint8) * 255)
        ys, xs = np.nonzero(measured)
        if len(xs) == 0:
            continue
        samples = depth[ys, xs]
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0 or not (zmin <= float(np.median(samples)) <= zmax):
            continue
        chosen = component, cv2.bitwise_and(front, component)
        # 没有地面时维持旧行为：取最大的一块。有地面时改选长边最接近方块的那块。
        if plane is None or intr is None:
            return chosen
        long_side = _span_long_side(component, depth, intr, plane)
        if long_side is None or long_side > cube_size * 2.5:
            continue
        err = abs(long_side - cube_size)
        if err < best_err:
            best_err = err
            best = chosen
    return best


def _erode_radius(distance_m: float) -> int:
    # 0.5 m 用 5 px，1.5 m 用 3 px。更远仍保留 3 px，避免把方块蚀穿。
    radius = int(round(5.0 - (distance_m - 0.5) * 2.0))
    return int(np.clip(radius, 3, 5))


def _drop_edge_jumps(depth: np.ndarray, mask: np.ndarray, thresh_m: float = 0.015) -> np.ndarray:
    """只去掉外轮廓附近深度跳变超过 15 mm 的点。面的内部保持连续。整幅图的 80 mm 飞点过滤在这之前已经做过。"""
    from camera.source import remove_flying_pixels

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    # 只在掩膜外接框里看邻域。框外的深度本来就不会拿来比较。
    y0 = max(int(ys.min()) - 1, 0)
    y1 = min(int(ys.max()) + 2, mask.shape[0])
    x0 = max(int(xs.min()) - 1, 0)
    x1 = min(int(xs.max()) + 2, mask.shape[1])
    sub = mask[y0:y1, x0:x1]
    # 外轮廓向内约 3 像素。红蓝接缝和斜侧面在内部，上面的跳变留下。
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rim = cv2.subtract(sub, cv2.erode(sub, rim_kernel))
    local = np.where(sub > 0, depth[y0:y1, x0:x1], np.nan)
    cleaned = remove_flying_pixels(local, thresh_m)
    updated = sub.copy()
    on_rim = rim > 0
    updated[on_rim] = np.where(np.isfinite(cleaned[on_rim]), 255, 0).astype(np.uint8)
    kept = mask.copy()
    kept[y0:y1, x0:x1] = updated
    return kept


def _fill_enclosed(mask: np.ndarray) -> np.ndarray:
    # 封在面内部的洞补上。种子必须落在背景上，掩膜碰到画面边缘时不能把整幅填满。
    if int(np.count_nonzero(mask)) == 0:
        return mask
    flood = mask.copy()
    height, width = mask.shape
    seed = None
    for x in range(width):
        if flood[0, x] == 0:
            seed = (x, 0)
            break
    if seed is None:
        for y in range(height):
            if flood[y, 0] == 0:
                seed = (0, y)
                break
    if seed is None:
        return mask
    pad = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, pad, seed, 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def _half_view(color_bgr: np.ndarray, depth: np.ndarray, intr):
    # 彩色用面积平均缩小，深度用最近邻，空洞先写成 0 再还原成无效，避免 NaN 被插值抹掉。
    height, width = color_bgr.shape[:2]
    small_w = max(1, width // 2)
    small_h = max(1, height // 2)
    color = cv2.resize(color_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    filled = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
    small_depth = cv2.resize(filled, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    small_depth[small_depth <= 0] = np.nan
    sx = small_w / float(width)
    sy = small_h / float(height)
    small_intr = replace(
        intr,
        width=small_w,
        height=small_h,
        fx=float(intr.fx) * sx,
        fy=float(intr.fy) * sy,
        cx=float(intr.cx) * sx,
        cy=float(intr.cy) * sy,
    )
    return color, small_depth, small_intr, sx


def _expand_roi(mask: np.ndarray, pad: int) -> np.ndarray | None:
    # 半分辨率只给出方块大概在哪。外接框向外留一圈，全分辨率重描时棱不会被裁掉。
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    roi = np.zeros(mask.shape, dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255
    return roi


def _crop_view(color_bgr: np.ndarray, depth: np.ndarray, intr, roi: np.ndarray):
    # 只把方块附近裁出来做全分辨率。主点跟着裁切原点移，焦距不变。
    ys, xs = np.nonzero(roi)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = replace(
        intr,
        width=x1 - x0,
        height=y1 - y0,
        cx=float(intr.cx) - x0,
        cy=float(intr.cy) - y0,
    )
    return color_bgr[y0:y1, x0:x1], depth[y0:y1, x0:x1], cropped, x0, y0


def _paste(shape: tuple[int, int], tile: np.ndarray, x0: int, y0: int) -> np.ndarray:
    full = np.zeros(shape, dtype=np.uint8)
    full[y0 : y0 + tile.shape[0], x0 : x0 + tile.shape[1]] = tile
    return full


def _mask_from_plane(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    plane: Plane,
    intr,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float,
    seam_full_px: int = 11,
    roi: np.ndarray | None = None,
):
    # 半分辨率只定位。全分辨率在方块附近用小接缝核重描，足迹用这一次的掩膜。
    full_hw = color_bgr.shape[:2]
    origin = (0, 0)
    cropped = False
    view_color, view_depth, view_intr = color_bgr, depth, intr
    if roi is not None and pixel_scale >= 0.99:
        view_color, view_depth, view_intr, x0, y0 = _crop_view(color_bgr, depth, intr, roi)
        origin = (x0, y0)
        cropped = True
    if pixel_scale < 0.99:
        color, depth_s, intr_s, pixel_scale = _half_view(view_color, view_depth, view_intr)
    else:
        color, depth_s, intr_s = view_color, view_depth, view_intr
    strict, loose, front = _color_layers(color, colors, pixel_scale=pixel_scale)
    mask = _cube_silhouette(
        strict,
        loose,
        depth_s,
        intr_s,
        plane,
        cube_size,
        zmin,
        zmax,
        pixel_scale=pixel_scale,
        seam_full_px=seam_full_px,
    )
    if mask is None:
        return None
    if mask.shape[:2] != view_color.shape[:2]:
        full = (view_color.shape[1], view_color.shape[0])
        mask = cv2.resize(mask, full, interpolation=cv2.INTER_NEAREST)
        front = cv2.resize(front, full, interpolation=cv2.INTER_NEAREST)
    front = cv2.bitwise_and(front, mask)
    if cropped:
        mask = _paste(full_hw, mask, origin[0], origin[1])
        front = _paste(full_hw, front, origin[0], origin[1])
    if int(np.count_nonzero(mask)) < 10:
        return None
    return mask, front


def _masks_from_plane(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    plane: Plane,
    intr,
    cube_size: float,
    zmin: float,
    zmax: float,
    pixel_scale: float,
    seam_full_px: int = 11,
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 整幅上最多留 3 块绿线。每一块单独成掩膜，后面按地面中心比较。
    if pixel_scale < 0.99:
        color, depth_s, intr_s, pixel_scale = _half_view(color_bgr, depth, intr)
    else:
        color, depth_s, intr_s = color_bgr, depth, intr
    strict, loose, front = _color_layers(color, colors, pixel_scale=pixel_scale)
    masks = _cube_silhouettes(
        strict,
        loose,
        depth_s,
        intr_s,
        plane,
        cube_size,
        zmin,
        zmax,
        pixel_scale=pixel_scale,
        seam_full_px=seam_full_px,
    )
    found: list[tuple[np.ndarray, np.ndarray]] = []
    full = (color_bgr.shape[1], color_bgr.shape[0])
    for mask in masks:
        if mask.shape[:2] != color_bgr.shape[:2]:
            mask = cv2.resize(mask, full, interpolation=cv2.INTER_NEAREST)
            front_full = cv2.resize(front, full, interpolation=cv2.INTER_NEAREST)
        else:
            front_full = front
        front_full = cv2.bitwise_and(front_full, mask)
        if int(np.count_nonzero(mask)) < 10:
            continue
        found.append((mask, front_full))
    return found


def prepare_candidate_masks(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    zmin: float,
    zmax: float,
    plane: Plane,
    intr,
    cube_size: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 半分辨率找出最多 3 块，再各自在全分辨率上重描。一块重描失败就留定位掩膜。
    located: list[tuple[np.ndarray, np.ndarray]] = []
    if min(color_bgr.shape[:2]) >= 8:
        located = _masks_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=0.5,
            seam_full_px=11,
        )
    if not located:
        one = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=11,
        )
        return [] if one is None else [one]
    refined: list[tuple[np.ndarray, np.ndarray]] = []
    for mask, _front in located[:3]:
        roi = _expand_roi(mask, 24)
        one = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=3,
            roi=roi,
        )
        if one is not None and int(np.count_nonzero(one[0])) * 2 < int(np.count_nonzero(mask)):
            wider = _mask_from_plane(
                color_bgr,
                depth,
                colors,
                plane,
                intr,
                cube_size,
                zmin,
                zmax,
                pixel_scale=1.0,
                seam_full_px=11,
                roi=roi,
            )
            if wider is not None:
                one = wider
        refined.append(one if one is not None else (mask, _front))
    return refined


def prepare_mask(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    colors: list[dict],
    zmin: float = 0.3,
    zmax: float = 3.0,
    plane: Plane | None = None,
    intr=None,
    cube_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray] | None:
    # 有地面时，半分辨率只找方块在哪。找到后在全分辨率上用 3 px 接缝重描，足迹用这一次。
    # 半分辨率没找到时，整幅用全分辨率再找一次。重描失败才退回定位掩膜。
    if plane is not None and intr is not None:
        located = None
        if min(color_bgr.shape[:2]) >= 8:
            located = _mask_from_plane(
                color_bgr,
                depth,
                colors,
                plane,
                intr,
                cube_size,
                zmin,
                zmax,
                pixel_scale=0.5,
                seam_full_px=11,
            )
        roi = _expand_roi(located[0], 24) if located is not None else None
        refined = _mask_from_plane(
            color_bgr,
            depth,
            colors,
            plane,
            intr,
            cube_size,
            zmin,
            zmax,
            pixel_scale=1.0,
            seam_full_px=3,
            roi=roi,
        )
        # 3 px 接不上红蓝时，掩膜会只剩一条。同一块区域再用定位核重描，足迹才盖住顶面。
        if located is not None and refined is not None:
            if int(np.count_nonzero(refined[0])) * 2 < int(np.count_nonzero(located[0])):
                refined = _mask_from_plane(
                    color_bgr,
                    depth,
                    colors,
                    plane,
                    intr,
                    cube_size,
                    zmin,
                    zmax,
                    pixel_scale=1.0,
                    seam_full_px=11,
                    roi=roi,
                )
        if refined is not None:
            return refined
        return located
    strict, _loose, front = _color_layers(color_bgr, colors)
    mask = strict
    front = cv2.bitwise_and(front, mask)
    selected = _select_component(
        mask, front, depth, zmin, zmax, plane=plane, intr=intr, cube_size=cube_size
    )
    if selected is None:
        return None
    mask, front = selected
    ys, xs = np.nonzero(mask)
    finite = depth[ys, xs]
    finite = finite[np.isfinite(finite)]
    if len(finite) < 10:
        return None
    # 已经按高度去掉地面时不再腐蚀。腐蚀会吃掉落地边，足迹比真实方块小一圈。
    if plane is None or intr is None:
        distance = float(np.median(finite))
        radius = _erode_radius(distance)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.erode(mask, kernel)
        front = cv2.bitwise_and(front, mask)
    mask = _drop_edge_jumps(depth, mask, 0.015)
    mask = _fill_enclosed(mask)
    front = cv2.bitwise_and(front, mask)
    if int(np.count_nonzero(mask)) < 10:
        return None
    return mask, front


def _circular_median_hue(hues: np.ndarray) -> float:
    # OpenCV 色相是 0–180 的环。取环形中值，避免 0 和 179 被当成差很远。
    values = np.asarray(hues, dtype=float)
    if len(values) > 2000:
        values = values[:: max(len(values) // 2000, 1)]
    best = float(values[0])
    best_cost = np.inf
    for candidate in values:
        delta = (values - candidate + 90.0) % 180.0 - 90.0
        cost = float(np.median(np.abs(delta)))
        if cost < best_cost:
            best_cost = cost
            best = float(candidate)
    return best


def calibrate_hsv(
    frames: list[tuple[np.ndarray, np.ndarray]],
    colors: list[dict],
    zmin: float = 0.3,
    zmax: float = 3.0,
) -> list[dict]:
    """用约 30 帧里已经能分开的方块像素收紧色相。

    OpenCV 色相是 0–180，对应 0°–360°。8° 色相在这里是 4 个单位。
    饱和度、明度下限取这批像素的 5% 分位，并限制在不会把灰白地面收进来的范围。
    """
    updated: list[dict] = []
    for spec in colors:
        # 白的门槛是饱和度上限和明度下限。按色相收紧会把白收成一段彩色。
        if _is_white_spec(spec):
            updated.append(dict(spec))
            continue
        hues: list[np.ndarray] = []
        sats: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        for color_bgr, depth in frames:
            hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
            part = cv2.inRange(hsv, low, high)
            part = cv2.bitwise_and(part, _valid_depth(depth, zmin, zmax).astype(np.uint8) * 255)
            ys, xs = np.nonzero(part)
            if len(xs) < 30:
                continue
            pix = hsv[ys, xs]
            hues.append(pix[:, 0].astype(float))
            sats.append(pix[:, 1].astype(float))
            vals.append(pix[:, 2].astype(float))
        if not hues:
            updated.append(dict(spec))
            continue
        hue = np.concatenate(hues)
        sat = np.concatenate(sats)
        val = np.concatenate(vals)
        center = _circular_median_hue(hue)
        delta = (hue - center + 90.0) % 180.0 - 90.0
        half = max(3.0 * float(np.std(delta)), 4.0)
        hue_low = int(np.clip(np.floor(center - half), 0, 179))
        hue_high = int(np.clip(np.ceil(center + half), hue_low + 1, 180))
        sat_low = int(np.clip(np.percentile(sat, 5), 40, 200))
        val_low = int(np.clip(np.percentile(val, 5), 30, 220))
        cloned = dict(spec)
        cloned["hsv_low"] = [hue_low, sat_low, val_low]
        cloned["hsv_high"] = [hue_high, 255, 255]
        updated.append(cloned)
    return updated


# --- 掩膜反投影 ---------------------------------------------------------------
# 只用腐蚀后仍有深度的彩色像素，按已知内参变到相机系。

def _mask_points(
    color_bgr,
    depth,
    intr,
    colors,
    zmin: float,
    zmax: float,
    plane: Plane,
    cube_size: float,
    prepared=None,
):
    if prepared is None:
        prepared = prepare_mask(
            color_bgr,
            depth,
            colors,
            zmin,
            zmax,
            plane=plane,
            intr=intr,
            cube_size=cube_size,
        )
    if prepared is None:
        return None
    mask, front = prepared
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    good = np.isfinite(z)
    if int(good.sum()) < 10:
        return None
    pixels = np.column_stack([xs[good], ys[good]])
    points = deproject_pixels(pixels, z[good], intr)
    front_flag = front[ys[good], xs[good]] > 0
    return points, front_flag, int(good.sum()), mask, pixels


# --- 地面足迹 -----------------------------------------------------------------
# 点沿地面法向压到平面上，再用最小外接矩形量水平跨度。
# 矩形只用接近顶面的一层。侧面深度偏近时会在靠近相机的一侧铺开，不拿来撑矩形。
# 不拟合上表面，因此没有上表面 RMS、12° 法向和 6 mm 正方形残差。
# 长边、短边、面积三道门限仍然把不合格的足迹丢掉。

def _footprint(points: np.ndarray, plane: Plane) -> tuple[np.ndarray, np.ndarray, tuple]:
    # 投影到地面后，顶面仍是约 0.2 m 的正方形；只看见一个竖直面时会塌成一条窄带。
    projected = plane.project(points)
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    coords = np.column_stack(
        [
            (projected - origin) @ axis_x,
            (projected - origin) @ axis_y,
        ]
    )
    # minAreaRect 给出中心、长宽和一条边的角度，单位与坐标相同（米、度）。
    rect = cv2.minAreaRect(coords.astype(np.float32))
    return coords, projected, rect


def _ring_px(meters: float, z_med: float, fx: float, lo: int, hi: int) -> int:
    return int(np.clip(round(float(meters) * float(fx) / float(z_med)), lo, hi))


def _ring_band(mask: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
    if inner_px >= outer_px:
        return np.zeros(mask.shape, dtype=np.uint8)
    outer_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_px + 1, 2 * outer_px + 1))
    inner_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_px + 1, 2 * inner_px + 1))
    return cv2.subtract(cv2.dilate(mask, outer_k), cv2.dilate(mask, inner_k))


_RING_SCALE = 4


def _coarse_ring(mask: np.ndarray, inner_px: int, outer_px: int) -> np.ndarray:
    # 整幅上膨胀两三百像素要一秒。缩到四分之一再画同一圈，放大回去对原来的深度。
    scale = _RING_SCALE
    height, width = mask.shape[:2]
    if min(height, width) < scale * 8:
        return _ring_band(mask, inner_px, outer_px)
    small = cv2.resize(mask, (width // scale, height // scale), interpolation=cv2.INTER_NEAREST)
    band = _ring_band(
        small,
        max(1, int(round(inner_px / scale))),
        max(2, int(round(outer_px / scale))),
    )
    return cv2.resize(band, (width, height), interpolation=cv2.INTER_NEAREST)


def _ring_on_plane(depth: np.ndarray, ring: np.ndarray, plane: Plane, intr) -> bool:
    # 这一圈不必绕满。采到的点里将近一半贴着平面，高度偏 2 cm、散布 3 cm 以内，就算对上。
    ry, rx = np.nonzero(ring)
    if len(rx) < 30:
        return False
    step = max(len(rx) // 800, 1)
    rx = rx[::step]
    ry = ry[::step]
    rz = depth[ry, rx]
    good = np.isfinite(rz) & (rz > 0.2)
    if int(good.sum()) < 25:
        return False
    points = deproject_pixels(np.column_stack([rx[good], ry[good]]), rz[good], intr)
    dist = plane.signed_distance(points)
    band = np.abs(dist) < 0.05
    if int(band.sum()) < 25 or int(band.sum()) < 0.45 * len(dist):
        return False
    sample = dist[band]
    if abs(float(np.median(sample))) > _GROUND_BIAS_M:
        return False
    return float(np.percentile(np.abs(sample), 75)) <= _GROUND_SPREAD_M


def _white_pixels(color_bgr: np.ndarray, colors: list[dict]) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    for spec in colors:
        if not _is_white_spec(spec):
            continue
        low = np.array(spec["hsv_low"], dtype=np.uint8)
        high = np.array(spec["hsv_high"], dtype=np.uint8)
        white = cv2.bitwise_or(white, cv2.inRange(hsv, low, high))
    return white


def _ground_under_cube(
    depth: np.ndarray,
    mask: np.ndarray,
    plane: Plane,
    intr,
    color_bgr: np.ndarray | None = None,
    colors: list[dict] | None = None,
) -> bool:
    # 先看轮廓外大约 2 到 6 cm。这一圈对上就留下，侧面的判断和原来一样。
    # 对不上再看挖掉的大约 8 cm 以外。别的白盒子跳过，能看见的几段红地板对上平面即可。
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return False
    z_mask = depth[ys, xs]
    z_mask = z_mask[np.isfinite(z_mask) & (z_mask > 0.2)]
    if len(z_mask) < 10:
        return False
    z_med = float(np.median(z_mask))
    outer = _ring_px(_GROUND_RING_M, z_med, intr.fx, 8, _GROUND_RING_PX_MAX)
    inner = _ring_px(_GROUND_RING_SKIP_M, z_med, intr.fx, 3, max(outer - 2, 3))
    if inner >= outer:
        inner = max(1, outer // 3)
    if _ring_on_plane(depth, _coarse_ring(mask, inner, outer), plane, intr):
        return True
    far_in = _ring_px(_GROUND_KEEPOUT_M, z_med, intr.fx, outer + 1, _GROUND_OUTER_PX_MAX)
    far_out = _ring_px(
        _GROUND_KEEPOUT_M + _GROUND_RING_M, z_med, intr.fx, far_in + 1, _GROUND_OUTER_PX_MAX
    )
    far = _coarse_ring(mask, far_in, far_out)
    if color_bgr is not None and colors is not None and far.shape == color_bgr.shape[:2]:
        far[_white_pixels(color_bgr, colors) > 0] = 0
    if _ring_on_plane(depth, far, plane, intr):
        return True
    # 两圈被上面那块或身后的杂物挡住，或者落到画面外、深度空洞里。
    # 改看画面里其余有深度、又不是白盒的地方。贴着地面就留下，白块贴不贴边都一样。
    # 一点都采不到，就沿用已经得到的地面。这些点对不上平面，仍然丢掉。
    return _visible_floor_ok(depth, mask, plane, intr, color_bgr, colors)


def _visible_floor_ok(
    depth: np.ndarray,
    mask: np.ndarray,
    plane: Plane,
    intr,
    color_bgr: np.ndarray | None,
    colors: list[dict] | None,
) -> bool:
    # 和橙线同一批点：画面里有深度、又不是白盒的地方。离平面 2 cm 以内、散布 3 cm 以内就继续认。
    # 采样点太少时沿用这块地面。对得上的点太少，或高度偏了，就丢掉。
    stride = 4
    vs, us = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    outside = mask[vs, us] == 0
    if color_bgr is not None and colors is not None and color_bgr.shape[:2] == depth.shape[:2]:
        outside &= _white_pixels(color_bgr, colors)[vs, us] == 0
    z = depth[vs, us]
    valid = outside & np.isfinite(z) & (z > 0.2)
    if int(valid.sum()) < 25:
        return True
    points = deproject_pixels(
        np.column_stack([us[valid].ravel(), vs[valid].ravel()]),
        np.asarray(z[valid], dtype=float).ravel(),
        intr,
    )
    dist = plane.signed_distance(points)
    near = np.abs(dist) < 0.05
    if int(near.sum()) < 25:
        return False
    sample = dist[near]
    if abs(float(np.median(sample))) > _GROUND_BIAS_M:
        return False
    return float(np.percentile(np.abs(sample), 75)) <= _GROUND_SPREAD_M


def _keep_far_span(hits: np.ndarray, plane: Plane, cube_size: float) -> np.ndarray | None:
    # 正面的射线先碰到这一层，交点在盒子和相机之间。近处正面像素多，不能用全部交点的平均位置定方向。
    # 先取离相机最远的那一团来定方向，再留下大约一个边长。贴边要转多少是另一件事，这里不动。
    if len(hits) < 30:
        return None
    normal = np.asarray(plane.normal, dtype=float).reshape(3)
    on_ground = hits - normal * float(cube_size)
    cam = np.asarray(plane.project(np.zeros(3)), dtype=float).reshape(3)
    flat = on_ground - cam
    flat = flat - normal * (flat @ normal)[:, None]
    reach = np.linalg.norm(flat, axis=1)
    far_reach = float(np.percentile(reach, 98))
    span = float(cube_size) + 0.02
    cluster = reach >= far_reach - span
    if int(cluster.sum()) < 30:
        cluster = np.ones(len(hits), dtype=bool)
    away = flat[cluster].mean(axis=0)
    length = float(np.linalg.norm(away))
    if length < 1e-4:
        return hits
    away = away / length
    along = flat @ away
    far = float(np.percentile(along[cluster], 98))
    keep = along >= far - span
    if int(keep.sum()) < 30:
        return None
    return hits[keep]


def _top_from_color(
    mask: np.ndarray,
    plane: Plane,
    intr,
    cube_size: float,
    lift: float | None = None,
) -> np.ndarray | None:
    # 白像素只定射线，和地面抬高 lift 的平面相交。不抄旁边的深度。
    # 一层的 lift 是一个边长。上面那块用两个边长，远处保留的长度仍是一个边长。
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return None
    step = max(len(xs) // 2500, 1)
    pixels = np.column_stack([xs[::step], ys[::step]]).astype(float)
    hits = _intersect_lifted_plane(pixels, plane, intr, cube_size if lift is None else lift)
    hits = hits[np.isfinite(hits[:, 2])]
    return _keep_far_span(hits, plane, cube_size)


def _intersect_lifted_plane(pixels: np.ndarray, plane: Plane, intr, lift: float) -> np.ndarray:
    # 像素只定射线。交点落在与地面平行、抬高 lift 的平面上，不乘测到的深度。
    dirs = deproject_pixels(pixels, np.ones(len(pixels)), intr)
    denom = dirs @ np.asarray(plane.normal, dtype=float)
    level = float(plane.offset) + float(lift)
    t = np.full(len(dirs), np.nan)
    ok = np.abs(denom) > 1e-6
    t[ok] = level / denom[ok]
    t[~(t > 0.0)] = np.nan
    return t[:, None] * dirs


def _ray_top_points(
    pixels: np.ndarray,
    measured: np.ndarray,
    front: np.ndarray,
    depth_shape: tuple[int, int],
    plane: Plane,
    intr,
    cube_size: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    # 顶面外圈两三像素是棱，深度常混进正面。先剥掉，点不够再退回没剥的那一版。
    # 测到的深度只用来丢掉侧面：侧面的射线要走得更远才碰到抬高的平面。
    if len(pixels) < 3:
        return None
    sel = np.zeros(depth_shape, dtype=np.uint8)
    sel[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(sel, kernel)
    keep = eroded[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] > 0
    if int(keep.sum()) < 20:
        keep = np.ones(len(pixels), dtype=bool)
    hits = _intersect_lifted_plane(pixels[keep], plane, intr, cube_size)
    z_hit = hits[:, 2]
    agree = np.isfinite(z_hit) & (np.abs(z_hit - measured[keep]) < _RAY_DEPTH_AGREE_M)
    if int(agree.sum()) < 3:
        return None
    median_dz = float(np.median(np.abs(z_hit[agree] - measured[keep][agree])))
    return hits[agree], front[keep][agree], median_dz


def _rectangle_on_plane(
    points: np.ndarray,
    plane: Plane,
    cube_size: float,
    min_footprint_fraction: float,
    size_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float] | None:
    # 投影后做最小外接矩形。长边、短边或面积不合格就不用这套点。
    if len(points) < 3:
        return None
    coords, _projected, rect = _footprint(points, plane)
    (center_u, center_v), (width, height_rect), angle_deg = rect
    long_side = max(width, height_rect)
    short_side = min(width, height_rect)
    fraction = float(width * height_rect) / (cube_size * cube_size)
    if fraction < min_footprint_fraction:
        return None
    z_ref = float(np.median(points[:, 2]))
    if not np.isfinite(z_ref) or z_ref <= 0.0:
        z_ref = 1.0
    # 0.2 m 方块近处大约 2.4 cm。0.35 m 白盒近处用满 size_tolerance，0.12 倍边长只有约 4 cm。
    if float(cube_size) >= 0.30:
        side_limit = float(size_tolerance)
    else:
        side_limit = min(float(size_tolerance), max(0.12 * cube_size, 0.02 * z_ref))
    if abs(long_side - cube_size) > side_limit or short_side < cube_size * 0.45:
        return None
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    bottom = origin + center_u * axis_x + center_v * axis_y
    box = cv2.boxPoints(rect)
    corners = origin + box[:, 0:1] * axis_x + box[:, 1:2] * axis_y
    angle = np.deg2rad(angle_deg)
    edge_2d = np.array([np.cos(angle), np.sin(angle)])
    return bottom, corners, coords, edge_2d, fraction, float(long_side), float(short_side)


def detect_visible_cube(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.2,
    min_pixels: int = 400,
    min_footprint_fraction: float = 0.45,
    size_tolerance: float = 0.06,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 3.0,
    rejects: dict | None = None,
    prepared=None,
    ground_color: np.ndarray | None = None,
) -> CubePose | None:
    # 颜色足迹给出底面中心和 yaw。面积、长边或短边不合格就丢帧，不拟合上表面。
    def fail(reason: str) -> None:
        # 记一次拒绝原因。返回值是 None，调用方据此丢掉这一帧。
        if rejects is not None:
            rejects[reason] = int(rejects.get(reason, 0)) + 1
        return None

    # 1. 颜色点是否够多，并且贴在地面以上、高度接近方块边长。
    # 足迹只投掩膜里接近顶面的那一层。侧面和落地边留在掩膜里，不参与矩形。
    # 掩膜之外的木台即使高度差不多，也不会被投到这个矩形上。
    packed = _mask_points(
        color_bgr, depth, intr, colors, zmin, zmax, plane, cube_size, prepared=prepared
    )
    if packed is None:
        return fail("mask")
    points, front_flag, count, mask, pixels = packed
    if count < min_pixels:
        return fail("depth")
    height = plane.signed_distance(points)
    # 掩膜已经去掉了方块占地以外的地面。这里再丢掉钻到地下和高过方块的点，落地边保留。
    on_cube = (height > -0.02) & (height < cube_size + 0.08)
    if int(on_cube.sum()) < min_pixels:
        return fail("height_band")
    points = points[on_cube]
    front_flag = front_flag[on_cube]
    pixels = pixels[on_cube]
    # 落地边和凸包里贴近地面的点不参与 90 分位。峰值只在离地超过半个边长的点上取。
    # 上半段没有深度时，这些点会少于 3 个。白块已经够大就改用射线补顶面，不在这里停。
    height = plane.signed_distance(points)
    body = height > max(GROUND_CLEARANCE_M, 0.5 * cube_size)
    filled = None
    if int(body.sum()) < 3:
        filled = _top_from_color(mask, plane, intr, cube_size)
        if filled is None:
            return fail("body")
        peak = float(cube_size)
    else:
        peak = float(np.percentile(height[body], 90))
        if abs(peak - cube_size) > size_tolerance + 0.05 and peak < cube_size * 0.5:
            return fail("peak")
        # 顶面往下 1 cm。正面上沿留在这层外面，投到地面时不会把框拉出绿线。
        top = body & (height >= peak - TOP_LAYER_BELOW_PEAK_M)
        if int(top.sum()) < 3:
            filled = _top_from_color(mask, plane, intr, cube_size)
            if filled is None:
                return fail("top")
    # 方块外面几厘米对不上这块平面时，深度足迹会落在翘起的面上。这一帧丢掉，不用这张平面做矩形。
    # 分割可以用涂掉上层之后的图。判断地面时仍用原图，旁边的白盒才会被跳过。
    seen = color_bgr if ground_color is None else ground_color
    ground_ok = _ground_under_cube(depth, mask, plane, intr, seen, colors)
    if not ground_ok:
        return fail("ground_ring")
    if filled is not None:
        rect = _rectangle_on_plane(
            filled,
            plane,
            cube_size,
            min_footprint_fraction,
            size_tolerance,
        )
        if rect is None:
            return fail("footprint_long")
        bottom, corners, coords, edge_2d, fraction, long_side, short_side = rect
        front_flag = np.zeros(len(coords), dtype=bool)
        source = "ray"
    else:
        # 2. 两套足迹。深度把测到的顶面点压到地面。射线和抬高一个边长的平面求交，中心用交点的外接矩形。
        # 射线点够就用射线。两套中心差过 2 cm 时丢帧，不在两套之间切换，避免多帧里出现两个位置。
        points_top = points[top]
        front_top = front_flag[top]
        pixels_top = pixels[top]
        # 脚下平面贴住时，正面棱的深度和抬高一个边长的平面对不上。这些点留下会把矩形拉向相机。
        hits = _intersect_lifted_plane(pixels_top, plane, intr, cube_size)
        agree = np.isfinite(hits[:, 2]) & (np.abs(hits[:, 2] - points_top[:, 2]) < _RAY_DEPTH_AGREE_M)
        if int(agree.sum()) >= 20:
            points_top = points_top[agree]
            front_top = front_top[agree]
            pixels_top = pixels_top[agree]
        old = _rectangle_on_plane(
            points_top,
            plane,
            cube_size,
            min_footprint_fraction,
            size_tolerance,
        )
        ray = _ray_top_points(
            pixels_top,
            points_top[:, 2],
            front_top,
            depth.shape[:2],
            plane,
            intr,
            cube_size,
        )
        new = None if ray is None else _rectangle_on_plane(
            ray[0],
            plane,
            cube_size,
            min_footprint_fraction,
            size_tolerance,
        )
        ray_front = None if ray is None or new is None else ray[1]
        if old is not None and new is not None and float(np.linalg.norm(old[0] - new[0])) > _SCHEME_SPLIT_M:
            return fail("footprint_split")
        if new is not None:
            bottom, corners, coords, edge_2d, fraction, long_side, short_side = new
            front_flag = ray_front
            source = "ray"
        elif old is not None:
            bottom, corners, coords, edge_2d, fraction, long_side, short_side = old
            front_flag = front_top
            source = "depth"
        else:
            return fail("footprint_long")
    # 3. 底面中心给位置。能分出正面颜色时，用颜色重心相对足迹中心的方向当 yaw；
    #    否则用矩形棱的方向，并折进 90° 周期。
    axis_x, axis_y = plane.basis()
    front_ratio = float(front_flag.mean()) if len(front_flag) else 0.0
    # 正面像素太少或几乎占满（整块都是同一颜色）时，分不出哪条棱朝前。
    front_seen = 0.08 < front_ratio < 0.85
    if front_seen:
        # 正面颜色的重心相对足迹中心的偏移，就是该面法向在地面上的投影。
        shift = coords[front_flag].mean(axis=0) - coords.mean(axis=0)
        if np.linalg.norm(shift) > 1e-4:
            edge_2d = shift / np.linalg.norm(shift)
    edge = edge_2d[0] * axis_x + edge_2d[1] * axis_y
    edge = edge / np.linalg.norm(edge)
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    # 棱方向只旋转、不平移。yaw 取它在世界 XY 上相对 X 轴的角。
    edge_world = rotation.T @ edge
    yaw = yaw_about_z(edge_world)
    if not front_seen:
        yaw = fold_yaw_quarter(yaw)
    return CubePose(
        bottom_center=bottom,
        yaw=yaw,
        edge_direction=edge,
        footprint_fraction=fraction,
        front_color=front_seen,
        inliers=count,
        corners=corners,
        source=source,
        long_side=float(long_side),
        short_side=float(short_side),
        peak=float(peak),
    )


# 上面那块要明显高于下层顶面。0.43 m 以下仍算下面那块，避免一层的顶面被当成第二块。
_UPPER_CLEAR_M = 0.08
_UPPER_MIN_POINTS = 80
_RIM_MIN_M = 0.05


def _away_on_plane(plane: Plane, point: np.ndarray) -> np.ndarray | None:
    # 从相机在地面上的落点指向这块白，得到离开相机的方向。
    cam = np.asarray(plane.project(np.zeros(3)), dtype=float).reshape(3)
    away = np.asarray(point, dtype=float).reshape(3) - cam
    normal = np.asarray(plane.normal, dtype=float)
    away = away - normal * float(np.dot(away, normal))
    length = float(np.linalg.norm(away))
    if length < 1e-4:
        return None
    return away / length


def _index_mask(shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if int(np.count_nonzero(chosen)) == 0:
        return mask
    mask[ys[chosen], xs[chosen]] = 255
    return mask


def _white_cloud(color_bgr, depth, intr, plane: Plane, colors, zmin: float, zmax: float):
    # 白像素里有深度的，记下离地高度。上面那块和下面那块靠这个高度切开。
    white = _white_pixels(color_bgr, colors)
    ys, xs = np.nonzero(white)
    empty = np.zeros(0, dtype=bool)
    if len(xs) == 0:
        return white, ys, xs, empty, np.zeros((0, 3)), np.zeros(0)
    z = depth[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    points = np.full((len(xs), 3), np.nan)
    heights = np.full(len(xs), np.nan)
    if int(good.sum()) >= 3:
        found = deproject_pixels(np.column_stack([xs[good], ys[good]]), z[good], intr)
        points[good] = found
        heights[good] = plane.signed_distance(found)
    return white, ys, xs, good, points, heights


def _blank_lower_view(color_bgr, depth, ys, xs, good, heights, plane: Plane, intr, cut: float):
    # 量下面那块时，离地超过大约一个边长的白不参与。没有深度时，射线已经朝向比这一层更高的方向，同样涂掉。
    if len(xs) == 0:
        return color_bgr, depth, False
    above = good & np.isfinite(heights) & (heights > cut)
    missing = np.flatnonzero(~good)
    if len(missing) > 0:
        dirs = deproject_pixels(
            np.column_stack([xs[missing], ys[missing]]),
            np.ones(len(missing)),
            intr,
        )
        rising = dirs @ np.asarray(plane.normal, dtype=float) >= 0.0
        above[missing[rising]] = True
    if int(above.sum()) == 0:
        return color_bgr, depth, False
    out = color_bgr.copy()
    cleared = np.array(depth, copy=True)
    mark = _index_mask(color_bgr.shape[:2], ys, xs, above)
    out[mark > 0] = (40, 40, 40)
    cleared[mark > 0] = np.nan
    return out, cleared, True


def _complete_short(points: np.ndarray, plane: Plane, cube_size: float, size_tolerance: float):
    # 长边已经像一个边长，短边只有大约 5 cm 以上、又不到半个面。缺的一半补向离开相机的方向。
    if len(points) < 8:
        return None
    _coords, _projected, rect = _footprint(points, plane)
    (center_u, center_v), (width, height_rect), angle_deg = rect
    long_side = float(max(width, height_rect))
    short_side = float(min(width, height_rect))
    z_ref = float(np.median(points[:, 2]))
    if not np.isfinite(z_ref) or z_ref <= 0.0:
        z_ref = 1.0
    if float(cube_size) >= 0.30:
        side_limit = float(size_tolerance)
    else:
        side_limit = min(float(size_tolerance), max(0.12 * cube_size, 0.02 * z_ref))
    if abs(long_side - cube_size) > side_limit:
        return None
    if short_side < _RIM_MIN_M or short_side >= cube_size * 0.45:
        return None
    axis_x, axis_y = plane.basis()
    origin = plane.origin()
    partial = origin + center_u * axis_x + center_v * axis_y
    away = _away_on_plane(plane, partial)
    if away is None:
        return None
    angle = np.deg2rad(angle_deg if width >= height_rect else angle_deg + 90.0)
    long_dir = np.cos(angle) * axis_x + np.sin(angle) * axis_y
    normal = np.asarray(plane.normal, dtype=float)
    long_dir = long_dir - normal * float(np.dot(long_dir, normal))
    length = float(np.linalg.norm(long_dir))
    if length < 1e-6:
        return None
    long_dir = long_dir / length
    side = np.cross(normal, long_dir)
    side = side / np.linalg.norm(side)
    if float(np.dot(side, away)) < 0.0:
        side = -side
    bottom = partial + side * (0.5 * (float(cube_size) - short_side))
    half = 0.5 * float(cube_size)
    corners = np.vstack(
        [
            bottom - half * long_dir - half * side,
            bottom + half * long_dir - half * side,
            bottom + half * long_dir + half * side,
            bottom - half * long_dir + half * side,
        ]
    )
    edge_2d = np.array([float(np.dot(long_dir, axis_x)), float(np.dot(long_dir, axis_y))])
    fraction = float(long_side * short_side) / (cube_size * cube_size)
    return bottom, corners, _coords, edge_2d, fraction, float(cube_size), float(cube_size)


def _fit_points(points, plane, cube_size, min_footprint_fraction, size_tolerance, allow_complete: bool):
    rect = _rectangle_on_plane(points, plane, cube_size, min_footprint_fraction, size_tolerance)
    if rect is not None:
        return rect
    if not allow_complete:
        return None
    return _complete_short(points, plane, cube_size, size_tolerance)


def _pose_from_rect(rect, peak: float, count: int, plane: Plane, rotation, translation, source: str) -> CubePose | None:
    bottom, corners, _coords, edge_2d, fraction, long_side, short_side = rect
    axis_x, axis_y = plane.basis()
    edge = edge_2d[0] * axis_x + edge_2d[1] * axis_y
    length = float(np.linalg.norm(edge))
    if length < 1e-8:
        return None
    edge = edge / length
    if rotation is None:
        rotation = np.eye(3)
    if translation is None:
        translation = np.zeros(3)
    yaw = fold_yaw_quarter(yaw_about_z(rotation.T @ edge))
    return CubePose(
        bottom_center=bottom,
        yaw=float(yaw),
        edge_direction=edge,
        footprint_fraction=float(fraction),
        front_color=False,
        inliers=int(count),
        corners=corners,
        source=source,
        long_side=float(long_side),
        short_side=float(short_side),
        peak=float(peak),
    )


def _measure_mask(
    mask: np.ndarray,
    depth_points: np.ndarray | None,
    plane: Plane,
    intr,
    cube_size: float,
    lift: float,
    min_footprint_fraction: float,
    size_tolerance: float,
    allow_complete: bool,
    rotation,
    translation,
    count: int,
) -> CubePose | None:
    # 顶面点够就直接量。不够，或只剩一条棱，就用射线交到这一层的高度，短棱再往远处补。
    if depth_points is not None and len(depth_points) >= 20:
        rect = _fit_points(depth_points, plane, cube_size, min_footprint_fraction, size_tolerance, allow_complete)
        if rect is not None:
            pose = _pose_from_rect(rect, lift, count, plane, rotation, translation, "depth")
            if pose is not None:
                pose.mask = mask
                return pose
    filled = _top_from_color(mask, plane, intr, cube_size, lift=lift)
    if filled is None:
        return None
    rect = _fit_points(filled, plane, cube_size, min_footprint_fraction, size_tolerance, allow_complete)
    if rect is None:
        return None
    pose = _pose_from_rect(rect, lift, count, plane, rotation, translation, "ray")
    if pose is not None:
        pose.mask = mask
    return pose


def _on_lower_shelf(points: np.ndarray, anchor: CubePose, plane: Plane, cube_size: float) -> np.ndarray:
    # 下层四周放开一块盒子，上面那块自己再伸出半个边长也留着。更远的白墙不算。
    center = plane.project(np.asarray(anchor.bottom_center, dtype=float).reshape(1, 3))[0]
    flat = plane.project(np.atleast_2d(np.asarray(points, dtype=float)))
    normal = np.asarray(plane.normal, dtype=float)
    delta = flat - center
    delta = delta - normal * (delta @ normal)[:, None]
    return np.linalg.norm(delta, axis=1) <= (2.0 * float(cube_size))


def _upper_pose(
    white, ys, xs, good, points, heights, depth, plane, intr, cube_size, min_footprint_fraction, size_tolerance, rotation, translation,
    anchor: CubePose | None = None,
) -> CubePose | None:
    # 离地下层顶面 8 cm 以上、又不超过两层加 8 cm 的白，才可能是上面那块。
    # 已经有下层时，只留这块青色框附近的白，中心仍自己量，不把下层中心往上抬。
    hi = (
        good
        & np.isfinite(heights)
        & (heights > cube_size + _UPPER_CLEAR_M)
        & (heights < 2.0 * cube_size + _UPPER_CLEAR_M)
    )
    if anchor is not None and int(hi.sum()) > 0:
        hi = hi & _on_lower_shelf(points, anchor, plane, cube_size)
    if int(hi.sum()) < _UPPER_MIN_POINTS:
        return None
    peak = float(np.percentile(heights[hi], 90))
    if peak < cube_size + 0.15:
        return None
    lift = 2.0 * float(cube_size)
    upper_mask = _index_mask(white.shape, ys, xs, hi)
    near = cv2.dilate(upper_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    missing = (white > 0) & ~np.isfinite(depth)
    upper_mask[missing & (near > 0)] = 255
    # 最高处不一定刚好像两个边长。先用这一层最上的一截点；对不上再交给射线。
    depth_points = None
    top = hi & (heights >= peak - TOP_LAYER_BELOW_PEAK_M)
    if int(top.sum()) >= 20:
        depth_points = points[top]
    pose = _measure_mask(
        upper_mask,
        depth_points,
        plane,
        intr,
        cube_size,
        lift,
        min_footprint_fraction,
        size_tolerance,
        True,
        rotation,
        translation,
        int(hi.sum()),
    )
    if pose is None or anchor is None:
        return pose
    # 补全以后的底面仍要落在下层附近，底面高度是下层顶面，不是把下层中心抬上去。
    if not bool(_on_lower_shelf(pose.bottom_center.reshape(1, 3), anchor, plane, cube_size)[0]):
        return None
    return pose


def _flat_dir(vector: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    flat = np.asarray(vector, dtype=float) - normal * float(np.dot(vector, normal))
    length = float(np.linalg.norm(flat))
    if length < 1e-6:
        return None
    return flat / length


def _edge_from_points(points: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    # 一条边上的点。返回中点、沿边方向、长度。
    if len(points) < 4:
        return None
    center = np.median(points, axis=0)
    delta = points - center
    delta = delta - normal * (delta @ normal)[:, None]
    direction = _flat_dir(np.linalg.eigh(delta.T @ delta)[1][:, -1], normal)
    if direction is None:
        return None
    span = float(np.ptp(points @ direction))
    mid = center - direction * float((center @ direction) - np.median(points @ direction))
    return mid, direction, span


def _square_from_edge(mid: np.ndarray, edge_dir: np.ndarray, away: np.ndarray, plane: Plane, cube_size: float, projected: np.ndarray):
    bottom = np.asarray(mid, dtype=float) + away * (0.5 * float(cube_size))
    half = 0.5 * float(cube_size)
    corners = np.vstack(
        [
            bottom - half * edge_dir - half * away,
            bottom + half * edge_dir - half * away,
            bottom + half * edge_dir + half * away,
            bottom - half * edge_dir + half * away,
        ]
    )
    axis_x, axis_y = plane.basis()
    edge_2d = np.array([float(edge_dir @ axis_x), float(edge_dir @ axis_y)])
    fraction = 0.02 / float(cube_size)
    return bottom, corners, projected, edge_2d, fraction, float(cube_size), float(cube_size)


def _front_among_two_faces(
    lip_pts: np.ndarray,
    corner: np.ndarray,
    away: np.ndarray,
    normal: np.ndarray,
    cube_size: float,
    size_tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    # 角朝向镜头时，近处是两条边。按离开角的方向拆开，只留更正对镜头、长度又接近一个边长的那一条。
    rel = lip_pts - corner
    rel = rel - normal * (rel @ normal)[:, None]
    dist = np.linalg.norm(rel, axis=1)
    keep = dist > 0.02
    rel = rel[keep]
    pts = lip_pts[keep]
    if len(pts) < 8:
        return None
    right = np.cross(normal, away)
    right_len = float(np.linalg.norm(right))
    if right_len < 1e-6:
        return None
    right = right / right_len
    ang = np.arctan2(rel @ right, rel @ away)
    order = np.argsort(ang)
    ang = ang[order]
    pts = pts[order]
    cuts = np.where(np.diff(ang) > np.deg2rad(20.0))[0]
    bounds = np.concatenate([[-1], cuts, [len(ang) - 1]])
    chosen = None
    for start, stop in zip(bounds[:-1], bounds[1:]):
        group = pts[start + 1 : stop + 1]
        edge = _edge_from_points(group, normal)
        if edge is None:
            continue
        mid, direction, span = edge
        if abs(span - float(cube_size)) > float(size_tolerance):
            continue
        facing = abs(float(direction @ away))
        if facing > 0.85:
            continue
        if chosen is None or facing < chosen[0]:
            chosen = (facing, mid, direction)
    if chosen is None:
        return None
    return chosen[1], chosen[2]


def _square_from_near_edge(
    points: np.ndarray,
    plane: Plane,
    cube_size: float,
    size_tolerance: float,
    heights: np.ndarray | None = None,
):
    # 正面底边落在地面上。中心向远处补半个边长。转角用这条底边，不用顶面。
    if len(points) < 8:
        return None
    projected = plane.project(points)
    cam = np.asarray(plane.project(np.zeros(3)), dtype=float).reshape(3)
    away = _away_on_plane(plane, np.median(projected, axis=0))
    if away is None:
        return None
    normal = np.asarray(plane.normal, dtype=float)
    if heights is None:
        band = np.ones(len(points), dtype=bool)
    else:
        column = np.asarray(heights, dtype=float)
        band = (column > 0.02) & (column < 0.12)
        if int(band.sum()) < 8:
            band = np.ones(len(points), dtype=bool)
    base = projected[band]
    along = (base - cam) @ away
    front = float(np.percentile(along, 5))
    lip = along <= front + 0.03
    if int(lip.sum()) < 4:
        return None
    lip_pts = base[lip]
    limit = float(size_tolerance)
    edge = _edge_from_points(lip_pts, normal)
    if edge is None:
        return None
    mid, direction, span = edge
    inward = _flat_dir(np.cross(normal, direction), normal)
    off = None if inward is None else np.abs((lip_pts - mid) @ inward)
    bent = off is not None and float(np.percentile(off, 75)) > 0.02
    corner = base[int(np.argmin((base - cam) @ away))]
    if bent or abs(span - float(cube_size)) > limit:
        split = _front_among_two_faces(base, corner, away, normal, cube_size, size_tolerance)
        if split is None:
            return None
        mid, direction = split
        inward = _flat_dir(np.cross(normal, direction), normal)
    elif abs(float(direction @ away)) > 0.85:
        return None
    if inward is None:
        return None
    if float(inward @ away) < 0.0:
        inward = -inward
    return _square_from_edge(mid, direction, inward, plane, cube_size, projected)


def _stack_lower_from_face(
    ys,
    xs,
    good,
    points,
    heights,
    plane,
    shape: tuple[int, int],
    cube_size,
    size_tolerance,
    rotation,
    translation,
) -> CubePose | None:
    # 两层时优先认正面。底边在地面上，缺的一半向远处补。没有上下成面的盒身，仍不认。
    body = good & np.isfinite(heights) & (heights > 0.02) & (heights <= float(cube_size) + 0.02)
    if int(body.sum()) < 80:
        return None
    column = heights[body]
    if float(np.percentile(column, 90) - np.percentile(column, 10)) < 0.15:
        return None
    rect = _square_from_near_edge(points[body], plane, cube_size, size_tolerance, column)
    if rect is None:
        return None
    pose = _pose_from_rect(rect, float(cube_size), int(body.sum()), plane, rotation, translation, "rim")
    if pose is None:
        return None
    pose.mask = _index_mask(shape, ys, xs, body)
    return pose


def _lower_from_white_face(
    color_bgr,
    depth,
    plane: Plane,
    intr,
    colors,
    cube_size,
    size_tolerance,
    rotation,
    translation,
) -> CubePose | None:
    # 两层叠在一起时，下面那块的正面往往一整面都是白，深度却很稀，成块的种子组不起来。
    # 用贴地的下沿当靠近镜头的边，向远处补成方块。上面没有更高的白，就不走这里，单独一块仍要有深度。
    white = _white_pixels(color_bgr, colors)
    if int(np.count_nonzero(white)) < 400:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count < 2:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[label, cv2.CC_STAT_AREA]) < 400:
        return None
    ys, xs = np.nonzero(labels == label)
    if int(ys.max() - ys.min()) < 20:
        return None
    # 只取贴着画面里这块白最下沿的一两行。再往上的点落到地面上会偏近，把边拉向镜头。
    bottom = ys >= int(ys.max()) - 1
    if int(bottom.sum()) < 12:
        return None
    dirs = deproject_pixels(np.column_stack([xs[bottom], ys[bottom]]), np.ones(int(bottom.sum())), intr)
    normal = np.asarray(plane.normal, dtype=float)
    denom = dirs @ normal
    scale = np.full(len(dirs), np.nan)
    facing = denom < -1e-4
    scale[facing] = float(plane.offset) / denom[facing]
    hit = facing & np.isfinite(scale) & (scale > 0.25) & (scale < 3.5) & (dirs[:, 2] > 0.05)
    if int(hit.sum()) < 12:
        return None
    ground_pts = dirs[hit] * scale[hit, None]
    away = _away_on_plane(plane, np.median(ground_pts, axis=0))
    if away is None:
        return None
    rect = _square_from_near_edge(ground_pts, plane, cube_size, size_tolerance)
    if rect is None:
        return None
    front = rect[0] - away * (0.5 * float(cube_size))
    x0 = int(xs.min()) - 6
    x1 = int(xs.max()) + 6
    wys, wxs = np.nonzero(white)
    column = (wxs >= x0) & (wxs <= x1)
    if int(column.sum()) < _UPPER_MIN_POINTS:
        return None
    face_dirs = deproject_pixels(np.column_stack([wxs[column], wys[column]]), np.ones(int(column.sum())), intr)
    face_denom = face_dirs @ away
    face_scale = np.full(len(face_dirs), np.nan)
    face_ok = face_denom > 1e-4
    face_scale[face_ok] = float(np.dot(front, away)) / face_denom[face_ok]
    on_face = face_ok & np.isfinite(face_scale) & (face_scale > 0.25) & (face_scale < 3.5)
    if int(on_face.sum()) < _UPPER_MIN_POINTS:
        return None
    face_pts = face_dirs[on_face] * face_scale[on_face, None]
    heights = plane.signed_distance(face_pts)
    cube = float(cube_size)
    higher = (heights > cube + _UPPER_CLEAR_M) & (heights < 2.0 * cube + _UPPER_CLEAR_M)
    if int(higher.sum()) < _UPPER_MIN_POINTS:
        return None
    if float(np.percentile(heights[higher], 90)) < cube + 0.15:
        return None
    body = (heights > 0.05) & (heights <= cube + 0.02)
    if int(body.sum()) < _UPPER_MIN_POINTS:
        return None
    if float(np.percentile(heights[body], 90) - np.percentile(heights[body], 10)) < 0.15:
        return None
    t_face = float(np.median(face_scale[on_face][body]))
    if not np.isfinite(t_face) or t_face <= 0.2:
        return None
    edge_px = cube * float(intr.fx) / t_face
    rows = wys[column][np.flatnonzero(on_face)[body]]
    if int(rows.max() - rows.min()) < 0.45 * edge_px:
        return None
    pose = _pose_from_rect(rect, cube, int(body.sum()), plane, rotation, translation, "rim")
    if pose is None:
        return None
    chosen_y = wys[column][np.flatnonzero(on_face)[body]]
    chosen_x = wxs[column][np.flatnonzero(on_face)[body]]
    pose.mask = _index_mask(color_bgr.shape[:2], chosen_y, chosen_x, np.ones(len(chosen_y), dtype=bool))
    if not _visible_floor_ok(depth, pose.mask, plane, intr, color_bgr, colors):
        return None
    return pose


def _lower_rim_pose(
    ys, xs, good, heights, depth, plane, intr, cube_size, min_footprint_fraction, size_tolerance, rotation, translation
) -> CubePose | None:
    # 下面的顶面被挡住时，只用不超过一个边长的白。射线交到这一层，缺的一半补向远处。
    low = good & np.isfinite(heights) & (heights > 0.5 * cube_size) & (heights <= cube_size + 0.02)
    if int(low.sum()) < 30:
        return None
    mask = _index_mask(depth.shape[:2], ys, xs, low)
    return _measure_mask(
        mask,
        None,
        plane,
        intr,
        cube_size,
        float(cube_size),
        min_footprint_fraction,
        size_tolerance,
        True,
        rotation,
        translation,
        int(low.sum()),
    )


def _pixels_on_plane(pixels: np.ndarray, plane: Plane, intr) -> np.ndarray:
    # 轮廓点只定射线，交到地面上。深度空洞里的拐点也能用。
    dirs = deproject_pixels(np.asarray(pixels, dtype=float), np.ones(len(pixels)), intr)
    normal = np.asarray(plane.normal, dtype=float)
    denom = dirs @ normal
    scale = np.full(len(dirs), np.nan)
    facing = denom < -1e-4
    scale[facing] = float(plane.offset) / denom[facing]
    ok = facing & np.isfinite(scale) & (scale > 0.05) & (scale < 3.5) & (dirs[:, 2] > 0.05)
    points = dirs * scale[:, None]
    points[~ok] = np.nan
    return points


def _lower_from_ground_corner(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    plane: Plane,
    intr,
    colors: list[dict],
    cube_size: float,
    size_tolerance: float,
    rotation,
    translation,
) -> CubePose | None:
    # 白还不成块时，看地面轮廓向里凹出的直角。旁边仍要有一些白，缺的一半向远处补。
    # 用这一眼全部有深度的点。只做很小的闭合，把断掉的地板连上，避免把 35 厘米的边缩短。
    finite = np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(finite)
    if len(xs) < 30:
        return None
    step = max(len(xs) // 80000, 1)
    ys = ys[::step]
    xs = xs[::step]
    samples = deproject_pixels(
        np.column_stack([xs, ys]).astype(float),
        np.asarray(depth[ys, xs], dtype=float),
        intr,
    )
    on = np.abs(plane.signed_distance(samples)) < 0.018
    if int(on.sum()) < 30:
        return None
    floor = np.zeros(depth.shape[:2], dtype=np.uint8)
    floor[ys[on], xs[on]] = 255
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _hierarchy = cv2.findContours(floor, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    white = _white_pixels(color_bgr, colors)
    normal = np.asarray(plane.normal, dtype=float)
    limit = float(size_tolerance)
    best = None
    for contour in contours:
        if len(contour) < 12 or cv2.contourArea(contour) < 80:
            continue
        z_hint = float(np.nanmedian(samples[on, 2])) if int(on.sum()) else 0.8
        epsilon = max(2.0, 0.02 * float(intr.fx) / max(z_hint, 0.2))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        pixels = approx[:, 0, :].astype(float)
        lifted = _pixels_on_plane(pixels, plane, intr)
        count = len(lifted)
        for index in range(count):
            corner = lifted[index]
            prev = lifted[(index - 1) % count]
            nxt = lifted[(index + 1) % count]
            if not (np.isfinite(corner).all() and np.isfinite(prev).all() and np.isfinite(nxt).all()):
                continue
            pix = pixels[index]
            if pix[0] < 4 or pix[1] < 4 or pix[0] > depth.shape[1] - 5 or pix[1] > depth.shape[0] - 5:
                continue
            arms = []
            for other in (prev, nxt):
                delta = other - corner
                delta = delta - normal * float(delta @ normal)
                span = float(np.linalg.norm(delta))
                if span < 0.08:
                    continue
                arms.append((other, delta / span, span))
            if len(arms) < 2:
                continue
            # 两条边从拐点出发。接近 90 度才像方块的底角，接近 180 度只是地板的直边。
            angle = float(np.degrees(np.arccos(np.clip(float(np.dot(arms[0][1], arms[1][1])), -1.0, 1.0))))
            if angle < 70.0 or angle > 110.0:
                continue
            for other, direction, span in arms:
                if abs(span - float(cube_size)) > limit:
                    continue
                mid = 0.5 * (corner + other)
                away = _away_on_plane(plane, mid)
                if away is None or abs(float(direction @ away)) > 0.85:
                    continue
                reach = float(np.linalg.norm(mid))
                radius = int(np.clip(round(0.25 * float(intr.fx) / max(reach, 0.2)), 12, 90))
                u = int(round(float(pix[0])))
                v = int(round(float(pix[1])))
                y0 = max(0, v - radius)
                y1 = min(white.shape[0], v + radius + 1)
                x0 = max(0, u - radius)
                x1 = min(white.shape[1], u + radius + 1)
                if int(np.count_nonzero(white[y0:y1, x0:x1])) < 40:
                    continue
                facing = abs(float(direction @ away))
                # 正对镜头的边里，留更靠近镜头的那一条。方块背后那条边同样接近直角，不能拿来补。
                if best is None or (reach, facing) < (best[1], best[0]):
                    best = (facing, reach, mid, direction, away, y0, y1, x0, x1)
    if best is None:
        return None
    _facing, _reach, mid, direction, away, y0, y1, x0, x1 = best
    inward = _flat_dir(np.cross(normal, direction), normal)
    if inward is None:
        return None
    if float(inward @ away) < 0.0:
        inward = -inward
    near_white = white[y0:y1, x0:x1] > 0
    rect = _square_from_edge(mid, direction, inward, plane, cube_size, np.vstack([mid, mid + direction]))
    pose = _pose_from_rect(rect, float(cube_size), int(near_white.sum()), plane, rotation, translation, "ground_corner")
    if pose is None:
        return None
    ys, xs = np.nonzero(white[y0:y1, x0:x1])
    pose.mask = _index_mask(color_bgr.shape[:2], ys + y0, xs + x0, np.ones(len(xs), dtype=bool))
    return pose


def _cube_fill(pose: CubePose, plane: Plane, intr, cube_size: float, shape: tuple[int, int]) -> np.ndarray | None:
    # 青色框在画面里围住的那一块，用来看里面是不是白。
    if pose.corners is None:
        return None
    bottom = np.asarray(pose.corners, dtype=float).reshape(-1, 3)
    if len(bottom) < 3:
        return None
    top = bottom + np.asarray(plane.normal, dtype=float).reshape(3) * float(cube_size)
    pts = np.vstack([bottom, top])
    front = pts[:, 2] > 0.05
    if int(front.sum()) < 3:
        return None
    pix = project_points(pts[front], intr)
    ok = np.isfinite(pix).all(axis=1)
    pix = pix[ok]
    if len(pix) < 3:
        return None
    hull = cv2.convexHull(np.round(pix).astype(np.int32))
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def _white_on_ground(color_bgr: np.ndarray, depth: np.ndarray, plane: Plane, intr, colors: list[dict]) -> np.ndarray | None:
    # 贴着地面的那块白。上面单独悬着的白不取，两层连在一起时用整块的下沿。
    white = _white_pixels(color_bgr, colors)
    if int(np.count_nonzero(white)) < 40:
        return None
    finite = np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(finite)
    if len(xs) < 30:
        return None
    step = max(len(xs) // 40000, 1)
    ys = ys[::step]
    xs = xs[::step]
    pts = deproject_pixels(np.column_stack([xs, ys]).astype(float), np.asarray(depth[ys, xs], dtype=float), intr)
    on = np.abs(plane.signed_distance(pts)) < 0.018
    if int(on.sum()) < 30:
        return None
    floor = np.zeros(depth.shape[:2], dtype=np.uint8)
    floor[ys[on], xs[on]] = 255
    floor = cv2.dilate(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # 正面的白常常是碎的。先把挨着的点连上，连不上就用所有贴地的白。
    solid = cv2.morphologyEx(
        np.where(white > 0, np.uint8(255), np.uint8(0)),
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )
    touch = (solid > 0) & (floor > 0)
    if int(touch.sum()) < 40:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    best = None
    for label in range(1, count):
        comp = labels == label
        if int((comp & touch).sum()) < 40:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if best is None or area > best[0]:
            best = (area, comp)
    if best is not None:
        return best[1]
    return touch


def _pose_from_ground_white(
    blob: np.ndarray,
    plane: Plane,
    intr,
    cube_size: float,
    size_tolerance: float,
    rotation,
    translation,
) -> CubePose | None:
    # 用这块白最下面一两行当贴地的边，缺的一半向远处补。边长仍要接近 35 厘米。
    ys, xs = np.nonzero(blob)
    if len(xs) < 40:
        return None
    bottom = ys >= int(ys.max()) - 1
    if int(bottom.sum()) < 12:
        return None
    lifted = _pixels_on_plane(np.column_stack([xs[bottom], ys[bottom]]).astype(float), plane, intr)
    ground_pts = lifted[np.isfinite(lifted).all(axis=1)]
    if len(ground_pts) < 12:
        return None
    rect = _square_from_near_edge(ground_pts, plane, cube_size, size_tolerance)
    if rect is None:
        edge = _edge_from_points(ground_pts, np.asarray(plane.normal, dtype=float))
        if edge is None:
            return None
        mid, direction, span = edge
        if span < 0.18 or abs(span - float(cube_size)) > max(float(size_tolerance), 0.10):
            return None
        away = _away_on_plane(plane, mid)
        inward = None if away is None else _flat_dir(np.cross(plane.normal, direction), plane.normal)
        if inward is None or away is None:
            return None
        if float(inward @ away) < 0.0:
            inward = -inward
        if abs(float(direction @ away)) > 0.85:
            return None
        rect = _square_from_edge(mid, direction, inward, plane, cube_size, ground_pts)
    pose = _pose_from_rect(rect, float(cube_size), int(bottom.sum()), plane, rotation, translation, "white_shift")
    if pose is None:
        return None
    pose.mask = _index_mask(blob.shape, ys, xs, np.ones(len(xs), dtype=bool))
    return pose


def _pull_lower_onto_white(
    pose: CubePose,
    color_bgr: np.ndarray,
    depth: np.ndarray,
    plane: Plane,
    intr,
    colors: list[dict],
    cube_size: float,
    size_tolerance: float,
    rotation,
    translation,
) -> CubePose | None:
    # 框里有大片不是白，或框比贴地的白小一圈，就按这块白的下沿重算，不再留在远处的折线上。
    fill = _cube_fill(pose, plane, intr, cube_size, color_bgr.shape)
    blob = _white_on_ground(color_bgr, depth, plane, intr, colors)
    if fill is None or blob is None:
        return pose
    inside = fill > 0
    box_n = int(inside.sum())
    if box_n < 80:
        return pose
    white = _white_pixels(color_bgr, colors) > 0
    foreign = int((inside & ~white).sum()) / float(box_n)
    white_n = int(blob.sum())
    by, bx = np.nonzero(blob)
    band = by >= int(by.max()) - max(6, int(0.15 * (int(by.max()) - int(by.min()))))
    bottom = np.zeros(blob.shape, dtype=bool)
    bottom[by[band], bx[band]] = True
    covered = int((bottom & inside).sum()) / max(int(bottom.sum()), 1)
    off_white = foreign >= 0.40
    # 上层的白会让整块变大。框已经罩住贴地的下沿，就不要因为上面那层再挪。
    too_small = white_n > 1.8 * box_n and covered < 0.45
    if not off_white and not too_small:
        return pose
    rebuilt = _pose_from_ground_white(blob, plane, intr, cube_size, size_tolerance, rotation, translation)
    if rebuilt is not None:
        return rebuilt
    # 贴地的白已经是一大块，远处的框对不上，就不要留。只有零星的白时，仍用原来的框。
    if too_small or (off_white and white_n > 400):
        return None
    return pose


def detect_white_layers(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.2,
    min_pixels: int = 400,
    min_footprint_fraction: float = 0.45,
    size_tolerance: float = 0.06,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 3.0,
    rejects: dict | None = None,
) -> list[CubePose]:
    # 没有明显高于一层的白时，只走原来的一块。有的话下面、上面各量各的，中心不互相抬高。
    white, ys, xs, good, points, heights = _white_cloud(
        color_bgr, depth, intr, plane, colors, zmin, zmax
    )
    hi = (
        good
        & np.isfinite(heights)
        & (heights > cube_size + _UPPER_CLEAR_M)
        & (heights < 2.0 * cube_size + _UPPER_CLEAR_M)
    )
    upper_evidence = int(hi.sum()) >= _UPPER_MIN_POINTS and (
        len(heights[hi]) > 0 and float(np.percentile(heights[hi], 90)) >= cube_size + 0.15
    )
    # 两层时先认下层正面。正面给出底边和转角。认不出再退回顶面。
    lower = None
    if upper_evidence:
        lower = _stack_lower_from_face(
            ys,
            xs,
            good,
            points,
            heights,
            plane,
            color_bgr.shape[:2],
            cube_size,
            size_tolerance,
            rotation,
            translation,
        )
        if lower is not None and not _ground_under_cube(depth, lower.mask, plane, intr, color_bgr, colors):
            lower = None
    # 下层始终只留离地不超过大约一个边长的白。地面那一圈仍用原来的彩色图。
    view, view_depth, hid_upper = _blank_lower_view(
        color_bgr, depth, ys, xs, good, heights, plane, intr, float(cube_size) + 0.02
    )
    if lower is None:
        lower = detect_visible_cube(
            view,
            view_depth,
            intr,
            plane,
            colors,
            cube_size=cube_size,
            min_pixels=min_pixels,
            min_footprint_fraction=min_footprint_fraction,
            size_tolerance=size_tolerance,
            rotation=rotation,
            translation=translation,
            zmin=zmin,
            zmax=zmax,
            rejects=rejects,
            ground_color=color_bgr if hid_upper else None,
        )
    # 正面一整面白、深度却稀时，成块失败。上面已经有更高的白，才用这块盒身补下层。
    if lower is None:
        lower = _lower_from_white_face(
            color_bgr,
            depth,
            plane,
            intr,
            colors,
            cube_size,
            size_tolerance,
            rotation,
            translation,
        )
    # 成块的白和正面都没有时，用地面轮廓上接近直角、旁边还有白的角补底层。
    if lower is None:
        lower = _lower_from_ground_corner(
            color_bgr,
            depth,
            plane,
            intr,
            colors,
            cube_size,
            size_tolerance,
            rotation,
            translation,
        )
    def settle(candidate: CubePose | None) -> CubePose | None:
        if candidate is None:
            return None
        return _pull_lower_onto_white(
            candidate,
            color_bgr,
            depth,
            plane,
            intr,
            colors,
            cube_size,
            size_tolerance,
            rotation,
            translation,
        )

    if not upper_evidence:
        lower = settle(lower)
        if lower is None:
            return []
        lower.layer = 1
        return [lower]
    upper = _upper_pose(
        white,
        ys,
        xs,
        good,
        points,
        heights,
        depth,
        plane,
        intr,
        cube_size,
        min_footprint_fraction,
        size_tolerance,
        rotation,
        translation,
        anchor=lower,
    )
    if upper is None:
        lower = settle(lower)
        if lower is None:
            return []
        lower.layer = 1
        return [lower]
    if lower is None:
        lower = _lower_rim_pose(
            ys,
            xs,
            good,
            heights,
            depth,
            plane,
            intr,
            cube_size,
            min_footprint_fraction,
            size_tolerance,
            rotation,
            translation,
        )
        if lower is not None and not _ground_under_cube(depth, lower.mask, plane, intr, color_bgr, colors):
            lower = None
    lower = settle(lower)
    poses: list[CubePose] = []
    if lower is not None:
        lower.layer = 1
        poses.append(lower)
    if lower is None and not _ground_under_cube(depth, upper.mask, plane, intr, color_bgr, colors):
        return []
    upper.layer = 2
    poses.append(upper)
    return poses


def detect_cube_candidates(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intr,
    plane: Plane,
    colors: list[dict],
    cube_size: float = 0.2,
    min_pixels: int = 400,
    min_footprint_fraction: float = 0.45,
    size_tolerance: float = 0.06,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
    zmin: float = 0.3,
    zmax: float = 3.0,
    rejects: dict | None = None,
) -> list[CubePose]:
    # 最多 3 块各自做足迹。绿线可以在这几块之间跳，选择留到多帧比较。
    # 一块都没通过时只记第一个拒绝原因，不把 height 和后面的 mask 叠在同一帧上。
    prepared_list = prepare_candidate_masks(
        color_bgr, depth, colors, zmin, zmax, plane, intr, cube_size
    )
    if not prepared_list:
        if rejects is not None:
            rejects["mask"] = int(rejects.get("mask", 0)) + 1
        return []
    poses: list[CubePose] = []
    first_reason: str | None = None
    for prepared in prepared_list[:3]:
        local: dict[str, int] = {}
        pose = detect_visible_cube(
            color_bgr,
            depth,
            intr,
            plane,
            colors,
            cube_size=cube_size,
            min_pixels=min_pixels,
            min_footprint_fraction=min_footprint_fraction,
            size_tolerance=size_tolerance,
            rotation=rotation,
            translation=translation,
            zmin=zmin,
            zmax=zmax,
            rejects=local,
            prepared=prepared,
        )
        if pose is None:
            if first_reason is None and local:
                first_reason = next(iter(local))
            continue
        pose.mask = prepared[0]
        poses.append(pose)
    if not poses and rejects is not None:
        reason = first_reason or "mask"
        rejects[reason] = int(rejects.get(reason, 0)) + 1
    return poses
