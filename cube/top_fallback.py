"""抬头未发现顶块后的低头检查；用第一层顶面的等面积采样判断露出与遮挡。"""
from dataclasses import dataclass

import numpy as np
import cv2

from cube.detect import _square_from_near_edge, _white_pixels
from cube.stack_look import _cloud, vote_rise_degrees
from geom.transform import deproject_pixels, project_points


def top_exposure(color, depth, intr, plane, corners, colors, size, zmin, zmax):
    # 在物理顶面均匀取样，避开边缘；每个样点代表相同面积，不比较透视像素面积。
    t = np.linspace(.06, .94, 24)
    a, b = np.meshgrid(t, t)
    base = (corners[0] + a.ravel()[:, None] * (corners[1] - corners[0])
            + b.ravel()[:, None] * (corners[3] - corners[0]))
    expected = base + plane.normal * size
    uv = project_points(expected, intr)
    inside = (np.isfinite(uv).all(axis=1) & (expected[:, 2] > .05)
              & (uv[:, 0] >= 3) & (uv[:, 0] < intr.width - 3)
              & (uv[:, 1] >= 3) & (uv[:, 1] < intr.height - 3))
    result = dict(layers=None, visible=float(inside.mean()), coverage=0., exposed=0., blocked=0., rgb_white=0.)
    if inside.mean() < .80:
        result['reason'] = '顶面投影未完整入画，当前可见面积不足80%，层数待确认'
        return result
    pixels = np.rint(uv[inside]).astype(int)
    z = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    result['coverage'] = float(valid.sum() / len(expected))
    white_mask = _white_pixels(color, colors) > 0
    result['rgb_white'] = float(white_mask[pixels[:, 1], pixels[:, 0]].mean())
    if result['coverage'] < .45:
        result['reason'] = '顶面深度不足，保持待确认'
        return result
    pixels = pixels[valid]
    points = deproject_pixels(pixels, z[valid], intr)
    targets = expected[inside][valid]
    white = _white_pixels(color, colors)[pixels[:, 1], pixels[:, 0]] > 0
    heights = plane.signed_distance(points)
    exposed = white & (np.linalg.norm(points - targets, axis=1) < .04)
    # 遮挡点必须比预期顶面靠近镜头，并落在该盒子足迹之内。
    flat = plane.project(points) - corners[0]
    x, y = corners[1] - corners[0], corners[3] - corners[0]
    u, v = flat @ x / (x @ x), flat @ y / (y @ y)
    in_box = (u >= -.05) & (u <= 1.05) & (v >= -.05) & (v <= 1.05)
    blocked = (white & in_box & (heights > size + .08) & (heights < 2 * size + .08)
               & (points[:, 2] < targets[:, 2] - .02))
    result['exposed'] = float(exposed.sum() / len(expected))
    result['blocked'] = float(blocked.sum() / len(expected))
    # RGB轮廓与分布在顶面各区的实测深度共同验证，空洞不作为露出证据。
    supported = np.zeros(len(expected), dtype=bool)
    supported[np.flatnonzero(inside)[valid]] = exposed
    grid = supported.reshape(24, 24)
    distributed = all(part.mean() >= .25 for part in
                      (grid[:12, :12], grid[:12, 12:], grid[12:, :12], grid[12:, 12:]))
    if (result['rgb_white'] >= .80 and distributed
            and exposed.mean() >= .85 and result['blocked'] < .05):
        result['layers'] = 1
    elif result['exposed'] < .65 and result['blocked'] >= .25:
        result['layers'] = 2
    result['reason'] = '露出与遮挡证据不足' if result['layers'] is None else '顶面面积与遮挡一致'
    return result


def visible_top_corners(points, heights, flat, plane, foot, size, tolerance):
    """Recover the first top footprint from a horizontal surface, without a bottom edge."""
    near = np.linalg.norm(flat - foot, axis=1) < size * 1.25
    selected = near & (np.abs(heights - size) < .025)
    if selected.sum() < 80:
        return None
    # A thin slice of a vertical face cannot supply a square-sized surface.
    a, b = plane.basis()
    xy = np.column_stack([flat[selected] @ a, flat[selected] @ b]).astype(np.float32)
    rect = cv2.minAreaRect(xy)
    lengths = np.asarray(rect[1])
    if np.any(np.abs(lengths - size) > tolerance):
        return None
    hull_area = cv2.contourArea(cv2.convexHull(xy))
    if hull_area < .6 * size * size:
        return None
    vertices = cv2.boxPoints(rect).astype(float)
    corners = plane.origin() + vertices[:, :1] * a + vertices[:, 1:] * b
    # Validate a fitted surface normal; reject sloping boards at the right height.
    surface = points[selected]
    _, _, vh = np.linalg.svd(surface - surface.mean(axis=0), full_matrices=False)
    if abs(float(vh[-1] @ plane.normal)) < np.cos(np.radians(8)):
        return None
    return corners


def associated_upper(points, heights, flat, pixels, bands, foot, size, lower_reference=None):
    """上层必须有实测深度，并关联当前下层或已经连续确认的下层参照。"""
    result = dict(layers=None, reason='未提供上下层候选区域', center=None)
    if not bands or not bands.get('upper_white'):
        result['reason'] = '未检测到粉色上层候选，不能据此确认两层'
        return result
    masks = []
    for key, lo, hi in [('lower_uv', .08, size - .03),
                        ('upper_uv', size + .08, 2 * size + .08)]:
        uv = np.asarray(bands.get(key, []), dtype=int).reshape(-1, 2)
        selected = np.isin(pixels[:, 0] + pixels[:, 1] * 100000,
                           uv[:, 0] + uv[:, 1] * 100000)
        masks.append(selected & (heights > lo) & (heights < hi))
    lower, upper = masks
    historical = int(lower.sum()) < 30
    if historical:
        if lower_reference is None:
            result['reason'] = '下层有效高度深度点不足（需至少30点），且无可靠历史参照'
            return result
        lower_center = np.asarray(lower_reference)
        if np.linalg.norm(lower_center - foot) > size * .35:
            result['reason'] = '历史下层参照与当前塔脚不匹配'
            return result
        # 只保留历史塔脚附近的实测点，RGB补出的整片背景不能参与计票。
        upper &= np.linalg.norm(flat - lower_center, axis=1) <= size * .6
    else:
        lower_center = np.median(flat[lower], axis=0)
    if int(upper.sum()) < 30:
        result['reason'] = '粉色上层有效高度深度点不足（需至少30点）'
        return result
    upper_center = np.median(flat[upper], axis=0)
    if (np.linalg.norm(lower_center - foot) > size * .6 or
            np.linalg.norm(upper_center - lower_center) > size * .6):
        result['reason'] = '上下层水平位置不匹配，尚不能确认为同一座塔'
        return result
    width = np.linalg.norm(np.percentile(flat[upper], 90, axis=0) -
                           np.percentile(flat[upper], 10, axis=0))
    overlap_ok = True
    if not historical:
        lx = np.percentile(pixels[lower, 0], [10, 90])
        ux = np.percentile(pixels[upper, 0], [10, 90])
        overlap = min(lx[1], ux[1]) - max(lx[0], ux[0])
        overlap_ok = overlap >= .5 * min(lx[1] - lx[0], ux[1] - ux[0])
    if width < size * .3 or not overlap_ok:
        result['reason'] = '粉色上层面积或上下层横向重叠不足，尚不能确认两层'
        return result
    result.update(layers=2, center=upper_center, used_history=historical,
                  reason='上层实测深度、高度与可靠下层参照一致' if historical else
                         '上下层有效深度、高度和水平归属一致')
    return result


@dataclass
class TopFallback:
    timeout: float = 15.
    started: float | None = None
    active: bool = False
    layers: int | None = None
    pending: int | None = None
    count: int = 0
    target_pitch: float | None = None
    top_target_pitch: float | None = None
    evidence: dict | None = None
    instruction: str = ''
    saw_top_view: bool = False
    cap_confirmed: bool = False
    cap_seen: bool = False
    upper_center: object = None
    upper_count: int = 0
    scan_base: float | None = None
    scan_index: int = 0
    failed_since: float | None = None
    scan_exhausted: bool = False
    angle_started: float | None = None
    angle_timeout: float = 8.
    observation_started: float | None = None
    observation_timeout: float = 8.
    upper_hold_used: bool = False
    arrived: bool = False
    lower_reference: object = None
    lower_reference_count: int = 0
    lower_seen_at: float | None = None

    def clear_reference(self):
        self.lower_reference = None
        self.lower_reference_count = 0
        self.lower_seen_at = None

    def reorient(self, rotation):
        if self.lower_reference is not None:
            self.lower_reference = rotation @ self.lower_reference
        if self.upper_center is not None:
            self.upper_center = rotation @ self.upper_center

    def remember_lower(self, now, bands, depth, intr, plane, foot, size, moving, cfg):
        if self.lower_reference is not None and (
                np.linalg.norm(self.lower_reference - foot) > size * .35
                or now - self.lower_seen_at > 45.):
            self.clear_reference()
        if moving or not bands:
            if self.lower_reference_count < 3:
                self.clear_reference()
            return
        uv = np.asarray(bands.get('lower_uv', []), dtype=int).reshape(-1, 2)
        if len(uv) < 30:
            if self.lower_reference_count < 3:
                self.clear_reference()
            return
        z = depth[uv[:, 1], uv[:, 0]]
        good = np.isfinite(z) & (z >= cfg.get('depth_min', .05)) & (z <= cfg.get('depth_max', 1.2))
        points = deproject_pixels(uv[good], z[good], intr)
        heights = plane.signed_distance(points)
        flat = plane.project(points[(heights > .08) & (heights < size - .03)])
        if len(flat) < 30:
            if self.lower_reference_count < 3:
                self.clear_reference()
            return
        center = np.median(flat, axis=0)
        width = np.linalg.norm(np.percentile(flat, 90, axis=0) - np.percentile(flat, 10, axis=0))
        if np.linalg.norm(center - foot) > size * .6:
            self.clear_reference()
            return
        if width < size * .3:
            if self.lower_reference_count < 3:
                self.clear_reference()
            return
        consistent = self.lower_reference is not None and np.linalg.norm(center - self.lower_reference) <= .08
        self.lower_reference_count = self.lower_reference_count + 1 if consistent else 1
        self.lower_reference = center.copy()
        self.lower_seen_at = now

    def reliable_lower(self):
        return self.lower_reference if self.lower_reference_count >= 3 else None

    def at_target(self, pitch):
        """3度内到位；到位后5度内保持，避免边界附近反复换向。"""
        if pitch is None or not np.isfinite(pitch) or self.target_pitch is None:
            return False
        return abs(pitch - self.target_pitch) <= (5. if self.arrived else 3.)

    def scan_failure(self, now, reason, advance=False):
        self.pending, self.count = None, 0
        if self.failed_since is None:
            self.failed_since = now
        if advance and not self.scan_exhausted:
            # 先检查较浅角度，再逐步低头，避免最后从+24度反向跳到-8度。
            offsets = (0., -8., 8., 16., 24.)
            if self.scan_index + 1 < len(offsets):
                self.scan_index += 1
                self.target_pitch = float(np.clip(self.scan_base + offsets[self.scan_index], 10., 75.))
                self.failed_since = None
                self.angle_started = now
                self.arrived = False
                self.observation_started = None
                self.upper_hold_used = False
                self.upper_center, self.upper_count = None, 0
            else:
                self.scan_exhausted = True
        if self.scan_exhausted:
            self.layers = 1
            self.evidence = dict(self.evidence or {}, layers=1, defaulted=True,
                                 reason='完整扫描后证据不足，按默认策略输出1层')
            self.instruction = '白盒 1 层已确认，保持当前俯仰，开启下一轮识别'
        else:
            self.instruction = (f'{reason}；层数待确认；俯仰扫描目标低头 {self.target_pitch:.1f} 度，'
                                '保持车体位置，旋转停止')

    @property
    def result(self):
        suffix = ('没有顶块' if self.scan_exhausted else
                  '顶块待确认' if self.cap_seen or not self.saw_top_view else '没有顶块')
        return f'白盒层数 {self.layers}，{suffix}'

    def update(self, now, anchor, success, valid, moving, pitch, color, depth, intr, plane, cfg, cap_visible=False, bands=None):
        self.cap_confirmed = bool(success and valid)
        # All rejected frames break the consecutive upper-region vote.
        previous_center, previous_count = self.upper_center, self.upper_count
        self.upper_center, self.upper_count = None, 0
        if not valid or anchor is None or plane is None:
            self.started = None
            self.pending, self.count = None, 0
            self.clear_reference()
            self.instruction = '姿态或目标参照不足，暂停俯视检查'
            return
        if success:
            self.started, self.active = None, False
            self.layers, self.pending, self.count = None, None, 0
            return
        self.cap_seen = self.cap_seen or bool(cap_visible)
        size = float(cfg['cube_size'])
        foot = plane.project(np.asarray(anchor).reshape(1, 3))[0]
        self.remember_lower(now, bands, depth, intr, plane, foot, size, moving, cfg)
        horizontal = float(np.linalg.norm(foot - plane.origin()))
        camera_height = -float(plane.offset)
        rise = vote_rise_degrees(horizontal)
        top_pitch = -rise if rise > .5 else 0.
        self.top_target_pitch = top_pitch
        at_top = pitch is not None and np.isfinite(pitch) and pitch <= (top_pitch + 3 if rise > .5 else 12.)
        if at_top and not moving:
            self.saw_top_view = True
        # 连续无结果计时不依赖精确抬头角度，也不被可见但无法测量的候选重置。
        if not self.active and (moving or pitch is None or not np.isfinite(pitch)):
            self.started = None
            self.instruction = '停止俯仰和旋转，等待画面稳定后检查塔顶'
            return
        if self.started is None:
            self.started = now
        self.active = self.active or now - self.started >= self.timeout
        if not self.active:
            remaining = max(0., self.timeout - (now - self.started))
            action = ('可以了，停止俯仰和旋转，保持检查塔顶' if at_top
                      else f'抬头到 {top_pitch:.0f} 度检查塔顶')
            self.instruction = f'{action}；连续无结果还剩 {remaining:.0f} 秒后转入低头检查'
            return
        # anchor 通常来自下层正面，先向里半个边长瞄准顶面中心。
        distance = max(.05, horizontal + .5 * size)
        if self.scan_base is None:
            self.scan_base = float(np.degrees(np.arctan2(camera_height - size, distance)))
            self.target_pitch = self.scan_base
        if self.layers is not None:
            self.instruction = f'白盒 {self.layers} 层已确认，保持当前俯仰，开启下一轮识别'
            return
        if self.angle_started is None:
            self.angle_started = now
        white = points = heights = flat = pixels = None
        upper = None
        top_depth_max = None
        finite_pitch = pitch is not None and np.isfinite(pitch)
        if not moving and finite_pitch and bands and bands.get('upper_white'):
            # 到位前也检查真实上层深度；纯颜色补出的轮廓不能触发保持。
            top_depth_max = max(float(cfg['depth_max']), float(np.asarray(anchor)[2]) + 2 * size)
            white = _white_pixels(color, cfg['colors'])
            points, heights, flat, pixels = _cloud(white, depth, intr, plane, cfg['depth_min'], top_depth_max)
            upper = associated_upper(points, heights, flat, pixels, bands, foot, size, self.reliable_lower())
            if not self.upper_hold_used and upper['layers'] == 2:
                # 每个扫描角度只允许一次保持，不能被候选反复出现无限续时。
                self.target_pitch = float(pitch)
                self.arrived = True
                self.upper_hold_used = True
                self.observation_started = now
                self.failed_since = None
        self.arrived = self.at_target(pitch)
        if self.observation_started is not None:
            if now - self.observation_started >= self.observation_timeout:
                self.scan_failure(now, '到位观察超时', advance=True)
                return
        elif now - self.angle_started >= self.angle_timeout:
            self.scan_failure(now, '等待角度到位超时', advance=True)
            return
        self.instruction = f'顶块识别超时，低头到 {self.target_pitch:.0f} 度检查第一层顶面'
        if moving or not self.arrived:
            self.pending, self.count = None, 0
            if moving:
                self.instruction = '停止俯仰和旋转，等待画面稳定；层数待确认'
            elif not finite_pitch:
                self.instruction = '俯仰读数无效，停止俯仰和旋转，等待姿态恢复'
            else:
                delta = self.target_pitch - pitch
                self.instruction += f'，还需{"低头" if delta > 0 else "抬头"} {abs(delta):.0f} 度'
            return
        if self.observation_started is None:
            self.observation_started = now
        arrival = '俯仰已到位，停止俯仰和旋转'
        self.instruction = arrival
        if white is None:
            top_depth_max = max(float(cfg['depth_max']), float(np.asarray(anchor)[2]) + 2 * size)
            white = _white_pixels(color, cfg['colors'])
            points, heights, flat, pixels = _cloud(white, depth, intr, plane, cfg['depth_min'], top_depth_max)
        if upper is None:
            upper = associated_upper(points, heights, flat, pixels, bands, foot, size, self.reliable_lower())
        if upper['layers'] == 2:
            center = upper['center']
            self.upper_count = (previous_count + 1 if previous_center is not None and
                                np.linalg.norm(center - previous_center) <= .06 else 1)
            self.upper_center = center
            self.pending, self.count = None, 0
            self.evidence = dict(upper, method='associated_upper', stable_frames=self.upper_count)
            self.instruction = (f'{arrival}；粉色上层与下层三维关联通过，'
                                f'连续稳定 {self.upper_count}/5 帧；层数待确认')
            if self.upper_count >= 5:
                self.layers = 2
                self.instruction = '白盒 2 层已确认（上下层三维关联连续5帧），保持当前俯仰，开启下一轮识别'
            return
        corners = visible_top_corners(points, heights, flat, plane, foot, size,
                                      cfg.get('size_tolerance', .06))
        method = 'visible_top'
        if corners is None:
            near = np.linalg.norm(flat - foot, axis=1) < size * 1.25
            body = near & (heights > .03) & (heights < size - .03)
            if bands is not None:
                uv = np.asarray(bands.get('lower_uv', []), dtype=int).reshape(-1, 2)
                body &= np.isin(pixels[:, 0] + pixels[:, 1] * 100000,
                                uv[:, 0] + uv[:, 1] * 100000)
            rect = _square_from_near_edge(points[body], plane, size, cfg.get('size_tolerance', .06), heights[body])
            if rect is None:
                self.evidence = dict(method='geometry_missing', upper_reason=upper['reason'])
                self.scan_failure(now, '顶面几何不足且下层底边拟合失败；' + upper['reason'])
                return
            corners = rect[1]
            method = 'bottom_edge'
        self.evidence = top_exposure(color, depth, intr, plane, corners, cfg['colors'], size,
                                     cfg['depth_min'], top_depth_max)
        self.evidence['method'] = method
        self.evidence['upper_reason'] = upper['reason']
        # An unresolved upper candidate must not be hidden by a contradictory one-layer vote.
        if self.evidence['layers'] == 1 and bands and bands.get('upper_white'):
            self.evidence['layers'] = None
            self.evidence['reason'] = '上层候选尚未排除，不能确认一层'
        guess = self.evidence['layers']
        if guess is None:
            self.scan_failure(now, self.evidence['reason'])
            return
        self.failed_since = None
        if guess == self.pending:
            self.count += 1
        else:
            self.pending, self.count = guess, 1
        self.instruction = (f'{arrival}；检查目标低头 {self.target_pitch:.0f} 度；顶面露出 {self.evidence["exposed"]:.0%}，上层遮挡 {self.evidence["blocked"]:.0%}；'
                            + self.evidence['reason'])
        if self.count >= 5:
            self.layers = guess
            self.instruction = f'白盒 {guess} 层已确认，保持当前俯仰，开启下一轮识别'

    def display(self, lines):
        if self.cap_confirmed:
            return lines
        start = next((i for i, line in enumerate(lines) if str(line).startswith('层数')), len(lines))
        body = list(lines[:start])
        if body:
            body[0] = ('备用流程：俯视第一层顶面，独立判断白盒层数' if self.active
                       else '先检查塔顶，返回低头检查后才判断白盒层数')
        label = '层数待确认；塔顶未确认' if self.layers is None else f'层数 {self.layers}（白盒）；' + self.result.split('，', 1)[1]
        return body + [label, self.instruction]
