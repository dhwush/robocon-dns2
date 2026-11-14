"""抬头未发现顶块后的低头检查；用第一层顶面的等面积采样判断露出与遮挡。"""
from dataclasses import dataclass

import numpy as np

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
    result = dict(layers=None, visible=float(inside.mean()), coverage=0., exposed=0., blocked=0.)
    if inside.mean() < .95:
        result['reason'] = '顶面没有完整入画，请后退或调整方向'
        return result
    pixels = np.rint(uv[inside]).astype(int)
    z = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    result['coverage'] = float(valid.sum() / len(expected))
    if result['coverage'] < .80:
        result['reason'] = '顶面深度不足，保持待确认'
        return result
    pixels = pixels[valid]
    points = deproject_pixels(pixels, z[valid], intr)
    targets = expected[inside][valid]
    white = _white_pixels(color, colors)[pixels[:, 1], pixels[:, 0]] > 0
    heights = plane.signed_distance(points)
    exposed = white & (np.linalg.norm(points - targets, axis=1) < .025)
    # 遮挡点必须比预期顶面靠近镜头，并落在该盒子足迹之内。
    flat = plane.project(points) - corners[0]
    x, y = corners[1] - corners[0], corners[3] - corners[0]
    u, v = flat @ x / (x @ x), flat @ y / (y @ y)
    in_box = (u >= -.05) & (u <= 1.05) & (v >= -.05) & (v <= 1.05)
    blocked = (white & in_box & (heights > size + .08) & (heights < 2 * size + .08)
               & (points[:, 2] < targets[:, 2] - .02))
    result['exposed'] = float(exposed.sum() / len(expected))
    result['blocked'] = float(blocked.sum() / len(expected))
    if result['exposed'] >= .78 and result['blocked'] < .05:
        result['layers'] = 1
    elif result['exposed'] < .65 and result['blocked'] >= .25:
        result['layers'] = 2
    result['reason'] = '露出与遮挡证据不足' if result['layers'] is None else '顶面面积与遮挡一致'
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
    evidence: dict | None = None
    instruction: str = ''
    saw_top_view: bool = False
    cap_confirmed: bool = False

    def update(self, now, anchor, success, valid, moving, pitch, color, depth, intr, plane, cfg, cap_visible=False):
        self.cap_confirmed = bool(success and valid)
        if not valid or anchor is None or plane is None:
            self.started = None
            self.pending, self.count = None, 0
            self.instruction = '姿态或目标参照不足，暂停俯视检查'
            return
        if success:
            self.started, self.active = None, False
            self.layers, self.pending, self.count = None, None, 0
            return
        size = float(cfg['cube_size'])
        foot = plane.project(np.asarray(anchor).reshape(1, 3))[0]
        horizontal = float(np.linalg.norm(foot - plane.origin()))
        camera_height = -float(plane.offset)
        rise = vote_rise_degrees(horizontal)
        top_pitch = -rise if rise > .5 else 0.
        at_top = pitch is not None and np.isfinite(pitch) and pitch <= (top_pitch + 3 if rise > .5 else 12.)
        if at_top and not moving:
            self.saw_top_view = True
        if not self.saw_top_view:
            self.instruction = f'先抬头到 {top_pitch:.0f} 度检查塔顶，暂不判断白盒层数'
            self.pending, self.count = None, 0
            return
        if not self.active and (not at_top or moving):
            self.started = None
            self.instruction = f'抬头到 {top_pitch:.0f} 度并保持，完成无顶块检查'
            return
        if cap_visible and not self.active:
            self.started = None
            self.instruction = '已看到顶块，继续完成顶块位置和颜色识别'
            return
        if self.started is None:
            self.started = now
        self.active = self.active or now - self.started >= self.timeout
        if not self.active:
            self.instruction = '正在检查塔顶，俯仰保持；返回低头检查前，白盒层数待确认'
            return
        # anchor 通常来自下层正面，先向里半个边长瞄准顶面中心。
        distance = max(.05, horizontal + .5 * size)
        self.target_pitch = float(np.degrees(np.arctan2(camera_height - size, distance)))
        if self.layers is not None:
            self.instruction = f'白盒 {self.layers} 层已确认，保持当前俯仰，开启下一轮识别'
            return
        self.instruction = f'抬头检查未发现顶块，低头到 {self.target_pitch:.0f} 度检查第一层顶面'
        if moving or pitch is None or pitch < 10 or abs(pitch - self.target_pitch) > 10:
            self.pending, self.count = None, 0
            if pitch is not None:
                delta = self.target_pitch - pitch
                self.instruction += f'，还需{"低头" if delta > 0 else "抬头"} {abs(delta):.0f} 度'
            return
        white = _white_pixels(color, cfg['colors'])
        points, heights, flat, pixels = _cloud(white, depth, intr, plane, cfg['depth_min'], cfg['depth_max'])
        near = np.linalg.norm(flat - foot, axis=1) < size * 1.5
        body = near & (heights > .03) & (heights < size - .03)
        rect = _square_from_near_edge(points[body], plane, size, cfg.get('size_tolerance', .06), heights[body])
        if rect is None:
            self.pending, self.count = None, 0
            self.evidence = None
            self.instruction += '；下层边缘不足，保持待确认'
            return
        center, corners = rect[:2]
        distance = float(np.linalg.norm(center - plane.origin()))
        self.target_pitch = float(np.degrees(np.arctan2(camera_height - size, max(distance, .05))))
        self.evidence = top_exposure(color, depth, intr, plane, corners, cfg['colors'], size,
                                     cfg['depth_min'], cfg['depth_max'])
        guess = self.evidence['layers']
        if guess is None:
            self.pending, self.count = None, 0
        elif guess == self.pending:
            self.count += 1
        else:
            self.pending, self.count = guess, 1
        self.instruction = (f'低头保持 {self.target_pitch:.0f} 度；顶面露出 {self.evidence["exposed"]:.0%}，上层遮挡 {self.evidence["blocked"]:.0%}；'
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
        label = '层数待确认；塔顶未确认' if self.layers is None else f'层数 {self.layers}（白盒）；没有顶块'
        return body + [label, self.instruction]
