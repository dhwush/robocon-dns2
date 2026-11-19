import numpy as np
import pytest
from cube.top_fallback import TopFallback, top_exposure
from geom.transform import deproject_pixels
from ground.plane import Plane
from sim.scene import intrinsics_for

COLORS = [{'name':'white','hsv_low':[0,0,140],'hsv_high':[180,110,255]}]
CFG = dict(cube_size=.35, colors=COLORS, depth_min=.05, depth_max=1.2, size_tolerance=.06)


def scene(layers=1, offset=0., pitch_deg=32., distance=.35):
    intr = intrinsics_for(320, 240)
    angle = np.radians(pitch_deg)
    up = np.array([0., -np.cos(angle), -np.sin(angle)])
    forward = np.array([0., -np.sin(angle), np.cos(angle)])
    plane = Plane(up, -.669)
    yy, xx = np.indices((240, 320))
    rays = deproject_pixels(np.column_stack([xx.ravel(), yy.ravel()]), np.ones(xx.size), intr)
    directions = np.column_stack([rays[:, 0], rays @ forward, rays @ up])
    origin = np.array([0.,0.,.669])
    nearest = np.full(xx.size, np.inf)
    for level in range(layers):
        lo = np.array([-.175 + (offset if level else 0), distance, level*.35])
        hi = lo + .35
        with np.errstate(divide='ignore', invalid='ignore'):
            t0, t1 = (lo-origin)/directions, (hi-origin)/directions
        enter = np.max(np.minimum(t0,t1), axis=1)
        leave = np.min(np.maximum(t0,t1), axis=1)
        hit = (leave >= enter) & (enter > 0)
        nearest[hit] = np.minimum(nearest[hit], enter[hit])
    depth = nearest.reshape(xx.shape)
    depth[~np.isfinite(depth)] = np.nan
    color = np.full((240,320,3),40,np.uint8)
    color[np.isfinite(depth)] = 210
    corners = np.array([np.array([x,0.,0.])+y*forward-.669*up
                        for x,y in [(-.175,distance),(.175,distance),(.175,distance+.35),(-.175,distance+.35)]])
    anchor = distance*forward-.669*up
    return intr, plane, anchor, color, depth, corners


@pytest.mark.parametrize('layers,offset', [(1,0.),(2,0.),(2,.08)])
def test_exposed_top_and_offset_upper_box(layers, offset):
    intr, plane, anchor, color, depth, corners = scene(layers, offset)
    info = top_exposure(color,depth,intr,plane,corners,COLORS,.35,.05,1.2)
    assert info['layers'] == layers, info


def test_missing_depth_and_clipped_view_are_unknown():
    intr, plane, anchor, color, depth, corners = scene()
    assert top_exposure(color,depth* np.nan,intr,plane,corners,COLORS,.35,.05,1.2)['layers'] is None
    shifted = corners + np.array([1.,0.,0.])
    info = top_exposure(color,depth,intr,plane,shifted,COLORS,.35,.05,1.2)
    assert info['layers'] is None
    assert info['visible'] < .95


def update(fallback, now, layers=1, **kwargs):
    intr, plane, anchor, color, depth, corners = scene(layers)
    args = dict(anchor=anchor, success=False, valid=True, moving=False, pitch=32.,
                color=color,depth=depth,intr=intr,plane=plane,cfg=CFG)
    if now == 0. and "pitch" not in kwargs:
        args["pitch"] = -6.
    args.update(kwargs)
    # 先在抬头位置完成等待，再按默认俯视角度检查顶面。
    if "pitch" not in kwargs and now > 0 and not fallback.active:
        fallback.update(now, **dict(args, pitch=-6.))
    fallback.update(now, **args)


def test_timeout_and_five_consistent_observations():
    fallback = TopFallback()
    update(fallback,0.)
    update(fallback,14.99, pitch=-6.)
    assert not fallback.active
    update(fallback,15.)
    assert fallback.active and fallback.layers is None
    assert 25 < fallback.target_pitch < 40
    for i in range(1,5):
        update(fallback,15+i*.1)
    assert fallback.layers == 1
    update(fallback,16.)
    assert '保持当前俯仰' in fallback.instruction
    assert '下一轮识别' in fallback.instruction


def test_success_resets_timeout_and_new_target_uses_new_instance():
    fallback = TopFallback()
    update(fallback,0.)
    update(fallback,14., success=True)
    update(fallback,20., pitch=-6.)
    assert not fallback.active
    update(fallback,35.)
    assert fallback.active
    assert not TopFallback().active


def test_wrong_pitch_and_motion_do_not_confirm():
    fallback = TopFallback()
    update(fallback,0.)
    update(fallback,15., pitch=-6.)
    for i in range(10):
        update(fallback,15.+i, pitch=0.)
    assert fallback.active and fallback.layers is None and fallback.count == 0
    assert '低头' in fallback.instruction
    for i in range(10):
        update(fallback,25.+i, moving=True)
    assert fallback.layers is None


def test_display_overrides_stale_layers_and_cap_absence():
    fallback = TopFallback(active=True, layers=2, instruction='继续检查塔顶', saw_top_view=True)
    lines = fallback.display(['定了：一层白', '诊断', '层数1，没有塔顶', '保持'])
    assert '一层白' not in ''.join(lines)
    assert '没有塔顶' not in ''.join(lines)
    assert '层数 2（白盒）；没有顶块' in lines


@pytest.mark.parametrize("headless", [False, True])
def test_live_loop_switches_after_timeout_and_shows_top_result(monkeypatch, headless, capsys):
    import sys
    from types import SimpleNamespace
    import view_tower as view
    from camera.source import Frame
    from cube.stack_look import StackSearch
    from ground.plane import GroundFix
    intr, plane, anchor, color, depth, corners = scene(2)
    clock, index, rendered = [0.], [-1], []
    class Source:
        def __init__(self, *a, **k): pass
        def read(self):
            index[0] += 1
            if headless and index[0] >= 8:
                raise KeyboardInterrupt
            clock[0] = 0. if index[0] == 0 else 16. + index[0]*.1
            return Frame(color, depth, -plane.normal, gravity_status='fresh', gravity_age_s=.01)
        def close(self): pass
    class Debug:
        def __init__(self): pass
        def update(self, *a, **k): pass
        def close(self): pass
        def key(self, *a): pass
    monkeypatch.setattr(sys, 'argv', ['view_tower.py'] + (['--headless'] if headless else []))
    monkeypatch.setattr(view, 'RealSenseSource', Source)
    monkeypatch.setattr(view, 'DebugWindow', Debug)
    monkeypatch.setattr(Source, 'intrinsics', intr, raising=False)
    monkeypatch.setattr(view.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(view, 'clean_frame', lambda *a: SimpleNamespace(color=color,depth=depth,gravity=-plane.normal))
    monkeypatch.setattr(view, 'tower_ground', lambda *a: GroundFix(plane, 'imu', height=.669))
    monkeypatch.setattr(view, 'coarse_bands', lambda *a, **k: dict(anchor=anchor,lower=True,
                           upper_white=False,cap=False,distance=.35,lower_uv=np.column_stack(np.where(np.isfinite(depth))[::-1])))
    monkeypatch.setattr(view, 'pitch_down_degrees', lambda *a: -6. if index[0] <= 1 else 32.)
    monkeypatch.setattr(view, 'StackSearch', lambda: StackSearch(anchor=anchor.copy()))
    monkeypatch.setattr(StackSearch, 'update', lambda *a, **k: '')
    monkeypatch.setattr(view, 'paint', lambda color,bands,lines,*a: rendered.append(lines.copy()) or color)
    monkeypatch.setattr(view.cv2,'imshow',lambda *a: None)
    monkeypatch.setattr(view.cv2,'destroyAllWindows',lambda: None)
    monkeypatch.setattr(view.cv2,'waitKey',lambda *a: ord('q') if index[0] >= 7 else -1)
    if headless:
        def forbidden(*a, **k):
            raise AssertionError('无窗口模式不能调用绘图或GUI')
        for name in ('imshow', 'namedWindow', 'resizeWindow', 'waitKey', 'destroyAllWindows'):
            monkeypatch.setattr(view.cv2, name, forbidden)
        monkeypatch.setattr(view, 'paint', forbidden)
        monkeypatch.setattr(view, '_write', forbidden)
    view.main()
    if headless:
        output = capsys.readouterr().out
        assert '当前俯仰' in output
        assert '结果：白盒层数 2，没有顶块' in output
        assert '预期俯仰：' in output
        assert '连续稳定' not in output
        assert '开始下一轮识别' not in output
        assert not rendered
        return
    assert not any('没有顶块' in line or '层数 1' in line for line in rendered[0])
    assert any('备用流程' in line for line in rendered[1])
    assert any('层数 2（白盒）；没有顶块' == line for line in rendered[-2])
    assert any('保持当前俯仰' in line for line in rendered[-2])
    assert any('层数待确认' in line for line in rendered[-1])
    assert any(line.startswith('抬头 ') for line in rendered[-1])
    assert any(line == '上一轮结果：白盒层数 2，没有顶块' for line in rendered[-1])
    assert not any('没有顶块' in line for line in rendered[-1] if not line.startswith('上一轮结果：'))


def test_active_fallback_target_loss_can_reset_without_a_reported_result():
    from cube.stack_look import StackSearch
    from view_tower import TowerWatch
    watch = TowerWatch(hold_s=.5)
    search = StackSearch(anchor=np.array([0., .6, .5]))
    assert not watch.depart(0., search, {}, tracking=True)
    assert watch.depart(.6, search, {}, tracking=True)


def test_invalid_observation_breaks_confirmation_run():
    fallback = TopFallback()
    update(fallback, 0.)
    update(fallback, 15.)
    assert fallback.count == 1
    update(fallback, 15.1, valid=False)
    assert fallback.count == 0 and fallback.layers is None
    for i in range(4):
        update(fallback, 16.+i*.1)
    assert fallback.layers is None
    update(fallback, 16.5)
    assert fallback.layers == 1


def test_initial_downward_view_times_out_but_cannot_claim_cap_absence():
    fallback = TopFallback()
    for now in (0., 15., 16., 17., 18., 19.):
        update(fallback, now, pitch=32.)
    assert fallback.active and fallback.layers == 1
    assert '顶块待确认' in fallback.result
    assert '没有顶块' not in ''.join(fallback.display(['层数待确认']))


def test_top_view_never_publishes_white_layers_before_return():
    fallback = TopFallback()
    update(fallback, 0., pitch=-6.)
    for i in range(10):
        update(fallback, 16.+i, pitch=-6.)
    assert fallback.active and fallback.saw_top_view
    assert fallback.layers is None and fallback.count == 0
    lines = fallback.display(['定了：一层白', '层数 1，没有塔顶', '保持'])
    assert '层数待确认；塔顶未确认' in lines
    assert '一层白' not in ''.join(lines)


def test_before_timeout_stale_white_result_is_hidden():
    fallback = TopFallback()
    update(fallback, 0., pitch=-6.)
    lines = fallback.display(['定了：两层白', '层数 2，没有塔顶', '保持'])
    assert '两层白' not in ''.join(lines)
    assert '层数待确认；塔顶未确认' in lines
    assert not fallback.active


def test_visible_unmeasured_cap_times_out_without_claiming_absence():
    fallback = TopFallback()
    update(fallback, 0., pitch=-6., cap_visible=True)
    update(fallback, 16., pitch=-6., cap_visible=True)
    assert fallback.active and '低头' in fallback.instruction
    for i in range(5):
        update(fallback, 17.+i, cap_visible=False)
    assert fallback.layers == 1
    assert '顶块待确认' in fallback.result
    assert '没有顶块' not in ''.join(fallback.display(['层数待确认']))


def test_timeout_can_switch_after_pitch_changes_without_publishing_early():
    fallback = TopFallback()
    update(fallback, 0., pitch=-6.)
    update(fallback, 16., pitch=32.)
    assert fallback.active and fallback.layers is None
    assert fallback.count == 1


def test_arrival_explains_missing_lower_edge(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, 'visible_top_corners', lambda *a, **k: None)
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback()
    update(fallback, 0.)
    update(fallback, 15.)
    update(fallback, 16., pitch=fallback.target_pitch)
    assert fallback.layers is None
    assert '俯仰扫描目标低头' in fallback.instruction
    assert '下层底边拟合失败' in fallback.instruction
    assert '后退' not in fallback.instruction
    assert '调整方向' not in fallback.instruction


def test_near_top_angle_does_not_reset_timeout():
    fallback = TopFallback()
    update(fallback, 0., pitch=-4.)
    update(fallback, 14., pitch=-3.)
    assert not fallback.active
    update(fallback, 15., pitch=-4.)
    assert fallback.active and '低头' in fallback.instruction


@pytest.mark.parametrize('offset', [-3., 3., -3.01, 3.01])
def test_arrival_tolerance_is_three_degrees(monkeypatch, offset):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback()
    update(fallback, 0.)
    update(fallback, 15., pitch=-6.)
    update(fallback, 16., pitch=fallback.target_pitch + offset)
    assert ('俯仰已到位' in fallback.instruction) == (abs(offset) <= 3.)


def test_motion_restarts_no_result_timer():
    fallback = TopFallback()
    update(fallback, 0., pitch=-4.)
    update(fallback, 14., pitch=-4., moving=True)
    update(fallback, 15., pitch=-4.)
    update(fallback, 29., pitch=-4.)
    assert not fallback.active
    update(fallback, 30., pitch=-4.)
    assert fallback.active


def upper_scene_bands():
    from cube.stack_look import _cloud
    from cube.detect import _white_pixels
    intr, plane, anchor, color, depth, corners = scene(2)
    points, heights, flat, pixels = _cloud(_white_pixels(color, COLORS), depth, intr, plane, .05, 1.2)
    bands = dict(upper_white=True, lower_uv=pixels[(heights > .08) & (heights < .32)],
                 upper_uv=pixels[(heights > .43) & (heights < .78)])
    return bands


def test_upper_depth_confirms_without_bottom_edge_and_reports_over_ssh(monkeypatch):
    import cube.top_fallback as module
    from cube.stack_look import StackSearch
    from view_tower import competition_lines
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback(active=True, saw_top_view=True)
    bands = upper_scene_bands()
    for i in range(5):
        update(fallback, 20 + i, layers=2, bands=bands)
        text = ' | '.join(competition_lines(StackSearch(), 32., None, None, fallback, True))
        if i < 4:
            assert fallback.layers is None
            assert f'连续稳定 {i+1}/5 帧' in fallback.instruction
            assert '识别结果：待确认' in text
            assert '停止俯仰和旋转' in text
    assert fallback.layers == 2
    assert '识别完成 白盒层数 2' in text
    assert '后退' not in text and '调整方向' not in text


def test_upper_vote_breaks_on_motion_and_depth_loss(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback(active=True)
    bands = upper_scene_bands()
    update(fallback, 20, layers=2, bands=bands)
    assert fallback.upper_count == 1
    update(fallback, 21, layers=2, bands=bands, moving=True)
    assert fallback.upper_count == 0
    assert '等待画面稳定' in fallback.instruction
    update(fallback, 22, layers=2, bands=bands)
    assert fallback.upper_count == 1
    intr, plane, anchor, color, depth, corners = scene(2)
    update(fallback, 23, layers=2, bands=bands, depth=depth * np.nan)
    assert fallback.layers is None and fallback.upper_count == 0
    assert '深度点不足' in fallback.instruction


def test_upper_region_rejects_unassociated_background():
    from cube.top_fallback import associated_upper
    n = 60
    pixels = np.column_stack([np.tile(np.arange(n), 2), np.repeat([100, 20], n)])
    flat = np.column_stack([np.tile(np.linspace(-.12, .12, n), 2),
                            np.repeat([.35, .95], n), np.zeros(2*n)])
    heights = np.repeat([.2, .55], n)
    bands = dict(upper_white=True, lower_uv=pixels[:n], upper_uv=pixels[n:])
    info = associated_upper(flat, heights, flat, pixels, bands, np.array([0,.35,0]), .35)
    assert info['layers'] is None
    assert '水平位置不匹配' in info['reason']


def test_one_layer_uses_top_surface_when_bottom_is_missing(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback(active=True, saw_top_view=True)
    for i in range(5):
        update(fallback, 20 + i * .1)
        if i < 4:
            assert fallback.layers is None
    assert fallback.layers == 1
    assert fallback.evidence['method'] == 'visible_top'
    assert fallback.evidence['exposed'] >= .78


def test_two_layers_never_become_one_from_missing_upper_candidate(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback(active=True)
    for i in range(8):
        update(fallback, 20 + i * .1, layers=2, bands={'upper_white': False})
    assert fallback.layers is None
    assert fallback.evidence['method'] == 'geometry_missing'


def test_missing_depth_advances_bounded_scan_and_ssh_stops():
    from cube.stack_look import StackSearch
    from view_tower import terminal_status
    intr, plane, anchor, color, depth, corners = scene()
    fallback = TopFallback(active=True)
    now = 20.
    visited = []
    for offset in (0., -8., 8., 16., 24.):
        target = 31.28366264107595 + offset
        visited.append(target)
        update(fallback, now, pitch=target, depth=depth * np.nan)
        assert fallback.layers is None
        update(fallback, now + 8.1, pitch=target, depth=depth * np.nan)
        now += 9.
    assert fallback.scan_exhausted
    assert '白盒 1 层已确认' in fallback.instruction
    text = terminal_status(StackSearch(), visited[-1], None, fallback, True)
    assert '角度已到达，可以了，停止俯仰和旋转' in text
    assert '证据不足' not in text
    assert '顶块待确认' not in text
    assert '结果：白盒层数 1' in text
    assert fallback.evidence['defaulted']
    assert '后退' not in text
    old_target = fallback.target_pitch
    update(fallback, now + 100, pitch=old_target, depth=depth * np.nan)
    assert fallback.target_pitch == old_target


def test_partial_top_and_vertical_slice_cannot_confirm_one(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    intr, plane, anchor, color, depth, corners = scene()
    # Keep a narrow strip of the visible top: insufficient square area.
    depth[:, :145] = np.nan
    depth[:, 175:] = np.nan
    fallback = TopFallback(active=True)
    for i in range(5):
        update(fallback, 20 + i * .1, depth=depth)
    assert fallback.layers is None


def test_upper_candidate_prevents_contradictory_one_layer(monkeypatch):
    import cube.top_fallback as module
    monkeypatch.setattr(module, '_square_from_near_edge', lambda *a, **k: None)
    fallback = TopFallback(active=True)
    for i in range(5):
        update(fallback, 20 + i * .1, bands={'upper_white': True})
    assert fallback.layers is None
    assert '上层候选尚未排除' in fallback.instruction


def test_rgb_and_distributed_partial_depth_confirm_one():
    intr, plane, anchor, color, depth, corners = scene()
    depth[::3, :] = np.nan
    info = top_exposure(color, depth, intr, plane, corners, COLORS, .35, .05, 1.2)
    assert info['layers'] == 1, info
    assert .45 <= info['coverage'] < .80
    assert info['rgb_white'] >= .80


def test_dark_rgb_cannot_confirm_from_depth_alone():
    intr, plane, anchor, color, depth, corners = scene()
    color[:] = 40
    info = top_exposure(color, depth, intr, plane, corners, COLORS, .35, .05, 1.2)
    assert info['layers'] is None



def test_arrival_holds_inside_three_degrees_and_corrects_outside():
    fallback = TopFallback(active=True)
    update(fallback, 20., depth=scene()[4] * np.nan)
    target = fallback.target_pitch
    update(fallback, 20.1, pitch=target + 2.9, moving=True)
    assert fallback.arrived
    update(fallback, 20.2, pitch=target + 3.0, moving=True)
    assert fallback.arrived and fallback.at_target(target + 3.0)
    update(fallback, 20.3, pitch=target + 3.01, moving=True)
    assert not fallback.arrived


def test_never_arriving_or_continuous_motion_finishes_bounded_scan():
    fallback = TopFallback(active=True)
    for i in range(6):
        update(fallback, 20. + i * 8.1, pitch=-20., moving=True)
    assert fallback.scan_exhausted and fallback.layers == 1
    assert fallback.result == '白盒层数 1，没有顶块'


def test_default_and_normal_one_layer_have_identical_terminal_output():
    from cube.stack_look import StackSearch
    from view_tower import terminal_status, competition_lines
    normal = TopFallback(active=True, layers=1, saw_top_view=True, target_pitch=32.)
    default = TopFallback(active=True, layers=1, scan_exhausted=True,
                          cap_seen=True, target_pitch=32.)
    normal.instruction = default.instruction = '白盒 1 层已确认，保持当前俯仰，开启下一轮识别'
    assert terminal_status(StackSearch(), 40., None, normal, True) == terminal_status(StackSearch(), 40., None, default, True)
    assert competition_lines(StackSearch(), 40., None, None, normal, True) == competition_lines(StackSearch(), 40., None, None, default, True)


def test_slow_arrival_does_not_consume_stable_observation_budget():
    fallback = TopFallback(active=True, scan_base=32., target_pitch=32.)
    update(fallback, 20., pitch=-10., moving=True)
    update(fallback, 27.5, pitch=32.)
    assert fallback.observation_started == 27.5
    for i in range(1, 5):
        update(fallback, 27.5 + i, pitch=32.)
    assert fallback.layers == 1 and not fallback.scan_exhausted
    assert fallback.scan_index == 0


def test_real_upper_depth_holds_current_view_before_nominal_arrival():
    from cube.stack_look import StackSearch
    from view_tower import competition_lines, terminal_status
    fallback = TopFallback(active=True, scan_base=55., target_pitch=55.)
    bands = upper_scene_bands()
    update(fallback, 20., layers=2, pitch=32., bands=bands)
    assert fallback.target_pitch == 32. and fallback.upper_hold_used
    assert fallback.upper_count == 1 and fallback.layers is None
    search = StackSearch()
    window = competition_lines(search, 32., None, None, fallback, True)[1]
    terminal = terminal_status(search, 32., None, fallback, True)
    assert '两层确认 1/5 帧' in window and window in terminal
    for i in range(1, 5):
        update(fallback, 20. + i, layers=2, pitch=32., bands=bands)
    assert fallback.layers == 2 and fallback.scan_index == 0


def test_upper_candidate_cannot_renew_hold_forever_or_confirm_without_lower():
    fallback = TopFallback(active=True, scan_base=55., target_pitch=55.)
    bands = upper_scene_bands()
    bands['lower_uv'] = np.zeros((0, 2), dtype=int)
    for i in range(50):
        update(fallback, 20. + i, layers=2, pitch=32., bands=bands)
        if fallback.layers is not None:
            break
    assert fallback.scan_exhausted and fallback.layers == 1
    assert fallback.evidence['defaulted']
    assert i < 50


def test_hold_gives_late_upper_candidate_its_own_observation_window():
    fallback = TopFallback(active=True, scan_base=32., target_pitch=32.)
    update(fallback, 20., pitch=32., depth=scene()[4] * np.nan)
    bands = upper_scene_bands()
    for i in range(5):
        update(fallback, 27. + i, layers=2, pitch=32., bands=bands)
    assert fallback.layers == 2 and fallback.scan_index == 0
    assert fallback.observation_started == 27.


@pytest.mark.parametrize('moving,missing_depth', [(True, False), (False, True)])
def test_motion_or_rgb_only_upper_does_not_lock_view(moving, missing_depth):
    fallback = TopFallback(active=True, scan_base=55., target_pitch=55.)
    bands = upper_scene_bands()
    depth = scene(2)[4]
    if missing_depth:
        depth = depth * np.nan
    update(fallback, 20., layers=2, pitch=32., bands=bands,
           moving=moving, depth=depth)
    assert not fallback.upper_hold_used
    assert fallback.target_pitch == 55. and fallback.upper_count == 0


def seed_lower_history(fallback):
    bands = upper_scene_bands()
    bands['upper_white'] = False
    for i in range(3):
        update(fallback, i * .1, layers=2, pitch=32., bands=bands)
    assert fallback.lower_reference_count == 3
    return bands


def test_remembered_lower_confirms_two_after_lower_leaves_image():
    fallback = TopFallback()
    seed_lower_history(fallback)
    bands = upper_scene_bands()
    bands['lower_uv'] = np.zeros((0, 2), dtype=int)
    fallback.active = True
    fallback.scan_base = fallback.target_pitch = 55.
    for i in range(5):
        update(fallback, 1. + i, layers=2, pitch=32., bands=bands)
    assert fallback.layers == 2
    assert fallback.evidence['used_history']
    assert fallback.upper_hold_used and fallback.target_pitch == 32.


@pytest.mark.parametrize('mode', ['expired', 'jumped', 'invalid'])
def test_unreliable_history_cannot_keep_upper_view(mode):
    fallback = TopFallback()
    seed_lower_history(fallback)
    bands = upper_scene_bands()
    bands['lower_uv'] = np.zeros((0, 2), dtype=int)
    kwargs = {}
    now = 1.
    if mode == 'expired':
        now = 46.
    elif mode == 'jumped':
        kwargs['anchor'] = scene(2)[2] + np.array([.5, 0., 0.])
    else:
        update(fallback, .5, valid=False)
    fallback.active = True
    fallback.scan_base = fallback.target_pitch = 55.
    update(fallback, now, layers=2, pitch=32., bands=bands, **kwargs)
    assert fallback.reliable_lower() is None
    assert fallback.layers is None and not fallback.upper_hold_used


def test_historical_lower_rejects_far_background_with_real_upper_depth():
    from cube.top_fallback import associated_upper
    n = 60
    flat = np.column_stack([np.linspace(-.12, .12, n), np.full(n, .95), np.zeros(n)])
    pixels = np.column_stack([np.arange(n), np.full(n, 20)])
    bands = dict(upper_white=True, lower_uv=np.zeros((0, 2), dtype=int), upper_uv=pixels)
    foot = np.array([0., .35, 0.])
    info = associated_upper(flat, np.full(n, .55), flat, pixels, bands, foot, .35,
                            lower_reference=foot)
    assert info['layers'] is None


def test_incomplete_lower_history_does_not_survive_missing_lower():
    fallback = TopFallback()
    bands = upper_scene_bands()
    bands['upper_white'] = False
    update(fallback, 0., layers=2, pitch=32., bands=bands)
    assert fallback.lower_reference_count == 1
    update(fallback, .1, layers=2, pitch=32., bands={})
    assert fallback.lower_reference_count == 0


def test_camera_rotation_keeps_history_in_current_camera_coordinates():
    from geom.transform import Rx
    fallback = TopFallback()
    seed_lower_history(fallback)
    point = fallback.lower_reference.copy()
    rotation = Rx(.3)
    fallback.reorient(rotation)
    assert np.allclose(fallback.reliable_lower(), rotation @ point)
    fallback.reorient(rotation.T)
    assert np.allclose(fallback.reliable_lower(), point)


def run_direct_frame(flow, now, layers=2, missing_depth=False, far_upper=False):
    from cube.stack_look import _cloud
    from cube.detect import _white_pixels
    intr, plane, anchor, color, depth, _ = scene(layers, pitch_deg=0., distance=1.)
    points, heights, flat, pixels = _cloud(_white_pixels(color, COLORS), depth, intr, plane, .05, 1.2)
    bands = dict(anchor=anchor, lower=True, distance=1., covers=True, cap=False,
                 upper_white=layers == 2,
                 lower_uv=pixels[(heights > .08) & (heights < .32)],
                 upper_uv=pixels[(heights > .43) & (heights < .78)])
    if far_upper:
        depth = depth.copy()
        uv = bands['upper_uv']
        depth[uv[:, 1], uv[:, 0]] = 2.5
    if missing_depth:
        depth = depth * np.nan
    raw_depth = np.where(np.isfinite(depth), depth, 3.)
    if missing_depth:
        raw_depth[:] = np.nan
    flow.update_search(bands, 0., -plane.normal, True, color, depth, intr, plane, raw_depth)
    spoken = flow.update_result(now, True, False, 0., color, depth, intr, plane,
                               bands, None, None, None, None, False)
    return spoken


def test_old_style_two_layer_reports_before_any_low_view_or_timeout():
    from view_tower import TowerFlow
    flow = TowerFlow(CFG)
    assert run_direct_frame(flow, 0.) is None
    assert run_direct_frame(flow, .1) is None
    assert run_direct_frame(flow, .2) == '识别完成 层数 2 没有顶块'
    assert not flow.fallback.active and flow.fallback.scan_index == 0


def test_direct_one_layer_requires_reliable_lower_and_clear_upper_space():
    from view_tower import TowerFlow
    flow = TowerFlow(CFG)
    for i in range(6):
        spoken = run_direct_frame(flow, i * .1, layers=1)
    assert flow.watch.line == '识别完成 层数 1 没有顶块'
    assert not flow.fallback.active


@pytest.mark.parametrize('mode', ['depth_holes', 'far_upper'])
def test_direct_layer_output_rejects_background_and_depth_holes(mode):
    from view_tower import TowerFlow
    flow = TowerFlow(CFG)
    for i in range(6):
        assert run_direct_frame(flow, i * .1, missing_depth=mode == 'depth_holes',
                                far_upper=mode == 'far_upper') is None
    assert flow.watch.line is None
    assert not flow.search.settled


def test_rejected_frame_breaks_direct_layer_consecutive_vote():
    from view_tower import TowerFlow
    flow = TowerFlow(CFG)
    run_direct_frame(flow, 0.)
    run_direct_frame(flow, .1)
    run_direct_frame(flow, .2, missing_depth=True)
    assert run_direct_frame(flow, .3) is None
    assert run_direct_frame(flow, .4) is None
    assert run_direct_frame(flow, .5) == '识别完成 层数 2 没有顶块'
