import numpy as np
import pytest
from cube.top_fallback import TopFallback, top_exposure
from geom.transform import deproject_pixels
from ground.plane import Plane
from sim.scene import intrinsics_for

COLORS = [{'name':'white','hsv_low':[0,0,140],'hsv_high':[180,110,255]}]
CFG = dict(cube_size=.35, colors=COLORS, depth_min=.05, depth_max=1.2, size_tolerance=.06)


def scene(layers=1, offset=0.):
    intr = intrinsics_for(320, 240)
    angle = np.radians(32.)
    up = np.array([0., -np.cos(angle), -np.sin(angle)])
    forward = np.array([0., -np.sin(angle), np.cos(angle)])
    plane = Plane(up, -.669)
    yy, xx = np.indices((240, 320))
    rays = deproject_pixels(np.column_stack([xx.ravel(), yy.ravel()]), np.ones(xx.size), intr)
    directions = np.column_stack([rays[:, 0], rays @ forward, rays @ up])
    origin = np.array([0.,0.,.669])
    nearest = np.full(xx.size, np.inf)
    for level in range(layers):
        lo = np.array([-.175 + (offset if level else 0), .35, level*.35])
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
                        for x,y in [(-.175,.35),(.175,.35),(.175,.70),(-.175,.70)]])
    anchor = .35*forward-.669*up
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
    fallback = TopFallback(active=True, layers=2, instruction='继续检查塔顶')
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
    monkeypatch.setattr(view, 'load_intrinsics', lambda *a: intr)
    monkeypatch.setattr(view.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(view, 'clean_frame', lambda *a: SimpleNamespace(color=color,depth=depth,gravity=-plane.normal))
    monkeypatch.setattr(view, 'tower_ground', lambda *a: GroundFix(plane, 'imu', height=.669))
    monkeypatch.setattr(view, 'coarse_bands', lambda *a, **k: dict(anchor=anchor,lower=True,
                           upper_white=False,cap=False,distance=.35,lower_uv=np.zeros((0,2),int)))
    monkeypatch.setattr(view, 'pitch_down_degrees', lambda *a: -6. if index[0] <= 1 else 32.)
    monkeypatch.setattr(view, 'StackSearch', lambda: StackSearch(kind='one',settled=True,anchor=anchor.copy()))
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
        assert '识别完成 白盒层数 2，没有顶块' in output
        assert '开始下一轮识别' in output
        assert not rendered
        return
    assert not any('没有顶块' in line or '层数 1' in line for line in rendered[0])
    assert any('备用流程' in line for line in rendered[1])
    assert any('层数 2（白盒）；没有顶块' == line for line in rendered[-2])
    assert any('保持当前俯仰' in line for line in rendered[-2])
    assert any('层数待确认' in line for line in rendered[-1])
    assert any('先抬头' in line for line in rendered[-1])
    assert not any('没有顶块' in line for line in rendered[-1])


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


def test_initial_downward_view_never_counts_without_prior_top_check():
    fallback = TopFallback()
    for now in (0., 15., 30., 60.):
        update(fallback, now, pitch=32.)
    assert not fallback.saw_top_view and not fallback.active
    assert fallback.layers is None and fallback.count == 0
    assert '先抬头' in fallback.instruction


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


def test_visible_cap_cannot_timeout_into_absent_branch():
    fallback = TopFallback()
    for now in (0., 16., 32.):
        update(fallback, now, pitch=-6., cap_visible=True)
    assert not fallback.active
    assert fallback.started is None
    assert '已看到顶块' in fallback.instruction


def test_lowering_before_absence_check_finishes_cannot_confirm_absence():
    fallback = TopFallback()
    update(fallback, 0., pitch=-6.)
    update(fallback, 16., pitch=32.)
    assert not fallback.active and fallback.layers is None
    assert fallback.started is None
