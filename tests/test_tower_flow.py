"""Lifecycle regression checks: partial resets must not lose retained evidence."""
import numpy as np
import pytest

from main import load_config
from view_tower import TowerFlow


@pytest.mark.parametrize('reason,keep_search,keep_memory', [
    ('attitude_invalid', True, False),
    ('target_changed', False, True),
    ('next_cycle', False, False),
])
def test_reset_preserves_original_evidence_scope(reason, keep_search, keep_memory):
    flow = TowerFlow(load_config('config/default.yaml'))
    search, fallback, caps, outlines, watch, color, memory, cycle = flow.resources()
    search.anchor = np.array([0., .3, .8])
    watch.line = '已确认'
    watch.status = '位置跳变连续确认：重新定位'
    color.latch('red', [0., .7, .8], 0.)
    cycle.cap_reported = True
    flow.reset(reason)
    assert (flow.search is search) == keep_search
    assert (flow.memory is memory) == keep_memory
    assert flow.fallback is not fallback and flow.caps is not caps
    assert flow.cycle is not cycle and not flow.cycle.cap_reported
    assert not flow.color_hold.latched
    assert flow.watch.line == ('已确认' if keep_search else None)
    if reason == 'target_changed':
        assert flow.watch.status == '位置跳变连续确认：重新定位'


def test_completed_cap_returns_before_new_cycle_and_counts_once():
    flow = TowerFlow(load_config('config/default.yaml'))
    flow.watch.line = '识别完成 顶块红色'
    count = flow.complete_frame(0, -6., True, False)
    assert count == 1 and not flow.cycle.restart
    count = flow.complete_frame(count, 48.8, True, True)
    assert count == 1 and not flow.cycle.restart
    count = flow.complete_frame(count, 48.8, True, False)
    assert count == 1 and flow.cycle.restart


def test_layer_result_keeps_immediate_restart_without_return():
    flow = TowerFlow(load_config('config/default.yaml'))
    flow.fallback.layers = 2
    assert flow.complete_frame(0, 30., True, False) == 1
    assert flow.cycle.restart


@pytest.mark.parametrize('blocked_by', ['invalid', 'reported', 'fallback'])
def test_departure_does_not_clear_target_during_protected_states(monkeypatch, blocked_by):
    flow = TowerFlow(load_config('config/default.yaml'))
    flow.cycle.cap_reported = blocked_by == 'reported'
    flow.fallback.active = blocked_by == 'fallback'
    def unexpected(*args, **kwargs):
        pytest.fail('protected state must not run departure detection')
    monkeypatch.setattr(flow.watch, 'depart', unexpected)
    assert not flow.target_departed(1., {}, blocked_by != 'invalid', False)


@pytest.mark.parametrize('evidence', ['presence', 'measured', 'world'])
def test_latest_cap_evidence_precedes_restored_direct_white_result(monkeypatch, evidence):
    flow = TowerFlow(load_config('config/default.yaml'))
    flow.search.kind = 'two'
    flow.search.settled = True
    monkeypatch.setattr(flow.fallback, 'update', lambda *args, **kwargs: None)
    bands = {'cap_presence_confirmed': evidence == 'presence'}
    measured = object() if evidence == 'measured' else None
    world = np.array([.1,.8,.8]) if evidence == 'world' else None
    spoken = flow.update_result(1., True, False, 0., None, None, None, None,
                                bands, measured, world, 0., 'red', False)
    if evidence == 'world':
        assert flow.search.kind == 'two_cap'
        assert spoken is not None and '红' in spoken
    else:
        assert spoken is None and flow.watch.line is None
    if spoken:
        assert '没有顶块' not in spoken
