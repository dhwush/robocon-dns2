"""Feedback stays consistent; optional diagnosis does not run on every frame."""
import numpy as np
import pytest

from cube.stack_look import StackSearch
from cube.top_fallback import TopFallback
from view_tower import competition_lines, terminal_status
import tower_debug_window as debug


@pytest.mark.parametrize('pitch,valid,moving,complete', [
    (32., True, False, False),
    (40., True, False, False),
    (40., True, True, False),
    (None, False, False, False),
    (40., True, False, True),
])
def test_window_and_terminal_give_same_action(pitch, valid, moving, complete):
    search = StackSearch()
    fallback = TopFallback(active=True, target_pitch=32., layers=1 if complete else None)
    lines = competition_lines(search, pitch, None, None, fallback, valid, moving)
    terminal = terminal_status(search, pitch, None, fallback, valid, moving=moving)
    assert terminal.split(' | 动作：')[1].split(' | ')[0] == lines[1]
    if moving or not valid:
        assert lines[1].startswith('停止俯仰和旋转')


def test_arrival_hysteresis_is_used_in_both_displays():
    fallback = TopFallback(active=True, target_pitch=32., arrived=True)
    search = StackSearch()
    for pitch in (28., 36.):
        assert '停止俯仰和旋转' in competition_lines(search, pitch, None, None, fallback, True)[1]
        assert '停止俯仰和旋转' in terminal_status(search, pitch, None, fallback, True)


def test_debug_is_opt_in_and_throttled_with_freeze_and_reopen(monkeypatch):
    calls, clock = [], [0.]
    monkeypatch.setattr(debug.time, 'monotonic', lambda: clock[0])
    def build(*args):
        calls.append(args)
        return np.zeros((20, 20, 3), np.uint8), {'frame': args[0]}
    monkeypatch.setattr(debug, 'build_panel', build)
    for name in ('namedWindow', 'resizeWindow', 'imshow', 'destroyWindow'):
        monkeypatch.setattr(debug.cv2, name, lambda *args: None)
    window = debug.DebugWindow()
    window.update(0)
    assert calls == []
    window.key(ord('d'))
    window.update(1)
    clock[0] = .5
    window.update(2)
    assert len(calls) == 1 and window.metadata['frame'] == 1
    clock[0] = 1.
    window.update(3)
    assert len(calls) == 2 and window.metadata['frame'] == 3
    window.key(ord('f'))
    clock[0] = 5.
    window.update(4)
    assert len(calls) == 2
    window.key(ord('f'))
    window.update(5)
    assert len(calls) == 3 and window.metadata['frame'] == 5
    window.key(ord('d'))
    clock[0] = 10.
    window.update(6)
    assert len(calls) == 3
    window.key(ord('d'))
    window.update(7)
    assert len(calls) == 4 and window.metadata['frame'] == 7


def test_scan_preserves_angles_and_reduces_largest_reverse_move():
    fallback = TopFallback(active=True, scan_base=32., target_pitch=32.)
    targets = [fallback.target_pitch]
    for i in range(4):
        fallback.scan_failure(i * 8., '超时', advance=True)
        targets.append(fallback.target_pitch)
    assert sorted(targets) == [24., 32., 40., 48., 56.]
    assert max(abs(b - a) for a, b in zip(targets, targets[1:])) == 16.
    assert not fallback.scan_exhausted and fallback.layers is None
