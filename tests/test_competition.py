import pytest

from cube.stack_look import StackSearch
from cube.top_fallback import TopFallback
from view_tower import competition_lines, park_result, settled_look_down_line


@pytest.mark.parametrize('pitch', [46.8, 48.8, 50.8])
def test_park_stops_within_tolerance(pitch):
    assert '已到位，可以停止' in settled_look_down_line(pitch)


def test_park_corrects_both_directions_and_rechecks_after_drift():
    assert '低头 13.8 度' in settled_look_down_line(35.)
    assert '抬头 6.2 度' in settled_look_down_line(55.)
    assert '已到位' in settled_look_down_line(48.8)
    assert '低头' in settled_look_down_line(40.)


@pytest.mark.parametrize('pitch', [None, float('nan'), float('inf')])
def test_invalid_pitch_never_reports_ready(pitch):
    assert '读数无效，停止' in settled_look_down_line(pitch)


def test_competition_holds_result_and_uses_live_pitch():
    search, fallback = StackSearch(), TopFallback()
    result = '识别完成 层数 1 没有顶块'
    lines = competition_lines(search, 48.8, '抬头 6 度', result, fallback, True)
    assert len(lines) == 3
    assert '+48.8' in lines[0]
    assert '已到位' in lines[1] and '旋转' in lines[1]
    assert lines[2] == result
    invalid = competition_lines(search, None, None, result, fallback, False)
    assert '等待姿态恢复' in invalid[1]
    assert '已到位' not in '\n'.join(invalid)


def test_competition_search_and_fallback_keep_actionable_instructions():
    search, fallback = StackSearch(), TopFallback()
    lines = competition_lines(search, 30., None, None, fallback, True)
    assert '先低头' in lines[1] and '旋转停止' in lines[1]
    assert lines[2] == '识别结果：待确认'
    fallback.active = True
    fallback.instruction = '低头到 40 度检查第一层顶面'
    fallback.layers = 2
    lines = competition_lines(search, 30., None, '旧结果', fallback, True)
    assert fallback.instruction in lines[1]
    assert '没有顶块' in lines[2]


def test_unfinished_result_does_not_park():
    lines = ['层数还没定', '继续抬头']
    assert park_result(lines, None, 48.8) == lines


def test_cap_cycle_restarts_only_after_return_and_keeps_latched_result():
    from view_tower import RecognitionCycle
    cycle = RecognitionCycle()
    assert not cycle.update('识别完成 顶块红色', None, -6., True, False)
    assert not cycle.update(None, None, 32., True, False)
    assert not cycle.update(None, None, 48.8, True, True)
    assert not cycle.update(None, None, 48.8, False, False)
    assert cycle.update(None, None, 48.8, True, False)
    assert not RecognitionCycle().update(None, None, 48.8, True, False)


def test_absent_cycle_keeps_pitch_and_restarts_after_layer_confirmation():
    from view_tower import RecognitionCycle
    cycle = RecognitionCycle()
    assert not cycle.update(None, None, 32., True, False)
    assert cycle.update(None, 2, 32., True, False)
    assert not RecognitionCycle().update(None, None, 32., True, False)
