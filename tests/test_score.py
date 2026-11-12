"""队内赛评分说明（2026-10-28）的计分。用例就是说明里的算例。

X/Y 得分 = 5 − 误差(cm)，角度得分 = (5 − 误差(°)) ÷ 2。每次每个小项都得正分。
方块 yaw 考虑 90° 旋转对称，误差取最小需要旋转的角度。
"""

from __future__ import annotations

import numpy as np

from fuse.median import average_scores, position_score, score_trial, yaw_score


def _trial(est_x_cm, est_y_cm, est_yaw_deg, gt_x_cm, gt_y_cm, gt_yaw_deg):
    return score_trial(
        np.array([est_x_cm / 100.0, est_y_cm / 100.0]),
        float(np.radians(est_yaw_deg)),
        np.array([gt_x_cm / 100.0, gt_y_cm / 100.0]),
        float(np.radians(gt_yaw_deg)),
    )


def test_single_trial_matches_the_worked_example():
    # 说明 2.3：X 40→42.37 得 2.63，Y 20→15.85 得 0.85，角度 12→15.26 得 0.87。这次达标。
    trial = _trial(42.37, 15.85, 15.26, 40.0, 20.0, 12.0)
    assert abs(trial["error_x_cm"] - 2.37) < 1e-9
    assert abs(trial["error_y_cm"] - 4.15) < 1e-9
    assert abs(trial["error_yaw_deg"] - 3.26) < 1e-9
    assert abs(trial["score_x"] - 2.63) < 1e-9
    assert abs(trial["score_y"] - 0.85) < 1e-9
    assert abs(trial["score_yaw"] - 0.87) < 1e-9
    assert trial["passed"]


def test_error_at_five_scores_zero_and_fails():
    # 说明 3.1：误差恰好 5 cm 或 5° 得 0 分，不满足"每次都得正分"。
    assert position_score(5.0) == 0.0
    assert yaw_score(5.0) == 0.0
    assert position_score(4.99) > 0.0
    assert position_score(5.01) < 0.0
    # 米换算回厘米在 5 cm 边界有浮点尾差，超线用明显超过的 5.01 cm 验证。
    trial = _trial(45.01, 20.0, 12.0, 40.0, 20.0, 12.0)
    assert trial["score_x"] < 0.0
    assert not trial["passed"]


def test_one_bad_item_fails_the_trial():
    # 说明 3.2：X 差 6 cm 是 −1，Y 差 1 cm 是 4，角度差 1° 是 2。X 为负，这次不达标。
    trial = _trial(46.0, 21.0, 13.0, 40.0, 20.0, 12.0)
    assert abs(trial["score_x"] - (-1.0)) < 1e-9
    assert abs(trial["score_y"] - 4.0) < 1e-9
    assert abs(trial["score_yaw"] - 2.0) < 1e-9
    assert not trial["passed"]


def test_yaw_error_uses_the_quarter_turn_symmetry():
    # 说明 5：89° 对 1° 只差 2°，得 1.5；0° 对 89° 差 1°，得 2。
    assert abs(_trial(40.0, 20.0, 1.0, 40.0, 20.0, 89.0)["score_yaw"] - 1.5) < 1e-9
    assert abs(_trial(40.0, 20.0, 89.0, 40.0, 20.0, 0.0)["score_yaw"] - 2.0) < 1e-9
    # 12° 对 17° 差 5° 得 0，不达标；10° 对 18° 差 8° 得 −1.5。
    assert abs(_trial(40.0, 20.0, 17.0, 40.0, 20.0, 12.0)["score_yaw"]) < 1e-9
    assert not _trial(40.0, 20.0, 17.0, 40.0, 20.0, 12.0)["passed"]
    assert abs(_trial(40.0, 20.0, 18.0, 40.0, 20.0, 10.0)["score_yaw"] - (-1.5)) < 1e-9


def test_average_is_five_minus_mean_absolute_error():
    # 说明 4.1：误差 1、2、4 cm，得分 4、3、1，平均 8/3，与 5 − 平均绝对误差一致。
    trials = [
        _trial(41.0, 20.0, 12.0, 40.0, 20.0, 12.0),
        _trial(42.0, 20.0, 12.0, 40.0, 20.0, 12.0),
        _trial(44.0, 20.0, 12.0, 40.0, 20.0, 12.0),
    ]
    result = average_scores(trials)
    assert abs(result["x"] - 8.0 / 3.0) < 1e-9
    assert abs(result["mean_error_x_cm"] - 7.0 / 3.0) < 1e-9
    assert result["passed"]
    assert result["trials"] == 3


def test_positive_average_with_one_negative_trial_still_fails():
    # 说明 4.2：误差 1、1、10 cm，得分 4、4、−5，平均 1 分，但第三次为负，不达标。
    trials = [
        _trial(41.0, 20.0, 12.0, 40.0, 20.0, 12.0),
        _trial(39.0, 20.0, 12.0, 40.0, 20.0, 12.0),
        _trial(50.0, 20.0, 12.0, 40.0, 20.0, 12.0),
    ]
    result = average_scores(trials)
    assert abs(result["x"] - 1.0) < 1e-9
    assert not result["passed"]


def test_zero_score_trial_fails_even_with_positive_average():
    # 说明 4.3：角度误差 1、3、5°，得分 2、1、0，平均 1 分，第三次 0 分不达标。
    trials = [
        _trial(40.0, 20.0, 13.0, 40.0, 20.0, 12.0),
        _trial(40.0, 20.0, 15.0, 40.0, 20.0, 12.0),
        _trial(40.0, 20.0, 17.0, 40.0, 20.0, 12.0),
    ]
    result = average_scores(trials)
    assert abs(result["yaw"] - 1.0) < 1e-9
    assert not result["passed"]


def test_absolute_error_taken_before_averaging():
    # 说明 4.4：+3 cm 和 −3 cm 不抵消。两次误差都是 3 cm，平均 2 分。
    trials = [
        _trial(43.0, 20.0, 12.0, 40.0, 20.0, 12.0),
        _trial(37.0, 20.0, 12.0, 40.0, 20.0, 12.0),
    ]
    result = average_scores(trials)
    assert abs(result["x"] - 2.0) < 1e-9
    assert result["passed"]


def test_empty_trials_fail():
    assert not average_scores([])["passed"]
