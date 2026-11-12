import cv2
import numpy as np
from cube.stack_look import cap_color_masks


def test_connected_weak_color_recovers_without_remote_or_white_fill():
    hsv = np.zeros((100, 160, 3), np.uint8)
    hsv[20:70, 20:70] = (174, 130, 200)
    hsv[30:50, 30:50] = (174, 210, 200)
    hsv[20:70, 100:140] = (174, 130, 200)
    hsv[35:45, 50:60] = (174, 0, 255)
    red, blue = cap_color_masks(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert red[25, 25] and not red[25, 110] and not red[40, 55]
    assert not blue.any()


def test_weak_colors_need_substantial_strict_seeds():
    hsv = np.full((100, 100, 3), (110, 100, 200), np.uint8)
    hsv[50, 50] = (110, 220, 200)
    red, blue = cap_color_masks(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert not red.any() and np.count_nonzero(blue) == 1


def test_blue_weak_region_recovers_from_strict_seed():
    hsv = np.zeros((100, 100, 3), np.uint8)
    hsv[20:70, 20:70] = (110, 100, 200)
    hsv[30:50, 30:50] = (110, 210, 200)
    _, blue = cap_color_masks(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert blue[25, 25]
