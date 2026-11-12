import cv2
import numpy as np
from types import SimpleNamespace
from cube.stack_look import cap_color_masks, _cap_faces


def test_weak_red_without_seeds_requires_adjacent_blue():
    hsv = np.zeros((240, 320, 3), np.uint8)
    hsv[40:100, 80:220] = (174, 140, 200)
    hsv[103:170, 80:220] = (110, 210, 180)
    hsv[40:100, 260:310] = (174, 140, 200)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    r,b = cap_color_masks(color)
    assert r[60,100] and not r[60,280]
    color[103:170,80:220] = 0
    r,b = cap_color_masks(color)
    assert not r.any()


def test_complete_pair_recovers_shorter_face_without_filling_gap():
    r=np.zeros((240,320),np.uint8);b=r.copy()
    r[60:200,70:250]=255;b[13:57,70:250]=255
    z=np.full(r.shape,.4);intr=SimpleNamespace(fx=500.,fy=500.)
    strict=_cap_faces(r|b,z,intr,.05,1.2)
    assert not (strict & b).any()
    fixed=_cap_faces(r|b,z,intr,.05,1.2,red=r,blue=b)
    assert (fixed & b).any() and not fixed[(r|b)==0].any()
