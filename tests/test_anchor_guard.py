import numpy as np
from camera.source import Frame
from cube.stack_look import StackSearch
from view_tower import TowerWatch, camera_turning_fast


def setup():
    return TowerWatch(), StackSearch(anchor=np.array([0.,.6,.5]),settled=True)


def candidate(x=.5):
    return dict(lower=True, upper_white=True,cap=False,anchor=np.array([x,.6,.5]),lower_uv=np.array([[2,3]]))


def test_jump_requires_three_consecutive_candidates_and_cannot_replace_search_anchor():
    w,s=setup();b=candidate()
    assert not w.depart(0,s,b)
    assert not b['lower'] and not b['upper_white']
    np.testing.assert_array_equal(b['anchor'],s.anchor)
    assert not w.depart(.3,s,candidate(.51))
    assert w.depart(.6,s,candidate(.49))


def test_return_or_inconsistent_candidate_breaks_confirmation():
    w,s=setup()
    w.depart(0,s,candidate())
    assert not w.depart(.2,s,candidate(0))
    w.depart(.3,s,candidate())
    w.depart(.5,s,candidate(-.5))
    assert w.pending_count==1
    assert not w.depart(2,s,candidate(-.5))
    assert w.pending_count==1


def test_motion_cancels_candidate_and_does_not_trigger_missing_timeout():
    w,s=setup();w.line='reported'
    w.depart(0,s,candidate())
    assert not w.depart(.2,s,candidate(),moving=True)
    assert w.pending_count==0
    assert not w.depart(4,s,{},moving=True)
    assert not w.depart(4.1,s,{})
    assert w.depart(4.7,s,{})


def test_gyro_uses_recent_valid_samples():
    f=Frame(np.zeros((1,1,3),np.uint8),np.zeros((1,1)))
    f.gyro=[(0,np.array([2.,0,0])),(1,np.zeros(3))]
    assert not camera_turning_fast(f)
    f.gyro.append((1.1,np.array([0,.6,0])))
    assert camera_turning_fast(f)
    f.gyro=None
    assert not camera_turning_fast(f)
