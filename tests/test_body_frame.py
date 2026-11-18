"""同一物理目标在不同相机俯仰下，输出必须保持车身右前点坐标。"""
from types import SimpleNamespace
import numpy as np
import pytest
from cube.body_frame import body_point, body_mount, body_record, calibrate_cap, DEFAULT_OPTICAL_CENTER, LEVEL_ROTATION
from cube.aim import _move, _cap_command
from view_tower import settled_look_down_line


def camera_rotation(degrees):
    a=np.radians(degrees)
    return np.array([[1.,0.,0.],[0.,np.cos(a),np.sin(a)],[0.,-np.sin(a),np.cos(a)]]) @ LEVEL_ROTATION


@pytest.mark.parametrize('pitch', [-40., -10., 0., 30., 60.])
def test_pitch_does_not_rotate_car_axes(pitch):
    rotation=camera_rotation(pitch)
    up=rotation[:,2]
    target=np.array([.2,.7,.8])
    optical=rotation @ (target-DEFAULT_OPTICAL_CENTER)
    np.testing.assert_allclose(body_point(optical,up),target,atol=1e-12)
    record=dict(zip(('x','y','z'),optical), edge=rotation @ np.array([1.,0.,0.]),yaw=0.)
    out=body_record(record,up)
    np.testing.assert_allclose([out['x'],out['y'],out['z']],target,atol=1e-12)
    assert out['frame']=='car_front_right' and abs(out['yaw'])<1e-12
    pose=SimpleNamespace(center=optical,edge=record['edge'],world=None,yaw=None)
    calibrate_cap(pose,up)
    np.testing.assert_allclose(pose.world,target,atol=1e-12)


@pytest.mark.parametrize('pitch', [-35., 0., 40.])
def test_offset_pivot_moves_optical_center_but_not_body_origin(pitch):
    rotation=camera_rotation(pitch)
    lever=np.array([.01,.08,.02])
    pivot=np.array([-.2,0.,.6])
    zero=pivot+lever
    origin=pivot+rotation.T @ LEVEL_ROTATION @ lever
    cfg={'camera_optical_center_m':zero,'camera_pivot_to_optical_m':lever}
    target=np.array([.1,.9,.8])
    point=rotation @ (target-origin)
    np.testing.assert_allclose(body_point(point,rotation[:,2],cfg),target,atol=1e-12)


@pytest.mark.parametrize('up', [None,[0,0,0],[np.nan,0,1]])
def test_missing_attitude_cannot_publish_body_coordinates(up):
    with pytest.raises(ValueError):
        body_point([0,0,1],up)


@pytest.mark.parametrize('degrees', [-3.,3.,-3.001,3.001])
def test_both_axes_use_inclusive_three_degree_stop(degrees):
    signed=np.radians(degrees)
    for axis in ('pitch','yaw'):
        assert (_move(axis,signed).direction=='none') == (abs(degrees)<=3.)
    assert (_cap_command(signed)==0.) == (abs(degrees)<=3.)


@pytest.mark.parametrize('delta',[-3.,3.,-3.01,3.01])
def test_parking_stop_boundary(delta):
    assert ('可以了' in settled_look_down_line(48.8+delta)) == (abs(delta)<=3.)


def test_command_line_publishes_body_frame_at_different_pitches(monkeypatch,capsys):
    import json
    import main as app
    from cube.aim import ViewAdjustment, _none_move
    from camera.source import Frame
    from ground.plane import Plane
    from sim.scene import intrinsics_for
    target=np.array([.15,.75,.175]); index=[-1]
    angles=[0.,30.,-20.]
    class Source:
        def __init__(self,*a,**kw): pass
        def read(self):
            index[0]+=1
            if index[0]>=len(angles): raise KeyboardInterrupt
            rotation=camera_rotation(angles[index[0]])
            return Frame(np.zeros((240,320,3),np.uint8),np.ones((240,320)), -rotation[:,2])
        def close(self): pass
    class Search:
        settled_one=True; settled=True; kind='one'; stage='one'; upper=None; anchor=None; looked_up=False
        def reorient(self,*a): pass
        def update(self,*a): pass
    def ground(*a,**kw):
        rotation=camera_rotation(angles[index[0]])
        return SimpleNamespace(plane=Plane(rotation[:,2],-.669)),False
    def estimate(*a,**kw):
        rotation=camera_rotation(angles[index[0]])
        point=rotation @ (target-DEFAULT_OPTICAL_CENTER)
        item=dict(zip(('x','y','z'),point),yaw=0.,quarter=True,layer=1,inliers=100,
                  long_side=.35,short_side=.35,ground='imu',edge=rotation[:,0],bottom=point)
        return [item]
    monkeypatch.setattr(Source, 'intrinsics', intrinsics_for(320,240), raising=False)
    monkeypatch.setattr(app,'RealSenseSource',Source)
    monkeypatch.setattr(app,'StackSearch',Search)
    monkeypatch.setattr(app,'ground_for_layers',ground)
    monkeypatch.setattr(app,'estimate_layers',estimate)
    monkeypatch.setattr(app,'coarse_bands',lambda *a,**k: {})
    monkeypatch.setattr(app,'plan_view',lambda *a,**k: ViewAdjustment(True,True,'inside',_none_move(),_none_move(),{}))
    monkeypatch.setattr(app.time,'monotonic',lambda: float(index[0]+1))
    cfg=app.load_config('config/default.yaml')
    cfg.update(min_samples=1,frames_to_fuse=1)
    with pytest.raises(KeyboardInterrupt):
        app.run_live(cfg,.669,None)
    outputs=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(outputs)==3
    for out in outputs:
        assert out['frame']=='car_front_right' and out['motion_arrived']
        assert isinstance(out['yaw'], (int,float)) and abs(out['yaw']) < 1e-8
        assert out['yaw_command']['direction']=='none'
        np.testing.assert_allclose([out['x'],out['y'],out['z']],target,atol=1e-8)
