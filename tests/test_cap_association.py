import numpy as np
from cube.stack_look import _at_tower_face, _rgb_cap_candidate, _near
from ground.plane import Plane
from sim.scene import intrinsics_for
from cube.top_fallback import TopFallback


def scene(width=.20, lateral=0., background=False):
    intr = intrinsics_for(640, 480)
    angle = np.radians(-10.)
    up = np.array([0., -np.cos(angle), -np.sin(angle)])
    forward = np.cross(up, [1.,0.,0.])
    plane = Plane(up, -.669)
    anchor = .65 * forward - up * .669
    yy, xx = np.indices((480,640))
    uv = np.column_stack([xx.ravel(), yy.ravel()])
    points = _at_tower_face(uv, intr, plane, anchor)
    heights = plane.signed_distance(points).reshape(xx.shape)
    lateral_x = points[:,0].reshape(xx.shape)
    body = (abs(lateral_x-lateral) < width/2) & (heights > .70) & (heights < .90)
    color = np.full((480,640,3),40,np.uint8)
    color[body & (heights < .80)] = [255,0,0]
    color[body & (heights >= .80)] = [0,0,255]
    depth = np.full(xx.shape,np.nan)
    if background:
        depth[body] = 2.5
    return intr, plane, anchor, color, depth


def candidate(**kwargs):
    intr, plane, anchor, color, depth = scene(**kwargs)
    debug = {}
    pixels = _rgb_cap_candidate(color, intr, plane, anchor, debug, depth)
    return pixels, debug


def test_valid_rgb_and_lateral_tolerance_without_narrowing_white_box_gate():
    pixels, debug = candidate(lateral=.20)
    assert len(pixels) and debug['associated']
    intr, plane, anchor, _, _ = scene()
    assert _near(np.array([anchor + [.40,0.,0.]]), anchor, plane)[0]


def test_same_image_background_depth_rejects_pair():
    assert len(candidate()[0])
    pixels, debug = candidate(background=True)
    assert not len(pixels) and debug['candidate'] == 'depth_conflict'


def test_oversized_red_blue_sign_is_rejected():
    pixels, debug = candidate(width=.65)
    assert not len(pixels) and debug['candidate'] == 'size_mismatch'


def test_unconfirmed_rgb_does_not_reset_absence_timer():
    intr, plane, anchor, color, depth = scene()
    cfg = dict(cube_size=.35)
    fallback = TopFallback(timeout=2.)
    for now in (0.,1.,2.):
        fallback.update(now,anchor,False,True,False,-10.,color,depth,intr,plane,cfg,cap_visible=False)
    assert fallback.active
    assert fallback.layers is None


def test_depthless_associated_candidate_is_not_presence_confirmation():
    from cube.stack_look import coarse_bands
    from main import load_config
    intr, plane, anchor, color, depth = scene(lateral=.20)
    cfg = load_config('config/default.yaml')
    bands = coarse_bands(color,depth,plane,intr,cfg['colors'],.05,1.2,anchor,pitch=-10.)
    assert bands['cap_rgb_candidate'] and bands['cap_candidate_associated']
    assert not bands['cap_presence_confirmed'] and not bands['cap']
