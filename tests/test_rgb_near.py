"""近处无深度、跨俯仰和截断 RGB 场景。"""
import numpy as np
from cube.stack_look import StackSearch, _at_tower_face, coarse_bands
from geom.transform import project_points
from ground.plane import Plane
from sim.scene import intrinsics_for
from main import load_config


def pose(pitch, distance=0.22):
    angle = np.radians(pitch)
    up = np.array([0., -np.cos(angle), -np.sin(angle)])
    forward = np.cross(up, [1., 0., 0.])
    plane = Plane(up, -0.7)
    foot = forward * distance - up * 0.7
    return up, forward, plane, foot


def test_remembered_anchor_survives_down_up_down_without_drift():
    search = StackSearch()
    up, _, _, foot = pose(50)
    search.anchor = foot.copy()
    search.reorient(-up)
    for pitch in (-15, 9, 50, -15, 50):
        current_up, _, _, expected = pose(pitch)
        search.reorient(-current_up)
        np.testing.assert_allclose(search.anchor, expected, atol=1e-10)
    search.reorient(None)
    assert search.anchor is None


def test_ray_face_intersection_recovers_heights_across_pitch():
    intr = intrinsics_for(320, 240)
    for pitch in (50, 9, -15):
        up, forward, plane, foot = pose(pitch, distance=0.6)
        points = np.array([forward * 0.6 + up * (height - 0.7) for height in (0.72, 0.8, 0.88)])
        uv = project_points(points, intr)
        actual = _at_tower_face(uv, intr, plane, foot)
        np.testing.assert_allclose(actual, points, atol=1e-10)
        np.testing.assert_allclose(plane.signed_distance(actual), [0.72, 0.8, 0.88], atol=1e-10)


def scene(pitch=-15, center=None):
    intr = intrinsics_for(320, 240)
    if center is not None:
        intr.cx = center
    up, _, plane, foot = pose(pitch)
    yy, xx = np.indices((intr.height, intr.width))
    uv = np.column_stack([xx.ravel(), yy.ravel()])
    xyz = _at_tower_face(uv, intr, plane, foot)
    height = plane.signed_distance(xyz).reshape(xx.shape)
    x = xyz[:, 0].reshape(xx.shape)
    color = np.full((intr.height, intr.width, 3), 40, np.uint8)
    red = (abs(x) < .1) & (height >= .7) & (height < .8)
    blue = (abs(x) < .1) & (height >= .8) & (height <= .9)
    color[red] = (0, 0, 255)
    color[blue] = (255, 0, 0)
    return intr, plane, foot, color, np.full(xx.shape, np.nan)


def test_close_depthless_rgb_is_visible_after_pitch_change():
    intr, plane, foot, color, depth = scene(-9)
    cfg = load_config('config/default.yaml')
    bands = coarse_bands(color, depth, plane, intr, cfg['colors'], .05, 1.2, foot, pitch=-9)
    assert bands['cap']
    assert bands['rgb_debug']['candidate'] == 'visible'


def test_border_clipped_pair_is_candidate_without_fabricated_point():
    intr, plane, foot, color, depth = scene(-15, center=20)
    # 只保留图像边缘附近的条带，模拟近处仅一部分入画，完整面检查应拒绝。
    color[:, 20:] = 40
    cfg = load_config('config/default.yaml')
    bands = coarse_bands(color, depth, plane, intr, cfg['colors'], .05, 1.2, foot, pitch=-15)
    assert bands['cap_rgb_candidate']
    assert bands['cap_point'] is None
    assert len(bands['cap_uv']) > 0
    # 同位置只有红色时，不能依靠截断放宽条件误认成红蓝候选。
    color[np.all(color == (255, 0, 0), axis=2)] = 40
    single = coarse_bands(color, depth, plane, intr, cfg['colors'], .05, 1.2, foot, pitch=-15)
    assert not single['cap_rgb_candidate']


def test_background_pair_without_tower_anchor_is_not_accepted():
    intr, plane, _, color, depth = scene()
    cfg = load_config('config/default.yaml')
    bands = coarse_bands(color, depth, plane, intr, cfg['colors'], .05, 1.2)
    assert not bands['cap']
    assert bands['rgb_debug']['candidate'] == 'no_anchor'


def test_upward_view_does_not_use_near_zero_ground_anchor_z_as_distance():
    intr, plane, foot, color, depth = scene(-15)
    assert foot[2] < 0.15
    cfg = load_config('config/default.yaml')
    bands = coarse_bands(color, depth, plane, intr, cfg['colors'], .05, 1.2, foot, pitch=-15)
    assert bands['cap']
    assert bands['cap_point'] is not None
    assert bands['rgb_debug']['stage'] == 'accepted'
    assert abs(bands['rgb_debug']['distance'] - 0.22) < 1e-8


def test_visible_candidate_survives_full_face_rejection_without_becoming_a_pose():
    from cube.cap_pose import color_seam_pixels
    intr, plane, foot, color, depth = scene(-15)
    yy, xx = np.indices(depth.shape)
    xyz = _at_tower_face(np.column_stack([xx.ravel(), yy.ravel()]), intr, plane, foot)
    height = plane.signed_distance(xyz).reshape(depth.shape)
    # Transition stripe separates the strong colors; each residual color fails the full-face test.
    color[np.abs(height - .8) < .008] = 40
    bands = coarse_bands(color, depth, plane, intr, load_config('config/default.yaml')['colors'],
                         .05, 1.2, foot, pitch=-15)
    assert bands['rgb_debug']['face'] == 0
    assert bands['cap'] and bands['cap_rgb_candidate']
    assert bands['cap_point'] is None
    # The visibility-only relaxation does not relax the precise seam extractor.
    assert len(color_seam_pixels(color)) == 0


def test_candidate_records_visible_border_without_completing_the_missing_cube():
    intr, plane, foot, color, depth = scene(-15, center=20)
    color[:, 20:] = 40
    bands = coarse_bands(color, depth, plane, intr, load_config('config/default.yaml')['colors'],
                         .05, 1.2, foot, pitch=-15)
    assert bands['cap_rgb_candidate'] and bands['cap_partial']
    assert 'left' in bands['rgb_debug']['clipped_edges']
    assert bands['cap_point'] is None


def test_unmeasured_candidate_message_does_not_claim_cached_pose_is_current():
    from view_tower import describe
    search = StackSearch(kind='two_cap', settled=True)
    lines = describe(search, {'cap': True, 'cap_rgb_candidate': True, 'cap_partial': True},
                     -15, cap_world=np.array([0., .4, .8]), cap_yaw=0.,
                     cap_reason='short', cap_unmeasured=True)
    text = '\n'.join(lines)
    assert '局部入画' in text and '目标可见，本帧无法精测' in text
    assert '塔顶 相对小车' not in text
    assert '浅蓝轮廓仅表示可见候选' in text
