"""Read-only, same-frame visual diagnosis; never feeds results into detection."""
from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np
from cube.cap_pose import _color_masks
from cube.stack_look import _cap_faces
from cube.color_boundary import boundary_pixels

TITLE = '识别分步诊断'


def build_panel(color, depth, intr, zmin, zmax, bands, result_image, write, metadata):
    red, blue = _color_masks(color)
    union = cv2.bitwise_or(red, blue)
    face = _cap_faces(union, depth, intr, zmin, zmax,
                      red=red, blue=blue, color_bgr=color)
    kept_red = cv2.bitwise_and(red, face)
    kept_blue = cv2.bitwise_and(blue, face)
    seam = boundary_pixels(kept_red, kept_blue, color)
    raw_seam = boundary_pixels(red, blue, color)
    counts = dict(red=int(np.count_nonzero(red)), blue=int(np.count_nonzero(blue)),
                  face_red=int(np.count_nonzero(kept_red)),
                  face_blue=int(np.count_nonzero(kept_blue)),
                  raw_seam=len(raw_seam), face_seam=len(seam),
                  candidate_pixels=0 if bands.get('cap_uv') is None else len(bands['cap_uv']))
    hsv_image = np.zeros_like(color)
    hsv_image[red > 0] = (0, 0, 255)
    hsv_image[blue > 0] = (255, 0, 0)
    face_image = np.zeros_like(color)
    face_image[kept_red > 0] = (0, 0, 255)
    face_image[kept_blue > 0] = (255, 0, 0)
    for x, y in seam:
        cv2.circle(face_image, (int(x), int(y)), 2, (255, 255, 255), -1)
    canvas = np.zeros((930, 1280, 3), np.uint8)
    source = metadata['draw_source']
    source = {'none':'无框', 'measured_latest':'本帧测量', 'historical_latest':'历史测量',
              'outline':'轮廓推算'}.get(source, source)
    header = [f"帧 {metadata['frame']}  测试 {metadata['elapsed_s']:.1f} 秒  框来源：{source}  路径：{metadata['via'] or '无'}",
              f"测量提示：{metadata['reason'] or '无'}  重置：{metadata['reset_reason'] or '无'}  光线：{metadata['light']}",
              'D 隐藏/打开此窗；F 冻结/恢复诊断图（识别继续）；S 保存当前诊断图和数据到 /tmp',
              '复算图只解释颜色和面筛选；完整检测还受高度、位置、搜索状态及投影检查约束。']
    header.append("定位状态：" + metadata.get("anchor_status", "未提供") + "；橙线为暂未接纳的下层候选")
    write(canvas, header)
    tiles = [(color, '① 原始彩色：观察遮挡、出画、反光'),
             (hsv_image, f"② HSV：红 {counts['red']}  蓝 {counts['blue']}  分界点 {len(raw_seam)}"),
             (face_image, f"③ 精测面筛选同帧复算：红 {counts['face_red']} 蓝 {counts['face_blue']} 分界 {len(seam)}"),
             (result_image, f"④ 实际显示结果：候选像素 {counts['candidate_pixels']}（轮廓不等于精测成功）")]
    for i, (im, label) in enumerate(tiles):
        x, y = (i % 2) * 640, 120 + (i // 2) * 405
        tile = np.zeros((405, 640, 3), np.uint8)
        write(tile, [label])
        scale = min(640 / im.shape[1], 360 / im.shape[0])
        w, h = int(im.shape[1]*scale), int(im.shape[0]*scale)
        tile[45:45+h, (640-w)//2:(640-w)//2+w] = cv2.resize(im, (w,h))
        canvas[y:y+405,x:x+640] = tile
    return canvas, dict(metadata, counts=counts)


class DebugWindow:
    def __init__(self):
        self.visible = True
        self.frozen = False
        self.image = None
        self.metadata = None
        self.opened = False

    def update(self, *args):
        if not self.visible:
            return
        if not self.frozen or self.image is None:
            self.image, self.metadata = build_panel(*args)
        if not self.opened:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, 1152, 837)
            self.opened = True
        shown = self.image.copy()
        if self.frozen:
            cv2.putText(shown, 'FROZEN - F to resume', (800, 110), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        cv2.imshow(TITLE, shown)

    def key(self, key):
        if key in (ord('d'), ord('D')):
            self.visible = not self.visible
            if self.opened:
                cv2.destroyWindow(TITLE)
                self.opened = False
        elif key in (ord('f'), ord('F')):
            self.frozen = not self.frozen
        elif key in (ord('s'), ord('S')) and self.image is not None:
            folder = Path('/tmp') / ('tower-debug-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            folder.mkdir()
            cv2.imwrite(str(folder / 'panel.png'), self.image)
            (folder / 'state.json').write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2))
            print(f'诊断窗口快照已保存：{folder}', flush=True)
