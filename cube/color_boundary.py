"""Associate vertically adjacent color regions across a bounded transition band."""
from __future__ import annotations

import cv2
import numpy as np


def _parts(mask):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    minimum = max(10, int(mask.size * 0.00004))
    ids = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    return [(labels == i, stats[i], centers[i]) for i in ids[:8]
            if stats[i, cv2.CC_STAT_AREA] >= minimum]


def adjacent_color_pairs(red, blue, color_bgr=None, *, gap_fraction=.08, max_gap=32):
    """Return (boundary pixels, original pair mask); never fill the color masks.

    Require horizontal overlap, vertical ordering and a gap bounded by target width.
    The boundary is one pixel per supporting column. In a transition band use the
    strongest local color change, rather than the arbitrary saturation cutoff edge.
    """
    pairs = []
    image = None if color_bgr is None else cv2.GaussianBlur(color_bgr.astype(np.float32), (3, 3), 0)
    for rr, rs, rc in _parts(red):
        for bb, bs, bc in _parts(blue):
            width = min(int(rs[cv2.CC_STAT_WIDTH]), int(bs[cv2.CC_STAT_WIDTH]))
            left = max(int(rs[cv2.CC_STAT_LEFT]), int(bs[cv2.CC_STAT_LEFT]))
            right = min(int(rs[cv2.CC_STAT_LEFT] + rs[cv2.CC_STAT_WIDTH]),
                        int(bs[cv2.CC_STAT_LEFT] + bs[cv2.CC_STAT_WIDTH]))
            if right - left < max(8, int(width * .35)) or abs(rc[1] - bc[1]) < 3:
                continue
            top, bottom = (rr, bb) if rc[1] < bc[1] else (bb, rr)
            reach = int(np.clip(round(width * gap_fraction), 4, max_gap))
            seam = []
            for x in range(left, right):
                upper = np.flatnonzero(top[:, x])
                lower = np.flatnonzero(bottom[:, x])
                if len(upper) < 2 or len(lower) < 2:
                    continue
                a, b = int(upper[-1]), int(lower[0])
                if not 0 < b - a <= reach + 1:
                    continue
                y = (a + b) // 2
                if image is not None and b - a > 2:
                    ys = np.arange(a, b + 1)
                    before = image[np.maximum(ys - 1, 0), x]
                    after = image[np.minimum(ys + 1, image.shape[0] - 1), x]
                    change = np.linalg.norm(after - before, axis=1)
                    y = int(ys[int(np.argmax(change))])
                seam.append((x, y))
            if len(seam) < max(8, int((right - left) * .35)):
                continue
            pairs.append((np.asarray(seam, dtype=int), rr | bb))
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def boundary_pixels(red, blue, color_bgr=None):
    # Preserve the established measured seam when the masks already touch.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    touch = (cv2.dilate(red, kernel) & blue) | (cv2.dilate(blue, kernel) & red)
    ys, xs = np.nonzero(touch)
    if len(xs) >= 16:
        return np.column_stack([xs, ys]).astype(int)
    pairs = adjacent_color_pairs(red, blue, color_bgr)
    return pairs[0][0] if pairs else np.zeros((0, 2), dtype=int)
