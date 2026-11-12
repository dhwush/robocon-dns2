"""Check cube scale, visible physical edges and reliable interior depth before use."""
import cv2
import numpy as np
from geom.transform import deproject_pixels, project_points


def _visible_edge_errors(observed, projected, image, axis):
    """Mask cutoffs alone are not physical edges; clipped sides are excluded."""
    if axis == 0:
        observed, projected = observed.T, projected.T
        image = image.transpose(1, 0, 2)
    height, width = observed.shape
    ys, xs = np.nonzero(observed)
    if len(xs) < 20:
        return []
    span = int(xs.max() - xs.min() + 1)
    search = int(np.clip(round(span * .12), 4, 60))
    tolerance = max(4., span * .08)
    gradient = np.zeros((height, width), np.float32)
    gradient[:, 1:-1] = np.linalg.norm(image[:, 2:] - image[:, :-2], axis=2)
    rows = np.unique(np.linspace(np.percentile(ys, 10), np.percentile(ys, 90), 24).astype(int))
    errors = [[], []]
    for y in rows:
        actual = np.flatnonzero(observed[y])
        predicted = np.flatnonzero(projected[y])
        if len(actual) < max(4, span * .25) or not len(predicted):
            continue
        for side, x in enumerate((int(actual[0]), int(actual[-1]))):
            # Clipping is established by observation, not the predicted box.
            if x <= 2 or x >= width - 3:
                continue
            lo, hi = max(2, x - search), min(width - 2, x + search + 1)
            strength = gradient[y, lo:hi]
            if not len(strength) or strength.max() < 25.:
                continue
            physical = lo + int(np.argmax(strength))
            if (side == 0 and physical > x + 3) or (side == 1 and physical < x - 3):
                continue
            predicted_x = int(predicted[0] if side == 0 else predicted[-1])
            errors[side].append(abs(predicted_x - physical) / tolerance)
    return [{"axis": axis, "side": side, "samples": len(values),
             "median_normalized_error": float(np.median(values)),
             "outside_fraction": float(np.mean(np.asarray(values) > 1.))}
            for side, values in enumerate(errors) if len(values) >= 6]


def _depth_check(pose, selected, projected, depth, intr, axes, zmin, zmax):
    """Compare interior z to the first ray/cube intersection, not just one face."""
    if depth is None or getattr(pose, "via", "depth") in ("close", "side"):
        return {"status": "unavailable_or_unreliable"}
    interior = cv2.erode(selected.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(interior & (projected > 0))
    if len(xs) < 30:
        return {"status": "insufficient_samples"}
    step = max(1, int(np.ceil(len(xs) / 1200)))
    xs, ys = xs[::step], ys[::step]
    z = np.asarray(depth)[ys, xs]
    good = np.isfinite(z) & (z >= zmin) & (z <= zmax)
    if good.sum() < 30 or good.mean() < .30:
        return {"status": "insufficient_depth"}
    z = z[good]
    rays = deproject_pixels(np.column_stack([xs[good], ys[good]]), np.ones(len(z)), intr) @ axes
    origin = -np.asarray(pose.center, dtype=float) @ axes
    parallel = np.abs(rays) < 1e-9
    safe = np.where(parallel, 1., rays)
    first, last = (-.10 - origin) / safe, (.10 - origin) / safe
    near = np.where(parallel, -np.inf, np.minimum(first, last))
    far = np.where(parallel, np.inf, np.maximum(first, last))
    outside = np.any(parallel & (np.abs(origin) > .10), axis=1)
    entry, exit = np.max(near, axis=1), np.min(far, axis=1)
    valid = ~outside & (entry > .05) & (entry <= exit)
    if valid.sum() < 30:
        return {"status": "no_supported_surface"}
    error = np.abs(z[valid] - entry[valid])
    tolerance = max(.02, .035 * float(np.median(z[valid])))
    fraction = float(np.mean(error <= tolerance))
    return {"status": "consistent" if fraction >= .65 else "mismatch",
            "samples": int(valid.sum()), "inlier_fraction": fraction,
            "median_error_m": float(np.median(error)), "tolerance_m": tolerance}


def pose_matches_color(pose, color_bgr, intr, plane, color_mask, depth=None,
                       zmin=.05, zmax=1.2, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.clear()
    diagnostics["reason"] = "invalid_geometry"
    if pose is None or pose.corners is None:
        return False
    corners = np.asarray(pose.corners, dtype=float)
    if corners.shape != (4, 3) or not np.isfinite(corners).all():
        return False
    up = np.asarray(plane.normal, dtype=float)
    norm = np.linalg.norm(up)
    if not np.isfinite(norm) or norm < 1e-8:
        return False
    up = up / norm
    edge, side = corners[1] - corners[0], corners[3] - corners[0]
    lengths = np.array([np.linalg.norm(edge), np.linalg.norm(side)])
    if np.any(np.abs(lengths - .20) > .02):
        return False
    axes = np.column_stack([edge / lengths[0], side / lengths[1], up])
    if not np.allclose(axes.T @ axes, np.eye(3), atol=.05):
        return False
    if not np.allclose(corners[2], corners[1] + corners[3] - corners[0], atol=.005):
        return False
    center = np.mean(corners, axis=0) + up * .10
    if not np.allclose(center, pose.center, atol=.005):
        return False
    wire = np.vstack([corners, corners + up * .20])
    if not np.isfinite(wire).all() or np.any(wire[:, 2] <= .05):
        return False
    pixels = project_points(wire, intr)
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 100000:
        return False
    projected = np.zeros(color_bgr.shape[:2], np.uint8)
    cv2.fillConvexPoly(projected, cv2.convexHull(np.rint(pixels).astype(np.int32)), 1)
    area = int(projected.sum())
    if area < 20:
        return False
    span = float(np.ptp(pixels, axis=0).min())
    margin = int(np.clip(round(span * .025), 2, 10))
    expanded = cv2.dilate(projected, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask.astype(np.uint8))
    overlaps = np.bincount(labels[expanded], minlength=count)
    largest = int(overlaps[1:].max()) if count > 1 else 0
    selected = np.zeros(projected.shape, bool)
    supported = observed = 0
    for index in range(1, count):
        size, overlap = int(stats[index, cv2.CC_STAT_AREA]), int(overlaps[index])
        if size < 20 or overlap < max(10, int(size * .10), int(largest * .08)):
            continue
        selected |= labels == index
        supported += overlap
        observed += size
    diagnostics.update(reason="color_coverage", observed=observed, projected_area=area)
    if observed < 20 or supported / observed < .75 or supported / area < .20:
        return False
    image = cv2.GaussianBlur(color_bgr.astype(np.float32), (3, 3), 0)
    edges = _visible_edge_errors(selected, projected, image, 0) + _visible_edge_errors(selected, projected, image, 1)
    diagnostics["edges"] = edges
    if any(e["median_normalized_error"] > 1. and e["outside_fraction"] >= .65 for e in edges):
        diagnostics["reason"] = "visible_edge_mismatch"
        return False
    checked = _depth_check(pose, selected, projected, depth, intr, axes, zmin, zmax)
    diagnostics["depth"] = checked
    if checked["status"] in ("mismatch", "no_supported_surface"):
        diagnostics["reason"] = "depth_mismatch"
        return False
    if checked["status"] != "consistent" and len(edges) < 2:
        diagnostics["reason"] = "insufficient_validation"
        return False
    diagnostics["reason"] = "accepted"
    return True
