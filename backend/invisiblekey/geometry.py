"""Homography-based mapping between camera image space and the desk surface.

Image coordinates are MediaPipe-normalized (x right, y down, 0..1).
Surface coordinates are normalized to the calibrated quad: (0,0) = top-left
(far edge of the desk area), (1,1) = bottom-right (edge closest to the user).
"""
from __future__ import annotations

import numpy as np

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
MIN_QUAD_AREA = 1e-3


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform for >= 4 point correspondences."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("need >= 4 matching (x, y) points")
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ h.T
    return homog[:, :2] / homog[:, 2:3]


def validate_quad(corners: np.ndarray) -> None:
    """Corners must be TL, TR, BR, BL forming a convex, non-degenerate quad."""
    c = np.asarray(corners, float)
    if c.shape != (4, 2) or not np.all(np.isfinite(c)):
        raise ValueError("expected 4 finite (x, y) corners")
    cross = []
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]))
    if not (all(z > 0 for z in cross) or all(z < 0 for z in cross)):
        raise ValueError("corners must form a convex quad in TL, TR, BR, BL order")
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < MIN_QUAD_AREA:
        raise ValueError("calibration area too small")


class CoordinateTransformer:
    def __init__(self, image_corners: np.ndarray | list) -> None:
        corners = np.asarray(image_corners, float)
        validate_quad(corners)
        self.image_corners = corners
        self.h = compute_homography(corners, UNIT_SQUARE)
        self.h_inv = np.linalg.inv(self.h)

    def to_surface(self, pts) -> np.ndarray:
        return apply_homography(self.h, pts)

    def to_image(self, pts) -> np.ndarray:
        return apply_homography(self.h_inv, pts)
