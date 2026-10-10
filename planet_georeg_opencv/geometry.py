"""Geographic overlap, clip-pixel offsets, keypoint thinning, and the
RANSAC-affine fit used to seed (or, on its own, directly estimate) a
geometric transform between two rasters.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_overlap(image1, image2):
    """Return (left, bottom, right, top) geographic bounding box of the
    overlap between two images, or None if they do not overlap.

    Both images must be in the same CRS before calling this function.
    """
    b1 = image1.rio.bounds()  # (left, bottom, right, top)
    b2 = image2.rio.bounds()
    left = max(b1[0], b2[0])
    bottom = max(b1[1], b2[1])
    right = min(b1[2], b2[2])
    top = min(b1[3], b2[3])
    if left >= right or bottom >= top:
        return None
    return left, bottom, right, top


def get_clip_offset(full_image, clip_image) -> tuple[int, int]:
    """Return the (col, row) pixel offset of clip_image's origin within
    full_image.

    Uses nearest-neighbor lookup on the coordinate arrays so it is robust to
    floating-point rounding from rio.clip_box.
    """
    col = int(np.argmin(np.abs(full_image.x.values - clip_image.x.values[0])))
    row = int(np.argmin(np.abs(full_image.y.values - clip_image.y.values[0])))
    return col, row


def spatially_thin_keypoints(fixed_pts, moving_pts, max_control_points, clip_w, clip_h, rng):
    """Subsample matched keypoint pairs to at most `max_control_points` using
    a regular spatial grid over the fixed-image clip extent.

    The image overlap is divided into a grid of cells whose size is chosen
    so that the expected number of cells approximately equals
    `max_control_points`. Within each occupied cell exactly one keypoint is
    kept (chosen at random). Cells with only one occupant are always
    retained; cells with many occupants (dense clusters) contribute only one
    representative.

    `fixed_pts` should be in clip-space pixel coordinates used to assign
    cells. `moving_pts` can be in any coordinate space -- they are
    subsampled in lock-step.
    """
    n = len(fixed_pts)
    if max_control_points is None or n <= max_control_points:
        return fixed_pts, moving_pts

    # Cell size chosen so that clip_area / cell_area ~= max_control_points.
    cell_area = (clip_w * clip_h) / max_control_points
    cell_size = max(1, int(np.sqrt(cell_area)))

    cell_map: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cell_col = int(fixed_pts[i, 0] / cell_size)
        cell_row = int(fixed_pts[i, 1] / cell_size)
        key = (cell_row, cell_col)
        cell_map.setdefault(key, []).append(i)

    kept = [int(rng.choice(indices)) for indices in cell_map.values()]
    kept = np.array(sorted(kept))
    return fixed_pts[kept], moving_pts[kept]


def estimate_affine_ransac(fixed_pts: np.ndarray, moving_pts: np.ndarray, ransac_reproj_threshold: float):
    """Fit a 2D affine transform fixed -> moving via RANSAC.

    `fixed_pts`/`moving_pts` are (N, 2) arrays of matched clip-pixel
    coordinates. Returns (M, inlier_mask), where M is the 2x3 affine matrix
    (None if RANSAC failed to produce a model) and inlier_mask is the (N, 1)
    uint8 mask cv2 returns (None if M is None).
    """
    fixed_arr = np.float32(fixed_pts).reshape(-1, 1, 2)
    moving_arr = np.float32(moving_pts).reshape(-1, 1, 2)
    return cv2.estimateAffine2D(
        fixed_arr, moving_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
