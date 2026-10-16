"""SIFT keypoint detection and FLANN-based matching, with support for a
geometrically-guided second pass.

Masking/blur conventions here are built around one real concern: nodata and
cloud-mask holes create false high-contrast edges that SIFT tends to lock
onto as if they were real texture. `masked_gaussian_blur` and
`erode_valid_mask` exist specifically to keep keypoints away from that.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
from scipy.spatial import cKDTree


def cap_for_sift(img_u8: np.ndarray, max_dim: int, interpolation=cv2.INTER_AREA):
    """Downsample img_u8 (if needed) so its larger side is at most max_dim.
    Works on both single-channel (H, W) and multi-channel (H, W, C) images.

    Returns (image_for_detection, scale), where scale = detection_size /
    original_size. Detected keypoint coordinates must be divided by `scale`
    to convert them back to the original image's pixel space.

    Pass interpolation=cv2.INTER_NEAREST when resizing a binary mask
    alongside its image (via a separate call with the same max_dim, which
    yields the same scale since it depends only on shape) to keep it
    strictly 0/255 rather than picking up intermediate averaged values.
    """
    h, w = img_u8.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img_u8, 1.0
    small = cv2.resize(
        img_u8, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=interpolation
    )
    return small, scale


def resolution_scaled_kernel_size(base_size: int, base_resolution: float, image_resolution: float) -> int:
    """Scale a Gaussian blur kernel size so it covers approximately the same
    real-world ground footprint at a different pixel resolution.

    E.g. base_size=3 at base_resolution=3.0 (m) gives 15 at
    image_resolution=0.6 (m). Rounds to the nearest odd integer (required by
    cv2.GaussianBlur) and never returns less than 1.
    """
    k = int(round(base_size * (base_resolution / image_resolution)))
    k = max(1, k)
    if k % 2 == 0:
        k += 1
    return k


def masked_gaussian_blur(img_u8: np.ndarray, valid_mask: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian-blur img_u8 using normalized convolution -- blur(data*mask) /
    blur(mask) -- so that invalid (valid_mask == False) pixels never bleed
    their arbitrary zero-filled value into neighboring valid pixels.

    A plain blur of a NaN-to-zero-filled image creates a false, artificially
    dark edge around every nodata/cloud-mask hole and scene boundary, which
    SIFT tends to lock onto as if it were real texture. Pixels with no valid
    support in their blur footprint fall back to 0 -- such pixels are
    excluded from detection anyway via the eroded mask passed to SIFT
    separately (see erode_valid_mask).
    """
    # In-place arithmetic throughout: at most four full-size float32 arrays
    # alive at once (the result is identical to the straightforward form).
    mask_f = valid_mask.astype(np.float32)
    data_f = img_u8.astype(np.float32)
    data_f *= mask_f
    blurred_data = cv2.GaussianBlur(data_f, (ksize, ksize), 0)
    del data_f
    blurred_mask = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    del mask_f
    supported = blurred_mask > 1e-6
    np.divide(blurred_data, blurred_mask, out=blurred_data, where=supported)
    del blurred_mask
    blurred_data[~supported] = 0
    np.round(blurred_data, out=blurred_data)
    np.clip(blurred_data, 0, 255, out=blurred_data)
    return blurred_data.astype(np.uint8)


def erode_valid_mask(valid_mask: np.ndarray, margin_px: float) -> np.ndarray:
    """Erode a boolean valid-data mask so only pixels at least margin_px away
    (in the same pixel space) from any nodata/cloud-masked region remain.

    Used to build the mask passed to SIFT's detectAndCompute, keeping
    keypoints (and their descriptor support window) away from holes and
    scene edges. Returns a uint8 0/255 mask suitable for cv2's mask
    parameter.
    """
    r = max(0, int(round(margin_px)))
    if r == 0:
        return valid_mask.astype(np.uint8) * 255
    kernel = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.erode(valid_mask.astype(np.uint8) * 255, kernel)


def keypoints_to_true_pixels(kp_list, sift_scale: float) -> np.ndarray:
    """Convert SIFT keypoints (in detection-image pixels; cv2.KeyPoint
    objects or an (N, 2) array of their .pt) to true clip-pixel
    coordinates."""
    if len(kp_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(kp_list, np.ndarray):
        return kp_list.astype(np.float32) / sift_scale
    return np.float32([kp.pt for kp in kp_list]) / sift_scale


# OpenCV's SIFT reports every keypoint a quarter of a detection pixel too
# far along both axes: it detects on a 2x enlargement of the image and
# halves the coordinates without allowing for the enlargement's half-pixel
# shift. Measured (OpenCV 5.0.0) by matching an image against 0.4x, 0.5x and
# 0.7x reductions of itself: 0.248 to 0.251 detection pixels each time.
SIFT_KEYPOINT_OFFSET_PX = 0.25


def detection_to_true_pixels(pts_det: np.ndarray, det_shape, true_shape) -> np.ndarray:
    """SIFT keypoint coordinates in a resized detection image of shape
    `det_shape` (rows, cols) as coordinates in the image it was resized
    from (`true_shape`), exactly: SIFT's own offset removed (see
    SIFT_KEYPOINT_OFFSET_PX), pixel centres mapped to pixel centres, and
    each axis given its own size ratio.

    The simpler `pt / scale` used elsewhere in this module ignores all
    three, which puts a keypoint off by (0.25 / scale - 0.5) pixels, plus
    up to half a detection pixel across the shorter axis (whose resized
    length is rounded). Those errors nearly cancel when two images are
    detected at nearly the same scale, but not when one side's keypoints
    come from a different scale -- as with keypoints detected beforehand
    on a reference image.
    """
    pts = np.asarray(pts_det, dtype=np.float64).reshape(-1, 2)
    ratio_x = true_shape[1] / det_shape[1]
    ratio_y = true_shape[0] / det_shape[0]
    centre = 0.5 - SIFT_KEYPOINT_OFFSET_PX
    return np.column_stack([(pts[:, 0] + centre) * ratio_x - 0.5, (pts[:, 1] + centre) * ratio_y - 0.5])


def prepare_channel_image(
    norm_arr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    blur_kernel: int,
    mask_margin_px: float,
    max_sift_dimension: int,
):
    """The image and mask SIFT runs on for one normalized ND-index channel:
    masked blur, eroded detection mask, both capped to max_sift_dimension.

    Returns (image_u8, mask_u8, sift_scale) -- small (at most
    max_sift_dimension per side) whatever the input size, so callers can
    prepare every channel first and free the full-size arrays before
    running SIFT (whose own working memory is large).
    """
    from .nd_index import to_uint8

    img_u8 = masked_gaussian_blur(to_uint8(norm_arr), valid_mask, blur_kernel)
    mask = erode_valid_mask(valid_mask, mask_margin_px)
    img_u8_det, scale = cap_for_sift(img_u8, max_sift_dimension)
    mask_det, _ = cap_for_sift(mask, max_sift_dimension, interpolation=cv2.INTER_NEAREST)
    return img_u8_det, mask_det, scale


def detect_channel_keypoints(
    norm_arr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    blur_kernel: int,
    mask_margin_px: float,
    max_sift_dimension: int,
    sift,
):
    """Detect SIFT keypoints/descriptors on one normalized ND-index channel
    (prepare_channel_image, then SIFT).

    Returns (keypoints, descriptors, sift_scale); keypoints are in the
    capped detection image's pixels -- divide by sift_scale for true clip
    pixels. `descriptors` is None (and `keypoints` empty) if detection
    found nothing.
    """
    img, mask, scale = prepare_channel_image(
        norm_arr, valid_mask, blur_kernel=blur_kernel, mask_margin_px=mask_margin_px,
        max_sift_dimension=max_sift_dimension,
    )
    kp, desc = sift.detectAndCompute(img, mask)
    return kp, desc, scale


def detect_keypoints(sift, img: np.ndarray, mask: np.ndarray, *, tile_size: int | None = None,
                     tile_overlap: int = 128, lock=None):
    """SIFT keypoints and descriptors of `img` within `mask`, as ((N, 2)
    float64 array of each keypoint's .pt, (N, 128) float32 descriptors) --
    descriptors are None if nothing was found.

    With `tile_size`, detection runs on tiles of that many pixels per side
    instead of on the whole image at once, each tile extended by
    `tile_overlap` pixels on every side so keypoints near its edge still
    have their neighbourhood; only keypoints inside a tile's own area are
    kept, so each is kept once. SIFT's working memory (about 0.25 GB per
    megapixel) then follows the tile size rather than the image size. The
    keypoints found are nearly, not exactly, those of one whole-image
    call: SIFT's coarsest scales and keypoints larger than the overlap
    differ near tile edges.

    `lock` (optional context manager) is held around each SIFT call.
    """
    def run(image, image_mask):
        with lock if lock is not None else contextlib.nullcontext():
            keypoints, descriptors = sift.detectAndCompute(image, image_mask)
        return np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2), descriptors

    height, width = img.shape[:2]
    if tile_size is None or (height <= tile_size + tile_overlap and width <= tile_size + tile_overlap):
        return run(img, mask)

    all_pts, all_desc = [], []
    for row0 in range(0, height, tile_size):
        for col0 in range(0, width, tile_size):
            row1, col1 = min(row0 + tile_size, height), min(col0 + tile_size, width)
            read_row0, read_col0 = max(0, row0 - tile_overlap), max(0, col0 - tile_overlap)
            rows = slice(read_row0, min(height, row1 + tile_overlap))
            cols = slice(read_col0, min(width, col1 + tile_overlap))
            tile_mask = None if mask is None else mask[rows, cols]
            if tile_mask is not None and not tile_mask.any():
                continue
            pts, desc = run(np.ascontiguousarray(img[rows, cols]),
                            None if tile_mask is None else np.ascontiguousarray(tile_mask))
            if desc is None or len(pts) == 0:
                continue
            pts[:, 0] += read_col0
            pts[:, 1] += read_row0
            in_core = (pts[:, 0] >= col0) & (pts[:, 0] < col1) & (pts[:, 1] >= row0) & (pts[:, 1] < row1)
            all_pts.append(pts[in_core])
            all_desc.append(desc[in_core])
    if not all_pts:
        return np.zeros((0, 2), dtype=np.float64), None
    return np.concatenate(all_pts), np.concatenate(all_desc)


def guided_match_per_descriptor_radius(
    moving_kp, moving_desc, fixed_kp, fixed_desc, moving_sift_scale, fixed_sift_scale,
    M_inv, distance_threshold, lowe_ratio_threshold,
):
    """Geometrically-guided re-matching: build a KD-tree over each moving
    keypoint's step-1-predicted fixed-image position (M_inv applied to it),
    then for each fixed descriptor individually, query the tree for moving
    keypoints predicted to fall within distance_threshold pixels and compare
    descriptor distances only within that geometric subset. Lowe's ratio
    test is applied to the two closest descriptors in the subset; a subset
    of size 1 is accepted without a ratio test.

    Keypoints may be cv2.KeyPoint sequences or (N, 2) arrays of their .pt;
    descriptors may be stored compactly as uint8 (they are compared as
    float32).

    Returns (moving_pts, fixed_pts): lists of matched (x, y) pairs in true
    clip-pixel coordinates.
    """
    moving_desc = np.asarray(moving_desc, dtype=np.float32)
    fixed_desc = np.asarray(fixed_desc, dtype=np.float32)
    moving_pts_true = keypoints_to_true_pixels(moving_kp, moving_sift_scale)
    fixed_pts_true = keypoints_to_true_pixels(fixed_kp, fixed_sift_scale)
    pred_fixed_pts = cv2.transform(moving_pts_true.reshape(-1, 1, 2), M_inv).reshape(-1, 2)

    tree = cKDTree(pred_fixed_pts)

    out_moving_pts, out_fixed_pts = [], []
    for fixed_i in range(len(fixed_kp)):
        fixed_pt = fixed_pts_true[fixed_i]
        candidate_idx = tree.query_ball_point(fixed_pt, r=distance_threshold)
        if not candidate_idx:
            continue
        cand_desc = moving_desc[candidate_idx]
        dists = np.linalg.norm(cand_desc - fixed_desc[fixed_i][None, :], axis=1)
        order = np.argsort(dists)
        if len(order) == 1:
            best_idx = candidate_idx[order[0]]
        else:
            if dists[order[0]] >= lowe_ratio_threshold * dists[order[1]]:
                continue
            best_idx = candidate_idx[order[0]]
        out_moving_pts.append((float(moving_pts_true[best_idx][0]), float(moving_pts_true[best_idx][1])))
        out_fixed_pts.append((float(fixed_pt[0]), float(fixed_pt[1])))
    return out_moving_pts, out_fixed_pts
