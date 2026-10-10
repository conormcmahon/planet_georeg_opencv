"""SIFT keypoint detection and FLANN-based matching, with support for a
geometrically-guided second pass.

Masking/blur conventions here are built around one real concern: nodata and
cloud-mask holes create false high-contrast edges that SIFT tends to lock
onto as if they were real texture. `masked_gaussian_blur` and
`erode_valid_mask` exist specifically to keep keypoints away from that.
"""

from __future__ import annotations

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
    mask_f = valid_mask.astype(np.float32)
    data_f = img_u8.astype(np.float32) * mask_f
    blurred_data = cv2.GaussianBlur(data_f, (ksize, ksize), 0)
    blurred_mask = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    out = np.divide(
        blurred_data, blurred_mask,
        out=np.zeros_like(blurred_data), where=blurred_mask > 1e-6
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


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
    """Convert SIFT keypoints (in detection-image pixels) to true clip-pixel
    coordinates."""
    if len(kp_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.float32([kp.pt for kp in kp_list]) / sift_scale


def detect_channel_keypoints(
    norm_arr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    blur_kernel: int,
    mask_margin_px: float,
    max_sift_dimension: int,
    sift,
):
    """Detect SIFT keypoints/descriptors on one normalized ND-index channel.

    Applies the masked blur and eroded detection mask, caps the working
    resolution for SIFT, and returns results already rescaled to true
    (un-capped) clip-pixel coordinates.

    Returns (keypoints, descriptors, sift_scale). `descriptors` is None (and
    `keypoints` empty) if detection found nothing.
    """
    from .nd_index import to_uint8

    img_u8 = masked_gaussian_blur(to_uint8(norm_arr), valid_mask, blur_kernel)
    mask = erode_valid_mask(valid_mask, mask_margin_px)

    img_u8_det, scale = cap_for_sift(img_u8, max_sift_dimension)
    mask_det, _ = cap_for_sift(mask, max_sift_dimension, interpolation=cv2.INTER_NEAREST)

    kp, desc = sift.detectAndCompute(img_u8_det, mask_det)
    return kp, desc, scale


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

    Returns (moving_pts, fixed_pts): lists of matched (x, y) pairs in true
    clip-pixel coordinates.
    """
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
