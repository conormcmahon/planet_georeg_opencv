"""Top-level registration orchestrators.

register_local  - two-step (coarse + guided) SIFT matching, RANSAC, and
                   either a thin-plate-spline local warp or a single
                   affine warp (warp_model="tps"/"affine") that corrects
                   `moving` onto ITS OWN pixel grid (self-referential: only
                   pixel values move, the grid/CRS/extent stay exactly as
                   given). Use this when `moving` and `fixed` are expected
                   to already share (or nearly share) a grid and only need
                   local/sub-pixel correction.

register_to_grid - one-step SIFT matching, RANSAC, and a single global
                   affine warp that resamples `moving` onto `fixed`'s own
                   full pixel grid. Use this to co-register two rasters
                   with different grids/extents onto one common grid.

Both accept `moving`/`fixed` as either a file path or an already-loaded
rioxarray-backed xr.DataArray, and both return a RegistrationResult without
writing anything to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

from . import affine_warp, band_metrics, geometry, nd_index, sift_matching, tps_warp

_RASTERIO_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}
_CV2_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "average": cv2.INTER_AREA,
}

WarpModel = Literal["tps", "affine"]

Status = Literal[
    "ok",
    "band_count_mismatch",
    "no_overlap",
    "insufficient_pixels",
    "insufficient_matches_coarse",
    "ransac_failed_coarse",
    "insufficient_matches_guided",
    "ransac_failed_guided",
    "insufficient_inliers",
]


@dataclass
class RegistrationResult:
    status: Status
    registered: xr.DataArray | None
    metrics: dict = field(default_factory=dict)
    message: str = ""
    diagnostics: dict = field(default_factory=dict)


def _load_raster(image) -> xr.DataArray:
    if isinstance(image, str):
        return rxr.open_rasterio(image, masked=True).squeeze()
    return image


def _validate_band_map(band_map: list[int]) -> list[tuple[int, int]]:
    matched = nd_index.get_matched_band_pairs(band_map)
    if len(matched) < 2:
        raise ValueError(
            "band_map must map at least two moving bands to fixed bands "
            "(need at least two matched bands to compute a normalized-difference index)."
        )
    return matched


def register_local(
    moving,
    fixed,
    band_map: list[int],
    *,
    min_pixel_count: int = 1000,
    min_keypoints: int = 10,
    lowe_ratio_threshold: float = 1000,
    distance_threshold_pixels: float = 50,
    ransac_reproj_threshold: float = 2.0,
    keypoint_match_distance_threshold_m: float = 15.0,
    blur_kernel_base_size: int = 3,
    blur_kernel_reference_resolution: float = 3.0,
    mask_erosion_blur_multiple: float = 2,
    max_sift_dimension: int = 4000,
    warp_model: WarpModel = "tps",
    fixed_buffer_m: float = 0.0,
    tps_max_control_points: int | None = 2000,
    tps_coarse_grid_size: int = 300,
    resampling_method: str = "bilinear",
    seed: int = 0,
) -> RegistrationResult:
    """Register `moving` onto its own grid using a warp fit from keypoint
    correspondences with `fixed`. See module docstring.

    `band_map[i]` is the 0-indexed `fixed` band matching 0-indexed `moving`
    band `i`, or -1 if `moving` band `i` has no match in `fixed`.

    `warp_model` picks the correction applied (the output stays on
    `moving`'s own grid either way):
      - "tps":    a thin-plate-spline displacement field through the RANSAC
                  inlier control points -- corrects spatially varying
                  (non-rigid) error, but can bend areas far from any
                  control point when matches are sparse or clustered.
      - "affine": the single refined RANSAC affine transform from step 2,
                  applied uniformly -- only corrects global shift/rotation/
                  scale/shear, but is stable with few or clustered matches.

    `fixed_buffer_m` widens the `fixed` area searched for keypoints beyond
    the strict moving/fixed overlap by this distance on every side (in
    `fixed`'s CRS units -- metres for a projected CRS such as UTM), so
    features near `moving`'s edges can still find their true match in
    `fixed` when `moving` is misplaced by up to about that distance. 0
    restricts the search to the strict geographic overlap. Before/after
    band metrics are always computed over the strict overlap.
    """
    if warp_model not in ("tps", "affine"):
        raise ValueError(f"warp_model must be 'tps' or 'affine', got {warp_model!r}")
    matched_band_pairs = _validate_band_map(band_map)
    rasterio_resampling = _RASTERIO_RESAMPLING[resampling_method]
    cv2_interpolation = _CV2_INTERPOLATION[resampling_method]
    rng = np.random.default_rng(seed=seed)

    moving_image = _load_raster(moving)
    fixed_image = _load_raster(fixed)

    if moving_image.shape[0] != len(band_map):
        return RegistrationResult(
            "band_count_mismatch", None,
            message=f"band_map has {len(band_map)} entries but moving image has {moving_image.shape[0]} bands.",
        )

    # moving_native keeps moving's approximate native resolution; it is used
    # for feature matching, the TPS warp, and the final registered output.
    if moving_image.rio.crs != fixed_image.rio.crs:
        moving_native = moving_image.rio.reproject(fixed_image.rio.crs, resampling=rasterio_resampling)
    else:
        moving_native = moving_image
    moving_native_h, moving_native_w = moving_native.shape[1], moving_native.shape[2]
    moving_native_transform = moving_native.rio.transform()
    fixed_transform = fixed_image.rio.transform()

    overlap = geometry.find_overlap(moving_native, fixed_image)
    if overlap is None:
        return RegistrationResult("no_overlap", None, message="No geographic overlap between moving and fixed.")
    left, bottom, right, top = overlap

    moving_clip = moving_native.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    # fixed_overlap_clip (strict overlap) is used for band metrics;
    # fixed_clip (overlap + fixed_buffer_m) is what keypoints are searched in.
    fixed_overlap_clip = fixed_image.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    if fixed_buffer_m > 0:
        fixed_clip = fixed_image.rio.clip_box(
            minx=left - fixed_buffer_m, miny=bottom - fixed_buffer_m,
            maxx=right + fixed_buffer_m, maxy=top + fixed_buffer_m,
        )
    else:
        fixed_clip = fixed_overlap_clip

    moving_valid = int(np.sum(~np.isnan(moving_clip.values)))
    fixed_valid = int(np.sum(~np.isnan(fixed_overlap_clip.values)))
    if min(moving_valid, fixed_valid) < min_pixel_count:
        return RegistrationResult(
            "insufficient_pixels", None,
            message=f"Too few valid pixels in overlap (moving={moving_valid}, fixed={fixed_valid}).",
        )

    metric_keys = band_metrics.build_metric_keys(matched_band_pairs)
    metrics = {k: np.nan for k in metric_keys}

    metrics_before = band_metrics.compute_band_metrics_matched_resolution(
        moving_clip, fixed_overlap_clip, matched_band_pairs, rng
    )
    band_metrics.store_band_metrics(metrics, metrics_before, matched_band_pairs, "before")

    moving_band_arrays = nd_index.extract_band_arrays(moving_clip, [i for i, _ in matched_band_pairs])
    fixed_band_arrays_raw = nd_index.extract_band_arrays(fixed_clip, [f for _, f in matched_band_pairs])
    fixed_band_arrays = {m: fixed_band_arrays_raw[f] for m, f in matched_band_pairs}
    moving_nd = nd_index.compute_nd_indices(moving_band_arrays)
    fixed_nd = nd_index.compute_nd_indices(fixed_band_arrays)
    del moving_band_arrays, fixed_band_arrays_raw, fixed_band_arrays

    # Where a fixed-clip pixel lands in moving-clip pixels if both rasters'
    # georeferencing were correct (via world coordinates, so it holds even
    # when the two clips differ in resolution or extent, e.g. with a
    # fixed_buffer_m margin).
    fixed_clip_to_moving_clip = (~moving_clip.rio.transform()) * fixed_clip.rio.transform()

    moving_res_x = abs(moving_clip.rio.transform().a)
    fixed_res_x = abs(fixed_clip.rio.transform().a)
    moving_blur_kernel = sift_matching.resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, moving_res_x
    )
    fixed_blur_kernel = sift_matching.resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, fixed_res_x
    )

    sift = cv2.SIFT_create()
    flann_index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    flann_search_params = dict(checks=50)

    # =====================================================================
    # Step 1 (coarse): detect + unguided match per ND-index channel, then a
    # coarse RANSAC affine. Produces no output -- only seeds step 2.
    # =====================================================================
    all_moving_pts, all_fixed_pts = [], []
    channel_features = []  # (idx_name, moving_kp, moving_desc, fixed_kp, fixed_desc, moving_scale, fixed_scale)

    for idx_name in moving_nd:
        moving_norm, fixed_norm = nd_index.normalize_pair(moving_nd[idx_name], fixed_nd[idx_name])
        moving_valid_mask = ~np.isnan(moving_nd[idx_name])
        fixed_valid_mask = ~np.isnan(fixed_nd[idx_name])

        moving_kp, moving_desc, moving_scale = sift_matching.detect_channel_keypoints(
            moving_norm, moving_valid_mask,
            blur_kernel=moving_blur_kernel, mask_margin_px=mask_erosion_blur_multiple * moving_blur_kernel,
            max_sift_dimension=max_sift_dimension, sift=sift,
        )
        fixed_kp, fixed_desc, fixed_scale = sift_matching.detect_channel_keypoints(
            fixed_norm, fixed_valid_mask,
            blur_kernel=fixed_blur_kernel, mask_margin_px=mask_erosion_blur_multiple * fixed_blur_kernel,
            max_sift_dimension=max_sift_dimension, sift=sift,
        )

        if (moving_desc is None or fixed_desc is None
                or len(moving_kp) < min_keypoints or len(fixed_kp) < min_keypoints):
            continue
        channel_features.append((idx_name, moving_kp, moving_desc, fixed_kp, fixed_desc, moving_scale, fixed_scale))

        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        raw_matches = flann.knnMatch(fixed_desc, moving_desc, k=2)

        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance >= lowe_ratio_threshold * n.distance:
                continue
            moving_pt_det = moving_kp[m.trainIdx].pt
            fixed_pt_det = fixed_kp[m.queryIdx].pt
            moving_pt = (moving_pt_det[0] / moving_scale, moving_pt_det[1] / moving_scale)
            fixed_pt = (fixed_pt_det[0] / fixed_scale, fixed_pt_det[1] / fixed_scale)
            expected_x, expected_y = fixed_clip_to_moving_clip * fixed_pt
            dist = np.sqrt((moving_pt[0] - expected_x) ** 2 + (moving_pt[1] - expected_y) ** 2)
            if dist < distance_threshold_pixels:
                all_moving_pts.append(moving_pt)
                all_fixed_pts.append(fixed_pt)

    if len(all_moving_pts) < 4:
        return RegistrationResult("insufficient_matches_coarse", None, metrics=metrics,
                                   message=f"Only {len(all_moving_pts)} coarse matches (need >= 4).")

    M_coarse, mask_coarse = geometry.estimate_affine_ransac(all_fixed_pts, all_moving_pts, ransac_reproj_threshold)
    if M_coarse is None:
        return RegistrationResult("ransac_failed_coarse", None, metrics=metrics,
                                   message="RANSAC failed to produce a coarse affine transform.")

    # =====================================================================
    # Step 2 (guided): re-match step 1's keypoints/descriptors, restricting
    # candidates to those within keypoint_match_distance_threshold_m of
    # where the coarse affine predicts they should fall.
    # =====================================================================
    M_inv = cv2.invertAffineTransform(M_coarse)
    keypoint_match_distance_threshold_px = keypoint_match_distance_threshold_m / fixed_res_x

    all_moving_pts, all_fixed_pts = [], []
    for idx_name, moving_kp, moving_desc, fixed_kp, fixed_desc, moving_scale, fixed_scale in channel_features:
        ch_moving_pts, ch_fixed_pts = sift_matching.guided_match_per_descriptor_radius(
            moving_kp, moving_desc, fixed_kp, fixed_desc, moving_scale, fixed_scale,
            M_inv, keypoint_match_distance_threshold_px, lowe_ratio_threshold,
        )
        all_moving_pts.extend(ch_moving_pts)
        all_fixed_pts.extend(ch_fixed_pts)

    if len(all_moving_pts) < 4:
        return RegistrationResult("insufficient_matches_guided", None, metrics=metrics,
                                   message=f"Only {len(all_moving_pts)} guided matches (need >= 4).")

    M_part, mask = geometry.estimate_affine_ransac(all_fixed_pts, all_moving_pts, ransac_reproj_threshold)
    if M_part is None:
        return RegistrationResult("ransac_failed_guided", None, metrics=metrics,
                                   message="RANSAC failed to produce a refined affine transform.")

    inlier_bool = (mask.ravel() == 1) if mask is not None else np.zeros(len(all_moving_pts), bool)
    inlier_count = int(inlier_bool.sum())

    if inlier_bool.sum() >= 1:
        inlier_moving = np.float32(all_moving_pts)[inlier_bool]
        inlier_fixed = np.float32(all_fixed_pts)[inlier_bool]
        expected_x, expected_y = fixed_clip_to_moving_clip * (inlier_fixed[:, 0], inlier_fixed[:, 1])
        diffs_before = inlier_moving - np.column_stack([expected_x, expected_y])
        metrics['mean_kp_dist_before'] = float(np.mean(np.sqrt(np.sum(diffs_before ** 2, axis=1))))
        fixed_h = np.c_[inlier_fixed, np.ones(len(inlier_fixed))]
        pred_moving = (M_part @ fixed_h.T).T
        diffs_after = inlier_moving - pred_moving
        metrics['mean_kp_dist_after'] = float(np.mean(np.sqrt(np.sum(diffs_after ** 2, axis=1))))

    if inlier_count < 4:
        return RegistrationResult("insufficient_inliers", None, metrics=metrics,
                                   message=f"Only {inlier_count} RANSAC inliers (need >= 4).")

    inlier_fixed_pts_clip = np.float32(all_fixed_pts)[inlier_bool]
    inlier_moving_pts_clip = np.float32(all_moving_pts)[inlier_bool]
    moving_col_off, moving_row_off = geometry.get_clip_offset(moving_native, moving_clip)

    # =====================================================================
    # Build the displacement field in moving_native's own pixel space, then
    # apply it as a self-referential remap.
    # =====================================================================
    if warp_model == "affine":
        full_ddx, full_ddy = tps_warp.affine_displacement_field(
            M_part, fixed_clip.rio.transform(), moving_clip.rio.transform(),
            moving_native_transform, moving_native_w, moving_native_h,
        )
        n_tps = 0
    else:
        full_ddx, full_ddy, n_tps = _tps_displacement_field(
            inlier_fixed_pts_clip, inlier_moving_pts_clip, fixed_image, fixed_clip, fixed_transform,
            moving_native_transform, moving_native_w, moving_native_h, moving_col_off, moving_row_off,
            tps_max_control_points, tps_coarse_grid_size, rng,
        )

    # Averaged over valid (non-nodata) moving pixels in the overlap only:
    # the field is extrapolated over nodata too (e.g. an AOI clip covering
    # only a corner of its raster), where it can be large but moves nothing.
    row_s, row_e = moving_row_off, min(moving_row_off + moving_clip.shape[1], moving_native_h)
    col_s, col_e = moving_col_off, min(moving_col_off + moving_clip.shape[2], moving_native_w)
    first_band = matched_band_pairs[0][0]
    valid = ~np.isnan(moving_native.isel(band=first_band).values[row_s:row_e, col_s:col_e])
    if valid.any():
        metrics['avg_dx_px'] = float(np.mean(full_ddx[row_s:row_e, col_s:col_e][valid]))
        metrics['avg_dy_px'] = float(np.mean(full_ddy[row_s:row_e, col_s:col_e][valid]))

    registered = tps_warp.apply_displacement_field(moving_native, full_ddx, full_ddy, cv2_interpolation)

    registered_clip = registered.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    metrics_after = band_metrics.compute_band_metrics_matched_resolution(
        registered_clip, fixed_overlap_clip, matched_band_pairs, rng
    )
    band_metrics.store_band_metrics(metrics, metrics_after, matched_band_pairs, "after")

    diagnostics = {
        "warp_model": warp_model,
        "num_step1_channels_used": len(channel_features),
        "coarse_ransac_inliers": int(np.sum(mask_coarse)) if mask_coarse is not None else 0,
        "guided_ransac_inliers": inlier_count,
        "n_tps_control_points": n_tps,
        "affine_matrix_clip": M_part,
        "moving_clip": moving_clip,
        "fixed_clip": fixed_clip,
        "inlier_moving_pts_clip": inlier_moving_pts_clip,
        "inlier_fixed_pts_clip": inlier_fixed_pts_clip,
        "band_metrics_before": metrics_before,
        "band_metrics_after": metrics_after,
    }

    return RegistrationResult("ok", registered, metrics=metrics, diagnostics=diagnostics)


def _tps_displacement_field(
    inlier_fixed_pts_clip, inlier_moving_pts_clip, fixed_image, fixed_clip, fixed_transform,
    moving_native_transform, moving_native_w, moving_native_h, moving_col_off, moving_row_off,
    tps_max_control_points, tps_coarse_grid_size, rng,
):
    """Fit a TPS displacement field through the inlier control points and
    evaluate it over moving_native's full pixel grid.

    Returns (full_ddx, full_ddy, n_control_points_used).
    """
    fixed_col_off, fixed_row_off = geometry.get_clip_offset(fixed_image, fixed_clip)
    clip_h, clip_w = fixed_clip.shape[1], fixed_clip.shape[2]

    tps_fixed_pts_clip, tps_moving_pts_clip = geometry.spatially_thin_keypoints(
        inlier_fixed_pts_clip, inlier_moving_pts_clip, tps_max_control_points, clip_w, clip_h, rng
    )

    # Deduplicate: keypoints from different spectral channels can land at
    # the same or nearly-same pixel, causing the TPS kernel matrix to be
    # singular. Round to the nearest 0.5 px in clip space and keep one
    # representative per site.
    rounded = (tps_fixed_pts_clip / 0.5).round().astype(np.int64)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    tps_fixed_pts_clip = tps_fixed_pts_clip[unique_idx]
    tps_moving_pts_clip = tps_moving_pts_clip[unique_idx]
    n_tps = len(tps_fixed_pts_clip)

    tps_fixed_pts_full = tps_fixed_pts_clip + np.array([[fixed_col_off, fixed_row_off]], dtype=np.float64)
    tps_moving_pts_full = tps_moving_pts_clip + np.array([[moving_col_off, moving_row_off]], dtype=np.float64)

    # The TPS field is fit and evaluated entirely in moving_native's own
    # pixel space, since the final registered output shares moving_native's
    # grid. Fixed control points are converted into that space via world
    # (map) coordinates, so the fit is correct even when moving and fixed
    # have different pixel sizes: fixed pixel -> world (fixed transform) ->
    # moving_native pixel (inverse of moving_native transform).
    fixed_world_x, fixed_world_y = fixed_transform * (tps_fixed_pts_full[:, 0], tps_fixed_pts_full[:, 1])
    domain_x, domain_y = (~moving_native_transform) * (fixed_world_x, fixed_world_y)
    tps_domain_pts_full = np.column_stack([domain_x, domain_y])

    disp_x = tps_moving_pts_full[:, 0] - tps_domain_pts_full[:, 0]
    disp_y = tps_moving_pts_full[:, 1] - tps_domain_pts_full[:, 1]

    rbf_dx, rbf_dy = tps_warp.fit_displacement_field(
        tps_domain_pts_full, disp_x, disp_y, (moving_native_w, moving_native_h)
    )
    full_ddx, full_ddy = tps_warp.evaluate_displacement_field(
        rbf_dx, rbf_dy, moving_native_w, moving_native_h, tps_coarse_grid_size
    )
    return full_ddx, full_ddy, n_tps


def register_to_grid(
    moving,
    fixed,
    band_map: list[int],
    *,
    min_pixel_count: int = 1000,
    min_keypoints: int = 10,
    lowe_ratio_threshold: float = 0.75,
    distance_threshold_pixels: float = 50,
    ransac_reproj_threshold: float = 2.0,
    blur_kernel_base_size: int = 3,
    blur_kernel_reference_resolution: float = 3.0,
    mask_erosion_blur_multiple: float = 2,
    max_sift_dimension: int = 4000,
    seed: int = 0,
) -> RegistrationResult:
    """Register `moving` onto `fixed`'s full pixel grid using a single
    global affine warp fit from one unguided SIFT matching pass. See module
    docstring.

    `band_map[i]` is the 0-indexed `fixed` band matching 0-indexed `moving`
    band `i`, or -1 if `moving` band `i` has no match in `fixed`.
    """
    matched_band_pairs = _validate_band_map(band_map)
    rng = np.random.default_rng(seed=seed)

    moving_image = _load_raster(moving)
    fixed_image = _load_raster(fixed)

    if moving_image.shape[0] != len(band_map):
        return RegistrationResult(
            "band_count_mismatch", None,
            message=f"band_map has {len(band_map)} entries but moving image has {moving_image.shape[0]} bands.",
        )

    working_fixed = fixed_image
    if fixed_image.rio.crs != moving_image.rio.crs:
        working_fixed = fixed_image.rio.reproject(moving_image.rio.crs)

    overlap = geometry.find_overlap(moving_image, working_fixed)
    if overlap is None:
        return RegistrationResult("no_overlap", None, message="No geographic overlap between moving and fixed.")
    left, bottom, right, top = overlap

    moving_clip = moving_image.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    fixed_clip = working_fixed.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)

    moving_valid = int(np.sum(~np.isnan(moving_clip.values)))
    fixed_valid = int(np.sum(~np.isnan(fixed_clip.values)))
    if min(moving_valid, fixed_valid) < min_pixel_count:
        return RegistrationResult(
            "insufficient_pixels", None,
            message=f"Too few valid pixels in overlap (moving={moving_valid}, fixed={fixed_valid}).",
        )

    metric_keys = band_metrics.build_metric_keys(matched_band_pairs)
    metrics = {k: np.nan for k in metric_keys}
    metrics_before = band_metrics.compute_band_metrics_matched_resolution(moving_clip, fixed_clip, matched_band_pairs, rng)
    band_metrics.store_band_metrics(metrics, metrics_before, matched_band_pairs, "before")

    moving_band_arrays = nd_index.extract_band_arrays(moving_clip, [i for i, _ in matched_band_pairs])
    fixed_band_arrays_raw = nd_index.extract_band_arrays(fixed_clip, [f for _, f in matched_band_pairs])
    fixed_band_arrays = {m: fixed_band_arrays_raw[f] for m, f in matched_band_pairs}
    moving_nd = nd_index.compute_nd_indices(moving_band_arrays)
    fixed_nd = nd_index.compute_nd_indices(fixed_band_arrays)

    scale_x = moving_clip.shape[2] / fixed_clip.shape[2]
    scale_y = moving_clip.shape[1] / fixed_clip.shape[1]
    moving_res_x = abs(moving_clip.rio.transform().a)
    fixed_res_x = abs(fixed_clip.rio.transform().a)
    moving_blur_kernel = sift_matching.resolution_scaled_kernel_size(blur_kernel_base_size, blur_kernel_reference_resolution, moving_res_x)
    fixed_blur_kernel = sift_matching.resolution_scaled_kernel_size(blur_kernel_base_size, blur_kernel_reference_resolution, fixed_res_x)

    sift = cv2.SIFT_create()
    flann_index_params = dict(algorithm=1, trees=5)
    flann_search_params = dict(checks=50)

    all_moving_pts, all_fixed_pts = [], []
    for idx_name in moving_nd:
        moving_norm, fixed_norm = nd_index.normalize_pair(moving_nd[idx_name], fixed_nd[idx_name])
        moving_valid_mask = ~np.isnan(moving_nd[idx_name])
        fixed_valid_mask = ~np.isnan(fixed_nd[idx_name])

        moving_kp, moving_desc, moving_scale = sift_matching.detect_channel_keypoints(
            moving_norm, moving_valid_mask, blur_kernel=moving_blur_kernel,
            mask_margin_px=mask_erosion_blur_multiple * moving_blur_kernel,
            max_sift_dimension=max_sift_dimension, sift=sift,
        )
        fixed_kp, fixed_desc, fixed_scale = sift_matching.detect_channel_keypoints(
            fixed_norm, fixed_valid_mask, blur_kernel=fixed_blur_kernel,
            mask_margin_px=mask_erosion_blur_multiple * fixed_blur_kernel,
            max_sift_dimension=max_sift_dimension, sift=sift,
        )
        if (moving_desc is None or fixed_desc is None
                or len(moving_kp) < min_keypoints or len(fixed_kp) < min_keypoints):
            continue

        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        raw_matches = flann.knnMatch(fixed_desc, moving_desc, k=2)
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance >= lowe_ratio_threshold * n.distance:
                continue
            moving_pt_det = moving_kp[m.trainIdx].pt
            fixed_pt_det = fixed_kp[m.queryIdx].pt
            moving_pt = (moving_pt_det[0] / moving_scale, moving_pt_det[1] / moving_scale)
            fixed_pt = (fixed_pt_det[0] / fixed_scale, fixed_pt_det[1] / fixed_scale)
            dist = np.sqrt((moving_pt[0] - scale_x * fixed_pt[0]) ** 2 + (moving_pt[1] - scale_y * fixed_pt[1]) ** 2)
            if dist < distance_threshold_pixels:
                all_moving_pts.append(moving_pt)
                all_fixed_pts.append(fixed_pt)

    if len(all_moving_pts) < 4:
        return RegistrationResult("insufficient_matches_coarse", None, metrics=metrics,
                                   message=f"Only {len(all_moving_pts)} matches (need >= 4).")

    M_part, mask = geometry.estimate_affine_ransac(all_fixed_pts, all_moving_pts, ransac_reproj_threshold)
    if M_part is None:
        return RegistrationResult("ransac_failed_coarse", None, metrics=metrics,
                                   message="RANSAC failed to produce a valid affine transform.")

    inlier_bool = (mask.ravel() == 1) if mask is not None else np.zeros(len(all_moving_pts), bool)
    inlier_count = int(inlier_bool.sum())
    if inlier_bool.sum() >= 1:
        inlier_moving = np.float32(all_moving_pts)[inlier_bool]
        inlier_fixed = np.float32(all_fixed_pts)[inlier_bool]
        diffs_before = inlier_moving - inlier_fixed * np.array([[scale_x, scale_y]])
        metrics['mean_kp_dist_before'] = float(np.mean(np.sqrt(np.sum(diffs_before ** 2, axis=1))))
        fixed_h = np.c_[inlier_fixed, np.ones(len(inlier_fixed))]
        pred_moving = (M_part @ fixed_h.T).T
        diffs_after = inlier_moving - pred_moving
        metrics['mean_kp_dist_after'] = float(np.mean(np.sqrt(np.sum(diffs_after ** 2, axis=1))))

    if inlier_count < 4:
        return RegistrationResult("insufficient_inliers", None, metrics=metrics,
                                   message=f"Only {inlier_count} RANSAC inliers (need >= 4).")

    M = np.eye(3, dtype=np.float32)
    M[0:2, 0:3] = M_part

    moving_col_off, moving_row_off = geometry.get_clip_offset(moving_image, moving_clip)
    fixed_col_off, fixed_row_off = geometry.get_clip_offset(working_fixed, fixed_clip)
    M_full = affine_warp.adjust_affine_for_offsets(M, moving_col_off, moving_row_off, fixed_col_off, fixed_row_off)

    registered = affine_warp.warp_onto_fixed_grid(moving_image, M_full, working_fixed)

    registered_clip = registered.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    if registered_clip.shape[1:] == fixed_clip.shape[1:]:
        metrics_after = band_metrics.compute_band_metrics_matched_resolution(registered_clip, fixed_clip, matched_band_pairs, rng)
        band_metrics.store_band_metrics(metrics, metrics_after, matched_band_pairs, "after")
    else:
        metrics_after = None

    diagnostics = {
        "ransac_inliers": inlier_count,
        "num_matches": len(all_moving_pts),
        "affine_matrix": M,
        "moving_clip": moving_clip,
        "fixed_clip": fixed_clip,
        "inlier_moving_pts_clip": np.float32(all_moving_pts)[inlier_bool],
        "inlier_fixed_pts_clip": np.float32(all_fixed_pts)[inlier_bool],
        "band_metrics_before": metrics_before,
        "band_metrics_after": metrics_after,
    }

    return RegistrationResult("ok", registered, metrics=metrics, diagnostics=diagnostics)
