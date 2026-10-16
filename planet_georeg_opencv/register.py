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

import copy
import functools
import inspect
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import cv2
import numpy as np
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling

from . import affine_warp, band_metrics, geometry, nd_index, sift_matching, tps_warp
from .reference_keypoints import DetectedFeatures, ReferenceKeypoints, features_in_clip

_RASTERIO_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}
CV2_INTERPOLATION = {
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
    "tps_fit_failed",
]


@dataclass
class RegistrationResult:
    status: Status
    registered: xr.DataArray | None
    metrics: dict = field(default_factory=dict)
    message: str = ""
    diagnostics: dict = field(default_factory=dict)
    # (map_x, map_y): for every pixel of the moving image's own grid, the
    # source pixel the correction samples (register_local only). Apply with
    # tps_warp.iter_remapped_bands, e.g. to stream the warp to disk.
    remap: tuple | None = None


def _load_raster(image) -> xr.DataArray:
    # cache=False keeps file-backed rasters lazy band-by-band: with xarray's
    # default in-memory cache, the first partial read loads every band.
    if isinstance(image, str):
        return rxr.open_rasterio(image, masked=True, cache=False).squeeze()
    return image


def _validate_band_map(band_map: list[int]) -> list[tuple[int, int]]:
    matched = nd_index.get_matched_band_pairs(band_map)
    if len(matched) < 2:
        raise ValueError(
            "band_map must map at least two moving bands to fixed bands "
            "(need at least two matched bands to compute a normalized-difference index)."
        )
    return matched


def _validate_mask(mask, image: xr.DataArray, name: str) -> np.ndarray | None:
    """`mask` (2D bool array or DataArray, True = usable) as a numpy bool
    array, checked against `image`'s own (rows, cols)."""
    if mask is None:
        return None
    arr = np.asarray(mask.values if isinstance(mask, xr.DataArray) else mask).astype(bool)
    if arr.shape != tuple(image.shape[-2:]):
        raise ValueError(f"{name} has shape {arr.shape} but its image is {tuple(image.shape[-2:])}")
    return arr


def _reproject_mask(mask: np.ndarray, source: xr.DataArray, like: xr.DataArray) -> np.ndarray:
    """Nearest-neighbour reprojection of a bool mask on `source`'s grid onto
    `like`'s grid; pixels outside `source` become False."""
    mask_da = xr.DataArray(mask.astype(np.uint8), coords={"y": source.y, "x": source.x}, dims=("y", "x"))
    mask_da = mask_da.rio.write_crs(source.rio.crs).rio.write_nodata(0)
    return mask_da.rio.reproject_match(like, resampling=Resampling.nearest).values.astype(bool)


def _clip_box_view(image: xr.DataArray, minx, miny, maxx, maxy) -> xr.DataArray:
    """Exactly the pixels image.rio.clip_box(...) selects, but as a slice of
    `image` -- a view for in-memory data (clip_box itself copies), still
    lazy for file-backed data. The window is found by running clip_box on
    a zero-cost uint8 stand-in with `image`'s coordinates."""
    stand_in = xr.DataArray(
        np.broadcast_to(np.uint8(0), tuple(image.shape[-2:])), coords={"y": image.y, "x": image.x}, dims=("y", "x"),
    ).rio.write_crs(image.rio.crs)
    window = stand_in.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    col, row = geometry.get_clip_offset(stand_in, window)
    return image.isel(y=slice(row, row + window.shape[0]), x=slice(col, col + window.shape[1]))


def _load_clip(clip: xr.DataArray, file_backed: bool) -> xr.DataArray:
    """clip.load(), except that a clip of a raster opened from a file is
    read one band at a time into a single buffer: loading several bands of
    a lazily opened raster in one call is many times slower than reading
    the same bands individually (measured: 8 s against under 1 s for four
    19-megapixel bands), with identical values either way."""
    if not file_backed or clip.ndim != 3:
        return clip.load()
    first = clip.isel(band=0).values
    data = np.empty(clip.shape, dtype=first.dtype)
    data[0] = first
    del first
    for band in range(1, clip.shape[0]):
        data[band] = clip.isel(band=band).values
    return clip.copy(data=data)


def _apply_mask_to_clip(clip: xr.DataArray, full_image: xr.DataArray, mask: np.ndarray | None,
                        in_place: bool = False) -> xr.DataArray:
    """Set pixels of `clip` (a clip of `full_image`) to NaN where `mask` (on
    `full_image`'s grid) is False. `in_place=True` overwrites clip's own
    buffer instead of copying -- only for a clip this module loaded itself,
    never one that is a view of a caller's array."""
    if mask is None:
        return clip
    col, row = geometry.get_clip_offset(full_image, clip)
    sub = mask[row: row + clip.shape[-2], col: col + clip.shape[-1]]
    if in_place:
        clip.values[..., ~sub] = np.nan
        return clip
    return clip.where(xr.DataArray(sub, coords={"y": clip.y, "x": clip.x}, dims=("y", "x")))


def _select_bands(image: xr.DataArray, bands: list[int]) -> xr.DataArray:
    """`image` restricted to `bands` (0-indexed), without a copy when that's
    every band in order."""
    if bands == list(range(image.shape[0])):
        return image
    return image.isel(band=bands)


def _relabel_band_metrics(result: dict, matched_band_pairs) -> dict:
    """compute_band_metrics_matched_resolution() results computed on
    band-subset clips are keyed 0..n-1; re-key per-band entries by the
    original moving band index so they line up with matched_band_pairs."""
    return {
        key: ({matched_band_pairs[k][0]: v for k, v in val.items()} if isinstance(val, dict) else val)
        for key, val in result.items()
    }


def _prepare_channel_pairs(
    moving_arrays, fixed_arrays, *, moving_blur_kernel, fixed_blur_kernel, mask_erosion_blur_multiple,
    max_sift_dimension,
):
    """For every ND-index channel, the (image, mask, scale) SIFT will run on
    in each image (see sift_matching.prepare_channel_image), jointly
    normalized exactly as nd_index.normalize_pair does. The results are
    small, so the full-size band arrays can be freed before SIFT runs."""
    prepared = []
    for _, moving_nd, fixed_nd in nd_index.iter_nd_index_pairs(moving_arrays, fixed_arrays):
        lo, hi = nd_index.joint_range(moving_nd, fixed_nd)
        moving_prepared = sift_matching.prepare_channel_image(
            nd_index.normalize_to_range(moving_nd, lo, hi), ~np.isnan(moving_nd),
            blur_kernel=moving_blur_kernel, mask_margin_px=mask_erosion_blur_multiple * moving_blur_kernel,
            max_sift_dimension=max_sift_dimension,
        )
        fixed_prepared = sift_matching.prepare_channel_image(
            nd_index.normalize_to_range(fixed_nd, lo, hi), ~np.isnan(fixed_nd),
            blur_kernel=fixed_blur_kernel, mask_margin_px=mask_erosion_blur_multiple * fixed_blur_kernel,
            max_sift_dimension=max_sift_dimension,
        )
        prepared.append((moving_prepared, fixed_prepared))
    return prepared


def _prepare_moving_channels(
    moving_arrays, fixed_features, *, moving_blur_kernel, mask_erosion_blur_multiple, max_sift_dimension,
):
    """_prepare_channel_pairs for a `fixed` whose keypoints were detected
    beforehand: only `moving`'s channels are prepared, each normalized
    over its own min/max, and paired with `fixed_features` (one
    DetectedFeatures per channel, in channel order).

    Each moving entry is (image, mask, true_shape) -- the full-size
    channel's (rows, cols) in place of _prepare_channel_pairs' scale -- so
    that its keypoints can be converted to clip pixels exactly (see
    sift_matching.detection_to_true_pixels): the fixed keypoints were
    detected at a scale of their own, so the approximations of `pt /
    scale` would not cancel between the two images."""
    prepared = []
    for (_, moving_nd), features in zip(nd_index.iter_nd_indices(moving_arrays), fixed_features, strict=True):
        lo, hi = nd_index.joint_range(moving_nd, moving_nd)
        image, mask, _ = sift_matching.prepare_channel_image(
            nd_index.normalize_to_range(moving_nd, lo, hi), ~np.isnan(moving_nd),
            blur_kernel=moving_blur_kernel, mask_margin_px=mask_erosion_blur_multiple * moving_blur_kernel,
            max_sift_dimension=max_sift_dimension,
        )
        prepared.append(((image, mask, moving_nd.shape), features))
    return prepared


def _match_keypoints_two_step(
    prepared, fixed_clip_to_moving_clip, *, fixed_res_x,
    min_keypoints, lowe_ratio_threshold, distance_threshold_pixels, ransac_reproj_threshold,
    keypoint_match_distance_threshold_m, sift_lock=None, sift_tile_size=None, sift_tile_overlap=128,
):
    """Steps 1 (coarse, unguided) and 2 (geometrically guided) of
    register_local's matching, on _prepare_channel_pairs' output. Returns
    (status, message, step-1 RANSAC mask, number of channels used, guided
    moving pts, guided fixed pts)."""
    sift = cv2.SIFT_create()
    flann_index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
    flann_search_params = dict(checks=50)

    # Step 1 (coarse): detect + unguided match per ND-index channel, then a
    # coarse RANSAC affine. Produces no output -- only seeds step 2.
    all_moving_pts, all_fixed_pts = [], []
    # Kept for step 2, compactly: keypoint coordinates as float64 arrays
    # (exactly the values cv2.KeyPoint.pt holds) instead of millions of
    # KeyPoint objects, and descriptors as uint8 (OpenCV's SIFT descriptors
    # are whole numbers 0-255 stored as float32, so this is lossless).
    channel_features = []  # (moving_pts, moving_desc, fixed_pts, fixed_desc, moving_scale, fixed_scale)

    while prepared:
        (moving_img, moving_mask, moving_scale), fixed_item = prepared.pop(0)
        detect = dict(tile_size=sift_tile_size, tile_overlap=sift_tile_overlap, lock=sift_lock)
        moving_pts_det, moving_desc = sift_matching.detect_keypoints(sift, moving_img, moving_mask, **detect)
        if isinstance(fixed_item, DetectedFeatures):
            # Detected beforehand, already in fixed-clip pixels (see
            # _prepare_moving_channels).
            fixed_pts_det, fixed_scale = fixed_item.points, 1.0
            fixed_desc = fixed_item.descriptors.astype(np.float32) if len(fixed_pts_det) else None
        else:
            fixed_img, fixed_mask, fixed_scale = fixed_item
            fixed_pts_det, fixed_desc = sift_matching.detect_keypoints(sift, fixed_img, fixed_mask, **detect)
            del fixed_img, fixed_mask
        moving_det_shape = moving_img.shape
        precomputed_fixed = isinstance(fixed_item, DetectedFeatures)
        del moving_img, moving_mask, fixed_item

        if (moving_desc is None or fixed_desc is None
                or len(moving_pts_det) < min_keypoints or len(fixed_pts_det) < min_keypoints):
            continue
        if precomputed_fixed:
            # moving_scale is the full-size channel's shape here (see
            # _prepare_moving_channels): convert exactly, once, and treat
            # the result as already being in clip pixels.
            moving_pts_det = sift_matching.detection_to_true_pixels(moving_pts_det, moving_det_shape, moving_scale)
            moving_scale = 1.0

        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        raw_matches = flann.knnMatch(fixed_desc, moving_desc, k=2)

        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance >= lowe_ratio_threshold * n.distance:
                continue
            moving_pt_det = moving_pts_det[m.trainIdx]
            fixed_pt_det = fixed_pts_det[m.queryIdx]
            moving_pt = (moving_pt_det[0] / moving_scale, moving_pt_det[1] / moving_scale)
            fixed_pt = (fixed_pt_det[0] / fixed_scale, fixed_pt_det[1] / fixed_scale)
            expected_x, expected_y = fixed_clip_to_moving_clip * fixed_pt
            dist = np.sqrt((moving_pt[0] - expected_x) ** 2 + (moving_pt[1] - expected_y) ** 2)
            if dist < distance_threshold_pixels:
                all_moving_pts.append(moving_pt)
                all_fixed_pts.append(fixed_pt)
        del raw_matches
        channel_features.append((
            moving_pts_det, moving_desc.astype(np.uint8), fixed_pts_det, fixed_desc.astype(np.uint8),
            moving_scale, fixed_scale,
        ))
        del moving_desc, fixed_desc

    if len(all_moving_pts) < 4:
        return ("insufficient_matches_coarse", f"Only {len(all_moving_pts)} coarse matches (need >= 4).",
                None, len(channel_features), None, None)

    M_coarse, mask_coarse = geometry.estimate_affine_ransac(all_fixed_pts, all_moving_pts, ransac_reproj_threshold)
    if M_coarse is None:
        return ("ransac_failed_coarse", "RANSAC failed to produce a coarse affine transform.",
                None, len(channel_features), None, None)

    # Step 2 (guided): re-match step 1's keypoints/descriptors, restricting
    # candidates to those within keypoint_match_distance_threshold_m of
    # where the coarse affine predicts they should fall.
    M_inv = cv2.invertAffineTransform(M_coarse)
    keypoint_match_distance_threshold_px = keypoint_match_distance_threshold_m / fixed_res_x

    all_moving_pts, all_fixed_pts = [], []
    for moving_pts_det, moving_desc, fixed_pts_det, fixed_desc, moving_scale, fixed_scale in channel_features:
        ch_moving_pts, ch_fixed_pts = sift_matching.guided_match_per_descriptor_radius(
            moving_pts_det, moving_desc, fixed_pts_det, fixed_desc, moving_scale, fixed_scale,
            M_inv, keypoint_match_distance_threshold_px, lowe_ratio_threshold,
        )
        all_moving_pts.extend(ch_moving_pts)
        all_fixed_pts.extend(ch_fixed_pts)

    if len(all_moving_pts) < 4:
        return ("insufficient_matches_guided", f"Only {len(all_moving_pts)} guided matches (need >= 4).",
                mask_coarse, len(channel_features), None, None)
    return "ok", "", mask_coarse, len(channel_features), all_moving_pts, all_fixed_pts


def register_local(
    moving,
    fixed,
    band_map: list[int],
    *,
    moving_valid_mask=None,
    fixed_valid_mask=None,
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
    return_registered: bool = True,
    on_warped_band=None,
    fixed_keypoints: ReferenceKeypoints | None = None,
    sift_lock=None,
    sift_tile_size: int | None = None,
    sift_tile_overlap: int = 128,
    seed: int = 0,
) -> RegistrationResult:
    """Register `moving` onto its own grid using a warp fit from keypoint
    correspondences with `fixed`. See module docstring.

    `band_map[i]` is the 0-indexed `fixed` band matching 0-indexed `moving`
    band `i`, or -1 if `moving` band `i` has no match in `fixed`. Only
    matched bands are read for matching and metrics; every `moving` band
    is warped.

    `moving_valid_mask` / `fixed_valid_mask` (optional 2D bool arrays on
    each image's own grid, True = usable, e.g. a cloud mask's clear
    pixels) exclude pixels from keypoint detection and from the before/
    after band metrics. They never affect the warped output itself.

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

    `return_registered=False` skips building the full warped image (the
    largest allocation for big rasters); `result.remap` still holds the
    correction, which tps_warp.iter_remapped_bands can apply one band at a
    time (e.g. while streaming the output to disk).

    `on_warped_band` (optional callable `(band_index, warped)`, requires
    return_registered=False) receives every band of the registered image
    in band order, as a (height, width) float32 array (NaN where there is
    no data) that the callee owns and may modify -- e.g. to write each
    band to disk as it is produced. Each band is then warped exactly once
    (the after-alignment metrics use the same warped arrays), where
    applying result.remap afterwards warps the matched bands a second
    time. It is only called once registration has succeeded.

    `fixed_keypoints` (optional, see reference_keypoints) supplies
    keypoints detected on `fixed` beforehand, so nothing is detected on
    `fixed` here -- for registering many images against one reference.
    The matching channels of `moving` are then normalized over their own
    range rather than the joint range of the two images, and the two
    images' keypoints may come from different detection scales, so
    results differ slightly from detecting on both.

    `sift_tile_size` (optional) runs SIFT detection on tiles of that many
    detection pixels per side, each extended by `sift_tile_overlap`,
    instead of on a whole detection image at once -- see
    sift_matching.detect_keypoints. Matching and the warp still use the
    keypoints of the whole image. It cuts SIFT's working memory from
    about 4 GB (at the default max_sift_dimension) to about 0.25 GB per
    megapixel of tile, at the cost of slightly different keypoints near
    tile edges.

    `sift_lock` (optional context manager) is held around each SIFT
    detection call. OpenCV's SIFT needs roughly 0.25 GB of working memory
    per megapixel of detection image (about 4 GB at the default
    max_sift_dimension), far more than the rest of registration; a lock
    shared between processes keeps several of them from hitting that peak
    at the same moment. It has no effect on results.
    """
    outcome = _register_local_models(
        moving, fixed, band_map, (warp_model,), {warp_model: on_warped_band},
        moving_valid_mask=moving_valid_mask,
        fixed_valid_mask=fixed_valid_mask,
        min_pixel_count=min_pixel_count,
        min_keypoints=min_keypoints,
        lowe_ratio_threshold=lowe_ratio_threshold,
        distance_threshold_pixels=distance_threshold_pixels,
        ransac_reproj_threshold=ransac_reproj_threshold,
        keypoint_match_distance_threshold_m=keypoint_match_distance_threshold_m,
        blur_kernel_base_size=blur_kernel_base_size,
        blur_kernel_reference_resolution=blur_kernel_reference_resolution,
        mask_erosion_blur_multiple=mask_erosion_blur_multiple,
        max_sift_dimension=max_sift_dimension,
        fixed_buffer_m=fixed_buffer_m,
        tps_max_control_points=tps_max_control_points,
        tps_coarse_grid_size=tps_coarse_grid_size,
        resampling_method=resampling_method,
        return_registered=return_registered,
        fixed_keypoints=fixed_keypoints,
        sift_lock=sift_lock,
        sift_tile_size=sift_tile_size,
        sift_tile_overlap=sift_tile_overlap,
        seed=seed,
    )
    return outcome if isinstance(outcome, RegistrationResult) else outcome[warp_model]


def register_local_models(moving, fixed, band_map: list[int], warp_models=("affine", "tps"), *,
                          on_warped_band=None, **settings) -> dict:
    """register_local for several warp models at once: keypoint detection
    and matching (most of the work) run once, then each model in
    `warp_models` is applied to the same matched keypoints. Each model's
    result is exactly what register_local(warp_model=model) returns.

    `on_warped_band` (optional, requires return_registered=False) is
    called as `(warp_model, band_index, warped)` -- register_local's
    callback with the model prepended. `settings` are register_local's
    other keyword arguments.

    Returns {warp_model: RegistrationResult}; if registration fails before
    any warp, every model maps to the same failed result.
    """
    warp_models = tuple(warp_models)
    if not warp_models or len(set(warp_models)) != len(warp_models):
        raise ValueError(f"warp_models must be distinct and non-empty, got {warp_models!r}")
    defaults = {
        name: parameter.default for name, parameter in inspect.signature(register_local).parameters.items()
        if parameter.default is not inspect.Parameter.empty and name not in ("warp_model", "on_warped_band")
    }
    unknown = sorted(set(settings) - set(defaults))
    if unknown:
        raise TypeError(f"register_local_models() got unexpected keyword argument(s) {unknown}")
    callbacks = {}
    if on_warped_band is not None:
        callbacks = {model: functools.partial(on_warped_band, model) for model in warp_models}
    outcome = _register_local_models(moving, fixed, band_map, warp_models, callbacks, **{**defaults, **settings})
    if isinstance(outcome, RegistrationResult):
        return {model: outcome for model in warp_models}
    return outcome


def _register_local_models(
    moving,
    fixed,
    band_map: list[int],
    warp_models: tuple,
    band_callbacks: dict,
    *,
    moving_valid_mask=None,
    fixed_valid_mask=None,
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
    fixed_buffer_m: float = 0.0,
    tps_max_control_points: int | None = 2000,
    tps_coarse_grid_size: int = 300,
    resampling_method: str = "bilinear",
    return_registered: bool = True,
    fixed_keypoints: ReferenceKeypoints | None = None,
    sift_lock=None,
    sift_tile_size: int | None = None,
    sift_tile_overlap: int = 128,
    seed: int = 0,
) -> RegistrationResult | dict:
    """register_local for one or several warp models: everything up to and
    including the matched keypoints is done once, then each model in
    `warp_models` is applied. `band_callbacks` maps a model to its
    on_warped_band (models without one get none).

    Returns {model: RegistrationResult}, or a single failed
    RegistrationResult if registration stopped before any warp."""
    for warp_model in warp_models:
        if warp_model not in ("tps", "affine"):
            raise ValueError(f"warp_model must be 'tps' or 'affine', got {warp_model!r}")
    if any(callback is not None for callback in band_callbacks.values()) and return_registered:
        raise ValueError("on_warped_band requires return_registered=False")
    matched_band_pairs = _validate_band_map(band_map)
    rasterio_resampling = _RASTERIO_RESAMPLING[resampling_method]
    cv2_interpolation = CV2_INTERPOLATION[resampling_method]
    rng = np.random.default_rng(seed=seed)
    # FLANN's randomized kd-trees draw from OpenCV's process-wide RNG, so
    # without this a result depends on what the process registered before.
    # (OpenCV treats seed 0 as its default initial state.)
    cv2.setRNGSeed(seed)

    moving_image = _load_raster(moving)
    fixed_image = _load_raster(fixed)

    if moving_image.shape[0] != len(band_map):
        return RegistrationResult(
            "band_count_mismatch", None,
            message=f"band_map has {len(band_map)} entries but moving image has {moving_image.shape[0]} bands.",
        )
    moving_mask = _validate_mask(moving_valid_mask, moving_image, "moving_valid_mask")
    fixed_mask = _validate_mask(fixed_valid_mask, fixed_image, "fixed_valid_mask")

    # moving_native keeps moving's approximate native resolution; it is used
    # for feature matching, the warp, and the final registered output.
    if moving_image.rio.crs != fixed_image.rio.crs:
        moving_native = moving_image.rio.reproject(fixed_image.rio.crs, resampling=rasterio_resampling)
        if moving_mask is not None:
            moving_mask = _reproject_mask(moving_mask, moving_image, moving_native)
    else:
        moving_native = moving_image
    moving_native_h, moving_native_w = moving_native.shape[1], moving_native.shape[2]
    moving_native_transform = moving_native.rio.transform()
    fixed_transform = fixed_image.rio.transform()

    overlap = geometry.find_overlap(moving_native, fixed_image)
    if overlap is None:
        return RegistrationResult("no_overlap", None, message="No geographic overlap between moving and fixed.")
    left, bottom, right, top = overlap

    # Clips hold only the matched bands, indexed 0..n-1 in matched_band_pairs
    # order; local_pairs indexes both.
    moving_bands = [m for m, _ in matched_band_pairs]
    fixed_bands = [f for _, f in matched_band_pairs]
    local_pairs = [(k, k) for k in range(len(matched_band_pairs))]

    # Clips are views/lazy slices (see _clip_box_view); .load() reads a
    # file-backed clip once and is a no-op for in-memory inputs. A moving
    # clip read from a file is this function's own buffer, so its mask is
    # applied in place; anything that may view a caller's array is copied.
    moving_from_file = isinstance(moving, str) and moving_native is moving_image
    fixed_from_file = isinstance(fixed, str)
    moving_clip = _load_clip(
        _clip_box_view(_select_bands(moving_native, moving_bands), left, bottom, right, top), moving_from_file,
    )
    moving_clip = _apply_mask_to_clip(moving_clip, moving_native, moving_mask, in_place=moving_from_file)
    fixed_sub = _select_bands(fixed_image, fixed_bands)
    # fixed_overlap_clip (strict overlap) is used for band metrics;
    # fixed_clip (overlap + fixed_buffer_m) is what keypoints are searched in.
    fixed_overlap_clip = _apply_mask_to_clip(
        _load_clip(_clip_box_view(fixed_sub, left, bottom, right, top), fixed_from_file), fixed_image, fixed_mask
    )
    if fixed_buffer_m > 0:
        fixed_clip = _apply_mask_to_clip(
            _load_clip(
                _clip_box_view(
                    fixed_sub, left - fixed_buffer_m, bottom - fixed_buffer_m,
                    right + fixed_buffer_m, top + fixed_buffer_m,
                ),
                fixed_from_file,
            ),
            fixed_image, fixed_mask,
        )
    else:
        fixed_clip = fixed_overlap_clip

    moving_valid = int(np.count_nonzero(~np.isnan(moving_clip.values)))
    fixed_valid = int(np.count_nonzero(~np.isnan(fixed_overlap_clip.values)))
    if min(moving_valid, fixed_valid) < min_pixel_count:
        return RegistrationResult(
            "insufficient_pixels", None,
            message=f"Too few valid pixels in overlap (moving={moving_valid}, fixed={fixed_valid}).",
        )

    metric_keys = band_metrics.build_metric_keys(matched_band_pairs)
    metrics = {k: np.nan for k in metric_keys}

    metrics_before = _relabel_band_metrics(
        band_metrics.compute_band_metrics_matched_resolution(moving_clip, fixed_overlap_clip, local_pairs, rng),
        matched_band_pairs,
    )
    band_metrics.store_band_metrics(metrics, metrics_before, matched_band_pairs, "before")

    # Where a fixed-clip pixel lands in moving-clip pixels if both rasters'
    # georeferencing were correct (via world coordinates, so it holds even
    # when the two clips differ in resolution or extent, e.g. with a
    # fixed_buffer_m margin).
    fixed_clip_to_moving_clip = (~moving_clip.rio.transform()) * fixed_clip.rio.transform()

    moving_res_x = abs(moving_clip.rio.transform().a)
    fixed_res_x = abs(fixed_clip.rio.transform().a)

    moving_blur_kernel = sift_matching.resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, moving_res_x)
    if fixed_keypoints is not None:
        # Channel k of the matched pairs is fixed bands (fixed_bands[a],
        # fixed_bands[b]) for the k-th a < b, as iter_nd_indices orders them.
        channel_keys = [(fixed_bands[a], fixed_bands[b]) for a, b in combinations(range(len(local_pairs)), 2)]
        prepared = _prepare_moving_channels(
            nd_index.extract_band_arrays(moving_clip, list(range(len(local_pairs)))),
            features_in_clip(fixed_keypoints, channel_keys, fixed_clip),
            moving_blur_kernel=moving_blur_kernel, mask_erosion_blur_multiple=mask_erosion_blur_multiple,
            max_sift_dimension=max_sift_dimension,
        )
    else:
        prepared = _prepare_channel_pairs(
            nd_index.extract_band_arrays(moving_clip, list(range(len(local_pairs)))),
            nd_index.extract_band_arrays(fixed_clip, list(range(len(local_pairs)))),
            moving_blur_kernel=moving_blur_kernel,
            fixed_blur_kernel=sift_matching.resolution_scaled_kernel_size(
                blur_kernel_base_size, blur_kernel_reference_resolution, fixed_res_x),
            mask_erosion_blur_multiple=mask_erosion_blur_multiple, max_sift_dimension=max_sift_dimension,
        )
    # From here on moving_clip is only needed for its geometry (shape,
    # coordinates, transform): swap its pixel data for a zero-cost
    # broadcast so the full-size array is freed before SIFT, whose own
    # working memory is several GB at max_sift_dimension=4000.
    moving_clip = moving_clip.isel(band=[0]).copy(
        data=np.broadcast_to(np.float32(0), (1,) + tuple(moving_clip.shape[1:]))
    )
    status, message, mask_coarse, n_channels, all_moving_pts, all_fixed_pts = _match_keypoints_two_step(
        prepared, fixed_clip_to_moving_clip, fixed_res_x=fixed_res_x,
        min_keypoints=min_keypoints, lowe_ratio_threshold=lowe_ratio_threshold,
        distance_threshold_pixels=distance_threshold_pixels, ransac_reproj_threshold=ransac_reproj_threshold,
        keypoint_match_distance_threshold_m=keypoint_match_distance_threshold_m, sift_lock=sift_lock,
        sift_tile_size=sift_tile_size, sift_tile_overlap=sift_tile_overlap,
    )
    if status != "ok":
        return RegistrationResult(status, None, metrics=metrics, message=message)

    M_part, mask = geometry.estimate_affine_ransac(all_fixed_pts, all_moving_pts, ransac_reproj_threshold)
    if M_part is None:
        return RegistrationResult("ransac_failed_guided", None, metrics=metrics,
                                   message="RANSAC failed to produce a refined affine transform.")

    inlier_bool = (mask.ravel() == 1) if mask is not None else np.zeros(len(all_moving_pts), bool)
    inlier_count = int(inlier_bool.sum())

    if inlier_count >= 1:
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

    # The fitted affine transform in moving's own pixels, recorded whatever
    # warp is applied (see tps_warp.describe_affine).
    affine_native = tps_warp.affine_in_native_pixels(
        M_part, fixed_clip.rio.transform(), moving_clip.rio.transform(), moving_native_transform)
    metrics.update(tps_warp.describe_affine(affine_native, moving_native_w, moving_native_h))

    # Everything above is the same whatever the warp; from here on each
    # requested warp model is applied in turn, starting from the same
    # random state so that a model's result does not depend on which
    # others were asked for.
    matched_metrics, matched_rng = metrics, rng
    results = {}
    for warp_model in warp_models:
        metrics = dict(matched_metrics)
        rng = copy.deepcopy(matched_rng)
        on_warped_band = band_callbacks.get(warp_model)

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
            try:
                full_ddx, full_ddy, n_tps = _tps_displacement_field(
                    inlier_fixed_pts_clip, inlier_moving_pts_clip, fixed_image, fixed_clip, fixed_transform,
                    moving_native_transform, moving_native_w, moving_native_h, moving_col_off, moving_row_off,
                    tps_max_control_points, tps_coarse_grid_size, rng,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                # A spline needs at least three distinct, non-collinear
                # control points; a handful of inliers at (nearly) one spot
                # is enough for the affine fit above but not for this. Only
                # this model fails -- any other requested model still runs.
                results[warp_model] = RegistrationResult(
                    "tps_fit_failed", None, metrics=metrics,
                    message=f"Could not fit a thin-plate spline to {inlier_count} inliers: {exc}",
                )
                continue

        # Averaged over valid (non-nodata) moving pixels in the overlap only:
        # the field is extrapolated over nodata too (e.g. an AOI clip covering
        # only a corner of its raster), where it can be large but moves nothing.
        row_s, row_e = moving_row_off, min(moving_row_off + moving_clip.shape[1], moving_native_h)
        col_s, col_e = moving_col_off, min(moving_col_off + moving_clip.shape[2], moving_native_w)
        valid = ~np.isnan(moving_native.isel(band=moving_bands[0]).values[row_s:row_e, col_s:col_e])
        if valid.any():
            metrics['avg_dx_px'] = float(np.mean(full_ddx[row_s:row_e, col_s:col_e][valid]))
            metrics['avg_dy_px'] = float(np.mean(full_ddy[row_s:row_e, col_s:col_e][valid]))
        del valid

        # From here on only the remap form of the field is needed: convert in
        # place rather than holding field and maps side by side.
        map_x, map_y = tps_warp.remap_maps(full_ddx, full_ddy, in_place=True)
        del full_ddx, full_ddy

        registered = None
        if return_registered:
            registered = xr.DataArray(
                np.stack(list(tps_warp.iter_remapped_bands(moving_native, map_x, map_y, cv2_interpolation))),
                coords={"band": np.arange(1, moving_native.shape[0] + 1), "y": moving_native.y, "x": moving_native.x},
                dims=("band", "y", "x"),
            ).rio.write_crs(moving_native.rio.crs)
            registered.attrs.update(moving_native.attrs)

        # After-alignment metrics: the matched bands, warped, over the same
        # window as moving_clip (the strict overlap), with the moving validity
        # mask warped alongside them. One band at a time: calling the metrics
        # once per band with the same rng, in band order, gives exactly the
        # results of one call over all bands.
        clip_h, clip_w = moving_clip.shape[1], moving_clip.shape[2]
        window = (slice(moving_row_off, moving_row_off + clip_h), slice(moving_col_off, moving_col_off + clip_w))
        warped_invalid = None
        if moving_mask is not None:
            mask_image = xr.DataArray(
                np.where(moving_mask, np.float32(1.0), np.float32(np.nan))[None],
                coords={"band": [1], "y": moving_native.y, "x": moving_native.x}, dims=("band", "y", "x"),
            )
            warped_invalid = np.isnan(next(tps_warp.iter_remapped_bands(mask_image, map_x, map_y, cv2.INTER_NEAREST))[window])
            del mask_image
        # Every band is warped with the same (map_x, map_y). With
        # on_warped_band, all bands are warped here, once, in band order: the
        # matched ones feed the metrics and every one is then handed to the
        # caller. Otherwise only the matched bands are needed.
        if registered is not None:
            warped_bands = ((b, registered.values[b]) for b in moving_bands)
        elif on_warped_band is not None:
            warped_bands = enumerate(tps_warp.iter_remapped_bands(moving_native, map_x, map_y, cv2_interpolation))
        else:
            warped_bands = zip(moving_bands, tps_warp.iter_remapped_bands(
                _select_bands(moving_native, moving_bands), map_x, map_y, cv2_interpolation))
        # The metrics' crop is masked in place below, so it must be a copy
        # whenever the warped band itself lives on (in `registered`, or with
        # the caller of on_warped_band).
        copy_crop = registered is not None or (on_warped_band is not None and warped_invalid is not None)
        matched_position = {band: k for k, band in enumerate(moving_bands)}
        metrics_after = {}
        for band, warped in warped_bands:
            k = matched_position.get(band)
            if k is not None:
                cropped = warped[window].copy() if copy_crop else warped[window]
                if warped_invalid is not None:
                    cropped[warped_invalid] = np.nan
                band_clip = xr.DataArray(
                    cropped[None], coords={"band": [1], "y": moving_clip.y, "x": moving_clip.x}, dims=("band", "y", "x"),
                ).rio.write_crs(moving_native.rio.crs)
                one = band_metrics.compute_band_metrics_matched_resolution(band_clip, fixed_overlap_clip, [(0, k)], rng)
                del band_clip, cropped
                for key, val in _relabel_band_metrics(one, [matched_band_pairs[k]]).items():
                    if isinstance(val, dict):
                        metrics_after.setdefault(key, {}).update(val)
                    else:
                        metrics_after[key] = val
            if on_warped_band is not None:
                on_warped_band(band, warped)
            del warped
        del warped_invalid
        band_metrics.store_band_metrics(metrics, metrics_after, matched_band_pairs, "after")

        diagnostics = {
            "warp_model": warp_model,
            "num_step1_channels_used": n_channels,
            "coarse_ransac_inliers": int(np.sum(mask_coarse)) if mask_coarse is not None else 0,
            "guided_ransac_inliers": inlier_count,
            "n_tps_control_points": n_tps,
            "affine_matrix_clip": M_part,
            "affine_matrix_native": affine_native[:2].copy(),
            # All-band clips with the same geometry as the matching clips (so
            # the clip-pixel keypoint coordinates below apply to them). Lazy
            # for file-backed inputs: nothing is read unless a caller uses them.
            "moving_clip": _clip_box_view(moving_native, left, bottom, right, top),
            "fixed_clip": _clip_box_view(
                fixed_image, left - fixed_buffer_m, bottom - fixed_buffer_m, right + fixed_buffer_m, top + fixed_buffer_m,
            ),
            "inlier_moving_pts_clip": inlier_moving_pts_clip,
            "inlier_fixed_pts_clip": inlier_fixed_pts_clip,
            "band_metrics_before": metrics_before,
            "band_metrics_after": metrics_after,
        }

        results[warp_model] = RegistrationResult(
            "ok", registered, metrics=metrics, diagnostics=diagnostics, remap=(map_x, map_y),
        )
        del map_x, map_y, registered
    return results


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
    moving_valid_mask=None,
    fixed_valid_mask=None,
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

    `moving_valid_mask` / `fixed_valid_mask`: as in register_local --
    pixels excluded from matching and metrics, never from the output.
    """
    matched_band_pairs = _validate_band_map(band_map)
    rng = np.random.default_rng(seed=seed)
    # FLANN's randomized kd-trees draw from OpenCV's process-wide RNG, so
    # without this a result depends on what the process registered before.
    # (OpenCV treats seed 0 as its default initial state.)
    cv2.setRNGSeed(seed)

    moving_image = _load_raster(moving)
    fixed_image = _load_raster(fixed)

    if moving_image.shape[0] != len(band_map):
        return RegistrationResult(
            "band_count_mismatch", None,
            message=f"band_map has {len(band_map)} entries but moving image has {moving_image.shape[0]} bands.",
        )
    moving_mask = _validate_mask(moving_valid_mask, moving_image, "moving_valid_mask")
    fixed_mask = _validate_mask(fixed_valid_mask, fixed_image, "fixed_valid_mask")

    working_fixed = fixed_image
    if fixed_image.rio.crs != moving_image.rio.crs:
        working_fixed = fixed_image.rio.reproject(moving_image.rio.crs)
        if fixed_mask is not None:
            fixed_mask = _reproject_mask(fixed_mask, fixed_image, working_fixed)

    overlap = geometry.find_overlap(moving_image, working_fixed)
    if overlap is None:
        return RegistrationResult("no_overlap", None, message="No geographic overlap between moving and fixed.")
    left, bottom, right, top = overlap

    moving_clip = _apply_mask_to_clip(
        moving_image.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top).load(), moving_image, moving_mask
    )
    fixed_clip = _apply_mask_to_clip(
        working_fixed.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top).load(), working_fixed, fixed_mask
    )

    moving_valid = int(np.count_nonzero(~np.isnan(moving_clip.values)))
    fixed_valid = int(np.count_nonzero(~np.isnan(fixed_clip.values)))
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
    for _, moving_nd, fixed_nd in nd_index.iter_nd_index_pairs(moving_band_arrays, fixed_band_arrays):
        moving_norm, fixed_norm = nd_index.normalize_pair(moving_nd, fixed_nd)
        channel_moving_valid = ~np.isnan(moving_nd)
        channel_fixed_valid = ~np.isnan(fixed_nd)
        del moving_nd, fixed_nd

        moving_kp, moving_desc, moving_scale = sift_matching.detect_channel_keypoints(
            moving_norm, channel_moving_valid, blur_kernel=moving_blur_kernel,
            mask_margin_px=mask_erosion_blur_multiple * moving_blur_kernel,
            max_sift_dimension=max_sift_dimension, sift=sift,
        )
        fixed_kp, fixed_desc, fixed_scale = sift_matching.detect_channel_keypoints(
            fixed_norm, channel_fixed_valid, blur_kernel=fixed_blur_kernel,
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

    registered_mask = None
    if moving_mask is not None:
        mask_image = xr.DataArray(
            np.where(moving_mask, 1.0, np.nan).astype(np.float32)[None],
            coords={"band": [1], "y": moving_image.y, "x": moving_image.x}, dims=("band", "y", "x"),
        ).rio.write_crs(moving_image.rio.crs)
        registered_mask = ~np.isnan(affine_warp.warp_onto_fixed_grid(mask_image, M_full, working_fixed).values[0])
    registered_clip = _apply_mask_to_clip(
        registered.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top), registered, registered_mask
    )
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
