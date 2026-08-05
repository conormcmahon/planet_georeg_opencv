"""Generic multi-sensor image registration using OpenCV feature matching and a
thin-plate-spline (TPS) local warp.

This is a generalized version of planetscope_registration_local.py. Unlike that
script, it makes no assumptions about sensor type, band count, band order,
spatial resolution, or coordinate reference system (CRS) of either image. It
is intended to register one or more "source" rasters onto a single, trusted
"target" raster, even when the two come from entirely different sensors
(e.g. PlanetScope source registered onto a NAIP target).

Matching happens in two sequential steps per source file:
    Step 1 (coarse): SIFT keypoints/descriptors are detected per ND-index
        channel and matched with an unguided FLANN k=2 + Lowe-ratio-test +
        loose pixel-distance search, then RANSAC fits a coarse affine
        transform. This step produces no output raster — it exists only to
        seed step 2.
    Step 2 (guided): the SAME per-channel keypoints/descriptors from step 1
        are re-matched via guided_match_per_descriptor_radius, which builds
        a KD-tree over predicted target positions and, per target
        descriptor, restricts candidates to source keypoints within
        keypoint_match_distance_threshold_m real-world meters (converted to
        target clip-pixels via that image's own resolution) of where step 1's
        coarse affine predicts a given source keypoint should fall — geometry is
        filtered first, and descriptor similarity (Lowe's ratio test) is
        only checked within that surviving geometric neighborhood. RANSAC,
        the TPS warp, and the final registered output are all built from
        this refined match set.

Inputs (see "Settings" section below):
    source_directory  - directory containing one or more source *.tif rasters
                         to be registered onto the target.
    target_filepath   - path to the single, well-geolocated target raster
                         that source rasters are registered onto.
    output_directory  - directory where registered rasters, diagnostic plots,
                         and an alignment-metrics CSV are written.
    band_map          - list of integers, one per source band (0-indexed).
                         band_map[i] is the 0-indexed target band that
                         corresponds to source band i, or -1 if source band i
                         has no corresponding band in the target image. At
                         least two source bands must have a match so that a
                         normalized-difference index can be computed.
    resampling_method - one of "nearest", "bilinear", "cubic", "average".
                         Controls only the final reprojected output raster:
                         the CRS reprojection to source_native (via rasterio)
                         and the final TPS pixel warp (via OpenCV). It does
                         NOT affect the R^2 / RMSE metrics — see "Design
                         notes" below.
    display_bands     - list of 3 0-indexed SOURCE band indices rendered as
                         (R, G, B) in the diagnostic PNGs under
                         output_plots/. Each must have a match in band_map.
                         Defaults to the first three bands; set to actual
                         red/green/blue band indices for true-color display.

Outputs (written under output_directory):
    registered/          - one registered GeoTIFF per source file, on a grid
                            that keeps the source's approximate native
                            resolution (reprojected into the target's CRS).
    output_plots/         - diagnostic PNGs per source file: a band-composite
                            comparison, all Lowe-threshold matches, RANSAC
                            inlier keypoints, inlier correspondence lines,
                            and (before/after) per-band spatial inlier/
                            outlier masks for the rescaling regression.
    alignment_metrics/    - a CSV log with one row per source file, giving
                            match/inlier counts for both step 1's coarse
                            global search (num_step1_ransac_inliers,
                            num_step1_good_matches, num_step1_raw_matches)
                            and step 2's guided local search
                            (num_ransac_inliers, num_good_matches,
                            num_raw_matches), plus, per band, R^2, RMSE, and the
                            RANSAC regression scale/intercept (source ~=
                            scale * target + intercept), each computed twice
                            — once using only RANSAC inliers, once using all
                            sampled pixels — plus the inlier fraction, before
                            and after alignment. See "Design notes" below.

Design notes on the two reprojected-source versions:
    Per source file, the source raster is reprojected into the target's CRS
    in two distinct ways:
      - source_native: reprojected to the target CRS but resampled at
        approximately the source's own native resolution, using
        `resampling_method`. This preserves source detail and is the raster
        that keypoint matching, the TPS warp, and the final registered
        output are all built from.
      - For R^2 / RMSE / regression only: whichever of source_native /
        target is the FINER resolution is resampled onto the other's
        (coarser) pixel grid — with area averaging if it is at least twice
        as fine, otherwise nearest-neighbor — regardless of
        `resampling_method`. Rather than materializing this as a full
        raster (which, via GDAL reprojection, can leave a multi-gigabyte,
        not-promptly-reclaimed memory footprint when one image is tens of
        megapixels), it is implemented as point-sampling: a bounded random
        sample of coarse-grid pixels is drawn and the finer image is
        resampled only at those points, so memory stays bounded regardless
        of image size.

Design notes on the robust (RANSAC) regression:
    Per band, a RANSAC linear regression fits source ~= scale*target +
    intercept (predicting source values from target, so the fitted scale
    maps target pixel values onto the source's radiometric scale). A pixel
    pair counts as an inlier if |source - (scale*target + intercept)| is
    less than 10% of the source band's own spatial standard deviation. This
    rejects pairs affected by real land-cover change between acquisition
    dates, clouds, or residual misregistration, which would otherwise
    distort a plain least-squares fit. R^2, RMSE, scale, and intercept are
    reported for both the inlier set and the full sample (see
    alignment_metrics.csv columns above), and the RANSAC-inlier fit is also
    used to rescale the target for the output_plots/ diagnostic PNGs (never
    applied to the registered GeoTIFF).

Ground-scale-aware processing:
    The Gaussian blur applied before SIFT is sized per image (not shared
    between source and target) so it covers approximately the same
    real-world ground footprint regardless of each image's resolution — see
    blur_kernel_base_size / blur_kernel_reference_resolution below.
"""

import rioxarray as rxr
import xarray as xr
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os
import glob
import gc
from itertools import combinations
from rasterio.enums import Resampling
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.spatial import cKDTree

# --- Settings ---
min_pixel_count = 1000        # minimum valid pixels in overlap region to attempt registration
min_keypoints = 10            # minimum keypoints per index channel to attempt matching
lowe_ratio_threshold = 1000   # Lowe's ratio test threshold
distance_threshold_pixels = 50  # max allowed pixel-space distance between matched keypoints
                                 # (evaluated in source clip-pixel units; see scale_x/scale_y below)
ransac_reproj_threshold = 2.0   # RANSAC reprojection error threshold (source clip pixels)

# --- Step 2: guided re-matching settings ---
# Step 1 detects keypoints/descriptors and does an unguided match to get a
# coarse affine transform (no output raster is written from step 1). Step 2
# reuses those same keypoints/descriptors and re-matches them via
# guided_match_per_descriptor_radius: a KD-tree spatial query restricts each
# target descriptor's candidates to source keypoints within
# keypoint_match_distance_threshold_m real-world meters of where step 1's
# coarse affine predicts they should fall, BEFORE any descriptor-similarity
# comparison; this refined match set is what RANSAC, the TPS warp, and the
# final registered output are built from. A fixed real-world distance (rather
# than a fixed pixel count) keeps the guided search radius comparable across
# source/target pairs with different resolutions; it is converted to target
# clip-pixel space per-file using that file's own pixel size (tgt_res_x)
# immediately before step 2 runs.
keypoint_match_distance_threshold_m = 15.0  # meters, real-world ground distance

# Gaussian blur kernel applied before SIFT, expressed as a pixel size at
# blur_kernel_reference_resolution (real-world units matching the target/
# source CRS, e.g. meters for UTM). Each image's actual kernel size is
# scaled by blur_kernel_reference_resolution / that image's own pixel size,
# rounded to the nearest odd integer, so the blur approximates the same
# real-world ground footprint regardless of the image's resolution (e.g. a
# 3 px kernel at 3 m becomes ~15 px at 0.6 m).
blur_kernel_base_size = 3
blur_kernel_reference_resolution = 3.0

# Nodata/cloud-mask handling before SIFT:
# Each ND-index channel's own valid-pixel mask (wherever neither contributing
# band is NaN) is used two ways: (1) the pre-SIFT Gaussian blur is a masked/
# normalized convolution (blur(data*mask)/blur(mask)) so invalid pixels never
# bleed a false zero-value edge into nearby valid pixels the way a plain blur
# of a NaN-to-zero-filled image would; (2) the mask is eroded by
# mask_erosion_blur_multiple * that image's own blur kernel size and passed
# to SIFT's detectAndCompute mask argument, so keypoints (and their
# descriptor support window) are never placed inside or near a hole or scene
# edge in the first place. The margin scales with the blur kernel (itself
# already scaled to a fixed real-world footprint) so it stays proportionate
# across resolutions. Larger values are more conservative (fewer keypoints
# survive near holes/edges).
mask_erosion_blur_multiple = 2

max_sift_dimension = 4000     # cap each ND-index channel's larger side before SIFT; clips
                               # larger than this (e.g. a high-resolution target) are downsampled
                               # for detection only, and matched keypoint coordinates are rescaled
                               # back to true clip-pixel space immediately after matching

# --- Local warp settings ---
# Maximum number of RANSAC inliers to use as control points for the TPS warp.
# When there are more inliers than this threshold, the excess is removed via
# spatial subsampling: dense clusters are thinned preferentially while isolated
# points in sparse regions are always retained.
# - Larger values → denser, more faithful warp field; slower RBF solve (O(N³))
# - Smaller values → coarser warp; faster solve
# Set to None to use all RANSAC inliers without any subsampling.
tps_max_control_points = 2000

# Resolution (in pixels, per axis) of the coarse grid on which the RBF
# displacement field is first evaluated. The full-resolution field is then
# obtained by upsampling with cv2.resize. Reducing this trades accuracy for
# speed; 200-400 is a practical range for 3-4K imagery.
tps_coarse_grid_size = 300

# Output plot parameters
max_lines_match_lines = 2000
max_lowe_match_display_points = 2000  # cap on points shown in the "all Lowe matches" plot

source_directory  = "/path/to/source_images/"   # rasters to be registered
target_filepath   = "/path/to/target.tif"        # well-geolocated target raster
output_directory  = "/path/to/output/"

# band_map[i] = 0-indexed target band matching 0-indexed source band i, or -1
# if source band i has no match in the target image.
band_map = [0, 1, 2, 3]

# 0-indexed SOURCE band indices rendered as (R, G, B) in the diagnostic PNGs
# under output_plots/. Each must have a match in band_map (its corresponding
# target band is looked up automatically). Defaults to the first three bands;
# set to actual red/green/blue band indices for true-color display.
display_bands = [0, 1, 2]

# Percentile (from each end) used to clip outlier pixels when contrast-
# stretching the output_plots/ band-composite images, e.g. 2.0 uses the
# 2nd-98th percentile range. A handful of extreme pixels (sun glint, sensor
# noise) would otherwise dominate a raw min/max stretch.
display_percentile_clip = 2.0

# One of: "nearest", "bilinear", "cubic", "average".
resampling_method = "bilinear"

_RASTERIO_RESAMPLING = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "average":  Resampling.average,
}
_CV2_INTERPOLATION = {
    "nearest":  cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "cubic":    cv2.INTER_CUBIC,
    "average":  cv2.INTER_AREA,
}


def get_matched_band_pairs(band_map):
    """Return [(source_band_idx, target_band_idx), ...] for all matched bands."""
    return [(i, t) for i, t in enumerate(band_map) if t != -1]


def extract_band_arrays(image, band_indices):
    """Return {band_idx: float32 numpy array} for the given 0-indexed bands."""
    return {
        idx: image.isel(band=idx).values.astype(np.float32)
        for idx in band_indices
    }


def compute_nd_indices(band_arrays):
    """
    Compute normalized-difference indices for every pair of bands in
    band_arrays, keyed by (band_idx_a, band_idx_b) with a < b.

    Using every combination rather than a fixed set of named indices avoids
    assuming which bands (if any) are present, while still producing diverse
    texture signals for feature matching.
    """
    eps = 1e-6
    keys = sorted(band_arrays)
    out = {}
    for a, b in combinations(keys, 2):
        out[(a, b)] = (band_arrays[a] - band_arrays[b]) / (band_arrays[a] + band_arrays[b] + eps)
    return out


def normalize_pair(arr1, arr2, percentile_clip=None):
    """
    Jointly normalize two float arrays to [0, 1] using their combined range.

    By default (percentile_clip=None) uses the exact combined min/max,
    matching the ND-index feature-matching pipeline's established behavior.
    If percentile_clip is given (e.g. 2.0), the shared display range is
    instead the union of each array's OWN [percentile_clip, 100 -
    percentile_clip] range (not the percentile of the two arrays pooled
    together, which lets whichever array has more pixels or a wider native
    spread dominate and crush the other's contrast to near-zero — e.g. an
    8-bit sensor pooled with a higher-dynamic-range one). Values outside the
    final range are clipped to [0, 1] — used for display composites, where
    a handful of extreme pixels (sun glint, sensor noise) would otherwise
    dominate a raw min/max stretch.
    """
    if percentile_clip is None:
        lo = min(np.nanmin(arr1), np.nanmin(arr2))
        hi = max(np.nanmax(arr1), np.nanmax(arr2))
    else:
        lo1, hi1 = np.nanpercentile(arr1, [percentile_clip, 100 - percentile_clip])
        lo2, hi2 = np.nanpercentile(arr2, [percentile_clip, 100 - percentile_clip])
        lo, hi = min(lo1, lo2), max(hi1, hi2)
    if not (hi > lo):
        return np.zeros_like(arr1), np.zeros_like(arr2)
    out1, out2 = (arr1 - lo) / (hi - lo), (arr2 - lo) / (hi - lo)
    if percentile_clip is not None:
        out1, out2 = np.clip(out1, 0.0, 1.0), np.clip(out2, 0.0, 1.0)
    return out1, out2


def normalize_single(arr, percentile_clip=None):
    """
    Normalize one float array to [0, 1] using its own range.

    Used for band-composite display images as a fallback when a joint
    (regression-calibrated) stretch isn't available: unlike ND indices
    (already unitless ratios in roughly [-1, 1]), raw band values from
    different sensors can have wildly different native scales, so each
    image needs its own contrast stretch. See normalize_pair for
    percentile_clip.
    """
    if percentile_clip is None:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    else:
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.zeros_like(arr)
        lo, hi = np.percentile(valid, [percentile_clip, 100 - percentile_clip])
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0) if percentile_clip is not None else out


def cap_for_sift(img_u8, max_dim, interpolation=cv2.INTER_AREA):
    """
    Downsample img_u8 (if needed) so its larger side is at most max_dim.
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


def resolution_scaled_kernel_size(base_size, base_resolution, image_resolution):
    """
    Scale a Gaussian blur kernel size so it covers approximately the same
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


def to_uint8(arr):
    """Convert a float [0, 1] array to uint8 [0, 255], mapping NaN to 0."""
    return np.nan_to_num(
        np.round(np.clip(arr * 255, 0, 255)), nan=0.0
    ).astype(np.uint8)


def masked_gaussian_blur(img_u8, valid_mask, ksize):
    """
    Gaussian-blur img_u8 using normalized convolution — blur(data*mask) /
    blur(mask) — so that invalid (valid_mask == False) pixels never bleed
    their arbitrary zero-filled value into neighboring valid pixels.

    A plain blur of a NaN-to-zero-filled image creates a false, artificially
    dark edge around every nodata/cloud-mask hole and scene boundary, which
    SIFT tends to lock onto as if it were real texture. This keeps blurred
    values at the edge of a hole a legitimate average of only the nearby
    valid pixels. Pixels with no valid support in their blur footprint
    (blurred mask ~ 0, i.e. deep inside a large hole) fall back to 0 — such
    pixels are excluded from detection anyway via the eroded mask passed to
    SIFT separately (see erode_valid_mask).
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


def erode_valid_mask(valid_mask, margin_px):
    """
    Erode a boolean valid-data mask so only pixels at least margin_px away
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


def ransac_linear_fit(x, y, threshold, rng, n_iterations=200, min_inliers=2):
    """
    Robust linear regression y ~= slope*x + intercept via RANSAC.

    Repeatedly fits a candidate line through two random points and keeps the
    one with the most inliers (|residual| < threshold); the final slope/
    intercept are then an ordinary-least-squares refit on that best inlier
    set, and the inlier mask is recomputed against the refit line.

    Returns (slope, intercept, inlier_mask). slope/intercept are NaN and
    inlier_mask is all False if no valid candidate model was found.
    """
    n = len(x)
    best_inliers = None
    best_count = -1
    for _ in range(n_iterations):
        i, j = rng.choice(n, 2, replace=False)
        if x[i] == x[j]:
            continue
        cand_slope = (y[j] - y[i]) / (x[j] - x[i])
        cand_intercept = y[i] - cand_slope * x[i]
        count = int(np.sum(np.abs(y - (cand_slope * x + cand_intercept)) < threshold))
        if count > best_count:
            best_count = count
            best_inliers = np.abs(y - (cand_slope * x + cand_intercept)) < threshold

    if best_inliers is None or best_count < min_inliers:
        return np.nan, np.nan, np.zeros(n, dtype=bool)

    slope, intercept = np.polyfit(x[best_inliers], y[best_inliers], 1)
    inlier_mask = np.abs(y - (slope * x + intercept)) < threshold
    return float(slope), float(intercept), inlier_mask


def compute_band_metrics_matched_resolution(src_clip, tgt_clip, matched_pairs, rng,
                                              max_sample_points=200_000,
                                              ransac_iterations=200,
                                              ransac_threshold_frac=0.2):
    """
    Estimate per-band agreement between src_clip and tgt_clip, robust to
    outlier pixels (e.g. real land-cover change between acquisition dates,
    clouds, or residual misregistration).

    src_clip and tgt_clip already share a CRS but may differ in pixel
    resolution. Whichever is finer is resampled onto the other (coarser)
    image's pixel grid: with area averaging if it is at least twice as fine,
    otherwise with nearest-neighbor sampling. This direction/method choice is
    automatic and independent of `resampling_method`, which only controls
    the final reprojected output raster.

    Rather than materializing a full resampled raster (which, via GDAL
    reprojection, can leave a multi-gigabyte, not-promptly-reclaimed memory
    footprint when one image is tens of megapixels — confirmed by profiling
    this pipeline against a 0.6 m target overlapping a 3 m source), this
    draws a bounded random sample of valid coarse-grid pixels and resamples
    only at those points, so memory stays bounded regardless of image size.

    Per band, a RANSAC linear regression source ~= scale*target + intercept
    is fit (predicting source values FROM target, so the fitted scale maps
    target pixel values onto the source's radiometric scale — used
    elsewhere to rescale the target for display). The inlier threshold is
    10% of the source band's own spatial standard deviation (computed over
    all of src_clip, not just the sample). R^2, RMSE, scale, and intercept
    are reported twice per band: once using only RANSAC inliers, and once
    using the full sample (plain OLS, no outlier rejection) — plus the
    fraction of the full sample classified as inliers.

    Also records, per band, the spatial sample locations and their inlier/
    outlier classification (mask_rows, mask_cols, mask_inlier — pixel row/
    col in the coarser image's clip grid, and a matching bool array), so
    callers can render a spatial inlier/outlier mask.

    Returns a dict of dicts, each keyed by source band index:
        {'r2_inliers', 'r2_all', 'rmse_inliers', 'rmse_all',
         'scale_inliers', 'scale_all', 'intercept_inliers', 'intercept_all',
         'inlier_frac', 'mask_rows', 'mask_cols', 'mask_inlier'}
    plus two call-level (not per-band) keys: 'coarse_shape' (the (h, w) of
    the grid mask_rows/mask_cols are expressed in) and 'coarse_is_source'
    (whether that grid is src_clip's, as opposed to tgt_clip's).
    """
    src_transform = src_clip.rio.transform()
    tgt_transform = tgt_clip.rio.transform()
    src_res = abs(src_transform.a)
    tgt_res = abs(tgt_transform.a)

    src_is_finer = src_res < tgt_res
    if src_is_finer:
        fine_clip, coarse_clip = src_clip, tgt_clip
        fine_transform, coarse_transform = src_transform, tgt_transform
        fine_res, coarse_res = src_res, tgt_res
    else:
        fine_clip, coarse_clip = tgt_clip, src_clip
        fine_transform, coarse_transform = tgt_transform, src_transform
        fine_res, coarse_res = tgt_res, src_res

    ratio = coarse_res / fine_res if fine_res > 0 else 1.0
    use_average = ratio >= 2.0
    box_size = max(1, int(round(ratio)))
    inv_fine_transform = ~fine_transform

    stat_names = ('r2_inliers', 'r2_all', 'rmse_inliers', 'rmse_all',
                  'scale_inliers', 'scale_all', 'intercept_inliers', 'intercept_all',
                  'inlier_frac')
    out = {name: {} for name in stat_names}
    out['mask_rows'], out['mask_cols'], out['mask_inlier'] = {}, {}, {}
    out['coarse_shape'] = coarse_clip.shape[1:]
    out['coarse_is_source'] = not src_is_finer

    def _set_nan(src_idx):
        for name in stat_names:
            out[name][src_idx] = np.nan
        out['mask_rows'][src_idx] = np.array([], dtype=np.int64)
        out['mask_cols'][src_idx] = np.array([], dtype=np.int64)
        out['mask_inlier'][src_idx] = np.array([], dtype=bool)

    for src_idx, tgt_idx in matched_pairs:
        fine_idx, coarse_idx = (src_idx, tgt_idx) if src_is_finer else (tgt_idx, src_idx)

        coarse_band = coarse_clip.isel(band=coarse_idx).values.astype(np.float32)
        rows, cols = np.where(~np.isnan(coarse_band))
        if len(rows) > max_sample_points:
            sel = rng.choice(len(rows), max_sample_points, replace=False)
            rows, cols = rows[sel], cols[sel]
        if len(rows) < 2:
            _set_nan(src_idx)
            continue

        world_x, world_y = coarse_transform * (cols.astype(np.float64), rows.astype(np.float64))
        fine_col, fine_row = inv_fine_transform * (world_x, world_y)

        fine_band = fine_clip.isel(band=fine_idx).values.astype(np.float32)
        fine_nan = np.isnan(fine_band)

        if use_average:
            # Local area average, respecting nodata: sum of filled values
            # over a box divided by the fraction of valid pixels in that box.
            filled = np.where(fine_nan, 0.0, fine_band)
            valid_frac = uniform_filter((~fine_nan).astype(np.float32), size=box_size,
                                         mode='constant', cval=0.0)
            mean_filled = uniform_filter(filled, size=box_size, mode='constant', cval=0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                fine_band_resampled = np.where(valid_frac > 0, mean_filled / valid_frac, np.nan)
        else:
            fine_band_resampled = fine_band

        resampled_nan = np.isnan(fine_band_resampled)
        sampled = map_coordinates(
            np.where(resampled_nan, 0.0, fine_band_resampled), [fine_row, fine_col],
            order=0, mode='constant', cval=0.0
        )
        sampled_nan = map_coordinates(
            resampled_nan.astype(np.float32), [fine_row, fine_col],
            order=0, mode='constant', cval=1.0
        ) > 0.5
        sampled[sampled_nan] = np.nan

        coarse_vals = coarse_band[rows, cols]
        s_vals, t_vals = (sampled, coarse_vals) if src_is_finer else (coarse_vals, sampled)

        valid = ~(np.isnan(s_vals) | np.isnan(t_vals))
        if valid.sum() < 2:
            _set_nan(src_idx)
            continue
        s, t = s_vals[valid], t_vals[valid]

        # --- Stats over all sampled pixels (no outlier rejection) ---
        out['rmse_all'][src_idx] = float(np.sqrt(np.mean((s - t) ** 2)))
        corr = np.corrcoef(s, t)[0, 1]
        out['r2_all'][src_idx] = float(corr ** 2) if np.isfinite(corr) else np.nan
        if np.std(t) > 0:
            _slope, _intercept = np.polyfit(t, s, 1)  # source ~= slope*target + intercept
            out['scale_all'][src_idx], out['intercept_all'][src_idx] = float(_slope), float(_intercept)
        else:
            out['scale_all'][src_idx], out['intercept_all'][src_idx] = np.nan, np.nan

        # --- RANSAC: robust source ~= scale*target + intercept, with inlier stats ---
        threshold = ransac_threshold_frac * float(
            np.nanstd(src_clip.isel(band=src_idx).values.astype(np.float32))
        )
        _, _, inlier_mask = ransac_linear_fit(t, s, threshold, rng, n_iterations=ransac_iterations)
        out['inlier_frac'][src_idx] = float(inlier_mask.mean()) if len(inlier_mask) else np.nan
        out['mask_rows'][src_idx] = rows[valid]
        out['mask_cols'][src_idx] = cols[valid]
        out['mask_inlier'][src_idx] = inlier_mask

        if inlier_mask.sum() >= 2:
            s_in, t_in = s[inlier_mask], t[inlier_mask]
            out['rmse_inliers'][src_idx] = float(np.sqrt(np.mean((s_in - t_in) ** 2)))
            corr_in = np.corrcoef(s_in, t_in)[0, 1]
            out['r2_inliers'][src_idx] = float(corr_in ** 2) if np.isfinite(corr_in) else np.nan
            if np.std(t_in) > 0:
                _slope_in, _intercept_in = np.polyfit(t_in, s_in, 1)
                out['scale_inliers'][src_idx] = float(_slope_in)
                out['intercept_inliers'][src_idx] = float(_intercept_in)
            else:
                out['scale_inliers'][src_idx], out['intercept_inliers'][src_idx] = np.nan, np.nan
        else:
            out['r2_inliers'][src_idx], out['rmse_inliers'][src_idx] = np.nan, np.nan
            out['scale_inliers'][src_idx], out['intercept_inliers'][src_idx] = np.nan, np.nan

    return out


def find_overlap(image1, image2):
    """
    Return (left, bottom, right, top) geographic bounding box of the overlap
    between two images, or None if they do not overlap.

    Both images must be in the same CRS before calling this function.
    """
    b1 = image1.rio.bounds()  # (left, bottom, right, top)
    b2 = image2.rio.bounds()
    left   = max(b1[0], b2[0])
    bottom = max(b1[1], b2[1])
    right  = min(b1[2], b2[2])
    top    = min(b1[3], b2[3])
    if left >= right or bottom >= top:
        return None
    return left, bottom, right, top


def get_clip_offset(full_image, clip_image):
    """
    Return the (col, row) pixel offset of clip_image's origin within full_image.

    Uses nearest-neighbor lookup on the coordinate arrays so it is robust to
    floating-point rounding from rio.clip_box.
    """
    col = int(np.argmin(np.abs(full_image.x.values - clip_image.x.values[0])))
    row = int(np.argmin(np.abs(full_image.y.values - clip_image.y.values[0])))
    return col, row


def spatially_thin_keypoints(tgt_pts, src_pts, max_control_points, clip_w, clip_h, rng):
    """
    Subsample matched keypoint pairs to at most `max_control_points` using a
    regular spatial grid over the target clip extent.

    The image overlap is divided into a grid of cells whose size is chosen so
    that the expected number of cells approximately equals `max_control_points`.
    Within each occupied cell exactly one keypoint is kept (chosen at random).
    Cells with only one occupant are always retained; cells with many occupants
    (dense clusters) contribute only one representative.

    `tgt_pts` should be in clip-space pixel coordinates used to assign cells.
    `src_pts` can be in any coordinate space — they are subsampled in lock-step.
    """
    n = len(tgt_pts)
    if max_control_points is None or n <= max_control_points:
        return tgt_pts, src_pts

    # Cell size chosen so that clip_area / cell_area ≈ max_control_points.
    cell_area = (clip_w * clip_h) / max_control_points
    cell_size = max(1, int(np.sqrt(cell_area)))

    # Assign each target keypoint to a grid cell.
    cell_map = {}
    for i in range(n):
        cell_col = int(tgt_pts[i, 0] / cell_size)
        cell_row = int(tgt_pts[i, 1] / cell_size)
        key = (cell_row, cell_col)
        if key not in cell_map:
            cell_map[key] = []
        cell_map[key].append(i)

    # From each occupied cell keep one randomly-chosen point.
    kept = [int(rng.choice(indices)) for indices in cell_map.values()]
    kept = np.array(sorted(kept))
    return tgt_pts[kept], src_pts[kept]


def keypoints_to_true_pixels(kp_list, sift_scale):
    """Convert SIFT keypoints (in detection-image pixels) to true clip-pixel coordinates."""
    if len(kp_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.float32([kp.pt for kp in kp_list]) / sift_scale


def guided_match_per_descriptor_radius(src_kp, src_desc, tgt_kp, tgt_desc, src_sift_scale, tgt_sift_scale,
                                        M_inv, distance_threshold, lowe_ratio_threshold):
    """
    Step-2 guided matching, approach 2: build a KD-tree over each source
    keypoint's step-1-predicted target position (M_inv applied to it), then
    for each target descriptor individually, query the tree for source
    keypoints predicted to fall within distance_threshold pixels and compare
    descriptor distances only within that geometric subset. Lowe's ratio
    test is applied to the two closest descriptors in the subset; a subset
    of size 1 is accepted without a ratio test.

    Returns (src_pts, tgt_pts): lists of matched (x, y) pairs in true
    clip-pixel coordinates.
    """
    src_pts_true = keypoints_to_true_pixels(src_kp, src_sift_scale)
    tgt_pts_true = keypoints_to_true_pixels(tgt_kp, tgt_sift_scale)
    pred_tgt_pts = cv2.transform(src_pts_true.reshape(-1, 1, 2), M_inv).reshape(-1, 2)

    tree = cKDTree(pred_tgt_pts)

    out_src_pts, out_tgt_pts = [], []
    for tgt_i in range(len(tgt_kp)):
        tgt_pt = tgt_pts_true[tgt_i]
        candidate_idx = tree.query_ball_point(tgt_pt, r=distance_threshold)
        if not candidate_idx:
            continue
        cand_desc = src_desc[candidate_idx]
        dists = np.linalg.norm(cand_desc - tgt_desc[tgt_i][None, :], axis=1)
        order = np.argsort(dists)
        if len(order) == 1:
            best_src_idx = candidate_idx[order[0]]
        else:
            if dists[order[0]] >= lowe_ratio_threshold * dists[order[1]]:
                continue
            best_src_idx = candidate_idx[order[0]]
        out_src_pts.append((float(src_pts_true[best_src_idx][0]), float(src_pts_true[best_src_idx][1])))
        out_tgt_pts.append((float(tgt_pt[0]), float(tgt_pt[1])))
    return out_src_pts, out_tgt_pts


def build_metric_keys(matched_pairs):
    keys = []
    for when in ("before", "after"):
        for src_idx, _ in matched_pairs:
            for stat in ("r2", "rmse", "scale", "intercept"):
                for subset in ("inliers", "all"):
                    keys.append(f"{stat}_src{src_idx}_{when}_{subset}")
            keys.append(f"inlier_frac_src{src_idx}_{when}")
    keys += ["mean_kp_dist_before", "mean_kp_dist_after", "avg_dx_px", "avg_dy_px"]
    return keys


def store_band_metrics(metrics, result, matched_pairs, when):
    """Copy a compute_band_metrics_matched_resolution() result into `metrics`."""
    for src_idx, _ in matched_pairs:
        for stat in ("r2", "rmse", "scale", "intercept"):
            for subset in ("inliers", "all"):
                metrics[f"{stat}_src{src_idx}_{when}_{subset}"] = result[f"{stat}_{subset}"][src_idx]
        metrics[f"inlier_frac_src{src_idx}_{when}"] = result["inlier_frac"][src_idx]


_RANSAC_MASK_CMAP = ListedColormap(['lightgray', 'red', 'green'])
_RANSAC_MASK_LEGEND = [
    Patch(color='lightgray', label='not sampled'),
    Patch(color='red', label='outlier'),
    Patch(color='green', label='inlier'),
]


def plot_ransac_mask_grid(result, matched_pairs, when, source_filename, out_path):
    """
    Save a PNG with one subplot per matched band, each a binary spatial mask
    of which sampled pixels were RANSAC inliers vs. outliers for the
    source ~= scale*target + intercept rescaling model (see
    compute_band_metrics_matched_resolution). Pixels that weren't part of
    the sample are shown as "not sampled" rather than left blank.
    """
    n = len(matched_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5), squeeze=False)
    coarse_shape = result['coarse_shape']
    for ax, (src_idx, _) in zip(axes[0], matched_pairs):
        mask_img = np.full(coarse_shape, -1, dtype=np.int8)
        rows, cols = result['mask_rows'][src_idx], result['mask_cols'][src_idx]
        if len(rows):
            mask_img[rows, cols] = result['mask_inlier'][src_idx].astype(np.int8)
        frac = result['inlier_frac'][src_idx]
        frac_str = f"{frac:.1%}" if np.isfinite(frac) else "n/a"
        ax.imshow(mask_img + 1, cmap=_RANSAC_MASK_CMAP, vmin=0, vmax=2, interpolation='nearest')
        ax.set_title(f'Band {src_idx} — inliers: {frac_str}')
        ax.axis('off')
    grid_space = 'source' if result['coarse_is_source'] else 'target'
    fig.suptitle(f'RANSAC rescaling model — {when} alignment ({grid_space}-clip pixel space)\n{source_filename}')
    fig.legend(handles=_RANSAC_MASK_LEGEND, loc='lower center', ncol=3)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def write_alignment_metrics(filepath, metric_keys, source_filename, target_filename,
                             num_step1_ransac_inliers, num_step1_good_matches, num_step1_raw_matches,
                             num_ransac, num_good, num_raw, metrics=None):
    """
    Append one row of local-registration quality metrics to the CSV log.

    num_step1_ransac_inliers/num_step1_good_matches/num_step1_raw_matches
    describe step 1's coarse, unguided global search (RANSAC inliers among
    the coarse affine fit, Lowe/distance-filtered matches, and raw FLANN
    matches, respectively); num_ransac/num_good/num_raw describe the same
    three quantities for step 2's geometrically-guided local search.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(
                "source_filename,target_filename,"
                "num_step1_ransac_inliers,num_step1_good_matches,num_step1_raw_matches,"
                "num_ransac_inliers,num_good_matches,num_raw_matches,"
                + ','.join(metric_keys) + '\n'
            )
    if metrics is None:
        metrics = {k: np.nan for k in metric_keys}
    with open(filepath, 'a') as f:
        extra_vals = ','.join(str(metrics.get(k, np.nan)) for k in metric_keys)
        f.write(
            f"{source_filename},{target_filename},"
            f"{num_step1_ransac_inliers},{num_step1_good_matches},{num_step1_raw_matches},"
            f"{num_ransac},{num_good},{num_raw},{extra_vals}\n"
        )


# ---------------------------------------------------------------------------
# Validate settings
# ---------------------------------------------------------------------------
if resampling_method not in _RASTERIO_RESAMPLING:
    raise ValueError(
        f"resampling_method must be one of {sorted(_RASTERIO_RESAMPLING)}, got {resampling_method!r}"
    )
rasterio_resampling = _RASTERIO_RESAMPLING[resampling_method]
cv2_interpolation = _CV2_INTERPOLATION[resampling_method]

matched_band_pairs = get_matched_band_pairs(band_map)
if len(matched_band_pairs) < 2:
    raise ValueError(
        "band_map must map at least two source bands to target bands "
        "(need at least two matched bands to compute a normalized-difference index)."
    )
metric_keys = build_metric_keys(matched_band_pairs)

if len(display_bands) != 3:
    raise ValueError(f"display_bands must have exactly 3 entries (R, G, B), got {display_bands!r}")
for _b in display_bands:
    if not (0 <= _b < len(band_map)) or band_map[_b] == -1:
        raise ValueError(
            f"display_bands entry {_b} is not a valid source band with a match in band_map."
        )
display_target_bands = [band_map[b] for b in display_bands]

# ---------------------------------------------------------------------------
# Initialise output directories
# ---------------------------------------------------------------------------
for subdir in ("registered", "output_plots", "alignment_metrics"):
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

alignment_metrics_filepath = os.path.join(
    output_directory, "alignment_metrics", "registration_metrics.csv"
)

# Use a non-interactive matplotlib backend (comment out when debugging interactively)
matplotlib.use("Agg")

# SIFT + FLANN configuration (shared across all images)
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann_index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
flann_search_params = dict(checks=50)

# RNG used for spatial thinning and plot subsampling (reproducible)
rng = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------------
# Load target image once; it stays fixed throughout
# ---------------------------------------------------------------------------
print("Loading target image:", target_filepath)
target_image = rxr.open_rasterio(target_filepath, masked=True).squeeze()
print(f"  Shape: {target_image.shape}, CRS: {target_image.rio.crs}, "
      f"resolution: {target_image.rio.resolution()}")
target_h, target_w = target_image.shape[1], target_image.shape[2]

# ---------------------------------------------------------------------------
# Main loop: process each source scene
# ---------------------------------------------------------------------------
source_files = sorted(glob.glob(os.path.join(source_directory, "*.tif")))
print(f"\nFound {len(source_files)} source file(s).\n")

for source_filepath in source_files:
    plt.close('all')
    gc.collect()

    source_filename = os.path.basename(source_filepath)
    target_filename = os.path.basename(target_filepath)
    metrics = {k: np.nan for k in metric_keys}
    print("=" * 70)
    print(f"Processing: {source_filename}")

    # --- Load source image ---
    source_image = rxr.open_rasterio(source_filepath, masked=True).squeeze()
    print(f"  Source shape: {source_image.shape}, CRS: {source_image.rio.crs}, "
          f"resolution: {source_image.rio.resolution()}")

    if source_image.shape[0] != len(band_map):
        print(f"  band_map has {len(band_map)} entries but source has "
              f"{source_image.shape[0]} bands. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            0, 0, 0, 0, 0, 0, metrics
        )
        continue

    # --- Reproject source to target CRS if needed ---
    # source_native retains the source's approximate original resolution; it is
    # used for feature matching, the TPS warp, and the final registered output.
    if source_image.rio.crs != target_image.rio.crs:
        print("  CRS mismatch — reprojecting source to target CRS.")
        source_native = source_image.rio.reproject(target_image.rio.crs, resampling=rasterio_resampling)
    else:
        source_native = source_image
    src_native_h, src_native_w = source_native.shape[1], source_native.shape[2]
    src_native_transform = source_native.rio.transform()
    tgt_transform = target_image.rio.transform()

    # --- Find geographic overlap (both images now share target's CRS) ---
    overlap = find_overlap(source_native, target_image)
    if overlap is None:
        print("  No geographic overlap between source and target. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            0, 0, 0, 0, 0, 0, metrics
        )
        continue

    left, bottom, right, top = overlap
    print(f"  Overlap: x=[{left:.1f}, {right:.1f}], y=[{bottom:.1f}, {top:.1f}]")

    # Clip both images to the overlap region
    src_clip = source_native.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    tgt_clip = target_image.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)

    # --- Pixel-coverage check ---
    src_valid = int(np.sum(~np.isnan(src_clip.values)))
    tgt_valid = int(np.sum(~np.isnan(tgt_clip.values)))
    print(f"  Valid pixels in overlap — source: {src_valid}, target: {tgt_valid}")
    if min(src_valid, tgt_valid) < min_pixel_count:
        print("  Too few valid pixels in overlap. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            0, 0, 0, 0, 0, 0, metrics
        )
        continue

    # --- Before-alignment band metrics ---
    # Computed by resampling the source (clipped to the overlap) onto the
    # target's exact pixel grid, one band at a time — this is the "matching
    # resolution to the target" reprojected source version, used only for
    # error metrics and never materialized as a full multi-band raster.
    _metrics_before = compute_band_metrics_matched_resolution(
        src_clip, tgt_clip, matched_band_pairs, rng
    )
    store_band_metrics(metrics, _metrics_before, matched_band_pairs, "before")
    plot_ransac_mask_grid(
        _metrics_before, matched_band_pairs, "before", source_filename,
        os.path.join(output_directory, "output_plots",
                      source_filename.replace('.tif', '_ransac_mask_before.png'))
    )

    # --- Extract matched bands and compute all pairwise ND indices ---
    src_band_arrays = extract_band_arrays(src_clip, [i for i, _ in matched_band_pairs])
    tgt_band_arrays = extract_band_arrays(tgt_clip, [t for _, t in matched_band_pairs])
    # Re-key target arrays by source band index so src_nd / tgt_nd share keys.
    tgt_band_arrays = {
        src_idx: tgt_band_arrays[tgt_idx] for src_idx, tgt_idx in matched_band_pairs
    }
    src_nd = compute_nd_indices(src_band_arrays)
    tgt_nd = compute_nd_indices(tgt_band_arrays)
    del src_band_arrays, tgt_band_arrays
    gc.collect()

    scale_x = src_clip.shape[2] / tgt_clip.shape[2]
    scale_y = src_clip.shape[1] / tgt_clip.shape[1]
    print(f"  Pixel scale ratio (source/target) — X: {scale_x:.4f}, Y: {scale_y:.4f}")

    # Blur kernel sizes scaled to approximate the same real-world ground
    # footprint in each image, regardless of its pixel resolution.
    src_res_x = abs(src_clip.rio.transform().a)
    tgt_res_x = abs(tgt_clip.rio.transform().a)
    src_blur_kernel = resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, src_res_x
    )
    tgt_blur_kernel = resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, tgt_res_x
    )
    print(f"  Blur kernel size — source: {src_blur_kernel}px, target: {tgt_blur_kernel}px")

    # --- Build RGB display composites for the diagnostic plots below ---
    # Independent of the ND-index channels used for matching: these show the
    # actual selected bands (display_bands, default the first three source
    # bands) so plots are visually interpretable. Raw band values from
    # different sensors can have very different native ranges (e.g.
    # PlanetScope surface reflectance vs. NAIP 8-bit DN), so before display
    # each target channel is rescaled onto the source's radiometric scale
    # using the before-alignment RANSAC-inlier fit (source ~= scale*target +
    # intercept, from compute_band_metrics_matched_resolution above) and the
    # two are then jointly normalized — this is only a display transform and
    # is never applied to the registered GeoTIFF output. Falls back to
    # independent per-image normalization if the fit is unavailable (e.g.
    # too few RANSAC inliers). Built here (rather than later, next to the
    # plotting code) because src_clip / tgt_clip are freed before plotting
    # to bound memory; capped immediately to a bounded resolution for the
    # same reason.
    _src_display_channels, _tgt_display_channels = [], []
    for _b, _t in zip(display_bands, display_target_bands):
        _s_arr = src_clip.isel(band=_b).values.astype(np.float32)
        _t_arr = tgt_clip.isel(band=_t).values.astype(np.float32)
        _b_scale = _metrics_before['scale_inliers'][_b]
        _b_intercept = _metrics_before['intercept_inliers'][_b]
        if np.isfinite(_b_scale) and np.isfinite(_b_intercept):
            _s_norm, _t_norm = normalize_pair(
                _s_arr, _b_scale * _t_arr + _b_intercept, percentile_clip=display_percentile_clip
            )
        else:
            _s_norm = normalize_single(_s_arr, percentile_clip=display_percentile_clip)
            _t_norm = normalize_single(_t_arr, percentile_clip=display_percentile_clip)
        _src_display_channels.append(to_uint8(_s_norm))
        _tgt_display_channels.append(to_uint8(_t_norm))
    src_display_rgb, src_display_scale = cap_for_sift(
        np.stack(_src_display_channels, axis=-1), max_sift_dimension
    )
    tgt_display_rgb, tgt_display_scale = cap_for_sift(
        np.stack(_tgt_display_channels, axis=-1), max_sift_dimension
    )
    del _src_display_channels, _tgt_display_channels

    # =========================================================================
    # Step 1: coarse matching. Detect SIFT keypoints/descriptors per ND-index
    # channel and do an unguided match to obtain a coarse affine transform.
    # This step never produces output raster imagery — its only purpose is
    # to seed step 2's geometrically-guided re-matching below.
    # =========================================================================
    all_src_pts = []
    all_tgt_pts = []
    total_raw_matches = 0
    channel_match_counts = {}
    channel_features = []  # (idx_name, src_kp, src_desc, tgt_kp, tgt_desc, src_sift_scale, tgt_sift_scale)

    for idx_name in src_nd:
        src_norm, tgt_norm = normalize_pair(src_nd[idx_name], tgt_nd[idx_name])

        # Per-channel valid-data mask: False wherever either contributing
        # band (and therefore this ND-index channel) is nodata/cloud-masked.
        src_valid = ~np.isnan(src_nd[idx_name])
        tgt_valid = ~np.isnan(tgt_nd[idx_name])

        src_u8 = masked_gaussian_blur(to_uint8(src_norm), src_valid, src_blur_kernel)
        tgt_u8 = masked_gaussian_blur(to_uint8(tgt_norm), tgt_valid, tgt_blur_kernel)

        # Erode each image's own valid mask by a margin tied to its own blur
        # kernel size, and pass it to SIFT's mask argument so keypoints (and
        # their descriptor support window) are never placed inside or near a
        # nodata/cloud-mask hole or scene edge.
        src_mask = erode_valid_mask(src_valid, mask_erosion_blur_multiple * src_blur_kernel)
        tgt_mask = erode_valid_mask(tgt_valid, mask_erosion_blur_multiple * tgt_blur_kernel)

        # Cap the working resolution for SIFT/FLANN — a fine-resolution target
        # clip can be tens of megapixels, which is impractical to run SIFT's
        # scale-space pyramid on. Detected keypoints are rescaled back to true
        # clip-pixel coordinates immediately below. The mask is downsampled
        # alongside its image (same shape -> same scale) with nearest-
        # neighbor interpolation to stay strictly binary.
        src_u8_det, src_sift_scale = cap_for_sift(src_u8, max_sift_dimension)
        tgt_u8_det, tgt_sift_scale = cap_for_sift(tgt_u8, max_sift_dimension)
        src_mask_det, _ = cap_for_sift(src_mask, max_sift_dimension, interpolation=cv2.INTER_NEAREST)
        tgt_mask_det, _ = cap_for_sift(tgt_mask, max_sift_dimension, interpolation=cv2.INTER_NEAREST)

        src_kp, src_desc = sift.detectAndCompute(src_u8_det, src_mask_det)
        tgt_kp, tgt_desc = sift.detectAndCompute(tgt_u8_det, tgt_mask_det)

        if (src_desc is None or tgt_desc is None or
                len(src_kp) < min_keypoints or len(tgt_kp) < min_keypoints):
            channel_match_counts[idx_name] = 0
            continue

        # Kept for step 2, which re-matches these same keypoints/descriptors
        # rather than re-running SIFT.
        channel_features.append(
            (idx_name, src_kp, src_desc, tgt_kp, tgt_desc, src_sift_scale, tgt_sift_scale)
        )

        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        raw_matches = flann.knnMatch(tgt_desc, src_desc, k=2)
        total_raw_matches += len(raw_matches)

        channel_good = 0
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]

            # Lowe's ratio test
            if m.distance >= lowe_ratio_threshold * n.distance:
                continue

            # Rescale from detection-image pixels back to true clip pixels.
            _src_pt_det = src_kp[m.trainIdx].pt
            _tgt_pt_det = tgt_kp[m.queryIdx].pt
            src_pt = (_src_pt_det[0] / src_sift_scale, _src_pt_det[1] / src_sift_scale)
            tgt_pt = (_tgt_pt_det[0] / tgt_sift_scale, _tgt_pt_det[1] / tgt_sift_scale)

            dist = np.sqrt(
                (src_pt[0] - scale_x * tgt_pt[0]) ** 2 +
                (src_pt[1] - scale_y * tgt_pt[1]) ** 2
            )
            if dist < distance_threshold_pixels:
                all_src_pts.append(src_pt)
                all_tgt_pts.append(tgt_pt)
                channel_good += 1

        channel_match_counts[idx_name] = channel_good

    print(f"  [Step 1] Per-channel good matches: {channel_match_counts}")
    print(f"  [Step 1] Total — raw: {total_raw_matches}, after filtering: {len(all_src_pts)}")

    if len(all_src_pts) < 4:
        print("  [Step 1] Not enough good matches for a coarse transform. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            0, len(all_src_pts), total_raw_matches,
            0, 0, 0, metrics
        )
        continue

    tgt_pts_arr = np.float32(all_tgt_pts).reshape(-1, 1, 2)
    src_pts_arr = np.float32(all_src_pts).reshape(-1, 1, 2)

    M_coarse, mask_coarse = cv2.estimateAffine2D(
        tgt_pts_arr, src_pts_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold
    )

    if M_coarse is None:
        print("  [Step 1] RANSAC failed to produce a coarse affine transform. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            0, len(all_src_pts), total_raw_matches,
            0, 0, 0, metrics
        )
        continue

    coarse_inlier_count = int(np.sum(mask_coarse)) if mask_coarse is not None else 0
    print(f"  [Step 1] Coarse RANSAC inliers: {coarse_inlier_count} / {len(all_src_pts)}")

    # Save step 1's (global-search) counts before step 2 reuses these variable
    # names for its own (local-search) counts below.
    step1_ransac_inliers = coarse_inlier_count
    step1_good_matches = len(all_src_pts)
    step1_raw_matches = total_raw_matches

    # =========================================================================
    # Step 2: guided re-matching. Reuse step 1's per-channel keypoints and
    # descriptors (no re-detection), restricting candidate matches to those
    # within keypoint_match_distance_threshold_m real-world meters (converted
    # to target clip-pixels via tgt_res_x) of where step 1's coarse affine
    # predicts a given source keypoint should fall in target clip-pixel space.
    # This refined match set is what RANSAC, the TPS warp, and the final
    # registered output are built from.
    # =========================================================================
    M_inv = cv2.invertAffineTransform(M_coarse)

    keypoint_match_distance_threshold_px = keypoint_match_distance_threshold_m / tgt_res_x
    print(f"  [Step 2] Guided match distance threshold: "
          f"{keypoint_match_distance_threshold_m} m -> {keypoint_match_distance_threshold_px:.2f} target px")

    all_src_pts = []
    all_tgt_pts = []
    total_raw_matches = 0
    channel_match_counts = {}

    for idx_name, src_kp, src_desc, tgt_kp, tgt_desc, src_sift_scale, tgt_sift_scale in channel_features:
        ch_src_pts, ch_tgt_pts = guided_match_per_descriptor_radius(
            src_kp, src_desc, tgt_kp, tgt_desc, src_sift_scale, tgt_sift_scale,
            M_inv, keypoint_match_distance_threshold_px, lowe_ratio_threshold
        )

        all_src_pts.extend(ch_src_pts)
        all_tgt_pts.extend(ch_tgt_pts)
        total_raw_matches += len(tgt_kp)
        channel_match_counts[idx_name] = len(ch_src_pts)

    print(f"  [Step 2] Per-channel guided matches: {channel_match_counts}")
    print(f"  [Step 2] Total guided matches: {len(all_src_pts)}")

    if len(all_src_pts) < 4:
        print("  [Step 2] Not enough guided matches for RANSAC. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            step1_ransac_inliers, step1_good_matches, step1_raw_matches,
            0, len(all_src_pts), total_raw_matches, metrics
        )
        continue

    # --- Estimate refined affine transform with RANSAC ---
    # This gives us an inlier mask for selecting geometrically consistent matches.
    # The affine parameters themselves are only used to compute mean_kp_dist_after;
    # the actual warp uses the TPS displacement field derived from the inliers.
    tgt_pts_arr = np.float32(all_tgt_pts).reshape(-1, 1, 2)
    src_pts_arr = np.float32(all_src_pts).reshape(-1, 1, 2)

    M_part, mask = cv2.estimateAffine2D(
        tgt_pts_arr, src_pts_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold
    )

    if M_part is None:
        print("  [Step 2] RANSAC failed to produce a valid affine transform. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            step1_ransac_inliers, step1_good_matches, step1_raw_matches,
            0, len(all_src_pts), total_raw_matches, metrics
        )
        continue

    inlier_count = int(np.sum(mask)) if mask is not None else 0
    print(f"  [Step 2] Refined RANSAC inliers: {inlier_count} / {len(all_src_pts)}")

    # --- Keypoint distances before/after (using the RANSAC affine as reference) ---
    _inlier_bool = (mask.ravel() == 1) if mask is not None else np.zeros(len(all_src_pts), bool)
    if _inlier_bool.sum() >= 1:
        _isrc = np.float32(all_src_pts)[_inlier_bool]
        _itgt = np.float32(all_tgt_pts)[_inlier_bool]
        _diffs_before = _isrc - _itgt * np.array([[scale_x, scale_y]])
        metrics['mean_kp_dist_before'] = float(
            np.mean(np.sqrt(np.sum(_diffs_before ** 2, axis=1)))
        )
        _tgt_h = np.c_[_itgt, np.ones(len(_itgt))]
        _pred_src = (M_part @ _tgt_h.T).T
        _diffs_after = _isrc - _pred_src
        metrics['mean_kp_dist_after'] = float(
            np.mean(np.sqrt(np.sum(_diffs_after ** 2, axis=1)))
        )

    if inlier_count < 4:
        print("  [Step 2] Too few RANSAC inliers. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename,
            step1_ransac_inliers, step1_good_matches, step1_raw_matches,
            inlier_count, len(all_src_pts), total_raw_matches, metrics
        )
        continue

    # --- Extract RANSAC inlier keypoint pairs (clip-space coordinates) ---
    inlier_tgt_pts_clip = np.float32(all_tgt_pts)[_inlier_bool]  # (N, 2) target clip coords
    inlier_src_pts_clip = np.float32(all_src_pts)[_inlier_bool]  # (N, 2) source clip coords

    # --- Clip pixel offsets within full images ---
    src_col_off, src_row_off = get_clip_offset(source_native, src_clip)
    tgt_col_off, tgt_row_off = get_clip_offset(target_image, tgt_clip)
    clip_h, clip_w = tgt_clip.shape[1], tgt_clip.shape[2]
    print(f"  Clip offsets — source: (col={src_col_off}, row={src_row_off}), "
          f"target: (col={tgt_col_off}, row={tgt_row_off})")

    # --- Spatial subsampling of TPS control points ---
    # Thinning is computed in target clip space so that the grid cell size is
    # proportional to the actual keypoint distribution in the overlap region.
    tps_tgt_pts_clip, tps_src_pts_clip = spatially_thin_keypoints(
        inlier_tgt_pts_clip, inlier_src_pts_clip,
        tps_max_control_points, clip_w, clip_h, rng
    )

    # Deduplicate: SIFT keypoints from different spectral channels can land at
    # the same or nearly-same pixel, causing the TPS kernel matrix to be singular.
    # Round to the nearest 0.5 px in clip space and keep one representative per site.
    _rounded = (tps_tgt_pts_clip / 0.5).round().astype(np.int64)
    _, _unique_idx = np.unique(_rounded, axis=0, return_index=True)
    tps_tgt_pts_clip = tps_tgt_pts_clip[_unique_idx]
    tps_src_pts_clip = tps_src_pts_clip[_unique_idx]

    n_tps = len(tps_tgt_pts_clip)
    print(f"  TPS control points: {n_tps} "
          f"(from {inlier_count} RANSAC inliers, max={tps_max_control_points})")

    # --- Convert control points to full-image pixel coordinates ---
    tps_tgt_pts_full = tps_tgt_pts_clip + np.array([[tgt_col_off, tgt_row_off]], dtype=np.float64)
    tps_src_pts_full = tps_src_pts_clip + np.array([[src_col_off, src_row_off]], dtype=np.float64)

    # The TPS field is fit and evaluated entirely in source_native's own pixel
    # space, since the final registered output shares source_native's grid.
    # Target control points are converted into that space via world (map)
    # coordinates, so the fit is correct even when source and target have
    # different pixel sizes: target pixel -> world (target transform) ->
    # source_native pixel (inverse of source_native transform).
    _tgt_world_x, _tgt_world_y = tgt_transform * (tps_tgt_pts_full[:, 0], tps_tgt_pts_full[:, 1])
    _domain_x, _domain_y = (~src_native_transform) * (_tgt_world_x, _tgt_world_y)
    tps_domain_pts_full = np.column_stack([_domain_x, _domain_y])

    disp_x = tps_src_pts_full[:, 0] - tps_domain_pts_full[:, 0]
    disp_y = tps_src_pts_full[:, 1] - tps_domain_pts_full[:, 1]

    # --- Fit thin-plate-spline RBF ---
    # Coordinates are normalised to [0, 1] before fitting: the TPS kernel
    # phi(r) = r^2 log(r) varies over many orders of magnitude when r spans
    # thousands of pixels, leading to an ill-conditioned kernel matrix.
    # A small smoothing (1e-3 in normalised space ~ sub-pixel in image space)
    # is added as a safety regulariser; it has negligible effect on accuracy.
    _coord_scale = np.array([[float(src_native_w), float(src_native_h)]])
    tps_domain_pts_norm = tps_domain_pts_full / _coord_scale

    print(f"  Fitting TPS RBF ({n_tps} control points)...")
    rbf_dx = RBFInterpolator(
        tps_domain_pts_norm, disp_x, kernel='thin_plate_spline', smoothing=1e-3
    )
    rbf_dy = RBFInterpolator(
        tps_domain_pts_norm, disp_y, kernel='thin_plate_spline', smoothing=1e-3
    )

    # --- Evaluate on a coarse grid covering the full source_native image ---
    nc_x = min(tps_coarse_grid_size, src_native_w)
    nc_y = min(tps_coarse_grid_size, src_native_h)
    c_xs = np.linspace(0, src_native_w - 1, nc_x)
    c_ys = np.linspace(0, src_native_h - 1, nc_y)
    c_xx, c_yy = np.meshgrid(c_xs, c_ys)
    coarse_pts = np.column_stack([c_xx.ravel(), c_yy.ravel()])
    coarse_pts_norm = coarse_pts / _coord_scale

    print(f"  Evaluating TPS on {nc_x}x{nc_y} coarse grid...")
    c_ddx = rbf_dx(coarse_pts_norm).reshape(nc_y, nc_x).astype(np.float32)
    c_ddy = rbf_dy(coarse_pts_norm).reshape(nc_y, nc_x).astype(np.float32)

    # --- Upsample displacement field to full source_native resolution ---
    full_ddx = cv2.resize(c_ddx, (src_native_w, src_native_h), interpolation=cv2.INTER_LINEAR)
    full_ddy = cv2.resize(c_ddy, (src_native_w, src_native_h), interpolation=cv2.INTER_LINEAR)

    # --- Average displacement over the overlap (clip) region ---
    # full_ddx / full_ddy are already expressed in source_native's own pixel
    # space (both the field's domain and its values live there), so the mean
    # over the source-side overlap window is directly the net geolocation
    # shift, with no extra offset correction needed.
    _r_s = src_row_off
    _r_e = min(src_row_off + src_clip.shape[1], src_native_h)
    _c_s = src_col_off
    _c_e = min(src_col_off + src_clip.shape[2], src_native_w)
    metrics['avg_dx_px'] = float(np.mean(full_ddx[_r_s:_r_e, _c_s:_c_e]))
    metrics['avg_dy_px'] = float(np.mean(full_ddy[_r_s:_r_e, _c_s:_c_e]))
    print(f"  Average displacement in overlap (source_native px) — "
          f"dx: {metrics['avg_dx_px']:.3f}, dy: {metrics['avg_dy_px']:.3f}")

    # --- Build cv2.remap lookup tables (source_native self-referential coordinates) ---
    # Build map_x / map_y in-place from full_ddx / full_ddy to avoid allocating
    # two additional (src_native_h x src_native_w) float32 arrays.
    full_ddx += np.arange(src_native_w, dtype=np.float32)           # add col index per column
    full_ddy += np.arange(src_native_h, dtype=np.float32).reshape(-1, 1)  # add row index per row
    map_x = full_ddx  # source_native column to sample for each output pixel
    map_y = full_ddy  # source_native row to sample for each output pixel
    del c_ddx, c_ddy, coarse_pts, coarse_pts_norm            # free coarse intermediates

    # Out-of-bounds mask: output pixels whose mapped sample position falls
    # outside source_native's own extent (will be set to NaN in the output).
    out_of_bounds = (
        (map_x < 0) | (map_x >= src_native_w) |
        (map_y < 0) | (map_y >= src_native_h)
    )

    # --- Warp source_native onto its own grid to correct local geolocation error ---
    source_registered = xr.DataArray(
        np.full((source_native.shape[0], src_native_h, src_native_w), np.nan, dtype=np.float32),
        coords={
            "band": np.arange(1, source_native.shape[0] + 1),
            "y": source_native.y,
            "x": source_native.x,
        },
        dims=["band", "y", "x"]
    )
    source_registered.rio.write_crs(source_native.rio.crs, inplace=True)
    source_registered.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    source_registered.attrs.update(source_native.attrs)

    for band in range(source_native.shape[0]):
        print(f"  Warping band {band + 1}/{source_native.shape[0]}")
        band_data = source_native.isel(band=band).values.astype(np.float32)

        src_nan = np.isnan(band_data)

        # Warp the NaN mask with nearest-neighbour so we can propagate nodata.
        warped_nan = cv2.remap(
            src_nan.astype(np.float32), map_x, map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
        ) > 0.5

        warped_band = cv2.remap(
            np.where(src_nan, 0.0, band_data), map_x, map_y,
            cv2_interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_band[out_of_bounds | warped_nan] = np.nan
        source_registered.values[band] = warped_band

    if 'long_name' in source_native.attrs:
        source_registered.attrs['long_name'] = source_native.attrs['long_name']

    output_filepath = os.path.join(
        output_directory, "registered",
        source_filename.replace(".tif", "_registered.tif")
    )
    source_registered.rio.to_raster(output_filepath)
    print(f"  Saved registered image: {output_filepath}")

    # --- After-alignment band metrics ---
    # The registered output is on source_native's grid; resample it to the
    # target's exact grid (over the overlap only, one band at a time) for a
    # fair pixel comparison.
    _warped_clip = source_registered.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    _metrics_after = compute_band_metrics_matched_resolution(
        _warped_clip, tgt_clip, matched_band_pairs, rng
    )
    store_band_metrics(metrics, _metrics_after, matched_band_pairs, "after")
    plot_ransac_mask_grid(
        _metrics_after, matched_band_pairs, "after", source_filename,
        os.path.join(output_directory, "output_plots",
                      source_filename.replace('.tif', '_ransac_mask_after.png'))
    )

    write_alignment_metrics(
        alignment_metrics_filepath, metric_keys, source_filename, target_filename,
        step1_ransac_inliers, step1_good_matches, step1_raw_matches,
        inlier_count, len(all_src_pts), total_raw_matches, metrics
    )

    # --- Early memory release before plots ---
    del (_warped_clip, source_registered, map_x, map_y, out_of_bounds,
         src_clip, tgt_clip,
         tps_tgt_pts_clip, tps_src_pts_clip, tps_tgt_pts_full, tps_src_pts_full,
         tps_domain_pts_full, disp_x, disp_y, rbf_dx, rbf_dy,
         inlier_tgt_pts_clip, inlier_src_pts_clip)
    gc.collect()

    # --- Diagnostic plot: band composite comparison in the overlap region ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title(f'Source — Band Composite (overlap)\n{source_filename}')
    ax1.imshow(src_display_rgb)
    ax2.set_title(f'Target — Band Composite (overlap)\n{os.path.basename(target_filepath)}')
    ax2.imshow(tgt_display_rgb)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            output_directory, "output_plots",
            source_filename.replace('.tif', '_band_comparison.png')
        )
    )
    plt.close(fig)

    # --- Diagnostic plot: all step-2 guided matches (before RANSAC) ---
    # Point coordinates (true clip-pixel space) are rescaled into the
    # (possibly downsampled) display-composite image space.
    _all_src_pts_arr = np.float32(all_src_pts)
    _all_tgt_pts_arr = np.float32(all_tgt_pts)
    if len(_all_src_pts_arr) > max_lowe_match_display_points:
        _lowe_idx = rng.choice(len(_all_src_pts_arr), max_lowe_match_display_points, replace=False)
        _lowe_src_pts = _all_src_pts_arr[_lowe_idx]
        _lowe_tgt_pts = _all_tgt_pts_arr[_lowe_idx]
    else:
        _lowe_src_pts = _all_src_pts_arr
        _lowe_tgt_pts = _all_tgt_pts_arr
    _lowe_src_pts_disp = _lowe_src_pts * src_display_scale
    _lowe_tgt_pts_disp = _lowe_tgt_pts * tgt_display_scale

    fig_lowe, (ax_lowe_src, ax_lowe_tgt) = plt.subplots(1, 2, figsize=(14, 6))
    ax_lowe_src.set_title(
        f'Source — step-2 guided matches ({len(all_src_pts)} total, {len(_lowe_src_pts)} shown)'
    )
    ax_lowe_src.imshow(src_display_rgb)
    if len(_lowe_src_pts_disp):
        ax_lowe_src.scatter(_lowe_src_pts_disp[:, 0], _lowe_src_pts_disp[:, 1],
                             s=6, c='yellow', linewidths=0)
    ax_lowe_tgt.set_title('Target — step-2 guided matches')
    ax_lowe_tgt.imshow(tgt_display_rgb)
    if len(_lowe_tgt_pts_disp):
        ax_lowe_tgt.scatter(_lowe_tgt_pts_disp[:, 0], _lowe_tgt_pts_disp[:, 1],
                             s=6, c='yellow', linewidths=0)
    fig_lowe.tight_layout()
    fig_lowe.savefig(
        os.path.join(
            output_directory, "output_plots",
            source_filename.replace('.tif', '_lowe_matches.png')
        )
    )
    plt.close(fig_lowe)
    del _all_src_pts_arr, _all_tgt_pts_arr, _lowe_src_pts, _lowe_tgt_pts, _lowe_src_pts_disp, _lowe_tgt_pts_disp

    # --- Diagnostic plot: RANSAC inlier keypoints ---
    inlier_mask = (mask.ravel() == 1) if mask is not None else np.ones(len(all_src_pts), dtype=bool)
    inlier_src = np.float32(all_src_pts)[inlier_mask] * src_display_scale
    inlier_tgt = np.float32(all_tgt_pts)[inlier_mask] * tgt_display_scale

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    ax3.set_title(f'Source — RANSAC inliers ({inlier_count})')
    ax3.imshow(src_display_rgb)
    if len(inlier_src):
        ax3.scatter(inlier_src[:, 0], inlier_src[:, 1], s=10, c='red', linewidths=0.5)
    ax4.set_title('Target — RANSAC inliers')
    ax4.imshow(tgt_display_rgb)
    if len(inlier_tgt):
        ax4.scatter(inlier_tgt[:, 0], inlier_tgt[:, 1], s=10, c='cyan', linewidths=0.5)
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(
            output_directory, "output_plots",
            source_filename.replace('.tif', '_ransac_inliers.png')
        )
    )
    plt.close(fig2)

    # --- Diagnostic plot: correspondence lines between inlier matched pairs ---
    # Source and target panels are independently rescaled to the same
    # display height so they are easy to visually compare side by side.
    target_display_h = 600
    src_h_clip, src_w_clip = src_display_rgb.shape[:2]
    tgt_h_clip, tgt_w_clip = tgt_display_rgb.shape[:2]
    corr_scale_src = target_display_h / src_h_clip
    corr_scale_tgt = target_display_h / tgt_h_clip

    def resize_display(arr, s):
        return cv2.resize(arr, (max(1, int(round(arr.shape[1] * s))), max(1, int(round(arr.shape[0] * s)))),
                          interpolation=cv2.INTER_AREA)

    src_disp = resize_display(src_display_rgb, corr_scale_src)
    tgt_disp = resize_display(tgt_display_rgb, corr_scale_tgt)

    disp_src_h, disp_src_w = src_disp.shape[:2]
    disp_tgt_h, disp_tgt_w = tgt_disp.shape[:2]
    canvas_h = max(disp_src_h, disp_tgt_h)
    canvas_w = disp_src_w + disp_tgt_w

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:disp_src_h, :disp_src_w] = src_disp
    canvas[:disp_tgt_h, disp_src_w:disp_src_w + disp_tgt_w] = tgt_disp

    if len(inlier_src) > max_lines_match_lines:
        idx = rng.choice(len(inlier_src), max_lines_match_lines, replace=False)
        plot_src_pts = inlier_src[idx]
        plot_tgt_pts = inlier_tgt[idx]
    else:
        plot_src_pts = inlier_src
        plot_tgt_pts = inlier_tgt

    fig3, ax5 = plt.subplots(figsize=(18, 7))
    ax5.set_title(
        f'Inlier Correspondences — {inlier_count} total'
        f' ({len(plot_src_pts)} shown)\n'
        f'Source: {source_filename}  |  Target: {target_filename}'
    )
    ax5.imshow(canvas)
    ax5.axvline(x=disp_src_w, color='white', linewidth=1, linestyle='--')

    for sp, tp in zip(plot_src_pts, plot_tgt_pts):
        ax5.plot(
            [sp[0] * corr_scale_src, tp[0] * corr_scale_tgt + disp_src_w],
            [sp[1] * corr_scale_src, tp[1] * corr_scale_tgt],
            color='lime', linewidth=0.5, alpha=0.6
        )
    if len(plot_src_pts):
        ax5.scatter(plot_src_pts[:, 0] * corr_scale_src, plot_src_pts[:, 1] * corr_scale_src,
                    s=6, c='red', zorder=5, linewidths=0)
        ax5.scatter(plot_tgt_pts[:, 0] * corr_scale_tgt + disp_src_w, plot_tgt_pts[:, 1] * corr_scale_tgt,
                    s=6, c='cyan', zorder=5, linewidths=0)

    ax5.set_xlabel('← Source clip          Target clip →')
    ax5.axis('off')
    fig3.tight_layout()
    fig3.savefig(
        os.path.join(
            output_directory, "output_plots",
            source_filename.replace('.tif', '_correspondences.png')
        ),
        dpi=150
    )
    plt.close(fig3)

    # --- Final memory cleanup before next iteration ---
    del (source_image, source_native, src_nd, tgt_nd, all_src_pts, all_tgt_pts,
         inlier_src, inlier_tgt, src_display_rgb, tgt_display_rgb, channel_features)
    gc.collect()

print("\nDone.")
