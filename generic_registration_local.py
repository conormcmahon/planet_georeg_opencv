"""Generic multi-sensor image registration using OpenCV feature matching and a
thin-plate-spline (TPS) local warp.

This is a generalized version of planetscope_registration_local.py. Unlike that
script, it makes no assumptions about sensor type, band count, band order,
spatial resolution, or coordinate reference system (CRS) of either image. It
is intended to register one or more "source" rasters onto a single, trusted
"target" raster, even when the two come from entirely different sensors
(e.g. PlanetScope source registered onto a NAIP target).

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
                         Controls both the CRS reprojection / resolution
                         matching steps (via rasterio) and the final TPS
                         pixel warp (via OpenCV). "average" is intended for
                         cases where the source has finer spatial resolution
                         than the target (area-weighted downsampling).

Outputs (written under output_directory):
    registered/          - one registered GeoTIFF per source file, on a grid
                            that keeps the source's approximate native
                            resolution (reprojected into the target's CRS).
    output_plots/         - diagnostic PNGs per source file (band-index
                            falsecolor comparison, RANSAC inlier keypoints,
                            inlier correspondence lines).
    alignment_metrics/    - a CSV log with one row per source file, giving
                            match counts and per-band R^2 / RMSE before and
                            after alignment.

Design notes on the two reprojected-source versions:
    Per source file, the source raster is reprojected into the target's CRS
    in two distinct ways:
      - source_native: reprojected to the target CRS but resampled at
        approximately the source's own native resolution. This preserves
        source detail and is the raster that keypoint matching, the TPS
        warp, and the final registered output are all built from.
      - source_matched: conceptually, the source resampled onto the target's
        exact pixel grid, used only to compute before/after R^2 and RMSE
        (which require comparable samples on both images). Rather than
        materializing this as a full raster (which, via GDAL reprojection,
        can leave a multi-gigabyte, not-promptly-reclaimed memory footprint
        when the target is tens of megapixels), it is implemented as
        point-sampling: a bounded random sample of target pixels is drawn
        and the source is interpolated at the corresponding world
        coordinates, so memory stays bounded regardless of image size.
"""

import rioxarray as rxr
import xarray as xr
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import gc
from itertools import combinations
from rasterio.enums import Resampling
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates

# --- Settings ---
min_pixel_count = 1000        # minimum valid pixels in overlap region to attempt registration
min_keypoints = 10            # minimum keypoints per index channel to attempt matching
lowe_ratio_threshold = 1000   # Lowe's ratio test threshold
distance_threshold_pixels = 50  # max allowed pixel-space distance between matched keypoints
                                 # (evaluated in source clip-pixel units; see scale_x/scale_y below)
ransac_reproj_threshold = 2.0   # RANSAC reprojection error threshold (source clip pixels)
blur_kernel_size = (3, 3)     # Gaussian blur kernel applied to both images before SIFT
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

source_directory  = "/path/to/source_images/"   # rasters to be registered
target_filepath   = "/path/to/target.tif"        # well-geolocated target raster
output_directory  = "/path/to/output/"

# band_map[i] = 0-indexed target band matching 0-indexed source band i, or -1
# if source band i has no match in the target image.
band_map = [0, 1, 2, 3]

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


def normalize_pair(arr1, arr2):
    """Jointly normalize two float arrays to [0, 1] using their combined range."""
    lo = min(np.nanmin(arr1), np.nanmin(arr2))
    hi = max(np.nanmax(arr1), np.nanmax(arr2))
    if hi == lo:
        return np.zeros_like(arr1), np.zeros_like(arr2)
    return (arr1 - lo) / (hi - lo), (arr2 - lo) / (hi - lo)


def cap_for_sift(img_u8, max_dim):
    """
    Downsample img_u8 (if needed) so its larger side is at most max_dim.

    Returns (image_for_detection, scale), where scale = detection_size /
    original_size. Detected keypoint coordinates must be divided by `scale`
    to convert them back to the original image's pixel space.
    """
    h, w = img_u8.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return img_u8, 1.0
    small = cv2.resize(
        img_u8, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA
    )
    return small, scale


def to_uint8(arr):
    """Convert a float [0, 1] array to uint8 [0, 255], mapping NaN to 0."""
    return np.nan_to_num(
        np.round(np.clip(arr * 255, 0, 255)), nan=0.0
    ).astype(np.uint8)


_METRIC_SAMPLE_INTERP_ORDER = {
    "nearest":  0,
    "bilinear": 1,
    "cubic":    3,
    # A true area-weighted average isn't meaningful for a point sample, so
    # this is approximated as bilinear — acceptable since this is only a
    # statistical error estimate, not the final resampled image.
    "average":  1,
}


def compute_band_metrics_matched_resolution(src_clip, tgt_clip, matched_pairs, resampling_method,
                                              rng, max_sample_points=2_000_000):
    """
    Estimate per-band Pearson R^2 and RMSE between src_clip and tgt_clip on
    the target's pixel grid, WITHOUT materializing a full "source resampled
    to target resolution" raster (which, via GDAL reprojection, can leave a
    multi-gigabyte, not-promptly-reclaimed memory footprint when the target
    is tens of megapixels — confirmed by profiling this pipeline against a
    0.6 m target overlapping a 3 m source).

    Instead, a bounded random sample of valid target pixels is drawn per
    band; each sampled pixel's world coordinate is converted into src_clip's
    pixel space (via each array's own affine transform) and the source is
    interpolated there with scipy.ndimage.map_coordinates. This only ever
    allocates arrays of size max_sample_points, regardless of image size.

    Returns two dicts keyed by source band index: r2, rmse.
    """
    interp_order = _METRIC_SAMPLE_INTERP_ORDER[resampling_method]
    tgt_transform = tgt_clip.rio.transform()
    src_transform = src_clip.rio.transform()
    inv_src_transform = ~src_transform

    r2, rmse = {}, {}
    for src_idx, tgt_idx in matched_pairs:
        tgt_band = tgt_clip.isel(band=tgt_idx).values.astype(np.float32)
        tgt_rows, tgt_cols = np.where(~np.isnan(tgt_band))
        if len(tgt_rows) > max_sample_points:
            sel = rng.choice(len(tgt_rows), max_sample_points, replace=False)
            tgt_rows, tgt_cols = tgt_rows[sel], tgt_cols[sel]

        if len(tgt_rows) < 2:
            r2[src_idx], rmse[src_idx] = np.nan, np.nan
            continue

        world_x, world_y = tgt_transform * (tgt_cols.astype(np.float64), tgt_rows.astype(np.float64))
        src_col, src_row = inv_src_transform * (world_x, world_y)

        # NaN nodata pixels are filled before interpolating: for order > 1,
        # map_coordinates' spline prefilter is a global recursive filter, so
        # even a single NaN in the input would otherwise contaminate the
        # entire output. Validity is instead tracked with a separate
        # nearest-neighbor sample of the nodata mask (mirroring how the final
        # TPS warp propagates nodata through cv2.remap).
        src_band = src_clip.isel(band=src_idx).values.astype(np.float32)
        src_band_nan = np.isnan(src_band)
        sampled = map_coordinates(
            np.where(src_band_nan, 0.0, src_band), [src_row, src_col],
            order=interp_order, mode='constant', cval=0.0
        )
        sampled_nan = map_coordinates(
            src_band_nan.astype(np.float32), [src_row, src_col],
            order=0, mode='constant', cval=1.0
        ) > 0.5
        sampled[sampled_nan] = np.nan

        t = tgt_band[tgt_rows, tgt_cols]
        valid = ~(np.isnan(sampled) | np.isnan(t))
        if valid.sum() < 2:
            r2[src_idx], rmse[src_idx] = np.nan, np.nan
            continue
        a, b = sampled[valid], t[valid]
        rmse[src_idx] = float(np.sqrt(np.mean((a - b) ** 2)))
        corr = np.corrcoef(a, b)[0, 1]
        r2[src_idx] = float(corr ** 2) if np.isfinite(corr) else np.nan
    return r2, rmse


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


def build_metric_keys(matched_pairs):
    keys = []
    for src_idx, _ in matched_pairs:
        keys += [f"r2_src{src_idx}_before", f"rmse_src{src_idx}_before"]
    for src_idx, _ in matched_pairs:
        keys += [f"r2_src{src_idx}_after", f"rmse_src{src_idx}_after"]
    keys += ["mean_kp_dist_before", "mean_kp_dist_after", "avg_dx_px", "avg_dy_px"]
    return keys


def write_alignment_metrics(filepath, metric_keys, source_filename, target_filename,
                             num_ransac, num_good, num_raw, metrics=None):
    """Append one row of local-registration quality metrics to the CSV log."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(
                "source_filename,target_filename,num_ransac_inliers,num_good_matches,num_raw_matches,"
                + ','.join(metric_keys) + '\n'
            )
    if metrics is None:
        metrics = {k: np.nan for k in metric_keys}
    with open(filepath, 'a') as f:
        extra_vals = ','.join(str(metrics.get(k, np.nan)) for k in metric_keys)
        f.write(
            f"{source_filename},{target_filename},{num_ransac},{num_good},{num_raw},{extra_vals}\n"
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
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, 0, 0, 0, metrics
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
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, 0, 0, 0, metrics
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
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, 0, 0, 0, metrics
        )
        continue

    # --- Before-alignment band metrics ---
    # Computed by resampling the source (clipped to the overlap) onto the
    # target's exact pixel grid, one band at a time — this is the "matching
    # resolution to the target" reprojected source version, used only for
    # error metrics and never materialized as a full multi-band raster.
    _r2b, _rmseb = compute_band_metrics_matched_resolution(
        src_clip, tgt_clip, matched_band_pairs, resampling_method, rng
    )
    for src_idx, _ in matched_band_pairs:
        metrics[f'r2_src{src_idx}_before']   = _r2b[src_idx]
        metrics[f'rmse_src{src_idx}_before'] = _rmseb[src_idx]

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

    # --- Feature matching across all ND index channels ---
    all_src_pts = []
    all_tgt_pts = []
    total_raw_matches = 0
    channel_match_counts = {}

    for idx_name in src_nd:
        src_norm, tgt_norm = normalize_pair(src_nd[idx_name], tgt_nd[idx_name])
        src_u8 = cv2.GaussianBlur(to_uint8(src_norm), blur_kernel_size, 0)
        tgt_u8 = cv2.GaussianBlur(to_uint8(tgt_norm), blur_kernel_size, 0)

        # Cap the working resolution for SIFT/FLANN — a fine-resolution target
        # clip can be tens of megapixels, which is impractical to run SIFT's
        # scale-space pyramid on. Detected keypoints are rescaled back to true
        # clip-pixel coordinates immediately below.
        src_u8_det, src_sift_scale = cap_for_sift(src_u8, max_sift_dimension)
        tgt_u8_det, tgt_sift_scale = cap_for_sift(tgt_u8, max_sift_dimension)

        src_kp, src_desc = sift.detectAndCompute(src_u8_det, None)
        tgt_kp, tgt_desc = sift.detectAndCompute(tgt_u8_det, None)

        if (src_desc is None or tgt_desc is None or
                len(src_kp) < min_keypoints or len(tgt_kp) < min_keypoints):
            channel_match_counts[idx_name] = 0
            continue

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

    print(f"  Per-channel good matches: {channel_match_counts}")
    print(f"  Total — raw: {total_raw_matches}, after filtering: {len(all_src_pts)}")

    if len(all_src_pts) < 4:
        print("  Not enough good matches for RANSAC. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, 0,
            len(all_src_pts), total_raw_matches, metrics
        )
        continue

    # --- Estimate affine transform with RANSAC ---
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
        print("  RANSAC failed to produce a valid affine transform. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, 0,
            len(all_src_pts), total_raw_matches, metrics
        )
        continue

    inlier_count = int(np.sum(mask)) if mask is not None else 0
    print(f"  RANSAC inliers: {inlier_count} / {len(all_src_pts)}")

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
        print("  Too few RANSAC inliers. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, metric_keys, source_filename, target_filename, inlier_count,
            len(all_src_pts), total_raw_matches, metrics
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
    _r2a, _rmsea = compute_band_metrics_matched_resolution(
        _warped_clip, tgt_clip, matched_band_pairs, resampling_method, rng
    )
    for src_idx, _ in matched_band_pairs:
        metrics[f'r2_src{src_idx}_after']   = _r2a[src_idx]
        metrics[f'rmse_src{src_idx}_after'] = _rmsea[src_idx]

    write_alignment_metrics(
        alignment_metrics_filepath, metric_keys, source_filename, target_filename, inlier_count,
        len(all_src_pts), total_raw_matches, metrics
    )

    # --- Early memory release before plots ---
    del (_warped_clip, source_registered, map_x, map_y, out_of_bounds,
         src_clip, tgt_clip,
         tps_tgt_pts_clip, tps_src_pts_clip, tps_tgt_pts_full, tps_src_pts_full,
         tps_domain_pts_full, disp_x, disp_y, rbf_dx, rbf_dy,
         inlier_tgt_pts_clip, inlier_src_pts_clip)
    gc.collect()

    # --- Diagnostic plot: falsecolor index comparison in the overlap region ---
    # Uses up to the first three available ND-index channels; if fewer than
    # three are available (small band_map), channels are repeated to fill
    # the RGB composite.
    _available_channels = sorted(src_nd.keys())
    display_channels = (_available_channels * 3)[:3]
    # Cap each channel to a bounded resolution before stacking — a
    # fine-resolution target clip can be tens of megapixels per channel.
    src_rgb = np.stack(
        [cap_for_sift(to_uint8(normalize_pair(src_nd[ch], tgt_nd[ch])[0]), max_sift_dimension)[0]
         for ch in display_channels],
        axis=-1
    )
    tgt_rgb = np.stack(
        [cap_for_sift(to_uint8(normalize_pair(src_nd[ch], tgt_nd[ch])[1]), max_sift_dimension)[0]
         for ch in display_channels],
        axis=-1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title(f'Source — Index Falsecolor (overlap)\n{source_filename}')
    ax1.imshow(src_rgb)
    ax2.set_title(f'Target — Index Falsecolor (overlap)\n{os.path.basename(target_filepath)}')
    ax2.imshow(tgt_rgb)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            output_directory, "output_plots",
            source_filename.replace('.tif', '_index_comparison.png')
        )
    )
    plt.close(fig)

    # --- Diagnostic plot: RANSAC inlier keypoints ---
    inlier_mask = (mask.ravel() == 1) if mask is not None else np.ones(len(all_src_pts), dtype=bool)
    inlier_src = np.float32(all_src_pts)[inlier_mask]
    inlier_tgt = np.float32(all_tgt_pts)[inlier_mask]

    _primary_channel = _available_channels[0]
    src_idx_u8, tgt_idx_u8 = [
        to_uint8(n) for n in normalize_pair(src_nd[_primary_channel], tgt_nd[_primary_channel])
    ]
    # Cap plot images to a bounded resolution — a fine-resolution target clip
    # can be tens of megapixels, which is unnecessarily expensive to rasterize
    # for a diagnostic PNG. Inlier point coordinates (in true clip-pixel
    # space) are rescaled into this same, possibly smaller, image space.
    src_idx_u8, _src_disp_scale = cap_for_sift(src_idx_u8, max_sift_dimension)
    tgt_idx_u8, _tgt_disp_scale = cap_for_sift(tgt_idx_u8, max_sift_dimension)
    inlier_src = inlier_src * _src_disp_scale
    inlier_tgt = inlier_tgt * _tgt_disp_scale

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    ax3.set_title(f'Source ND index {_primary_channel} — RANSAC inliers ({inlier_count})')
    ax3.imshow(src_idx_u8, cmap='gray', vmin=0, vmax=255)
    if len(inlier_src):
        ax3.scatter(inlier_src[:, 0], inlier_src[:, 1], s=10, c='red', linewidths=0.5)
    ax4.set_title(f'Target ND index {_primary_channel} — RANSAC inliers')
    ax4.imshow(tgt_idx_u8, cmap='gray', vmin=0, vmax=255)
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
    max_display_h = 600
    src_h_clip, src_w_clip = src_idx_u8.shape
    tgt_h_clip, tgt_w_clip = tgt_idx_u8.shape
    scale = min(1.0, max_display_h / max(src_h_clip, tgt_h_clip))

    def resize_display(arr, s):
        return cv2.resize(arr, (max(1, int(arr.shape[1] * s)), max(1, int(arr.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    src_disp = resize_display(src_idx_u8, scale)
    tgt_disp = resize_display(tgt_idx_u8, scale)

    disp_src_h, disp_src_w = src_disp.shape
    disp_tgt_h, disp_tgt_w = tgt_disp.shape
    canvas_h = max(disp_src_h, disp_tgt_h)
    canvas_w = disp_src_w + disp_tgt_w

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
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
    ax5.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax5.axvline(x=disp_src_w, color='white', linewidth=1, linestyle='--')

    for sp, tp in zip(plot_src_pts, plot_tgt_pts):
        ax5.plot(
            [sp[0] * scale, tp[0] * scale + disp_src_w],
            [sp[1] * scale, tp[1] * scale],
            color='lime', linewidth=0.5, alpha=0.6
        )
    if len(plot_src_pts):
        ax5.scatter(plot_src_pts[:, 0] * scale, plot_src_pts[:, 1] * scale,
                    s=6, c='red', zorder=5, linewidths=0)
        ax5.scatter(plot_tgt_pts[:, 0] * scale + disp_src_w, plot_tgt_pts[:, 1] * scale,
                    s=6, c='cyan', zorder=5, linewidths=0)

    ax5.set_xlabel('← Source ND index clip          Target ND index clip →')
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
    del source_image, source_native, src_nd, tgt_nd, all_src_pts, all_tgt_pts, inlier_src, inlier_tgt
    gc.collect()

print("\nDone.")
