
import rioxarray as rxr
import xarray as xr
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import gc
from scipy.interpolate import RBFInterpolator

# --- Settings ---
min_pixel_count = 1000        # minimum valid pixels in overlap region to attempt registration
min_keypoints = 10            # minimum keypoints per index channel to attempt matching
lowe_ratio_threshold = 0.75   # Lowe's ratio test threshold
distance_threshold_pixels = 50  # max allowed pixel-space distance between matched keypoints
ransac_reproj_threshold = 2.0   # RANSAC reprojection error threshold (pixels)
blur_kernel_size = (5, 5)     # Gaussian blur kernel; same for both images (both are 3 m PlanetScope)

# --- Local warp settings ---
# Maximum number of RANSAC inliers to use as control points for the TPS warp.
# When there are more inliers than this threshold, the excess is removed via
# spatial subsampling: dense clusters are thinned preferentially while isolated
# points in sparse regions are always retained.
# - Larger values → denser, more faithful warp field; slower RBF solve (O(N³))
# - Smaller values → coarser warp; faster solve
# Set to None to use all RANSAC inliers without any subsampling.
tps_max_control_points = 500

# Resolution (in pixels, per axis) of the coarse grid on which the RBF
# displacement field is first evaluated. The full-resolution field is then
# obtained by bicubic upsampling with cv2.resize. Reducing this trades
# accuracy for speed; 200–400 is a practical range for 3–4K imagery.
tps_coarse_grid_size = 300

source_directory = "/path/to/source_planetscope/"   # PlanetScope scenes to register
reference_filepath = "/path/to/reference.tif"       # well-geolocated reference image
output_directory = "/path/to/output/"

# --- PlanetScope band definitions (0-indexed) ---
# Dove Classic / Dove-R (4-band): Blue, Green, Red, NIR
DOVE_BANDS = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}

# SuperDove / PSB.SD (8-band): Coastal Blue, Blue, Green I, Green, Yellow, Red, Red Edge, NIR
# We keep only the four bands common to both sensor types.
SUPERDOVE_BANDS = {'blue': 1, 'green': 3, 'red': 5, 'nir': 7}


def identify_band_layout(image):
    """Return sensor type string and band-index dict based on the image's band count."""
    n = image.shape[0]
    if n == 4:
        return 'Dove', DOVE_BANDS
    elif n == 8:
        return 'SuperDove', SUPERDOVE_BANDS
    else:
        raise ValueError(
            f"Unexpected band count {n}; expected 4 (Dove) or 8 (SuperDove)."
        )


def extract_bgrnir(image, band_indices):
    """Return a dict of float32 numpy arrays for the four common PlanetScope bands."""
    return {
        name: image.isel(band=idx).values.astype(np.float32)
        for name, idx in band_indices.items()
    }


def compute_nd_indices(bands):
    """
    Compute all six pairwise normalized-difference indices from Blue, Green, Red, NIR.

    Using every combination rather than a fixed set avoids the need for SWIR bands
    (which PlanetScope does not carry) while still producing diverse texture signals
    for feature matching.
    """
    eps = 1e-6
    b, g, r, nir = bands['blue'], bands['green'], bands['red'], bands['nir']
    return {
        'ndvi':    (nir - r) / (nir + r + eps),   # NIR vs Red   (vegetation)
        'nd_bg':   (b   - g) / (b   + g + eps),   # Blue vs Green
        'nd_br':   (b   - r) / (b   + r + eps),   # Blue vs Red
        'nd_bnir': (b - nir) / (b + nir + eps),   # Blue vs NIR
        'nd_gr':   (g   - r) / (g   + r + eps),   # Green vs Red
        'nd_gnir': (g - nir) / (g + nir + eps),   # Green vs NIR
    }


def normalize_pair(arr1, arr2):
    """Jointly normalize two float arrays to [0, 1] using their combined range."""
    lo = min(np.nanmin(arr1), np.nanmin(arr2))
    hi = max(np.nanmax(arr1), np.nanmax(arr2))
    if hi == lo:
        return np.zeros_like(arr1), np.zeros_like(arr2)
    return (arr1 - lo) / (hi - lo), (arr2 - lo) / (hi - lo)


def to_uint8(arr):
    """Convert a float [0, 1] array to uint8 [0, 255], mapping NaN to 0."""
    return np.nan_to_num(
        np.round(np.clip(arr * 255, 0, 255)), nan=0.0
    ).astype(np.uint8)


def compute_band_metrics(src_da, ref_da, src_band_idx, ref_band_idx):
    """
    Compute per-band Pearson R² and RMSE for blue/green/red/NIR.

    Both DataArrays must be on the same pixel grid. Valid (non-NaN) pixels
    in both arrays are compared. R² is the square of the Pearson correlation
    coefficient, which measures structural agreement independent of any
    systematic radiometric offset between the two images.

    Returns two dicts keyed by band name: r2, rmse.
    """
    r2, rmse = {}, {}
    for name in ('blue', 'green', 'red', 'nir'):
        s = src_da.isel(band=src_band_idx[name]).values.astype(np.float32).ravel()
        r = ref_da.isel(band=ref_band_idx[name]).values.astype(np.float32).ravel()
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() < 2:
            r2[name] = np.nan
            rmse[name] = np.nan
            continue
        a, b = s[valid], r[valid]
        rmse[name] = float(np.sqrt(np.mean((a - b) ** 2)))
        corr = np.corrcoef(a, b)[0, 1]
        r2[name] = float(corr ** 2) if np.isfinite(corr) else np.nan
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


def spatially_thin_keypoints(ref_pts, src_pts, max_control_points, clip_w, clip_h, rng):
    """
    Subsample matched keypoint pairs to at most `max_control_points` using a
    regular spatial grid over the reference clip extent.

    The image overlap is divided into a grid of cells whose size is chosen so
    that the expected number of cells approximately equals `max_control_points`.
    Within each occupied cell exactly one keypoint is kept (chosen at random).
    Cells with only one occupant are always retained; cells with many occupants
    (dense clusters) contribute only one representative.

    This guarantees that:
    - Sparse regions (≤1 keypoint per cell) lose nothing.
    - Dense clusters are thinned proportionally, not preferentially dropped.
    - The total retained count is at most min(max_control_points, n_occupied_cells).

    `ref_pts` should be in clip-space pixel coordinates used to assign cells.
    `src_pts` can be in any coordinate space — they are subsampled in lock-step.
    """
    n = len(ref_pts)
    if max_control_points is None or n <= max_control_points:
        return ref_pts, src_pts

    # Cell size chosen so that clip_area / cell_area ≈ max_control_points.
    cell_area = (clip_w * clip_h) / max_control_points
    cell_size = max(1, int(np.sqrt(cell_area)))

    # Assign each reference keypoint to a grid cell.
    cell_map = {}
    for i in range(n):
        cell_col = int(ref_pts[i, 0] / cell_size)
        cell_row = int(ref_pts[i, 1] / cell_size)
        key = (cell_row, cell_col)
        if key not in cell_map:
            cell_map[key] = []
        cell_map[key].append(i)

    # From each occupied cell keep one randomly-chosen point.
    kept = [int(rng.choice(indices)) for indices in cell_map.values()]
    kept = np.array(sorted(kept))
    return ref_pts[kept], src_pts[kept]


_METRIC_KEYS = [
    'r2_blue_before',   'r2_green_before',   'r2_red_before',   'r2_nir_before',
    'r2_blue_after',    'r2_green_after',     'r2_red_after',    'r2_nir_after',
    'rmse_blue_before', 'rmse_green_before',  'rmse_red_before', 'rmse_nir_before',
    'rmse_blue_after',  'rmse_green_after',   'rmse_red_after',  'rmse_nir_after',
    'mean_kp_dist_before', 'mean_kp_dist_after',
    'avg_dx_px', 'avg_dy_px',
]


def _blank_metrics():
    return {k: np.nan for k in _METRIC_KEYS}


def write_alignment_metrics(filepath, source_filename, target_filename,
                             num_ransac, num_good, num_raw, metrics=None):
    """Append one row of local-registration quality metrics to the CSV log.

    Unlike the global-affine version this function does not write affine
    matrix parameters (M_0..M_8).  Instead it includes avg_dx_px and
    avg_dy_px — the mean displacement over all reference pixels in the
    overlap region.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(
                "source_filename,target_filename,num_ransac_inliers,num_good_matches,num_raw_matches,"
                + ','.join(_METRIC_KEYS) + '\n'
            )
    if metrics is None:
        metrics = _blank_metrics()
    with open(filepath, 'a') as f:
        extra_vals = ','.join(str(metrics.get(k, np.nan)) for k in _METRIC_KEYS)
        f.write(
            f"{source_filename},{target_filename},{num_ransac},{num_good},{num_raw},{extra_vals}\n"
        )


# ---------------------------------------------------------------------------
# Initialise output directories
# ---------------------------------------------------------------------------
for subdir in ("registered", "output_plots", "alignment_metrics"):
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

alignment_metrics_filepath = os.path.join(
    output_directory, "alignment_metrics", "ps_registration_metrics_local.csv"
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
# Load reference image once; it stays fixed throughout
# ---------------------------------------------------------------------------
print("Loading reference image:", reference_filepath)
reference_image = rxr.open_rasterio(reference_filepath, masked=True).squeeze()
ref_type, ref_band_indices = identify_band_layout(reference_image)
print(f"  Type: {ref_type}, shape: {reference_image.shape}, CRS: {reference_image.rio.crs}")

# ---------------------------------------------------------------------------
# Main loop: process each source PlanetScope scene
# ---------------------------------------------------------------------------
source_files = sorted(glob.glob(os.path.join(source_directory, "*.tif")))
print(f"\nFound {len(source_files)} source file(s).\n")

for source_filepath in source_files:
    plt.close('all')
    gc.collect()

    source_filename = os.path.basename(source_filepath)
    reference_filename = os.path.basename(reference_filepath)
    metrics = _blank_metrics()
    print("=" * 70)
    print(f"Processing: {source_filename}")

    # --- Load source image ---
    source_image = rxr.open_rasterio(source_filepath, masked=True).squeeze()
    src_type, src_band_indices = identify_band_layout(source_image)
    print(f"  Source type: {src_type}, shape: {source_image.shape}")

    # --- Ensure matching CRS ---
    working_reference = reference_image
    if reference_image.rio.crs != source_image.rio.crs:
        print("  CRS mismatch — reprojecting reference to source CRS.")
        working_reference = reference_image.rio.reproject(source_image.rio.crs)

    # --- Find geographic overlap ---
    overlap = find_overlap(source_image, working_reference)
    if overlap is None:
        print("  No geographic overlap between source and reference. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, reference_filename, 0, 0, 0, metrics
        )
        continue

    left, bottom, right, top = overlap
    print(f"  Overlap: x=[{left:.1f}, {right:.1f}], y=[{bottom:.1f}, {top:.1f}]")

    # Clip both images to the overlap region
    src_clip = source_image.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    ref_clip = working_reference.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)

    # --- Pixel-coverage check ---
    src_valid = int(np.sum(~np.isnan(src_clip.values)))
    ref_valid = int(np.sum(~np.isnan(ref_clip.values)))
    print(f"  Valid pixels in overlap — source: {src_valid}, reference: {ref_valid}")
    if min(src_valid, ref_valid) < min_pixel_count:
        print("  Too few valid pixels in overlap. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, reference_filename, 0, 0, 0, metrics
        )
        continue

    # --- Before-alignment band metrics (clip pixel comparison in overlap region) ---
    if src_clip.shape[1:] == ref_clip.shape[1:]:
        _r2b, _rmseb = compute_band_metrics(src_clip, ref_clip, src_band_indices, ref_band_indices)
        for _bn in ('blue', 'green', 'red', 'nir'):
            metrics[f'r2_{_bn}_before']   = _r2b[_bn]
            metrics[f'rmse_{_bn}_before'] = _rmseb[_bn]

    # --- Extract common bands and compute all six ND indices ---
    src_bands = extract_bgrnir(src_clip, src_band_indices)
    ref_bands = extract_bgrnir(ref_clip, ref_band_indices)
    src_nd = compute_nd_indices(src_bands)
    ref_nd = compute_nd_indices(ref_bands)

    scale_x = src_clip.shape[2] / ref_clip.shape[2]
    scale_y = src_clip.shape[1] / ref_clip.shape[1]
    print(f"  Pixel scale ratio (source/reference) — X: {scale_x:.4f}, Y: {scale_y:.4f}")

    # --- Feature matching across all six ND index channels ---
    all_src_pts = []
    all_ref_pts = []
    total_raw_matches = 0
    channel_match_counts = {}

    for idx_name in src_nd:
        src_norm, ref_norm = normalize_pair(src_nd[idx_name], ref_nd[idx_name])
        src_u8 = cv2.GaussianBlur(to_uint8(src_norm), blur_kernel_size, 0)
        ref_u8 = cv2.GaussianBlur(to_uint8(ref_norm), blur_kernel_size, 0)

        src_kp, src_desc = sift.detectAndCompute(src_u8, None)
        ref_kp, ref_desc = sift.detectAndCompute(ref_u8, None)

        if (src_desc is None or ref_desc is None or
                len(src_kp) < min_keypoints or len(ref_kp) < min_keypoints):
            channel_match_counts[idx_name] = 0
            continue

        flann = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)
        raw_matches = flann.knnMatch(ref_desc, src_desc, k=2)
        total_raw_matches += len(raw_matches)

        channel_good = 0
        for match_pair in raw_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]

            # Lowe's ratio test
            if m.distance >= lowe_ratio_threshold * n.distance:
                continue

            src_pt = src_kp[m.trainIdx].pt
            ref_pt = ref_kp[m.queryIdx].pt

            dist = np.sqrt(
                (src_pt[0] - scale_x * ref_pt[0]) ** 2 +
                (src_pt[1] - scale_y * ref_pt[1]) ** 2
            )
            if dist < distance_threshold_pixels:
                all_src_pts.append(src_pt)
                all_ref_pts.append(ref_pt)
                channel_good += 1

        channel_match_counts[idx_name] = channel_good

    print(f"  Per-channel good matches: {channel_match_counts}")
    print(f"  Total — raw: {total_raw_matches}, after filtering: {len(all_src_pts)}")

    if len(all_src_pts) < 4:
        print("  Not enough good matches for RANSAC. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, reference_filename, 0, len(all_src_pts),
            total_raw_matches, metrics
        )
        continue

    # --- Estimate affine transform with RANSAC ---
    # This gives us an inlier mask for selecting geometrically consistent matches.
    # The affine parameters themselves are only used to compute mean_kp_dist_after;
    # the actual warp uses the TPS displacement field derived from the inliers.
    ref_pts_arr = np.float32(all_ref_pts).reshape(-1, 1, 2)
    src_pts_arr = np.float32(all_src_pts).reshape(-1, 1, 2)

    M_part, mask = cv2.estimateAffine2D(
        ref_pts_arr, src_pts_arr,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold
    )

    if M_part is None:
        print("  RANSAC failed to produce a valid affine transform. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, reference_filename, 0, len(all_src_pts),
            total_raw_matches, metrics
        )
        continue

    inlier_count = int(np.sum(mask)) if mask is not None else 0
    print(f"  RANSAC inliers: {inlier_count} / {len(all_src_pts)}")

    # --- Keypoint distances before/after (using the RANSAC affine as reference) ---
    _inlier_bool = (mask.ravel() == 1) if mask is not None else np.zeros(len(all_src_pts), bool)
    if _inlier_bool.sum() >= 1:
        _isrc = np.float32(all_src_pts)[_inlier_bool]
        _iref = np.float32(all_ref_pts)[_inlier_bool]
        _diffs_before = _isrc - _iref * np.array([[scale_x, scale_y]])
        metrics['mean_kp_dist_before'] = float(
            np.mean(np.sqrt(np.sum(_diffs_before ** 2, axis=1)))
        )
        _ref_h = np.c_[_iref, np.ones(len(_iref))]
        _pred_src = (M_part @ _ref_h.T).T
        _diffs_after = _isrc - _pred_src
        metrics['mean_kp_dist_after'] = float(
            np.mean(np.sqrt(np.sum(_diffs_after ** 2, axis=1)))
        )

    if inlier_count < 4:
        print("  Too few RANSAC inliers. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, reference_filename, inlier_count,
            len(all_src_pts), total_raw_matches, metrics
        )
        continue

    # --- Extract RANSAC inlier keypoint pairs (clip-space coordinates) ---
    inlier_ref_pts_clip = np.float32(all_ref_pts)[_inlier_bool]  # (N, 2) ref clip coords
    inlier_src_pts_clip = np.float32(all_src_pts)[_inlier_bool]  # (N, 2) src clip coords

    # --- Clip pixel offsets within full images ---
    src_col_off, src_row_off = get_clip_offset(source_image, src_clip)
    ref_col_off, ref_row_off = get_clip_offset(working_reference, ref_clip)
    clip_h, clip_w = ref_clip.shape[1], ref_clip.shape[2]
    print(f"  Clip offsets — source: (col={src_col_off}, row={src_row_off}), "
          f"reference: (col={ref_col_off}, row={ref_row_off})")

    # --- Spatial subsampling of TPS control points ---
    # Thinning is computed in clip space so that the grid cell size is proportional
    # to the actual keypoint distribution in the overlap region.
    tps_ref_pts_clip, tps_src_pts_clip = spatially_thin_keypoints(
        inlier_ref_pts_clip, inlier_src_pts_clip,
        tps_max_control_points, clip_w, clip_h, rng
    )
    n_tps = len(tps_ref_pts_clip)
    print(f"  TPS control points: {n_tps} "
          f"(from {inlier_count} RANSAC inliers, max={tps_max_control_points})")

    # --- Convert control points to full-image coordinates for RBF fitting ---
    # The displacement vector at each control point is:
    #   disp = src_full - ref_full
    #        = (src_clip + src_offset) - (ref_clip + ref_offset)
    # The RBF is fitted in full-image space so that we can evaluate it directly
    # at every reference full-image pixel without a coordinate conversion.
    tps_ref_pts_full = tps_ref_pts_clip + np.array([[ref_col_off, ref_row_off]], dtype=np.float64)
    tps_src_pts_full = tps_src_pts_clip + np.array([[src_col_off, src_row_off]], dtype=np.float64)
    disp_x = tps_src_pts_full[:, 0] - tps_ref_pts_full[:, 0]
    disp_y = tps_src_pts_full[:, 1] - tps_ref_pts_full[:, 1]

    # --- Fit thin-plate-spline RBF in full-image coordinate space ---
    # smoothing=0 forces exact interpolation through all control points.
    print(f"  Fitting TPS RBF ({n_tps} control points)...")
    rbf_dx = RBFInterpolator(
        tps_ref_pts_full, disp_x, kernel='thin_plate_spline', smoothing=0
    )
    rbf_dy = RBFInterpolator(
        tps_ref_pts_full, disp_y, kernel='thin_plate_spline', smoothing=0
    )

    # --- Evaluate on a coarse grid covering the full reference image ---
    ref_h = working_reference.shape[1]
    ref_w = working_reference.shape[2]
    nc_x = min(tps_coarse_grid_size, ref_w)
    nc_y = min(tps_coarse_grid_size, ref_h)
    c_xs = np.linspace(0, ref_w - 1, nc_x)
    c_ys = np.linspace(0, ref_h - 1, nc_y)
    c_xx, c_yy = np.meshgrid(c_xs, c_ys)
    coarse_pts = np.column_stack([c_xx.ravel(), c_yy.ravel()])

    print(f"  Evaluating TPS on {nc_x}×{nc_y} coarse grid...")
    c_ddx = rbf_dx(coarse_pts).reshape(nc_y, nc_x).astype(np.float32)
    c_ddy = rbf_dy(coarse_pts).reshape(nc_y, nc_x).astype(np.float32)

    # --- Upsample displacement field to full reference image resolution ---
    full_ddx = cv2.resize(c_ddx, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    full_ddy = cv2.resize(c_ddy, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

    # --- Average displacement over the overlap (clip) region ---
    # Computed in the clip region where control points exist and the field
    # is well-constrained by data, rather than in extrapolated margins.
    _r_s = ref_row_off
    _r_e = min(ref_row_off + clip_h, ref_h)
    _c_s = ref_col_off
    _c_e = min(ref_col_off + clip_w, ref_w)
    metrics['avg_dx_px'] = float(np.mean(full_ddx[_r_s:_r_e, _c_s:_c_e]))
    metrics['avg_dy_px'] = float(np.mean(full_ddy[_r_s:_r_e, _c_s:_c_e]))
    print(f"  Average displacement in overlap — "
          f"dx: {metrics['avg_dx_px']:.3f} px, dy: {metrics['avg_dy_px']:.3f} px")

    # --- Build cv2.remap lookup tables (source full-image coordinates) ---
    # For each reference full-image pixel (col_f, row_f):
    #   src_full_col = col_f + full_ddx[row_f, col_f]
    #   src_full_row = row_f + full_ddy[row_f, col_f]
    base_xx, base_yy = np.meshgrid(
        np.arange(ref_w, dtype=np.float32),
        np.arange(ref_h, dtype=np.float32)
    )
    map_x = base_xx + full_ddx  # source full-image column for each reference pixel
    map_y = base_yy + full_ddy  # source full-image row for each reference pixel

    # Out-of-bounds mask: reference pixels whose mapped source position falls
    # outside the source image extent (will be set to NaN in the output).
    src_h_full = source_image.shape[1]
    src_w_full = source_image.shape[2]
    out_of_bounds = (
        (map_x < 0) | (map_x >= src_w_full) |
        (map_y < 0) | (map_y >= src_h_full)
    )

    # --- Warp source image to the reference image's pixel grid ---
    source_warped = xr.DataArray(
        np.full((source_image.shape[0], ref_h, ref_w), np.nan, dtype=np.float32),
        coords={
            "band": np.arange(1, source_image.shape[0] + 1),
            "y": working_reference.y,
            "x": working_reference.x,
        },
        dims=["band", "y", "x"]
    )
    source_warped.rio.write_crs(working_reference.rio.crs, inplace=True)
    source_warped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    source_warped.attrs.update(source_image.attrs)

    for band in range(source_image.shape[0]):
        print(f"  Warping band {band + 1}/{source_image.shape[0]}")
        band_data = source_image.isel(band=band).values.astype(np.float32)

        src_nan = np.isnan(band_data)

        # Warp the NaN mask with nearest-neighbour so we can propagate nodata.
        warped_nan = cv2.remap(
            src_nan.astype(np.float32), map_x, map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
        ) > 0.5

        warped_band = cv2.remap(
            np.where(src_nan, 0.0, band_data), map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_band[out_of_bounds | warped_nan] = np.nan
        source_warped.values[band] = warped_band

    if 'long_name' in source_image.attrs:
        source_warped.attrs['long_name'] = source_image.attrs['long_name']

    output_filepath = os.path.join(
        output_directory, "registered",
        source_filename.replace(".tif", "_registered.tif")
    )
    source_warped.rio.to_raster(output_filepath)
    print(f"  Saved registered image: {output_filepath}")

    # --- After-alignment band metrics ---
    _warped_clip = source_warped.rio.clip_box(minx=left, miny=bottom, maxx=right, maxy=top)
    if _warped_clip.shape[1:] == ref_clip.shape[1:]:
        _r2a, _rmsea = compute_band_metrics(
            _warped_clip, ref_clip, src_band_indices, ref_band_indices
        )
        for _bn in ('blue', 'green', 'red', 'nir'):
            metrics[f'r2_{_bn}_after']   = _r2a[_bn]
            metrics[f'rmse_{_bn}_after'] = _rmsea[_bn]

    write_alignment_metrics(
        alignment_metrics_filepath, source_filename, reference_filename, inlier_count,
        len(all_src_pts), total_raw_matches, metrics
    )

    # --- Diagnostic plot: falsecolor index comparison in the overlap region ---
    display_channels = ['ndvi', 'nd_gr', 'nd_bnir']
    src_rgb = np.stack(
        [to_uint8(normalize_pair(src_nd[ch], ref_nd[ch])[0]) for ch in display_channels],
        axis=-1
    )
    ref_rgb = np.stack(
        [to_uint8(normalize_pair(src_nd[ch], ref_nd[ch])[1]) for ch in display_channels],
        axis=-1
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title(f'Source — Index Falsecolor (overlap)\n{source_filename}')
    ax1.imshow(src_rgb)
    ax2.set_title(f'Reference — Index Falsecolor (overlap)\n{os.path.basename(reference_filepath)}')
    ax2.imshow(ref_rgb)
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
    inlier_ref = np.float32(all_ref_pts)[inlier_mask]

    src_ndvi_u8, ref_ndvi_u8 = [
        to_uint8(n) for n in normalize_pair(src_nd['ndvi'], ref_nd['ndvi'])
    ]

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    ax3.set_title(f'Source NDVI — RANSAC inliers ({inlier_count})')
    ax3.imshow(src_ndvi_u8, cmap='gray', vmin=0, vmax=255)
    if len(inlier_src):
        ax3.scatter(inlier_src[:, 0], inlier_src[:, 1], s=10, c='red', linewidths=0.5)
    ax4.set_title('Reference NDVI — RANSAC inliers')
    ax4.imshow(ref_ndvi_u8, cmap='gray', vmin=0, vmax=255)
    if len(inlier_ref):
        ax4.scatter(inlier_ref[:, 0], inlier_ref[:, 1], s=10, c='cyan', linewidths=0.5)
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
    src_h_clip, src_w_clip = src_ndvi_u8.shape
    ref_h_clip, ref_w_clip = ref_ndvi_u8.shape
    scale = min(1.0, max_display_h / max(src_h_clip, ref_h_clip))

    def resize_display(arr, s):
        return cv2.resize(arr, (max(1, int(arr.shape[1] * s)), max(1, int(arr.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    src_disp = resize_display(src_ndvi_u8, scale)
    ref_disp = resize_display(ref_ndvi_u8, scale)

    disp_src_h, disp_src_w = src_disp.shape
    disp_ref_h, disp_ref_w = ref_disp.shape
    canvas_h = max(disp_src_h, disp_ref_h)
    canvas_w = disp_src_w + disp_ref_w

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas[:disp_src_h, :disp_src_w] = src_disp
    canvas[:disp_ref_h, disp_src_w:disp_src_w + disp_ref_w] = ref_disp

    max_lines = 300
    if len(inlier_src) > max_lines:
        idx = rng.choice(len(inlier_src), max_lines, replace=False)
        plot_src_pts = inlier_src[idx]
        plot_ref_pts = inlier_ref[idx]
    else:
        plot_src_pts = inlier_src
        plot_ref_pts = inlier_ref

    fig3, ax5 = plt.subplots(figsize=(18, 7))
    ax5.set_title(
        f'Inlier Correspondences — {inlier_count} total'
        f' ({len(plot_src_pts)} shown)\n'
        f'Source: {source_filename}  |  Reference: {reference_filename}'
    )
    ax5.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax5.axvline(x=disp_src_w, color='white', linewidth=1, linestyle='--')

    for sp, rp in zip(plot_src_pts, plot_ref_pts):
        ax5.plot(
            [sp[0] * scale, rp[0] * scale + disp_src_w],
            [sp[1] * scale, rp[1] * scale],
            color='lime', linewidth=0.5, alpha=0.6
        )
    if len(plot_src_pts):
        ax5.scatter(plot_src_pts[:, 0] * scale, plot_src_pts[:, 1] * scale,
                    s=6, c='red', zorder=5, linewidths=0)
        ax5.scatter(plot_ref_pts[:, 0] * scale + disp_src_w, plot_ref_pts[:, 1] * scale,
                    s=6, c='cyan', zorder=5, linewidths=0)

    ax5.set_xlabel('← Source NDVI clip          Reference NDVI clip →')
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

print("\nDone.")
