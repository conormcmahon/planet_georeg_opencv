
import rioxarray as rxr
import xarray as xr
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import gc

# --- Settings ---
min_pixel_count = 1000        # minimum valid pixels in overlap region to attempt registration
min_keypoints = 10            # minimum keypoints per index channel to attempt matching
lowe_ratio_threshold = 0.75   # Lowe's ratio test threshold
distance_threshold_pixels = 50  # max allowed pixel-space distance between matched keypoints
ransac_reproj_threshold = 2.0   # RANSAC reprojection error threshold (pixels)
blur_kernel_size = (5, 5)     # Gaussian blur kernel; same for both images (both are 3 m PlanetScope)

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


def adjust_homography_for_offsets(M_3x3, src_col, src_row, ref_col, ref_row):
    """
    Convert a 3x3 affine-as-homography estimated on clipped images so that it maps
    full-source pixel coords to full-reference pixel coords.

        M_full = T_ref_offset  @  M_clip  @  T_src_offset_inv
    """
    T_src_inv = np.float32([[1, 0, -src_col], [0, 1, -src_row], [0, 0, 1]])
    T_ref     = np.float32([[1, 0,  ref_col], [0, 1,  ref_row], [0, 0, 1]])
    return T_ref @ M_3x3 @ T_src_inv


def write_alignment_metrics(filepath, source_filename, num_ransac, num_good, num_raw, M_mat):
    """Append one row of registration quality metrics to the CSV log."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(
                "source_filename,num_ransac_inliers,num_good_matches,num_raw_matches,"
                "M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8\n"
            )
    with open(filepath, 'a') as f:
        if M_mat is not None:
            m_vals = ','.join(str(v) for v in M_mat.flatten())
        else:
            m_vals = ','.join(['0'] * 9)
        f.write(f"{source_filename},{num_ransac},{num_good},{num_raw},{m_vals}\n")


# ---------------------------------------------------------------------------
# Initialise output directories
# ---------------------------------------------------------------------------
for subdir in ("registered", "output_plots", "alignment_metrics"):
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

alignment_metrics_filepath = os.path.join(
    output_directory, "alignment_metrics", "ps_registration_metrics.csv"
)

# Use a non-interactive matplotlib backend (comment out when debugging interactively)
matplotlib.use("Agg")

# SIFT + FLANN configuration (shared across all images)
sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1
flann_index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
flann_search_params = dict(checks=50)

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
    # Images may not share the exact same extent; we restrict feature matching
    # to the region where both images have data.
    overlap = find_overlap(source_image, working_reference)
    if overlap is None:
        print("  No geographic overlap between source and reference. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, 0, 0, 0, None
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
            alignment_metrics_filepath, source_filename, 0, 0, 0, None
        )
        continue

    # --- Extract common bands and compute all six ND indices ---
    src_bands = extract_bgrnir(src_clip, src_band_indices)
    ref_bands = extract_bgrnir(ref_clip, ref_band_indices)
    src_nd = compute_nd_indices(src_bands)
    ref_nd = compute_nd_indices(ref_bands)

    # Pixel-size scale ratio between clips (source vs reference).
    # Used to filter matches by expected spatial proximity.
    scale_x = src_clip.shape[2] / ref_clip.shape[2]
    scale_y = src_clip.shape[1] / ref_clip.shape[1]
    print(f"  Pixel scale ratio (source/reference) — X: {scale_x:.4f}, Y: {scale_y:.4f}")

    # --- Feature matching across all six ND index channels ---
    # Running SIFT independently on each index and pooling the spatial
    # correspondences lets all six image representations contribute to the
    # final transform estimate, while keeping descriptor matching within the
    # same spectral channel (which avoids cross-channel false matches).
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

            # Distance filter: in a well-registered pair the matched keypoints
            # should be close to their expected position given the scale ratio.
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
            alignment_metrics_filepath, source_filename, 0, len(all_src_pts),
            total_raw_matches, None
        )
        continue

    # --- Estimate affine transform with RANSAC ---
    # ref_pts = query (the fixed reference), src_pts = train (the image to warp)
    # estimateAffine2D maps ref → src, i.e. we get the transform that takes
    # reference clip coords and produces source clip coords.
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
            alignment_metrics_filepath, source_filename, 0, len(all_src_pts),
            total_raw_matches, None
        )
        continue

    # Embed the 2×3 affine matrix into a 3×3 homogeneous matrix
    M = np.eye(3, dtype=np.float32)
    M[0:2, 0:3] = M_part

    inlier_count = int(np.sum(mask)) if mask is not None else 0
    print(f"  RANSAC inliers: {inlier_count} / {len(all_src_pts)}")

    if inlier_count < 4:
        print("  Too few RANSAC inliers. Skipping.")
        write_alignment_metrics(
            alignment_metrics_filepath, source_filename, inlier_count,
            len(all_src_pts), total_raw_matches, M
        )
        continue

    write_alignment_metrics(
        alignment_metrics_filepath, source_filename, inlier_count,
        len(all_src_pts), total_raw_matches, M
    )
    print(f"  Affine transform (clip space):\n{M}")

    # --- Adjust homography from clip pixel space to full-image pixel space ---
    # The transform was estimated on the cropped overlap region. To apply it
    # to the full source image and produce output on the full reference grid,
    # we translate by the clip origin offsets on both sides.
    src_col_off, src_row_off = get_clip_offset(source_image, src_clip)
    ref_col_off, ref_row_off = get_clip_offset(working_reference, ref_clip)
    M_full = adjust_homography_for_offsets(
        M, src_col_off, src_row_off, ref_col_off, ref_row_off
    )
    print(f"  Clip offsets — source: (col={src_col_off}, row={src_row_off}), "
          f"reference: (col={ref_col_off}, row={ref_row_off})")
    print(f"  Affine transform (full-image space):\n{M_full}")

    # --- Warp source image to the reference image's pixel grid ---
    ref_h = working_reference.shape[1]
    ref_w = working_reference.shape[2]

    source_warped = xr.DataArray(
        np.zeros((source_image.shape[0], ref_h, ref_w), dtype=np.float32),
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
        warped_band = cv2.warpPerspective(
            source_image.isel(band=band).values.astype(np.float32),
            M_full,
            (ref_w, ref_h),
            flags=cv2.INTER_LINEAR
        )
        source_warped.values[band] = warped_band

    if 'long_name' in source_image.attrs:
        source_warped.attrs['long_name'] = source_image.attrs['long_name']

    output_filepath = os.path.join(
        output_directory, "registered",
        source_filename.replace(".tif", "_registered.tif")
    )
    source_warped.rio.to_raster(output_filepath)
    print(f"  Saved registered image: {output_filepath}")

    # --- Diagnostic plot: falsecolor index comparison in the overlap region ---
    # Use NDVI, Green-Red, and Blue-NIR as the R/G/B display channels; these
    # three capture vegetation brightness, greenness, and water/shadow contrast.
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
    # Plot the inlier matched points on the NDVI channel of each clip.
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

print("\nDone.")
