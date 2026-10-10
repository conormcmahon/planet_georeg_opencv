"""CLI script: register one or more "moving" rasters onto their own pixel
grid using a local thin-plate-spline or a single affine warp (see
warp_model below; planet_georeg_opencv.register_local),
making no assumptions about sensor type, band count, band order, spatial
resolution, or CRS of either image (e.g. PlanetScope moving registered onto
a NAIP fixed image).

Thin wrapper over the planet_georeg_opencv library: this script only knows
about an explicit band_map, file I/O, plotting, and CSV logging. All of the
actual matching/warping logic lives in the library.
"""

import gc
import glob
import os

import numpy as np
import rioxarray as rxr
from rasterio.enums import Resampling

import planet_georeg_opencv as pgo
from planet_georeg_opencv import band_metrics, nd_index, plotting

# --- Settings ---
min_pixel_count = 1000        # minimum valid pixels in overlap region to attempt registration
min_keypoints = 10            # minimum keypoints per index channel to attempt matching
lowe_ratio_threshold = 1000   # Lowe's ratio test threshold
distance_threshold_pixels = 50  # max allowed pixel-space distance between matched keypoints
ransac_reproj_threshold = 2.0   # RANSAC reprojection error threshold (moving clip pixels)
keypoint_match_distance_threshold_m = 15.0  # meters, real-world ground distance for step 2's guided search
blur_kernel_base_size = 3
blur_kernel_reference_resolution = 3.0
mask_erosion_blur_multiple = 2
max_sift_dimension = 4000
tps_max_control_points = 2000
tps_coarse_grid_size = 300
warp_model = "tps"  # "tps" (local, non-rigid) or "affine" (one global transform); output stays on the moving image's grid either way
fixed_buffer_m = 0.0  # extra fixed-image margin (CRS units, metres for UTM) searched for keypoints beyond the strict overlap
resampling_method = "bilinear"  # one of "nearest", "bilinear", "cubic", "average"

display_percentile_clip = 2.0  # percentile (from each end) used to clip outliers when contrast-stretching output_plots/

source_directory = "/path/to/source_images/"   # rasters to be registered
fixed_filepath = "/path/to/fixed.tif"          # well-geolocated fixed raster
output_directory = "/path/to/output/"

# band_map[i] = 0-indexed fixed-image band matching 0-indexed moving band i,
# or -1 if moving band i has no match in the fixed image.
band_map = [0, 1, 2, 3]

# 0-indexed MOVING band indices rendered as (R, G, B) in the diagnostic
# PNGs under output_plots/. Each must have a match in band_map (its
# corresponding fixed band is looked up automatically). Defaults to the
# first three bands; set to actual red/green/blue band indices for
# true-color display.
display_bands = [0, 1, 2]


# ---------------------------------------------------------------------------
# Validate settings
# ---------------------------------------------------------------------------
matched_band_pairs = nd_index.get_matched_band_pairs(band_map)
if len(matched_band_pairs) < 2:
    raise ValueError(
        "band_map must map at least two moving bands to fixed bands "
        "(need at least two matched bands to compute a normalized-difference index)."
    )
metric_keys = band_metrics.build_metric_keys(matched_band_pairs)

if len(display_bands) != 3:
    raise ValueError(f"display_bands must have exactly 3 entries (R, G, B), got {display_bands!r}")
for _b in display_bands:
    if not (0 <= _b < len(band_map)) or band_map[_b] == -1:
        raise ValueError(f"display_bands entry {_b} is not a valid moving band with a match in band_map.")
display_fixed_bands = [band_map[b] for b in display_bands]

# ---------------------------------------------------------------------------
# Initialise output directories
# ---------------------------------------------------------------------------
for subdir in ("registered", "output_plots", "alignment_metrics"):
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

alignment_metrics_filepath = os.path.join(output_directory, "alignment_metrics", "registration_metrics.csv")

print("Loading fixed image:", fixed_filepath)
fixed_image = rxr.open_rasterio(fixed_filepath, masked=True).squeeze()
print(f"  Shape: {fixed_image.shape}, CRS: {fixed_image.rio.crs}, resolution: {fixed_image.rio.resolution()}")

source_files = sorted(glob.glob(os.path.join(source_directory, "*.tif")))
print(f"\nFound {len(source_files)} source file(s).\n")

for source_filepath in source_files:
    gc.collect()
    source_filename = os.path.basename(source_filepath)
    fixed_filename = os.path.basename(fixed_filepath)
    print("=" * 70)
    print(f"Processing: {source_filename}")

    moving_image = rxr.open_rasterio(source_filepath, masked=True).squeeze()
    print(f"  Moving shape: {moving_image.shape}, CRS: {moving_image.rio.crs}, resolution: {moving_image.rio.resolution()}")

    if moving_image.shape[0] != len(band_map):
        print(f"  band_map has {len(band_map)} entries but moving image has {moving_image.shape[0]} bands. Skipping.")
        band_metrics.write_alignment_metrics_csv(
            alignment_metrics_filepath, metric_keys, source_filename, fixed_filename, 0, 0, 0, 0, 0, 0,
        )
        continue

    result = pgo.register_local(
        moving_image, fixed_image, band_map,
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
        tps_max_control_points=tps_max_control_points,
        tps_coarse_grid_size=tps_coarse_grid_size,
        warp_model=warp_model,
        fixed_buffer_m=fixed_buffer_m,
        resampling_method=resampling_method,
    )
    print(f"  Status: {result.status}" + (f" -- {result.message}" if result.message else ""))

    band_metrics.write_alignment_metrics_csv(
        alignment_metrics_filepath, metric_keys, source_filename, fixed_filename,
        result.diagnostics.get("coarse_ransac_inliers", 0), 0, 0,
        result.diagnostics.get("guided_ransac_inliers", 0), 0, 0,
        result.metrics,
    )

    if result.status != "ok":
        continue

    output_filepath = os.path.join(output_directory, "registered", source_filename.replace(".tif", "_registered.tif"))
    result.registered.rio.to_raster(output_filepath)
    print(f"  Saved registered image: {output_filepath}")

    # --- Diagnostic plots ---
    # Raw band values from different sensors can have very different native
    # ranges (e.g. PlanetScope surface reflectance vs. NAIP 8-bit DN), so
    # before display each fixed channel is linearly rescaled to match the
    # moving image's own percentile range (percentile_histogram_match) and
    # the two are then jointly normalized.
    moving_clip, fixed_clip = result.diagnostics["moving_clip"], result.diagnostics["fixed_clip"]
    moving_disp_channels, fixed_disp_channels = [], []
    for b, f in zip(display_bands, display_fixed_bands):
        m_arr = moving_clip.isel(band=b).values.astype(np.float32)
        f_arr = fixed_clip.isel(band=f).values.astype(np.float32)
        f_scale, f_intercept = nd_index.percentile_histogram_match(f_arr, m_arr, display_percentile_clip)
        if np.isfinite(f_scale) and np.isfinite(f_intercept):
            m_norm, f_norm = nd_index.normalize_pair(m_arr, f_scale * f_arr + f_intercept, percentile_clip=display_percentile_clip)
        else:
            m_norm = nd_index.normalize_single(m_arr, percentile_clip=display_percentile_clip)
            f_norm = nd_index.normalize_single(f_arr, percentile_clip=display_percentile_clip)
        moving_disp_channels.append(nd_index.to_uint8(m_norm))
        fixed_disp_channels.append(nd_index.to_uint8(f_norm))
    moving_rgb = np.stack(moving_disp_channels, axis=-1)
    fixed_rgb = np.stack(fixed_disp_channels, axis=-1)

    plotting.plot_band_comparison(
        moving_rgb, fixed_rgb, source_filename, fixed_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_band_comparison.png')),
    )
    inlier_moving = result.diagnostics["inlier_moving_pts_clip"]
    inlier_fixed = result.diagnostics["inlier_fixed_pts_clip"]
    plotting.plot_keypoints(
        moving_rgb, fixed_rgb, inlier_moving, inlier_fixed,
        f"RANSAC inliers ({result.diagnostics['guided_ransac_inliers']})",
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_ransac_inliers.png')),
        point_color='red',
    )
    plotting.plot_correspondences(
        moving_rgb, fixed_rgb, inlier_moving, inlier_fixed, source_filename, fixed_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_correspondences.png')),
    )
    plotting.plot_ransac_mask_grid(
        result.diagnostics["band_metrics_before"], matched_band_pairs, "before", source_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_ransac_mask_before.png')),
    )
    plotting.plot_ransac_mask_grid(
        result.diagnostics["band_metrics_after"], matched_band_pairs, "after", source_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_ransac_mask_after.png')),
    )

print("\nDone.")
