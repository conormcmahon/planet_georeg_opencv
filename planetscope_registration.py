"""CLI script: register PlanetScope scenes onto a reference scene's full
pixel grid using a single global affine warp (planet_georeg_opencv.register_to_grid).

Thin wrapper over the planet_georeg_opencv library: this script only knows
about PlanetScope's Dove/SuperDove band layouts, file I/O, plotting, and CSV
logging. All of the actual matching/warping logic lives in the library.
"""

import gc
import glob
import os

import numpy as np
import rioxarray as rxr

import planet_georeg_opencv as pgo
from planet_georeg_opencv import band_metrics, nd_index, plotting, sensors

# --- Settings ---
min_pixel_count = 1000
min_keypoints = 10
lowe_ratio_threshold = 0.75
distance_threshold_pixels = 50
ransac_reproj_threshold = 2.0
blur_kernel_base_size = 3
blur_kernel_reference_resolution = 3.0
mask_erosion_blur_multiple = 2
max_sift_dimension = 4000

source_directory = "/home/conor/src/planet_georeg_opencv/example_imagery/"
reference_filepath = (
    "/home/conor/src/planet_georeg_opencv/example_imagery/"
    "20250318_161156_07_24e6_3B_AnalyticMS_SR_harmonized_clip.tif"
)
output_directory = "/home/conor/test_outputs/"
#source_directory = "/path/to/source_planetscope/"   # PlanetScope scenes to register
#reference_filepath = "/path/to/reference.tif"       # well-geolocated reference image
#output_directory = "/path/to/output/"


# ---------------------------------------------------------------------------
# Initialise output directories
# ---------------------------------------------------------------------------
for subdir in ("registered", "output_plots", "alignment_metrics"):
    os.makedirs(os.path.join(output_directory, subdir), exist_ok=True)

alignment_metrics_filepath = os.path.join(output_directory, "alignment_metrics", "ps_registration_metrics.csv")

print("Loading reference image:", reference_filepath)
reference_image = rxr.open_rasterio(reference_filepath, masked=True).squeeze()
ref_type, ref_band_indices = sensors.identify_band_layout(reference_image)
print(f"  Type: {ref_type}, shape: {reference_image.shape}, CRS: {reference_image.rio.crs}")

source_files = sorted(glob.glob(os.path.join(source_directory, "*.tif")))
print(f"\nFound {len(source_files)} source file(s).\n")

for source_filepath in source_files:
    gc.collect()
    source_filename = os.path.basename(source_filepath)
    reference_filename = os.path.basename(reference_filepath)
    print("=" * 70)
    print(f"Processing: {source_filename}")

    if os.path.abspath(source_filepath) == os.path.abspath(reference_filepath):
        print("  Source is the reference itself. Skipping.")
        continue

    source_image = rxr.open_rasterio(source_filepath, masked=True).squeeze()
    src_type, src_band_indices = sensors.identify_band_layout(source_image)
    print(f"  Source type: {src_type}, shape: {source_image.shape}")

    band_map = sensors.band_map_from_layouts(src_band_indices, ref_band_indices)
    matched_band_pairs = nd_index.get_matched_band_pairs(band_map)
    metric_keys = band_metrics.build_metric_keys(matched_band_pairs)

    result = pgo.register_to_grid(
        source_image, reference_image, band_map,
        min_pixel_count=min_pixel_count,
        min_keypoints=min_keypoints,
        lowe_ratio_threshold=lowe_ratio_threshold,
        distance_threshold_pixels=distance_threshold_pixels,
        ransac_reproj_threshold=ransac_reproj_threshold,
        blur_kernel_base_size=blur_kernel_base_size,
        blur_kernel_reference_resolution=blur_kernel_reference_resolution,
        mask_erosion_blur_multiple=mask_erosion_blur_multiple,
        max_sift_dimension=max_sift_dimension,
    )
    print(f"  Status: {result.status}" + (f" -- {result.message}" if result.message else ""))

    band_metrics.write_alignment_metrics_csv(
        alignment_metrics_filepath, metric_keys, source_filename, reference_filename,
        result.diagnostics.get("ransac_inliers", 0), result.diagnostics.get("num_matches", 0), 0,
        result.diagnostics.get("ransac_inliers", 0), result.diagnostics.get("num_matches", 0), 0,
        result.metrics,
    )

    if result.status != "ok":
        continue

    output_filepath = os.path.join(output_directory, "registered", source_filename.replace(".tif", "_registered.tif"))
    result.registered.rio.to_raster(output_filepath)
    print(f"  Saved registered image: {output_filepath}")

    # --- Diagnostic plots: display the red/green/blue bands directly ---
    moving_clip, fixed_clip = result.diagnostics["moving_clip"], result.diagnostics["fixed_clip"]
    display_names = ('red', 'green', 'blue')
    moving_disp_bands = nd_index.extract_band_arrays(moving_clip, [src_band_indices[n] for n in display_names])
    fixed_disp_bands = nd_index.extract_band_arrays(fixed_clip, [ref_band_indices[n] for n in display_names])
    moving_rgb = np.stack(
        [nd_index.to_uint8(nd_index.normalize_single(moving_disp_bands[src_band_indices[n]], percentile_clip=2.0)) for n in display_names],
        axis=-1,
    )
    fixed_rgb = np.stack(
        [nd_index.to_uint8(nd_index.normalize_single(fixed_disp_bands[ref_band_indices[n]], percentile_clip=2.0)) for n in display_names],
        axis=-1,
    )

    plotting.plot_band_comparison(
        moving_rgb, fixed_rgb, source_filename, reference_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_band_comparison.png')),
    )
    inlier_moving = result.diagnostics["inlier_moving_pts_clip"]
    inlier_fixed = result.diagnostics["inlier_fixed_pts_clip"]
    plotting.plot_keypoints(
        moving_rgb, fixed_rgb, inlier_moving, inlier_fixed,
        f"RANSAC inliers ({len(inlier_moving)})",
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_ransac_inliers.png')),
        point_color='red',
    )
    plotting.plot_correspondences(
        moving_rgb, fixed_rgb, inlier_moving, inlier_fixed, source_filename, reference_filename,
        os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_correspondences.png')),
    )
    if result.diagnostics.get("band_metrics_after") is not None:
        plotting.plot_ransac_mask_grid(
            result.diagnostics["band_metrics_after"], matched_band_pairs, "after", source_filename,
            os.path.join(output_directory, "output_plots", source_filename.replace('.tif', '_ransac_mask_after.png')),
        )

print("\nDone.")
