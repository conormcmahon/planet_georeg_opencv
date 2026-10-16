
import rioxarray as rxr
import xarray as xr
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import gc

# Settings
min_pixel_count = 1000
min_keypoints = 15
target_directory = "E:/cmcmahon/EMIT_MESMA_v3/"

# Landsat Band Centers and Widths
landsat_band_left_edge = [450, 530, 640, 850, 1570, 2110]
landsat_band_right_edge = [510, 590, 670, 880, 1650, 2290]

# EMIT Wavelengths
EMIT_wavelength_max = 2493
EMIT_wavelength_min = 381
EMIT_wavelength_count = 285
EMIT_wavelengths = np.linspace(EMIT_wavelength_min, EMIT_wavelength_max, EMIT_wavelength_count)

# Get Wavelength Mapping
def get_wavelength_mapping(emit_wavelengths, landsat_left_edges, landsat_right_edges):
    mapping = []
    for left, right in zip(landsat_left_edges, landsat_right_edges):
        indices = np.where((emit_wavelengths >= left) & (emit_wavelengths <= right))[0]
        mapping.append(indices)
    return mapping
wavelength_mapping = get_wavelength_mapping(EMIT_wavelengths, landsat_band_left_edge, landsat_band_right_edge)

# Function to load one EMIT image and generate synthetic Landsat equivalent
def create_emit_synthetic_image(file_path):
    emit_image = rxr.open_rasterio(file_path, masked=True)
    landsat_synthetic = emit_image.sel(band=slice(0, len(wavelength_mapping))).copy()
    landsat_synthetic.values = np.zeros_like(landsat_synthetic.values)
    landsat_synthetic.attrs['long_name'] = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    valid_band_count = landsat_synthetic.copy(deep=True)
    # Example synthetic transformation (placeholder)
    for i, indices in enumerate(wavelength_mapping):
        for j in indices:
            landsat_synthetic[i, :, :] += emit_image.isel(band=indices).mean(dim='band')
            valid_band_count[i, :, :] += ~np.isnan(emit_image.isel(band=indices).mean(dim='band'))
        landsat_synthetic[i, :, :] /= valid_band_count[i, :, :].where(valid_band_count[i, :, :] != 0, other=1)

    return landsat_synthetic

# Function to load one Landsat image
def load_landsat_image(file_path):
    landsat_image = rxr.open_rasterio(file_path, masked=True).squeeze()
    return landsat_image

def get_landsat_band_subset(image, emit_num_bands):
    # Drop extra bands from Landsat image
    landsat_num_bands = image.shape[0]
    image = image.isel(band=slice(landsat_num_bands - emit_num_bands, landsat_num_bands))
    return image

# Define some spectral indices
def getNDVI(image):
    # Assuming image is in (bands, height, width) format
    nir_band = image.isel(band=3)  # NIR band
    red_band = image.isel(band=2)  # Red band
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)  # Adding small value to avoid division by zero
    return ndvi
def getNBR(image):
    # Assuming image is in (bands, height, width) format
    nir_band = image.isel(band=3)  # NIR band
    swir2_band = image.isel(band=5)  # SWIR2 band
    nbr = (nir_band - swir2_band) / (nir_band + swir2_band + 1e-6)  # Adding small value to avoid division by zero
    return nbr
def getNDMI(image):
    # Assuming image is in (bands, height, width) format
    nir_band = image.isel(band=3)  # NIR band
    swir1_band = image.isel(band=4)  # SWIR1 band
    ndwi = (nir_band - swir1_band) / (nir_band + swir1_band + 1e-6)  # Adding small value to avoid division by zero
    return ndwi

# Jointly normalize two images to scale from 0 to 1
def normalizeImages(img1, img2):
    min_val = min(np.nanmin(img1.values), np.nanmin(img2.values))
    max_val = max(np.nanmax(img1.values), np.nanmax(img2.values))
    img1_norm = (img1 - min_val) / (max_val - min_val)
    img2_norm = (img2 - min_val) / (max_val - min_val)
    return img1_norm, img2_norm

# Get list of EMIT files in target directory
emit_file_search_pattern = os.path.join(target_directory, "emit_reflectance/*EMIT_L2A_RFL*[0-9].tif")
emit_files = glob.glob(emit_file_search_pattern)

print(" ")
print(" ")
print(" ")

# Use a non-interactive and less memory-intensive backend for matplotlib
# Note - if debugging make sure to comment this out
matplotlib.use("Agg")

# Iterate over all EMIT files and find alignment to Landsat images
for emit_filepath in emit_files:
    #  Try to prevent memory leak issues
    plt.close('all')
    gc.collect()

    print(" ")
    print('Processing EMIT file: ' + emit_filepath)

    emit_filename = os.path.basename(emit_filepath)
    emit_landsat_equiv = create_emit_synthetic_image(emit_filepath)
    emit_landsat_equiv_filename = os.path.basename(emit_filepath).replace('.tif', '_landsat_equiv.tif')
    emit_landsat_equiv.rio.to_raster(target_directory + "emit_landsat_equivalent/" + emit_landsat_equiv_filename)
    print('  Generated EMIT synthetic Landsat equivalent: ' + emit_landsat_equiv_filename)

    landsat_filepath = target_directory + "landsat_synthetic_imagery/" + emit_filename.replace('.tif', '_Landsat_Phenology.tif')
    if not os.path.exists(landsat_filepath):
        print('  Landsat file not found, skipping: ' + landsat_filepath)
        continue
    landsat_image = load_landsat_image(landsat_filepath)
    landsat_image_band_subset = get_landsat_band_subset(landsat_image, emit_landsat_equiv.shape[0])
    print("  Loaded Landsat file: " + landsat_filepath)

    # Count non-nan values within the EMIT and Landsat imagery to make sure they're extensive enough to register
    emit_non_nan_pixel_count = emit_landsat_equiv.count().item()
    landsat_non_nan_pixel_count = landsat_image_band_subset.count().item()
    print("  Number of non-nan pixels in EMIT image: ", str(emit_non_nan_pixel_count))
    print("  Number of non-nan pixels in EMIT image: ", str(landsat_non_nan_pixel_count))
    if np.min((emit_non_nan_pixel_count, landsat_non_nan_pixel_count)) < min_pixel_count:
        print("  Too few pixels, skipping this set of images.")
        continue

    emit_ndvi = getNDVI(emit_landsat_equiv)
    landsat_ndvi = getNDVI(landsat_image_band_subset)
    emit_nbr = getNBR(emit_landsat_equiv)
    landsat_nbr = getNBR(landsat_image_band_subset)
    emit_ndmi = getNDMI(emit_landsat_equiv)
    landsat_ndmi = getNDMI(landsat_image_band_subset)

    emit_ndvi, landsat_ndvi = normalizeImages(emit_ndvi, landsat_ndvi)
    emit_nbr, landsat_nbr = normalizeImages(emit_nbr, landsat_nbr)
    emit_ndmi, landsat_ndmi = normalizeImages(emit_ndmi, landsat_ndmi)

    # Clip and convert to uint8 for OpenCV
    def to_uint8(image):
        return np.nan_to_num(np.round(np.clip((image.values) * 255, 0, 255)), nan=0.0).astype(np.uint8)
    emit_ndvi_cv2 = to_uint8(emit_ndvi)
    landsat_ndvi_cv2 = to_uint8(landsat_ndvi)
    emit_nbr_cv2 = to_uint8(emit_nbr)
    landsat_nbr_cv2 = to_uint8(landsat_nbr)
    emit_ndmi_cv2 = to_uint8(emit_ndmi)
    landsat_ndmi_cv2 = to_uint8(landsat_ndmi)

    # Apply a Gaussian blur to the Landsat image
    emit_ndvi_cv2 = cv2.GaussianBlur(emit_ndvi_cv2, (3, 3), 0)
    landsat_ndvi_cv2 = cv2.GaussianBlur(landsat_ndvi_cv2, (7, 7), 0)
    emit_nbr_cv2 = cv2.GaussianBlur(emit_nbr_cv2, (3, 3), 0)
    landsat_nbr_cv2 = cv2.GaussianBlur(landsat_nbr_cv2, (7, 7), 0)
    emit_ndmi_cv2 = cv2.GaussianBlur(emit_ndmi_cv2, (3, 3), 0)
    landsat_ndmi_cv2 = cv2.GaussianBlur(landsat_ndmi_cv2, (7, 7), 0)

    emit_indices = np.transpose(np.stack((emit_ndvi_cv2, emit_nbr_cv2, emit_ndmi_cv2)),
                            (1,2,0))
    landsat_indices = np.transpose(np.stack((landsat_ndvi_cv2, landsat_nbr_cv2, landsat_ndmi_cv2)),
                                (1,2,0))

    print("  Prepared spectral index images for alignment.")

    fig_init, (ax_init_1, ax_init_2) = plt.subplots(1, 2, figsize=(12, 6))
    ax_init_1.set_title('Landsat Index Falsecolor')
    ax_init_1.imshow(emit_indices, vmin=0, vmax=255)
    ax_init_2.set_title('EMIT Index Falsecolor')
    ax_init_2.imshow(landsat_indices, vmin=0, vmax=255)

    fig_init.savefig(target_directory + "output_plots/index_comparison/" + emit_filename.replace('.tif', '_alignment_initial.png'))

    # Generate Keypoints and Descriptors
    sift = cv2.SIFT_create()
    emit_keypoints, emit_descriptors = sift.detectAndCompute(emit_indices, None)
    landsat_keypoints, landsat_descriptors = sift.detectAndCompute(landsat_indices, None)

    # Check if descriptors were generated correctly
    if (emit_keypoints is None) or (emit_descriptors is None) or (landsat_keypoints is None) or (landsat_descriptors is None):
        print("  For some reason, descriptor definition failed for this image. Skipping it.")
        continue

    # Similarly, if there are too few descriptors generated, matching isn't possible. We'll use  a user-specified threshold requirement
    if (len(emit_keypoints) < min_keypoints) or (len(landsat_keypoints) < min_keypoints):
        print("  Too few keypoints generated. We only have ", str(len(emit_keypoints)), " for EMIT and ", str(len(landsat_keypoints)), " for Landsat. Skipping this image.")
        continue

    # Generate Matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(landsat_descriptors, emit_descriptors, k=10)
    print('On initial search, found ' + str(len(matches)) + ' total matches.')

    if len(matches) < 4:  # RANSAC requires at least 4 points for homography
        print('  Not enough good matches for RANSAC, skipping this EMIT file.')
        # Write homography matrix and number of inliers to a data frame
        alignment_metrics_filepath = target_directory + "alignment_metrics/emit_landsat_alignment_metrics.csv"
        if not os.path.exists(alignment_metrics_filepath):
            with open(alignment_metrics_filepath, 'w') as f:
                f.write("emit_filename,num_ransac_matches,num_allowed_matches,num_total_matches,M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8\n")
        with open(alignment_metrics_filepath, 'a') as f:
            f.write(f"{emit_filename},{0},{0},{len(matches)},{0},{0},{0},{0},{0},{0},{0},{0},{0}\n")
        continue

    # Lowes Ratio Test
    good_matches = []
    lowe_threshold = 1000
    good_matches = []
    distance_threshold = 10 # pixels
    emit_vs_landsat_scale_x = emit_indices.shape[1] / landsat_indices.shape[1]
    emit_vs_landsat_scale_y = emit_indices.shape[0] / landsat_indices.shape[0]
    print('Scale X:', emit_vs_landsat_scale_x)
    print('Scale Y:', emit_vs_landsat_scale_y)
    for mset in matches:
        mset = sorted(mset, key=lambda x: x.distance)
        if len(mset) >= 2:
            for ind in range(0, len(mset)-1):
                if mset[ind].distance < lowe_threshold * mset[ind+1].distance:
                    distance_from_expected = np.sqrt( (emit_keypoints[mset[ind].trainIdx].pt[0] - emit_vs_landsat_scale_x*landsat_keypoints[mset[ind].queryIdx].pt[0])**2 +
                                                    (emit_keypoints[mset[ind].trainIdx].pt[1] - emit_vs_landsat_scale_y*landsat_keypoints[mset[ind].queryIdx].pt[1])**2 )
                    if(distance_from_expected < distance_threshold):
                        good_matches.append(mset[ind])


    fig_alignment, ax_alignment = plt.subplots(figsize=(12, 6))
    img_matches_init = cv2.drawMatches(landsat_indices, landsat_keypoints, emit_indices, emit_keypoints,
                                good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax_alignment.imshow(img_matches_init)
    fig_alignment.savefig(target_directory + "output_plots/alignment_images_initial/" + emit_filename.replace('.tif', '_alignment_initial.png'))

    if len(good_matches) > 4:  # RANSAC requires at least 4 points for homography
            src_pts = np.float32([landsat_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([emit_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    else:
        print('  Not enough good matches for RANSAC, skipping this EMIT file.')
        # Write homography matrix and number of inliers to a data frame
        alignment_metrics_filepath = target_directory + "alignment_metrics/emit_landsat_alignment_metrics.csv"
        if not os.path.exists(alignment_metrics_filepath):
            with open(alignment_metrics_filepath, 'w') as f:
                f.write("emit_filename,num_ransac_matches,num_allowed_matches,num_total_matches,M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8\n")
        with open(alignment_metrics_filepath, 'a') as f:
            f.write(f"{emit_filename},{0},{len(good_matches)},{len(matches)},{0},{0},{0},{0},{0},{0},{0},{0},{0}\n")
        continue

    # Estimate homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0) # 5.0 is the reprojection threshold

    # Filter matches based on the inlier mask
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]

    print('After RANSAC, retained ' + str(len(inlier_matches)) + ' inlier matches.')

    if len(inlier_matches) < 4:
        print('  RANSAC ran, but found too few inliers in common for the homography to be good.')
        # Write homography matrix and number of inliers to a data frame
        alignment_metrics_filepath = target_directory + "alignment_metrics/emit_landsat_alignment_metrics.csv"
        if M is not None: # Homography can be None if there are no valid homographies found
            output_string = f"{emit_filename},{len(inlier_matches)},{len(good_matches)},{len(matches)},{M[0,0]},{M[0,1]},{M[0,2]},{M[1,0]},{M[1,1]},{M[1,2]},{M[2,0]},{M[2,1]},{M[2,2]}\n"
        else:
            output_string = f"{emit_filename},{len(inlier_matches)},{len(good_matches)},{len(matches)},{0},{0},{0},{0},{0},{0},{0},{0},{0}\n"
        if not os.path.exists(alignment_metrics_filepath):
            with open(alignment_metrics_filepath, 'w') as f:
                f.write("emit_filename,num_ransac_matches,num_allowed_matches,num_total_matches,M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8\n")
        with open(alignment_metrics_filepath, 'a') as f:
            f.write(output_string)
        continue

    fig_ransac, ax_ransac = plt.subplots(figsize=(12, 6))

    img_matches_ransac = cv2.drawMatches(landsat_indices, landsat_keypoints, emit_indices, emit_keypoints,
                                inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ax_ransac.imshow(img_matches_ransac)
    fig_ransac.savefig(target_directory + "output_plots/alignment_images_ransac/" + emit_filename.replace('.tif', '_alignment_ransac.png'))

    print('The homography matrix is: ' + str(M))

    # Write homography matrix and number of inliers to a data frame
    alignment_metrics_filepath = target_directory + "alignment_metrics/emit_landsat_alignment_metrics.csv"
    if not os.path.exists(alignment_metrics_filepath):
        with open(alignment_metrics_filepath, 'w') as f:
            f.write("emit_filename,num_ransac_matches,num_allowed_matches,num_total_matches,M_0,M_1,M_2,M_3,M_4,M_5,M_6,M_7,M_8\n")
    with open(alignment_metrics_filepath, 'a') as f:
        f.write(f"{emit_filename},{len(inlier_matches)},{len(good_matches)},{len(matches)},{M[0,0]},{M[0,1]},{M[0,2]},{M[1,0]},{M[1,1]},{M[1,2]},{M[2,0]},{M[2,1]},{M[2,2]}\n")

    landsat_output_filepath = target_directory + "synthetic_imagery_aligned/" + os.path.basename(emit_filename).replace(".tif", "") + "_landsat_aligned.tif"
    print(landsat_output_filepath)

    landsat_image_warped = xr.DataArray(
        np.zeros((landsat_image.shape[0], emit_ndvi_cv2.shape[0], emit_ndvi_cv2.shape[1])),
        coords={
            "band": np.arange(1, landsat_image.shape[0] + 1), # Assign new band numbers
            "y": emit_landsat_equiv.y,
            "x": emit_landsat_equiv.x
        },
        dims=["band", "y", "x"]
    )
    landsat_image_warped.rio.write_crs(emit_landsat_equiv.rio.crs, inplace=True)
    landsat_image_warped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    landsat_image_warped.attrs.update(landsat_image.attrs)

    for band in range(landsat_image.shape[0]):
        print('Warping band ' + str(band+1) + ' of ' + str(landsat_image.shape[0]))
        warped_band = cv2.warpPerspective(landsat_image.isel(band=band).values, M,
                                        (emit_ndvi_cv2.shape[1], emit_ndvi_cv2.shape[0]),
                                        flags=cv2.INTER_AREA)
        landsat_image_warped.values[band, :, :] = warped_band
    landsat_image_warped.attrs['long_name'] = landsat_image.attrs['long_name']
    landsat_image_warped.rio.to_raster(landsat_output_filepath)


    pt_str = emit_filename.split('_')[1]
    other_correlates_filepath = target_directory + "other_correlates/pt_" + pt_str + "_other_correlates.tif"
    other_correlates_output_filepath = target_directory + "other_correlates_aligned/" + os.path.basename(emit_filename).replace(".tif", "") + "_other_correlates_aligned.tif"
    print(other_correlates_filepath)
    print(other_correlates_output_filepath)
    if not os.path.exists(other_correlates_filepath):
        print('  Other correlates file not found, skipping: ' + other_correlates_filepath)
        continue
    other_correlates = rxr.open_rasterio(other_correlates_filepath, masked=True).squeeze()

    other_correlates_warped = xr.DataArray(
        np.zeros((other_correlates.shape[0], emit_ndvi_cv2.shape[0], emit_ndvi_cv2.shape[1])),
        coords={
            "band": np.arange(1, other_correlates.shape[0] + 1), # Assign new band numbers
            "y": emit_landsat_equiv.y,
            "x": emit_landsat_equiv.x
        },
        dims=["band", "y", "x"]
    )
    other_correlates_warped.rio.write_crs(emit_landsat_equiv.rio.crs, inplace=True)
    other_correlates_warped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    other_correlates_warped.attrs.update(other_correlates.attrs)

    for band in range(other_correlates_warped.shape[0]):
        print('Warping band ' + str(band+1) + ' of ' + str(other_correlates_warped.shape[0]))
        warped_band = cv2.warpPerspective(other_correlates.isel(band=band).values, M,
                                        (emit_ndvi_cv2.shape[1], emit_ndvi_cv2.shape[0]),
                                        flags=cv2.INTER_AREA)
        other_correlates_warped.values[band, :, :] = warped_band
    other_correlates_warped.attrs['long_name'] = other_correlates.attrs['long_name']
    other_correlates_warped.rio.to_raster(other_correlates_output_filepath)
    
    # Phenology warping
    #   Get date from filename
    date_str = emit_filename.split('_')[6][0:8]
    pt_number = emit_filename.split('_')[1]
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    day_of_year = np.datetime64(f'{year}-{month:02d}-{day:02d}') - np.datetime64(f'{year}-01-01')
    day_of_year = int(day_of_year.astype(int)) + 1
    print('  Phenology date: ' + str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2))
    current_year_phenology_filepath = target_directory + "landsat_phenology/" + "pt_" + pt_number + "_" + str(year) + "_phenology.tif"
    previous_year_phenology_filepath = target_directory + "landsat_phenology/" + "pt_" + pt_number + "_" + str(year-1) + "_phenology.tif"
    phenology_output_filepath = target_directory + "landsat_phenology_aligned/" + os.path.basename(emit_filename).replace(".tif", "") + "_phenology_aligned.tif"
    print('  Loading phenology file: ' + current_year_phenology_filepath + ' and ' + previous_year_phenology_filepath)
    print('  Saving aligned phenology file to: ' + phenology_output_filepath)
    if not (os.path.exists(current_year_phenology_filepath) and os.path.exists(previous_year_phenology_filepath)):
        print('  At least one required phenology file not found, skipping: ' + current_year_phenology_filepath + ' and ' + previous_year_phenology_filepath)
        continue
    # Load phenology files
    previous_year_phenology = rxr.open_rasterio(previous_year_phenology_filepath, masked=True).squeeze()
    current_year_phenology = rxr.open_rasterio(current_year_phenology_filepath, masked=True).squeeze()
    yearly_time_steps = previous_year_phenology.shape[0]
    # Combine previous and current year based on current date, keeping only the 52 weeks before the current date
    combined_phenology = xr.concat([previous_year_phenology, current_year_phenology], dim='band')
    target_time_step = int(np.ceil(day_of_year / 365 * yearly_time_steps))
    start_band = int(np.floor(combined_phenology.shape[0]/2)) + (target_time_step - yearly_time_steps)
    combined_phenology_subset = combined_phenology.isel(band=slice(start_band, start_band + yearly_time_steps))
    # Warp phenology data
    combined_phenology_warped = xr.DataArray(
        np.zeros((combined_phenology_subset.shape[0], emit_ndvi_cv2.shape[0], emit_ndvi_cv2.shape[1])),
        coords={
            "band": np.arange(1, combined_phenology_subset.shape[0] + 1), # Assign new band numbers
            "y": emit_landsat_equiv.y,
            "x": emit_landsat_equiv.x
        },
        dims=["band", "y", "x"]
    )
    combined_phenology_warped.rio.write_crs(emit_landsat_equiv.rio.crs, inplace=True)
    combined_phenology_warped.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    combined_phenology_warped.attrs.update(combined_phenology_subset.attrs)
    for band in range(combined_phenology_warped.shape[0]):
        print('Warping phenology time step ' + str(band+1) + ' of ' + str(combined_phenology_warped.shape[0]))
        warped_band = cv2.warpPerspective(combined_phenology_subset.isel(band=band).values, M, 
                                        (emit_ndvi_cv2.shape[1], emit_ndvi_cv2.shape[0]),
                                        flags=cv2.INTER_AREA)
        combined_phenology_warped.values[band, :, :] = warped_band
    combined_phenology_warped.attrs['long_name'] = combined_phenology_subset.attrs['long_name']
    combined_phenology_warped.rio.to_raster(phenology_output_filepath)
