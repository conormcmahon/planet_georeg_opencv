"""Global affine warp of a full raster onto another raster's full pixel grid
(as opposed to tps_warp's self-referential, grid-preserving local warp).
"""

from __future__ import annotations

import cv2
import numpy as np
import xarray as xr


def adjust_affine_for_offsets(M_3x3: np.ndarray, moving_col, moving_row, fixed_col, fixed_row) -> np.ndarray:
    """Convert a 3x3 affine-as-homography estimated on clipped images so
    that it maps full-fixed-image pixel coords to full-moving-image pixel
    coords (fixed_full -> moving_full).

    M_clip maps fixed_clip -> moving_clip. To lift that into full-image
    space:
      1. T_fixed_inv: fixed_full -> fixed_clip (subtract the fixed clip origin)
      2. M_clip:      fixed_clip -> moving_clip
      3. T_moving:    moving_clip -> moving_full (add the moving clip origin)

        M_full = T_moving @ M_clip @ T_fixed_inv
    """
    T_fixed_inv = np.float32([[1, 0, -fixed_col], [0, 1, -fixed_row], [0, 0, 1]])
    T_moving = np.float32([[1, 0, moving_col], [0, 1, moving_row], [0, 0, 1]])
    return T_moving @ M_3x3 @ T_fixed_inv


def warp_onto_fixed_grid(moving_image: xr.DataArray, M_full: np.ndarray, fixed_image: xr.DataArray) -> xr.DataArray:
    """Warp the full `moving_image` onto `fixed_image`'s exact pixel grid
    using the full-image-space affine `M_full` (fixed pixel coords -> moving
    pixel coords, i.e. the direction produced by adjust_affine_for_offsets).

    Nodata (NaN) is propagated: any output pixel sampled from a NaN source
    pixel, or from outside moving_image's footprint, is set to NaN.
    """
    fixed_h, fixed_w = fixed_image.shape[1], fixed_image.shape[2]
    moving_h, moving_w = moving_image.shape[1], moving_image.shape[2]

    # WARP_INVERSE_MAP tells warpPerspective to use M_full directly as the
    # dst->src lookup (output(p_fixed) = moving(M_full(p_fixed))) rather than
    # inverting it, which would incorrectly apply the moving->fixed direction.
    warp_flags_nn = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    warp_flags_lin = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP

    coverage = cv2.warpPerspective(
        np.ones((moving_h, moving_w), dtype=np.float32), M_full, (fixed_w, fixed_h), flags=warp_flags_lin
    )
    out_of_bounds = coverage < 0.5

    registered = xr.DataArray(
        np.full((moving_image.shape[0], fixed_h, fixed_w), np.nan, dtype=np.float32),
        coords={"band": np.arange(1, moving_image.shape[0] + 1), "y": fixed_image.y, "x": fixed_image.x},
        dims=["band", "y", "x"],
    )
    registered.rio.write_crs(fixed_image.rio.crs, inplace=True)
    registered.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    registered.attrs.update(moving_image.attrs)

    for band in range(moving_image.shape[0]):
        band_data = moving_image.isel(band=band).values.astype(np.float32)
        moving_nan = np.isnan(band_data)

        warped_nan = cv2.warpPerspective(
            moving_nan.astype(np.float32), M_full, (fixed_w, fixed_h), flags=warp_flags_nn
        ) > 0.5
        warped_band = cv2.warpPerspective(
            np.where(moving_nan, 0.0, band_data), M_full, (fixed_w, fixed_h), flags=warp_flags_lin
        )
        warped_band[out_of_bounds | warped_nan] = np.nan
        registered.values[band] = warped_band

    return registered
