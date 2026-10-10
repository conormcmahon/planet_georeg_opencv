"""Displacement fields -- either a thin-plate-spline (TPS) local field or a
uniform affine one -- and their application as a self-referential remap
that keeps a raster on its own pixel grid while correcting the per-pixel
geolocation error estimated from matched keypoints.
"""

from __future__ import annotations

import cv2
import numpy as np
import xarray as xr
from scipy.interpolate import RBFInterpolator


def fit_displacement_field(
    control_pts: np.ndarray,
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    coord_scale: tuple[float, float],
    smoothing: float = 1e-3,
):
    """Fit a thin-plate-spline RBF mapping control_pts -> (disp_x, disp_y).

    `control_pts` are (N, 2) coordinates in the displacement field's own
    pixel space (e.g. the moving image's native grid). `coord_scale` (width,
    height) normalizes coordinates to [0, 1] before fitting: the TPS kernel
    phi(r) = r^2 log(r) varies over many orders of magnitude when r spans
    thousands of pixels, leading to an ill-conditioned kernel matrix. A small
    smoothing (1e-3 in normalized space ~ sub-pixel in image space) is added
    as a safety regularizer; it has negligible effect on accuracy.

    Returns (rbf_dx, rbf_dy). Pass the same (width, height) used here as
    `coord_scale` to evaluate_displacement_field's width/height arguments.
    """
    scale_arr = np.array([[float(coord_scale[0]), float(coord_scale[1])]])
    control_pts_norm = control_pts / scale_arr
    rbf_dx = RBFInterpolator(control_pts_norm, disp_x, kernel='thin_plate_spline', smoothing=smoothing)
    rbf_dy = RBFInterpolator(control_pts_norm, disp_y, kernel='thin_plate_spline', smoothing=smoothing)
    return rbf_dx, rbf_dy


def evaluate_displacement_field(rbf_dx, rbf_dy, width: int, height: int, coarse_grid_size: int):
    """Evaluate a fitted displacement field on a coarse grid and upsample to
    (height, width) via bilinear resizing.

    Reducing `coarse_grid_size` trades accuracy for speed (RBF evaluation is
    the expensive step for a dense field); 200-400 is a practical range for
    3-4K imagery.

    Returns (full_ddx, full_ddy), each a (height, width) float32 array of
    pixel displacements in the same space the field was fit in.
    """
    scale_arr = np.array([[float(width), float(height)]])
    nc_x = min(coarse_grid_size, width)
    nc_y = min(coarse_grid_size, height)
    c_xs = np.linspace(0, width - 1, nc_x)
    c_ys = np.linspace(0, height - 1, nc_y)
    c_xx, c_yy = np.meshgrid(c_xs, c_ys)
    coarse_pts = np.column_stack([c_xx.ravel(), c_yy.ravel()])
    coarse_pts_norm = coarse_pts / scale_arr

    c_ddx = rbf_dx(coarse_pts_norm).reshape(nc_y, nc_x).astype(np.float32)
    c_ddy = rbf_dy(coarse_pts_norm).reshape(nc_y, nc_x).astype(np.float32)

    full_ddx = cv2.resize(c_ddx, (width, height), interpolation=cv2.INTER_LINEAR)
    full_ddy = cv2.resize(c_ddy, (width, height), interpolation=cv2.INTER_LINEAR)
    return full_ddx, full_ddy


def affine_displacement_field(
    M_fixed_clip_to_moving_clip: np.ndarray,
    fixed_clip_transform,
    moving_clip_transform,
    moving_native_transform,
    width: int,
    height: int,
):
    """Displacement field (in moving_native pixel space) equivalent to a
    single affine transform estimated between two clips.

    `M_fixed_clip_to_moving_clip` is the 2x3 matrix mapping fixed-clip pixel
    coords to moving-clip pixel coords (as returned by
    geometry.estimate_affine_ransac). It is lifted into moving_native's own
    pixel space via world coordinates:
      moving_native px -> world -> fixed clip px -> (M) -> moving clip px
      -> world -> moving_native px
    and each pixel's displacement is (where that chain lands) - (itself),
    the same convention evaluate_displacement_field uses, so the result
    can be passed straight to apply_displacement_field.

    Returns (full_ddx, full_ddy), each a (height, width) float32 array.
    """
    def as_3x3(affine):
        return np.array(tuple(affine)[:9], dtype=np.float64).reshape(3, 3)

    M = np.vstack([np.asarray(M_fixed_clip_to_moving_clip, dtype=np.float64), [0.0, 0.0, 1.0]])
    native_to_world = as_3x3(moving_native_transform)
    A = (
        np.linalg.inv(native_to_world)
        @ as_3x3(moving_clip_transform)
        @ M
        @ np.linalg.inv(as_3x3(fixed_clip_transform))
        @ native_to_world
    )

    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]
    full_ddx = ((A[0, 0] - 1.0) * xs + A[0, 1] * ys + A[0, 2]).astype(np.float32)
    full_ddy = (A[1, 0] * xs + (A[1, 1] - 1.0) * ys + A[1, 2]).astype(np.float32)
    return full_ddx, full_ddy


def apply_displacement_field(image: xr.DataArray, ddx: np.ndarray, ddy: np.ndarray, cv2_interpolation) -> xr.DataArray:
    """Warp every band of `image` onto its OWN pixel grid using a
    self-referential displacement field: output(p) = image(p + displacement(p)).

    The returned array keeps `image`'s exact coordinates/CRS/geotransform --
    only pixel VALUES move, never the grid itself. This is what makes the
    result trivially grid-aligned with whatever `image` was already aligned
    with. Nodata (NaN) is propagated: any output pixel sampled from a NaN
    source pixel, or from outside the image's extent, is set to NaN.
    """
    height, width = image.shape[1], image.shape[2]

    # Build cv2.remap lookup tables in-place to avoid allocating two more
    # (height x width) float32 arrays.
    map_x = ddx + np.arange(width, dtype=np.float32)
    map_y = ddy + np.arange(height, dtype=np.float32).reshape(-1, 1)

    out_of_bounds = (map_x < 0) | (map_x >= width) | (map_y < 0) | (map_y >= height)

    registered = xr.DataArray(
        np.full((image.shape[0], height, width), np.nan, dtype=np.float32),
        coords={"band": np.arange(1, image.shape[0] + 1), "y": image.y, "x": image.x},
        dims=["band", "y", "x"],
    )
    registered.rio.write_crs(image.rio.crs, inplace=True)
    registered.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    registered.attrs.update(image.attrs)

    for band in range(image.shape[0]):
        band_data = image.isel(band=band).values.astype(np.float32)
        src_nan = np.isnan(band_data)

        warped_nan = cv2.remap(
            src_nan.astype(np.float32), map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0,
        ) > 0.5
        warped_band = cv2.remap(
            np.where(src_nan, 0.0, band_data), map_x, map_y, cv2_interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )
        warped_band[out_of_bounds | warped_nan] = np.nan
        registered.values[band] = warped_band

    return registered
