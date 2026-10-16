"""Per-band agreement metrics (R^2, RMSE, robust scale/intercept) between a
moving and a fixed raster, used to quantify registration quality before and
after alignment.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter


def ransac_linear_fit(x, y, threshold, rng, n_iterations=200, min_inliers=2):
    """Robust linear regression y ~= slope*x + intercept via RANSAC.

    Repeatedly fits a candidate line through two random points and keeps
    the one with the most inliers (|residual| < threshold); the final
    slope/intercept are then an ordinary-least-squares refit on that best
    inlier set, and the inlier mask is recomputed against the refit line.

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


def sample_valid_pixels(valid: np.ndarray, max_points: int, rng):
    """(rows, cols) of every True pixel of the 2D `valid` mask in row-major
    order, or, if there are more than `max_points`, of a uniform random
    sample of them -- the same sample as
        flat = np.flatnonzero(valid); flat = flat[rng.choice(len(flat), max_points, replace=False)]
    but without materializing an int64 index of every valid pixel (8 bytes
    per pixel for a raster of tens of megapixels): sample positions are
    mapped to pixels through per-row counts instead.
    """
    n = int(np.count_nonzero(valid))
    if n <= max_points:
        return np.nonzero(valid)
    sel = rng.choice(n, max_points, replace=False)
    row_counts = np.count_nonzero(valid, axis=1)
    row_ends = np.cumsum(row_counts)
    order = np.argsort(sel, kind="stable")
    sel_sorted = sel[order]
    rows_sorted = np.searchsorted(row_ends, sel_sorted, side="right")
    within_row = sel_sorted - (row_ends[rows_sorted] - row_counts[rows_sorted])
    cols_sorted = np.empty_like(sel_sorted)
    breaks = np.flatnonzero(np.diff(rows_sorted)) + 1
    for start, end in zip(np.r_[0, breaks], np.r_[breaks, len(sel_sorted)]):
        cols_sorted[start:end] = np.flatnonzero(valid[rows_sorted[start]])[within_row[start:end]]
    rows, cols = np.empty_like(rows_sorted), np.empty_like(cols_sorted)
    rows[order], cols[order] = rows_sorted, cols_sorted
    return rows, cols


def compute_band_metrics_matched_resolution(
    moving_clip, fixed_clip, matched_pairs, rng,
    max_sample_points=200_000, ransac_iterations=200, ransac_threshold_frac=0.2,
):
    """Estimate per-band agreement between moving_clip and fixed_clip,
    robust to outlier pixels (e.g. real land-cover change between
    acquisition dates, clouds, or residual misregistration).

    moving_clip and fixed_clip already share a CRS but may differ in pixel
    resolution. Whichever is finer is resampled onto the other (coarser)
    image's pixel grid: with area averaging if it is at least twice as
    fine, otherwise with nearest-neighbor sampling.

    Rather than materializing a full resampled raster (which, via GDAL
    reprojection, can leave a multi-gigabyte, not-promptly-reclaimed memory
    footprint when one image is tens of megapixels), this draws a bounded
    random sample of valid coarse-grid pixels and resamples only at those
    points, so memory stays bounded regardless of image size.

    Per band, a RANSAC linear regression moving ~= scale*fixed + intercept
    is fit. The inlier threshold is 10% of the moving band's own spatial
    standard deviation. R^2, RMSE, scale, and intercept are reported twice
    per band: once using only RANSAC inliers, and once using the full
    sample (plain OLS, no outlier rejection) -- plus the fraction of the
    full sample classified as inliers.

    Also records, per band, the spatial sample locations and their
    inlier/outlier classification (mask_rows, mask_cols, mask_inlier --
    pixel row/col in the coarser image's clip grid, and a matching bool
    array), so callers can render a spatial inlier/outlier mask.

    Returns a dict of dicts, each keyed by moving band index:
        {'r2_inliers', 'r2_all', 'rmse_inliers', 'rmse_all',
         'scale_inliers', 'scale_all', 'intercept_inliers', 'intercept_all',
         'inlier_frac', 'mask_rows', 'mask_cols', 'mask_inlier'}
    plus two call-level (not per-band) keys: 'coarse_shape' (the (h, w) of
    the grid mask_rows/mask_cols are expressed in) and 'coarse_is_moving'
    (whether that grid is moving_clip's, as opposed to fixed_clip's).
    """
    moving_transform = moving_clip.rio.transform()
    fixed_transform = fixed_clip.rio.transform()
    moving_res = abs(moving_transform.a)
    fixed_res = abs(fixed_transform.a)

    moving_is_finer = moving_res < fixed_res
    if moving_is_finer:
        fine_clip, coarse_clip = moving_clip, fixed_clip
        fine_transform, coarse_transform = moving_transform, fixed_transform
        fine_res, coarse_res = moving_res, fixed_res
    else:
        fine_clip, coarse_clip = fixed_clip, moving_clip
        fine_transform, coarse_transform = fixed_transform, moving_transform
        fine_res, coarse_res = fixed_res, moving_res

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
    out['coarse_is_moving'] = not moving_is_finer

    def _set_nan(moving_idx):
        for name in stat_names:
            out[name][moving_idx] = np.nan
        out['mask_rows'][moving_idx] = np.array([], dtype=np.int64)
        out['mask_cols'][moving_idx] = np.array([], dtype=np.int64)
        out['mask_inlier'][moving_idx] = np.array([], dtype=bool)

    for moving_idx, fixed_idx in matched_pairs:
        fine_idx, coarse_idx = (moving_idx, fixed_idx) if moving_is_finer else (fixed_idx, moving_idx)

        coarse_band = coarse_clip.isel(band=coarse_idx).values.astype(np.float32, copy=False)
        rows, cols = sample_valid_pixels(~np.isnan(coarse_band), max_sample_points, rng)
        if len(rows) < 2:
            _set_nan(moving_idx)
            continue

        world_x, world_y = coarse_transform * (cols.astype(np.float64), rows.astype(np.float64))
        fine_col, fine_row = inv_fine_transform * (world_x, world_y)

        fine_band = fine_clip.isel(band=fine_idx).values.astype(np.float32, copy=False)
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
        m_vals, f_vals = (sampled, coarse_vals) if moving_is_finer else (coarse_vals, sampled)

        valid = ~(np.isnan(m_vals) | np.isnan(f_vals))
        if valid.sum() < 2:
            _set_nan(moving_idx)
            continue
        m, f = m_vals[valid], f_vals[valid]

        # --- Stats over all sampled pixels (no outlier rejection) ---
        out['rmse_all'][moving_idx] = float(np.sqrt(np.mean((m - f) ** 2)))
        corr = np.corrcoef(m, f)[0, 1]
        out['r2_all'][moving_idx] = float(corr ** 2) if np.isfinite(corr) else np.nan
        if np.std(f) > 0:
            _slope, _intercept = np.polyfit(f, m, 1)  # moving ~= slope*fixed + intercept
            out['scale_all'][moving_idx], out['intercept_all'][moving_idx] = float(_slope), float(_intercept)
        else:
            out['scale_all'][moving_idx], out['intercept_all'][moving_idx] = np.nan, np.nan

        # --- RANSAC: robust moving ~= scale*fixed + intercept, with inlier stats ---
        threshold = ransac_threshold_frac * float(
            np.nanstd(moving_clip.isel(band=moving_idx).values.astype(np.float32, copy=False))
        )
        _, _, inlier_mask = ransac_linear_fit(f, m, threshold, rng, n_iterations=ransac_iterations)
        out['inlier_frac'][moving_idx] = float(inlier_mask.mean()) if len(inlier_mask) else np.nan
        out['mask_rows'][moving_idx] = rows[valid]
        out['mask_cols'][moving_idx] = cols[valid]
        out['mask_inlier'][moving_idx] = inlier_mask

        if inlier_mask.sum() >= 2:
            m_in, f_in = m[inlier_mask], f[inlier_mask]
            out['rmse_inliers'][moving_idx] = float(np.sqrt(np.mean((m_in - f_in) ** 2)))
            corr_in = np.corrcoef(m_in, f_in)[0, 1]
            out['r2_inliers'][moving_idx] = float(corr_in ** 2) if np.isfinite(corr_in) else np.nan
            if np.std(f_in) > 0:
                _slope_in, _intercept_in = np.polyfit(f_in, m_in, 1)
                out['scale_inliers'][moving_idx] = float(_slope_in)
                out['intercept_inliers'][moving_idx] = float(_intercept_in)
            else:
                out['scale_inliers'][moving_idx], out['intercept_inliers'][moving_idx] = np.nan, np.nan
        else:
            out['r2_inliers'][moving_idx], out['rmse_inliers'][moving_idx] = np.nan, np.nan
            out['scale_inliers'][moving_idx], out['intercept_inliers'][moving_idx] = np.nan, np.nan

    return out


def build_metric_keys(matched_pairs):
    keys = []
    for when in ("before", "after"):
        for moving_idx, _ in matched_pairs:
            for stat in ("r2", "rmse", "scale", "intercept"):
                for subset in ("inliers", "all"):
                    keys.append(f"{stat}_band{moving_idx}_{when}_{subset}")
            keys.append(f"inlier_frac_band{moving_idx}_{when}")
    keys += ["mean_kp_dist_before", "mean_kp_dist_after", "avg_dx_px", "avg_dy_px"]
    # The global affine transform fitted to the matched keypoints, in the
    # moving image's own pixels (see tps_warp.describe_affine). It is the
    # warp applied by warp_model="affine"; for "tps" it is the fit the
    # control points were selected with.
    from .tps_warp import AFFINE_METRIC_KEYS
    keys += list(AFFINE_METRIC_KEYS)
    return keys


def store_band_metrics(metrics: dict, result: dict, matched_pairs, when: str) -> None:
    """Copy a compute_band_metrics_matched_resolution() result into `metrics`."""
    for moving_idx, _ in matched_pairs:
        for stat in ("r2", "rmse", "scale", "intercept"):
            for subset in ("inliers", "all"):
                metrics[f"{stat}_band{moving_idx}_{when}_{subset}"] = result[f"{stat}_{subset}"][moving_idx]
        metrics[f"inlier_frac_band{moving_idx}_{when}"] = result["inlier_frac"][moving_idx]


def write_alignment_metrics_csv(filepath, metric_keys, moving_filename, fixed_filename,
                                 num_step1_ransac_inliers, num_step1_good_matches, num_step1_raw_matches,
                                 num_ransac, num_good, num_raw, metrics=None):
    """Append one row of registration quality metrics to a CSV log.

    num_step1_* describe the coarse, unguided global search (RANSAC
    inliers among the coarse affine fit, Lowe/distance-filtered matches,
    and raw FLANN matches, respectively); the num_ransac/num_good/num_raw
    describe the same three quantities for the (optionally guided) final
    search.
    """
    import os

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = (
        "moving_filename,fixed_filename,"
        "num_step1_ransac_inliers,num_step1_good_matches,num_step1_raw_matches,"
        "num_ransac_inliers,num_good_matches,num_raw_matches,"
        + ','.join(metric_keys)
    )
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write(header + '\n')
    else:
        # Appending under a different header would silently misalign the
        # new row's columns (e.g. a log started before a metric was added).
        with open(filepath) as f:
            existing_header = f.readline().rstrip('\n')
        if existing_header != header:
            raise ValueError(
                f"'{filepath}' has different columns from the row being appended "
                f"({len(existing_header.split(','))} against {len(header.split(','))}); "
                f"write to a new file, or move the old one aside."
            )
    if metrics is None:
        metrics = {k: np.nan for k in metric_keys}
    with open(filepath, 'a') as f:
        extra_vals = ','.join(str(metrics.get(k, np.nan)) for k in metric_keys)
        f.write(
            f"{moving_filename},{fixed_filename},"
            f"{num_step1_ransac_inliers},{num_step1_good_matches},{num_step1_raw_matches},"
            f"{num_ransac},{num_good},{num_raw},{extra_vals}\n"
        )
