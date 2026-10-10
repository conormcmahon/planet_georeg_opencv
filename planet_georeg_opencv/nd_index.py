"""Band extraction and normalized-difference index computation.

Using every pairwise combination of a band map's matched bands (rather than a
fixed set of named indices, e.g. NDVI only) avoids assuming which bands are
present or what sensor they came from, while still producing diverse texture
signals for feature matching.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def get_matched_band_pairs(band_map: list[int]) -> list[tuple[int, int]]:
    """Return [(moving_band_idx, fixed_band_idx), ...] for all matched bands.

    `band_map[i]` is the 0-indexed fixed-image band matching 0-indexed moving
    band `i`, or -1 if moving band `i` has no match in the fixed image.
    """
    return [(i, f) for i, f in enumerate(band_map) if f != -1]


def extract_band_arrays(image, band_indices: list[int]) -> dict[int, np.ndarray]:
    """Return {band_idx: float32 numpy array} for the given 0-indexed bands."""
    return {
        idx: image.isel(band=idx).values.astype(np.float32)
        for idx in band_indices
    }


def compute_nd_indices(band_arrays: dict[int, np.ndarray]) -> dict[tuple[int, int], np.ndarray]:
    """Normalized-difference index for every pair of bands in `band_arrays`,
    keyed by (band_idx_a, band_idx_b) with a < b."""
    eps = 1e-6
    keys = sorted(band_arrays)
    out = {}
    for a, b in combinations(keys, 2):
        out[(a, b)] = (band_arrays[a] - band_arrays[b]) / (band_arrays[a] + band_arrays[b] + eps)
    return out


def normalize_pair(arr1: np.ndarray, arr2: np.ndarray, percentile_clip: float | None = None):
    """Jointly normalize two float arrays to [0, 1] using their combined range.

    By default (percentile_clip=None) uses the exact combined min/max. If
    percentile_clip is given (e.g. 2.0), the shared display range is instead
    the union of each array's OWN [percentile_clip, 100 - percentile_clip]
    range (not the percentile of the two arrays pooled together, which lets
    whichever array has more pixels or a wider native spread dominate and
    crush the other's contrast to near-zero). Values outside the final range
    are clipped to [0, 1].
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


def normalize_single(arr: np.ndarray, percentile_clip: float | None = None) -> np.ndarray:
    """Normalize one float array to [0, 1] using its own range.

    Used as a fallback when a joint (regression-calibrated) stretch isn't
    available: raw band values from different sensors can have wildly
    different native scales, so each image may need its own contrast
    stretch.
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


def percentile_histogram_match(sample_arr: np.ndarray, reference_arr: np.ndarray, percentile_clip: float):
    """Return (scale, intercept) such that scale*sample_arr + intercept has
    the same [percentile_clip, 100 - percentile_clip] range as reference_arr.

    Used to visually harmonize two band-composite display images of
    different sensors/units: matching each image's own value distribution
    needs no pixel-to-pixel correspondence between the two rasters, so it is
    unaffected by residual misregistration between them -- unlike a
    regression fit calibrated from paired pixels.

    Returns (nan, nan) if sample_arr's percentile range is degenerate (e.g.
    a uniform or all-nodata band).
    """
    lo_ref, hi_ref = np.nanpercentile(reference_arr, [percentile_clip, 100 - percentile_clip])
    lo_smp, hi_smp = np.nanpercentile(sample_arr, [percentile_clip, 100 - percentile_clip])
    if not (hi_smp > lo_smp):
        return np.nan, np.nan
    scale = (hi_ref - lo_ref) / (hi_smp - lo_smp)
    intercept = lo_ref - scale * lo_smp
    return float(scale), float(intercept)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a float [0, 1] array to uint8 [0, 255], mapping NaN to 0."""
    return np.nan_to_num(
        np.round(np.clip(arr * 255, 0, 255)), nan=0.0
    ).astype(np.uint8)
