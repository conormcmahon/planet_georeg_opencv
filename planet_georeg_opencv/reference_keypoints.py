"""SIFT keypoints detected once on a reference ("fixed") image, for
registering many images against that one reference.

register_local normally detects keypoints on both images for every pair it
registers. When many `moving` images share one `fixed` reference, the
reference half of that work is the same every time: detect_reference_keypoints
does it once, and the result is passed to register_local as
`fixed_keypoints`.

Two things make the keypoints independent of any particular `moving` image:

  - each normalized-difference channel is scaled to uint8 over a stated
    range (default [-1, 1], the bounds of a normalized difference of
    non-negative bands) instead of the joint min/max of the two images
    being registered. SIFT descriptors do not depend on that scaling;
    only how many keypoints pass SIFT's contrast threshold does. `moving`
    is then scaled over its own min/max, which is what the joint range
    nearly always amounts to for a raw scene against a composite;
  - detection runs at one stated `detection_scale` rather than at whatever
    scale brings a particular clip under max_sift_dimension.

Detection is tiled, so its working memory depends on the tile size and not
on the size of the reference (SIFT needs about 0.25 GB per megapixel of
detection image).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import NamedTuple

import cv2
import numpy as np
import rioxarray as rxr
import xarray as xr

from . import nd_index, sift_matching


class DetectedFeatures(NamedTuple):
    """Keypoints of one matching channel: `points` is (N, 2) x/y pixel
    coordinates, `descriptors` the matching (N, 128) uint8 SIFT
    descriptors (OpenCV's SIFT descriptors are whole numbers 0-255)."""
    points: np.ndarray
    descriptors: np.ndarray


@dataclass
class ReferenceKeypoints:
    """Keypoints of a reference image, per normalized-difference channel.

    `channels[(band_a, band_b)]` holds the DetectedFeatures of the channel
    (band_a - band_b) / (band_a + band_b), for 0-indexed bands of the
    reference image. Point coordinates are pixels of the grid described by
    `transform` (pixel -> world, in `crs`) -- the reference's own full
    resolution grid, whatever `detection_scale` detection ran at. It may
    hold only the keypoints of part of that grid.
    """
    channels: dict
    transform: object
    crs: object
    # {(band_a, band_b): (lo, hi)} -- the value range each channel was
    # scaled over for detection (informational).
    value_ranges: dict = field(default_factory=dict)
    detection_scale: float = 1.0


def _tile_starts(length: int, tile: int) -> list[int]:
    return list(range(0, length, tile))


def detect_reference_keypoints(
    fixed,
    bands: list[int],
    *,
    valid_mask=None,
    detection_scale: float = 1.0,
    blur_kernel_base_size: int = 3,
    blur_kernel_reference_resolution: float = 3.0,
    mask_erosion_blur_multiple: float = 2,
    value_range: tuple[float, float] = (-1.0, 1.0),
    value_percentiles: tuple[float, float] | None = None,
    tile_size: int = 4000,
    tile_overlap: int = 256,
    sift_lock=None,
) -> ReferenceKeypoints:
    """Detect SIFT keypoints on every normalized-difference channel formed
    from `bands` (0-indexed, at least two) of `fixed` (a file path or a
    rioxarray-backed DataArray with NaN for nodata).

    The channels are those register_local builds for a band_map whose
    matched fixed bands are `bands`, in the same order: (bands[i],
    bands[j]) for every i < j. Each is prepared as register_local prepares
    it (masked blur sized from the image resolution, detection mask eroded
    away from nodata), except for the value range and `detection_scale`
    described in the module docstring.

    Every channel is scaled over `value_range` = (lo, hi). The default
    [-1, 1] is close to what the joint range of a raw scene and a
    reference usually comes to (a few extreme pixels set it), so it finds
    about as many keypoints as detecting for each pair does. Passing
    `value_percentiles` (e.g. (0.1, 99.9)) instead scales each channel
    over its own percentiles of the whole image (estimated from every 8th
    pixel in each direction; values beyond them saturate). That stretches
    contrast much harder and finds many times more keypoints -- measured
    on a 93-megapixel composite: 330,000-460,000 per channel against
    3,000-170,000 -- which roughly doubles matching time.

    `detection_scale` (0 < scale <= 1) is the size of the detection image
    relative to `fixed`. SIFT is scale invariant, but matches best against
    keypoints detected at a similar scale, so use roughly the scale the
    `moving` images are detected at (min(1, max_sift_dimension / their
    larger side)).

    `valid_mask` (optional 2D bool array on `fixed`'s grid, True = usable)
    excludes pixels from detection, as register_local's fixed_valid_mask
    does.

    Detection runs on tiles of `tile_size` detection pixels, each read
    with a margin of `tile_overlap` detection pixels; only keypoints in a
    tile's own (non-margin) area are kept, so each keypoint is kept once
    and has its full neighbourhood available when it is described.
    `sift_lock` is held around each SIFT call (see register_local).
    """
    if len(bands) < 2:
        raise ValueError("need at least two bands to form a normalized-difference channel")
    if not 0 < detection_scale <= 1:
        raise ValueError(f"detection_scale must be in (0, 1], got {detection_scale!r}")
    image = rxr.open_rasterio(fixed, masked=True, cache=False).squeeze() if isinstance(fixed, str) else fixed
    height, width = image.shape[-2:]
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (height, width):
            raise ValueError(f"valid_mask has shape {valid_mask.shape} but the image is {(height, width)}")

    resolution = abs(image.rio.transform().a)
    blur_kernel = sift_matching.resolution_scaled_kernel_size(
        blur_kernel_base_size, blur_kernel_reference_resolution, resolution)
    mask_margin_px = mask_erosion_blur_multiple * blur_kernel
    channel_keys = [(bands[i], bands[j]) for i, j in combinations(range(len(bands)), 2)]

    if value_percentiles is None:
        value_ranges = {key: (float(value_range[0]), float(value_range[1])) for key in channel_keys}
    else:
        # One range per channel for the whole image, so that tiles (and
        # windows of the image used later) are all scaled alike.
        step = 8
        sample = {}
        for position, band in enumerate(bands):
            arr = image.isel(band=band, y=slice(None, None, step), x=slice(None, None, step)).values
            arr = arr.astype(np.float32, copy=True)
            if valid_mask is not None:
                arr[~valid_mask[::step, ::step]] = np.nan
            sample[position] = arr
        value_ranges = {}
        for (i, j), nd in nd_index.iter_nd_indices(sample):
            finite = nd[np.isfinite(nd)]
            lo, hi = np.percentile(finite, value_percentiles) if finite.size else (0.0, 0.0)
            value_ranges[(bands[i], bands[j])] = (float(lo), float(hi))
        del sample

    # Tile geometry in the reference's own (full resolution) pixels.
    core = max(1, int(round(tile_size / detection_scale)))
    margin = int(np.ceil(tile_overlap / detection_scale))
    points = {key: [] for key in channel_keys}
    descriptors = {key: [] for key in channel_keys}
    sift = cv2.SIFT_create()

    for row0 in _tile_starts(height, core):
        for col0 in _tile_starts(width, core):
            row1, col1 = min(row0 + core, height), min(col0 + core, width)
            read_row0, read_col0 = max(0, row0 - margin), max(0, col0 - margin)
            read_rows = slice(read_row0, min(height, row1 + margin))
            read_cols = slice(read_col0, min(width, col1 + margin))
            # Keyed by position in `bands`, so channels come out in
            # channel_keys order whatever the band numbers are.
            arrays = {}
            for position, band in enumerate(bands):
                arr = image.isel(band=band, y=read_rows, x=read_cols).values.astype(np.float32, copy=True)
                if valid_mask is not None:
                    arr[~valid_mask[read_rows, read_cols]] = np.nan
                arrays[position] = arr
            if not any(np.isfinite(arr).any() for arr in arrays.values()):
                continue

            for (i, j), nd in nd_index.iter_nd_indices(arrays):
                key = (bands[i], bands[j])
                valid = ~np.isnan(nd)
                if not valid.any():
                    continue
                lo, hi = value_ranges[key]
                img = sift_matching.masked_gaussian_blur(
                    nd_index.to_uint8(nd_index.normalize_to_range(nd, lo, hi)), valid, blur_kernel)
                mask = sift_matching.erode_valid_mask(valid, mask_margin_px)
                del nd, valid
                tile_h, tile_w = img.shape
                if detection_scale < 1:
                    det_size = (max(1, int(round(tile_w * detection_scale))),
                                max(1, int(round(tile_h * detection_scale))))
                    img = cv2.resize(img, det_size, interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, det_size, interpolation=cv2.INTER_NEAREST)
                with sift_lock if sift_lock is not None else contextlib.nullcontext():
                    keypoints, desc = sift.detectAndCompute(img, mask)
                if desc is None or len(keypoints) == 0:
                    continue
                # Detection pixels -> this tile's full resolution pixels ->
                # the reference's own pixel grid.
                pts = sift_matching.detection_to_true_pixels(
                    np.array([kp.pt for kp in keypoints], dtype=np.float64), img.shape, (tile_h, tile_w))
                pts[:, 0] += read_col0
                pts[:, 1] += read_row0
                in_core = (pts[:, 0] >= col0) & (pts[:, 0] < col1) & (pts[:, 1] >= row0) & (pts[:, 1] < row1)
                points[key].append(pts[in_core].astype(np.float32))
                descriptors[key].append(desc[in_core].astype(np.uint8))
            del arrays

    channels = {
        key: DetectedFeatures(
            np.concatenate(points[key]) if points[key] else np.zeros((0, 2), np.float32),
            np.concatenate(descriptors[key]) if descriptors[key] else np.zeros((0, 128), np.uint8),
        )
        for key in channel_keys
    }
    return ReferenceKeypoints(
        channels=channels, transform=image.rio.transform(), crs=image.rio.crs,
        value_ranges=value_ranges, detection_scale=float(detection_scale),
    )


def features_in_clip(reference: ReferenceKeypoints, channel_keys, clip: xr.DataArray) -> list[DetectedFeatures]:
    """For each (band_a, band_b) in `channel_keys`, the reference keypoints
    that fall inside `clip` (a raster in the reference's CRS), with their
    coordinates converted to `clip`'s own pixels."""
    if reference.crs != clip.rio.crs:
        raise ValueError(f"fixed_keypoints are in {reference.crs} but the fixed image is in {clip.rio.crs}")
    to_clip = (~clip.rio.transform()) * reference.transform
    clip_h, clip_w = clip.shape[-2:]
    out = []
    for key in channel_keys:
        if key not in reference.channels:
            hint = " (it has the same two bands in the other order)" if key[::-1] in reference.channels else ""
            raise ValueError(
                f"fixed_keypoints has no channel for fixed bands {key}{hint}; it has {sorted(reference.channels)}. "
                f"The band_map's matched fixed bands must be the bands the keypoints were detected with, in order."
            )
        features = reference.channels[key]
        x, y = to_clip * (features.points[:, 0].astype(np.float64), features.points[:, 1].astype(np.float64))
        inside = (x >= 0) & (x < clip_w) & (y >= 0) & (y < clip_h)
        out.append(DetectedFeatures(np.column_stack([x[inside], y[inside]]), features.descriptors[inside]))
    return out
