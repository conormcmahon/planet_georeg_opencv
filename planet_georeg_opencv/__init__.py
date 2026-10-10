"""planet_georeg_opencv: OpenCV/SIFT-based image co-registration.

A sensor/CRS/resolution-agnostic library for registering one raster ("moving")
onto another ("fixed") using SIFT keypoint matching over normalized-difference
index channels, RANSAC-filtered correspondences, and either a thin-plate-spline
(TPS) local warp or a single global affine warp.

Public entry points:
    register_local   - two-step (coarse + geometrically-guided) matching and
                        either a local TPS warp or a single affine warp
                        (warp_model="tps"/"affine") that corrects `moving` onto its own
                        pixel grid (used when `moving` and `fixed` already
                        share, or are expected to share, the same grid and
                        only need sub-pixel/local correction).
    register_to_grid - one-step matching and a single global affine warp that
                        resamples `moving` onto `fixed`'s own full pixel grid
                        (used to co-register two rasters with different grids
                        onto one common grid).

Both return a `RegistrationResult` and never write to disk -- callers decide
whether/where to save the registered raster (e.g. via
`result.registered.rio.to_raster(path)`).

This library has no knowledge of any particular downstream pipeline (e.g.
psnorm) -- it only knows about rasters, band maps, and pixels.
"""

from .register import RegistrationResult, register_local, register_to_grid

__all__ = ["RegistrationResult", "register_local", "register_to_grid"]
