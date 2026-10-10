"""PlanetScope sensor band-layout convenience.

This is a thin convenience for THIS library's own PlanetScope-focused CLI
scripts (planetscope_registration.py / planetscope_registration_local.py) --
it lets them build a band_map from "4-band Dove" / "8-band SuperDove" without
the caller having to know the band order. It is intentionally not used by
register_local/register_to_grid themselves, which only ever see a band_map
and have no sensor-specific knowledge. Callers with their own band-detection
logic (e.g. psnorm's psnorm.sensors, which also knows about composite
reference rasters) should build their own band_map instead of using this.
"""

from __future__ import annotations

# Dove Classic / Dove-R (4-band): Blue, Green, Red, NIR
DOVE_BANDS = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}

# SuperDove / PSB.SD (8-band): Coastal Blue, Blue, Green I, Green, Yellow,
# Red, Red Edge, NIR. Only the four bands common to both sensor types are
# listed here.
SUPERDOVE_BANDS = {'blue': 1, 'green': 3, 'red': 5, 'nir': 7}


def identify_band_layout(image) -> tuple[str, dict[str, int]]:
    """Return (sensor type string, band-index dict) based on the image's
    band count."""
    n = image.shape[0]
    if n == 4:
        return 'Dove', DOVE_BANDS
    elif n == 8:
        return 'SuperDove', SUPERDOVE_BANDS
    else:
        raise ValueError(f"Unexpected band count {n}; expected 4 (Dove) or 8 (SuperDove).")


def band_map_from_layouts(moving_band_indices: dict[str, int], fixed_band_indices: dict[str, int]) -> list[int]:
    """Build a band_map (as used by register_local/register_to_grid) between
    two band-index dicts keyed by logical name (e.g. identify_band_layout's
    second return value), ordered by moving_band_indices' own band order.
    """
    n_moving = max(moving_band_indices.values()) + 1
    band_map = [-1] * n_moving
    for name, moving_idx in moving_band_indices.items():
        if name in fixed_band_indices:
            band_map[moving_idx] = fixed_band_indices[name]
    return band_map
