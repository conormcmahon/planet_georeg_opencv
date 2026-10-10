"""Diagnostic PNGs for a registration run: band composites, keypoint
overlays, correspondence lines, and RANSAC inlier/outlier spatial masks.

Purely optional -- register_local/register_to_grid never call these
themselves. A caller (a CLI script, or psnorm) imports this module only if
it wants the plots, so importing the core library never requires
matplotlib.
"""

from __future__ import annotations

import cv2
import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_RANSAC_MASK_CMAP = ListedColormap(['lightgray', 'red', 'green'])
_RANSAC_MASK_LEGEND = [
    Patch(color='lightgray', label='not sampled'),
    Patch(color='red', label='outlier'),
    Patch(color='green', label='inlier'),
]


def plot_ransac_mask_grid(result: dict, matched_pairs, when: str, label: str, out_path: str) -> None:
    """Save a PNG with one subplot per matched band, each a binary spatial
    mask of which sampled pixels were RANSAC inliers vs. outliers for the
    moving ~= scale*fixed + intercept rescaling model (see
    band_metrics.compute_band_metrics_matched_resolution). Pixels that
    weren't part of the sample are shown as "not sampled" rather than left
    blank.
    """
    n = len(matched_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5), squeeze=False)
    coarse_shape = result['coarse_shape']
    for ax, (moving_idx, _) in zip(axes[0], matched_pairs):
        mask_img = np.full(coarse_shape, -1, dtype=np.int8)
        rows, cols = result['mask_rows'][moving_idx], result['mask_cols'][moving_idx]
        if len(rows):
            mask_img[rows, cols] = result['mask_inlier'][moving_idx].astype(np.int8)
        frac = result['inlier_frac'][moving_idx]
        frac_str = f"{frac:.1%}" if np.isfinite(frac) else "n/a"
        ax.imshow(mask_img + 1, cmap=_RANSAC_MASK_CMAP, vmin=0, vmax=2, interpolation='nearest')
        ax.set_title(f'Band {moving_idx} — inliers: {frac_str}')
        ax.axis('off')
    grid_space = 'moving' if result['coarse_is_moving'] else 'fixed'
    fig.suptitle(f'RANSAC rescaling model — {when} alignment ({grid_space}-clip pixel space)\n{label}')
    fig.legend(handles=_RANSAC_MASK_LEGEND, loc='lower center', ncol=3)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path)
    plt.close(fig)


def plot_band_comparison(moving_rgb: np.ndarray, fixed_rgb: np.ndarray, moving_label: str, fixed_label: str, out_path: str) -> None:
    """Side-by-side band-composite display images."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title(f'Moving — Band Composite\n{moving_label}')
    ax1.imshow(moving_rgb)
    ax2.set_title(f'Fixed — Band Composite\n{fixed_label}')
    ax2.imshow(fixed_rgb)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_keypoints(
    moving_rgb: np.ndarray, fixed_rgb: np.ndarray,
    moving_pts: np.ndarray, fixed_pts: np.ndarray,
    title_suffix: str, out_path: str,
    point_color: str = 'yellow',
) -> None:
    """Side-by-side display images with keypoints overlaid (used for both
    "all matches" and "RANSAC inliers" plots -- just pass the appropriate
    point set and color)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.set_title(f'Moving — {title_suffix}')
    ax1.imshow(moving_rgb)
    if len(moving_pts):
        ax1.scatter(moving_pts[:, 0], moving_pts[:, 1], s=8, c=point_color, linewidths=0)
    ax2.set_title(f'Fixed — {title_suffix}')
    ax2.imshow(fixed_rgb)
    if len(fixed_pts):
        ax2.scatter(fixed_pts[:, 0], fixed_pts[:, 1], s=8, c=point_color, linewidths=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correspondences(
    moving_rgb: np.ndarray, fixed_rgb: np.ndarray,
    moving_pts: np.ndarray, fixed_pts: np.ndarray,
    moving_label: str, fixed_label: str, out_path: str,
    target_display_h: int = 600, max_lines: int = 2000, rng=None,
) -> None:
    """Correspondence-line plot: both display images resized to a common
    height and placed side by side, with lines connecting matched points."""
    if rng is None:
        rng = np.random.default_rng(seed=0)

    moving_h, moving_w = moving_rgb.shape[:2]
    fixed_h, fixed_w = fixed_rgb.shape[:2]
    scale_moving = target_display_h / moving_h
    scale_fixed = target_display_h / fixed_h

    def resize_display(arr, s):
        return cv2.resize(arr, (max(1, int(round(arr.shape[1] * s))), max(1, int(round(arr.shape[0] * s)))),
                           interpolation=cv2.INTER_AREA)

    moving_disp = resize_display(moving_rgb, scale_moving)
    fixed_disp = resize_display(fixed_rgb, scale_fixed)

    disp_moving_h, disp_moving_w = moving_disp.shape[:2]
    disp_fixed_h, disp_fixed_w = fixed_disp.shape[:2]
    canvas_h = max(disp_moving_h, disp_fixed_h)
    canvas_w = disp_moving_w + disp_fixed_w
    is_color = moving_disp.ndim == 3

    canvas_shape = (canvas_h, canvas_w, 3) if is_color else (canvas_h, canvas_w)
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas[:disp_moving_h, :disp_moving_w] = moving_disp
    canvas[:disp_fixed_h, disp_moving_w:disp_moving_w + disp_fixed_w] = fixed_disp

    if len(moving_pts) > max_lines:
        idx = rng.choice(len(moving_pts), max_lines, replace=False)
        plot_moving_pts, plot_fixed_pts = moving_pts[idx], fixed_pts[idx]
    else:
        plot_moving_pts, plot_fixed_pts = moving_pts, fixed_pts

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_title(
        f'Correspondences — {len(moving_pts)} total ({len(plot_moving_pts)} shown)\n'
        f'Moving: {moving_label}  |  Fixed: {fixed_label}'
    )
    ax.imshow(canvas, cmap='gray' if not is_color else None)
    ax.axvline(x=disp_moving_w, color='white', linewidth=1, linestyle='--')

    for mp, fp in zip(plot_moving_pts, plot_fixed_pts):
        ax.plot(
            [mp[0] * scale_moving, fp[0] * scale_fixed + disp_moving_w],
            [mp[1] * scale_moving, fp[1] * scale_fixed],
            color='lime', linewidth=0.5, alpha=0.6,
        )
    if len(plot_moving_pts):
        ax.scatter(plot_moving_pts[:, 0] * scale_moving, plot_moving_pts[:, 1] * scale_moving,
                    s=6, c='red', zorder=5, linewidths=0)
        ax.scatter(plot_fixed_pts[:, 0] * scale_fixed + disp_moving_w, plot_fixed_pts[:, 1] * scale_fixed,
                    s=6, c='cyan', zorder=5, linewidths=0)

    ax.set_xlabel('← Moving          Fixed →')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
