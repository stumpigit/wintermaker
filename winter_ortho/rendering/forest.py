from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, white_tophat

from winter_ortho.features.texture import band_limited_noise
from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import desaturate, luminance, shade_snow_layer


def _normalize_percentile(
    values: np.ndarray,
    active: np.ndarray,
    lo_pct: float = 35.0,
    hi_pct: float = 92.0,
) -> np.ndarray:
    if not active.any():
        return np.zeros_like(values, dtype=np.float32)
    sample = values[active]
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~active] = 0.0
    return out


def render_forest(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    forest_snow_intensity: np.ndarray,
    snow_color: np.ndarray,
    hillshade: np.ndarray,
    contrast_reduction: float,
    original_texture_visibility: float,
    crown_tophat_radius_px: int = 4,
    hillshade_strength: float = 0.18,
    crown_highlight_strength: float = 0.42,
    crown_noise_strength: float = 0.04,
    noise_scale_px: int = 8,
    **_: object,
) -> np.ndarray:
    """Blend snow color onto canopy; crown peaks receive more snow, gaps stay dark."""
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    src = out.copy()
    mean = src[active].mean(axis=0, keepdims=True)
    src[active] = src[active] * (1.0 - contrast_reduction) + mean * contrast_reduction
    src = desaturate(src, np.where(active, 0.55, 0.0))

    forest_lum = luminance(src)
    radius = max(2, crown_tophat_radius_px)
    crown_signal = _normalize_percentile(
        white_tophat(forest_lum.astype(np.float64), disk(radius)).astype(np.float32),
        active,
        40.0,
        88.0,
    )
    sigma = max(1.0, radius * 0.35)
    highpass = forest_lum - ndimage.gaussian_filter(forest_lum, sigma=sigma)
    structure = np.clip(
        crown_signal * 0.65 + np.clip(highpass / 0.08, 0.0, 1.0) * 0.35,
        0.0,
        1.0,
    )

    snow_cover = np.clip(snow_fraction * forest_snow_intensity, 0.0, 1.0)
    snow_layer = shade_snow_layer(snow_color, hillshade, hillshade_strength)
    noise = band_limited_noise(rgb.shape[:2], scale_px=noise_scale_px, seed=17)
    snow_layer = np.clip(
        snow_layer + noise[..., np.newaxis] * crown_noise_strength * (0.3 + 0.7 * structure)[..., np.newaxis],
        0.0,
        1.0,
    )

    # Snow on crowns and moderate blanket on canopy — not hue-preserving summer brighten.
    alpha = np.clip(
        snow_cover * (0.62 + 0.38 * structure),
        0.0,
        0.94,
    )
    winter = blend(src, snow_layer, alpha[..., np.newaxis])

    gap = np.clip((1.0 - structure) * (1.0 - snow_cover * 0.7), 0.0, 1.0)
    preserve = np.clip(original_texture_visibility * gap, 0.0, 0.12)
    final = blend(winter, desaturate(src, 0.85), preserve[..., np.newaxis])

    out[active] = final[active]
    return out
