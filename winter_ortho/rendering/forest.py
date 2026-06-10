from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk, white_tophat

from winter_ortho.features.texture import band_limited_noise
from winter_ortho.rendering.base import blend
from winter_ortho.rendering.relief import desaturate, luminance, shade_snow_layer
from winter_ortho.rendering.summer_structure import remap_luminance_structure, summer_luminance_map


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
    winter_luminance_lo: float = 0.08,
    winter_luminance_hi: float = 0.34,
    summer_structure_strength: float = 0.82,
    green_suppression: float = 0.78,
    max_crown_snow_alpha: float = 0.48,
    canopy_blanket: float = 0.50,
    **_: object,
) -> np.ndarray:
    """Winter forest: summer shadows plus canopy blanket and crown snow."""
    out = rgb.copy()
    active = mask > 0
    if not active.any():
        return out

    summer = out.copy()
    src = desaturate(summer, np.where(active, green_suppression, 0.0))
    src = remap_luminance_structure(
        src,
        summer,
        summer_structure_strength,
        target_lo=winter_luminance_lo,
        target_hi=winter_luminance_hi,
        mask=active,
        summer_mask=active,
    )

    forest_lum = luminance(src)
    radius = max(2, crown_tophat_radius_px)
    crown_signal = _normalize_percentile(
        white_tophat(forest_lum.astype(np.float64), disk(radius)).astype(np.float32),
        active,
        45.0,
        90.0,
    )
    sigma = max(1.0, radius * 0.35)
    highpass = forest_lum - ndimage.gaussian_filter(forest_lum, sigma=sigma)
    s_map = summer_luminance_map(summer, active)
    structure = np.clip(
        crown_signal * 0.70
        + np.clip(highpass / 0.10, 0.0, 1.0) * 0.20
        + s_map * 0.10,
        0.0,
        1.0,
    )

    snow_cover = np.clip(snow_fraction * forest_snow_intensity, 0.0, 1.0)
    snow_layer = shade_snow_layer(
        snow_color,
        hillshade,
        hillshade_strength * 0.72,
        shadow_boost=1.05,
        highlight_cap=0.52,
    )
    noise = band_limited_noise(rgb.shape[:2], scale_px=noise_scale_px, seed=17)
    snow_layer = np.clip(
        snow_layer
        + noise[..., np.newaxis]
        * crown_noise_strength
        * crown_highlight_strength
        * structure[..., np.newaxis],
        0.0,
        1.0,
    )

    canopy_mix = np.clip(
        canopy_blanket + (1.0 - canopy_blanket) * np.clip(structure * 0.75 + crown_signal * 0.25, 0.0, 1.0),
        0.0,
        1.0,
    )
    alpha = np.clip(snow_cover * canopy_mix, 0.0, max_crown_snow_alpha)
    winter = blend(src, snow_layer, alpha[..., np.newaxis])

    gap = np.clip((1.0 - structure) * (1.0 - snow_cover * 0.55), 0.0, 1.0)
    preserve = np.clip(original_texture_visibility * gap, 0.0, 0.14)
    final = blend(winter, src, preserve[..., np.newaxis])

    out[active] = final[active]
    return out
