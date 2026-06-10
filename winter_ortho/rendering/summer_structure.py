from __future__ import annotations

import numpy as np
from scipy import ndimage

from winter_ortho.rendering.relief import compress_hillshade, luminance


def summer_luminance_map(
    summer_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> np.ndarray:
    """Normalized 0–1 luminance field from summer imagery (shadow structure)."""
    lum = luminance(summer_rgb)
    if mask is not None and np.any(mask > 0):
        sample = lum[mask > 0]
    else:
        sample = lum.ravel()
    if sample.size == 0:
        return np.zeros_like(lum, dtype=np.float32)
    lo, hi = np.percentile(sample, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(lum, dtype=np.float32)
    return np.clip((lum - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def blend_macro_hillshade(
    hillshade: np.ndarray,
    hillshade_generalized: np.ndarray | None,
    *,
    macro_weight: float = 0.85,
    compression: float = 0.45,
) -> np.ndarray:
    """Prefer smoothed relief on meadows; keep a little fine detail on ridges."""
    fine = compress_hillshade(hillshade, compression * 0.55)
    if hillshade_generalized is None:
        return fine
    macro = compress_hillshade(hillshade_generalized, compression)
    weight = float(np.clip(macro_weight, 0.0, 1.0))
    return np.clip(macro * weight + fine * (1.0 - weight), 0.0, 1.0).astype(np.float32)


def combined_shade_field(
    hillshade: np.ndarray,
    summer_rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    hillshade_weight: float,
    summer_weight: float,
    compression: float,
    hillshade_generalized: np.ndarray | None = None,
    macro_hillshade_weight: float = 0.85,
    cast_shadow: np.ndarray | None = None,
    cast_shadow_weight: float = 0.0,
) -> np.ndarray:
    """Blend generalized DEM hillshade with summer luminance and cast shadows."""
    hill = blend_macro_hillshade(
        hillshade,
        hillshade_generalized,
        macro_weight=macro_hillshade_weight,
        compression=compression,
    )
    summer_shade = summer_luminance_map(summer_rgb, mask)
    if cast_shadow is not None and cast_shadow_weight > 0:
        summer_shade = np.clip(
            summer_shade * (1.0 - cast_shadow_weight)
            + (1.0 - cast_shadow) * cast_shadow_weight,
            0.0,
            1.0,
        )
    total = hillshade_weight + summer_weight
    if total <= 0:
        return hill
    return np.clip(
        (hillshade_weight * hill + summer_weight * summer_shade) / total,
        0.0,
        1.0,
    ).astype(np.float32)


def summer_cast_shadow_field(
    summer_rgb: np.ndarray,
    *,
    building_mask: np.ndarray | None = None,
    forest_mask: np.ndarray | None = None,
    reference_sigma_px: float = 32.0,
    fine_sigma_px: float = 6.0,
    building_radius_px: int = 10,
    building_boost: float = 1.6,
    forest_boost: float = 1.25,
    base_strength: float = 0.35,
) -> np.ndarray:
    """
    Extract visible cast shadows from summer orthophoto (buildings, forest canopy).
    Returns 0–1 where 1 = strong shadow to preserve on snow.
    """
    lum = luminance(summer_rgb)
    ref_coarse = ndimage.gaussian_filter(lum, sigma=max(1.0, reference_sigma_px))
    ref_fine = ndimage.gaussian_filter(lum, sigma=max(0.5, fine_sigma_px))
    dark_coarse = np.clip((ref_coarse - lum) / np.maximum(ref_coarse, 0.05), 0.0, 1.0)
    dark_fine = np.clip((ref_fine - lum) / np.maximum(ref_fine, 0.05), 0.0, 1.0)

    weight = np.full(lum.shape, base_strength, dtype=np.float32)
    if building_mask is not None and building_mask.any():
        near_building = ndimage.binary_dilation(
            building_mask > 0,
            iterations=max(1, building_radius_px),
        )
        weight[near_building] = np.maximum(weight[near_building], building_boost)
        cast = dark_coarse * 0.55 + dark_fine * 0.45
    else:
        cast = dark_coarse

    if forest_mask is not None and forest_mask.any():
        weight[forest_mask > 0] = np.maximum(weight[forest_mask > 0], forest_boost)
        canopy = dark_coarse * 0.70 + dark_fine * 0.30
        cast = np.where(forest_mask > 0, np.maximum(cast, canopy), cast)

    return np.clip(cast * weight, 0.0, 1.0).astype(np.float32)


def apply_summer_cast_shadows(
    rgb: np.ndarray,
    cast_shadow: np.ndarray,
    *,
    snow_fraction: np.ndarray,
    strength: float = 0.55,
    min_snow: float = 0.25,
    max_darken: float = 0.42,
) -> np.ndarray:
    """Paint summer cast shadows onto snowy areas."""
    snow_weight = np.clip(
        (snow_fraction - min_snow) / max(1.0 - min_snow, 1e-3),
        0.0,
        1.0,
    )
    darken = np.clip(cast_shadow * float(strength) * snow_weight * float(max_darken), 0.0, max_darken)
    lum = luminance(rgb)
    scaled = np.clip(lum - darken, 0.0, 1.0)
    lum_safe = np.maximum(lum, 1e-4)
    return np.clip(rgb * (scaled / lum_safe)[..., np.newaxis], 0.0, 1.0)


def remap_luminance_structure(
    rgb: np.ndarray,
    summer_rgb: np.ndarray,
    strength: np.ndarray | float,
    *,
    target_lo: float,
    target_hi: float,
    mask: np.ndarray | None = None,
    summer_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remap luminance to follow summer shadow layout within a winter target range."""
    s_map = summer_luminance_map(summer_rgb, summer_mask if summer_mask is not None else mask)
    target_lum = target_lo + s_map * (target_hi - target_lo)
    cur_lum = luminance(rgb)
    if np.ndim(strength) == 0:
        w = float(strength)
        new_lum = cur_lum * (1.0 - w) + target_lum * w
    else:
        w = np.clip(strength, 0.0, 1.0)
        new_lum = cur_lum * (1.0 - w) + target_lum * w
    lum_safe = np.maximum(cur_lum, 1e-4)
    scaled = np.clip(rgb * (new_lum / lum_safe)[..., np.newaxis], 0.0, 1.0)
    if mask is None:
        return scaled
    out = rgb.copy()
    active = mask > 0
    out[active] = scaled[active]
    return out


def soft_highlight_rolloff(
    rgb: np.ndarray,
    *,
    knee: float = 0.82,
    compression: float = 0.55,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compress blown highlights while keeping midtones and shadows."""
    lum = luminance(rgb)
    knee = float(np.clip(knee, 0.5, 0.98))
    comp = float(np.clip(compression, 0.1, 1.0))
    excess = np.clip(lum - knee, 0.0, 1.0) / max(1.0 - knee, 1e-4)
    rolled = np.where(
        lum > knee,
        knee + excess * comp * (1.0 - knee),
        lum,
    )
    lum_safe = np.maximum(lum, 1e-4)
    scaled = np.clip(rgb * (rolled / lum_safe)[..., np.newaxis], 0.0, 1.0)
    if mask is None:
        return scaled
    out = rgb.copy()
    active = mask > 0
    out[active] = scaled[active]
    return out


def apply_summer_anchored_grade(
    winter_rgb: np.ndarray,
    summer_rgb: np.ndarray,
    class_masks: dict[str, np.ndarray],
    grade_cfg: dict,
) -> np.ndarray:
    """Final pass: one global summer-anchored tone curve (preserves landscape structure)."""
    protect = np.zeros(winter_rgb.shape[:2], dtype=bool)
    for name in grade_cfg.get("protect_masks", ["water_mask"]):
        mask = class_masks.get(name)
        if mask is not None:
            protect |= mask > 0

    apply_mask = ~protect
    result = winter_rgb.copy()
    class_cfgs = grade_cfg.get("classes", {})
    if class_cfgs:
        for mask_name, params in class_cfgs.items():
            class_mask = class_masks.get(mask_name)
            if class_mask is None:
                continue
            effective = (class_mask > 0) & apply_mask
            if not effective.any():
                continue
            result = remap_luminance_structure(
                result,
                summer_rgb,
                float(params.get("strength", 0.45)),
                target_lo=float(params.get("lum_lo", 0.20)),
                target_hi=float(params.get("lum_hi", 0.85)),
                mask=effective.astype(np.uint8),
                summer_mask=effective.astype(np.uint8),
            )
    else:
        remapped = remap_luminance_structure(
            result,
            summer_rgb,
            float(grade_cfg.get("global_strength", 0.52)),
            target_lo=float(grade_cfg.get("global_lum_lo", 0.14)),
            target_hi=float(grade_cfg.get("global_lum_hi", 0.86)),
            mask=apply_mask.astype(np.uint8),
        )
        result[apply_mask] = remapped[apply_mask]

    if grade_cfg.get("highlight_rolloff", True):
        result = soft_highlight_rolloff(
            result,
            knee=float(grade_cfg.get("highlight_knee", 0.83)),
            compression=float(grade_cfg.get("highlight_compression", 0.52)),
            mask=apply_mask.astype(np.uint8),
        )
    return result
