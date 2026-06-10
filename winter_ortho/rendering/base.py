from __future__ import annotations

import numpy as np


def to_float_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4):
        rgb = np.moveaxis(rgb[:3], 0, -1)
    return rgb.astype(np.float32) / 255.0


def to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1) * 255.0
    return rgb.astype(np.uint8)


def blend(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: np.ndarray | float,
) -> np.ndarray:
    if np.ndim(alpha) == 0:
        weight = float(alpha)
        return base * (1.0 - weight) + overlay * weight
    alpha = alpha[..., np.newaxis] if alpha.ndim == 2 else alpha
    return base * (1.0 - alpha) + overlay * alpha


def snow_cover_alpha(
    snow_fraction: np.ndarray,
    *,
    max_snow_blend: float = 1.0,
    strength: float = 1.0,
) -> np.ndarray:
    """Cover fraction only — brightness must not reduce how much summer shows through."""
    alpha = np.clip(snow_fraction * strength, 0.0, 1.0)
    if max_snow_blend < 1.0:
        alpha = np.minimum(alpha, max_snow_blend)
    return alpha


def modulate_snow_layer_brightness(
    snow_layer: np.ndarray,
    snow_brightness: np.ndarray,
    snow_color: np.ndarray,
) -> np.ndarray:
    """Modulate shading depth while keeping the snow chroma — not a flat darken."""
    weight = snow_brightness[..., np.newaxis] if snow_brightness.ndim == 2 else snow_brightness
    flat = np.broadcast_to(snow_color, snow_layer.shape)
    return np.clip(blend(flat, snow_layer, weight), 0.0, 1.0)
