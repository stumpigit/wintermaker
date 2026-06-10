from __future__ import annotations

from typing import Any

import numpy as np

from winter_ortho.utils.paths import TilePaths
from winter_ortho.utils.raster import read_raster, write_cog


def build_protect_mask(
    class_rules: dict[str, Any],
    paths: TilePaths,
    class_masks: dict[str, np.ndarray],
    *,
    transform,
    crs: str,
) -> np.ndarray:
    hard_masks = class_rules.get("protect_hard_masks", [])
    protect = np.zeros_like(next(iter(class_masks.values())), dtype=np.uint8)
    for name in hard_masks:
        mask = class_masks.get(name)
        if mask is not None:
            protect = np.maximum(protect, (mask > 0).astype(np.uint8))

    write_cog(
        str(paths.protect_mask),
        protect,
        transform=transform,
        crs=crs,
        nodata=0,
    )
    return protect
