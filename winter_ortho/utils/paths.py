from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def get_project_root(anchor: Path | None = None) -> Path:
    if anchor is not None:
        anchor = anchor.resolve()
        if anchor.is_file():
            anchor = anchor.parent
        current = anchor
        for _ in range(12):
            if (current / "pyproject.toml").exists() or (
                current / "config" / "default.yaml"
            ).exists():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        if anchor.name == "config":
            return anchor.parent
        return anchor
    return Path(__file__).resolve().parents[2]


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return get_project_root() / candidate


@dataclass(frozen=True)
class TilePaths:
    tile_id: str
    intermediate_dir: Path
    output_dir: Path

    @property
    def metadata(self) -> Path:
        return self.intermediate_dir / "metadata.json"

    @property
    def rgb_summer(self) -> Path:
        return self.intermediate_dir / "rgb_summer.tif"

    @property
    def dem(self) -> Path:
        return self.intermediate_dir / "dem_2m.tif"

    @property
    def nodata_mask(self) -> Path:
        return self.intermediate_dir / "nodata_mask.tif"

    @property
    def tlm_masks(self) -> Path:
        return self.intermediate_dir / "tlm_masks.tif"

    @property
    def protect_mask(self) -> Path:
        return self.intermediate_dir / "protect_mask.tif"

    @property
    def terrain_features(self) -> Path:
        return self.intermediate_dir / "terrain_features.tif"

    @property
    def snow_surface_dem(self) -> Path:
        return self.intermediate_dir / "snow_surface_dem.tif"

    @property
    def snow_thickness_m(self) -> Path:
        return self.intermediate_dir / "snow_thickness_m.tif"

    @property
    def blanket_thickness_m(self) -> Path:
        return self.intermediate_dir / "blanket_thickness_m.tif"

    @property
    def accumulation_mask(self) -> Path:
        return self.intermediate_dir / "accumulation_mask.tif"

    @property
    def snow_cover_weight(self) -> Path:
        return self.intermediate_dir / "snow_cover_weight.tif"

    @property
    def snow_fraction(self) -> Path:
        return self.output_dir / f"{self.tile_id}_snow_fraction.tif"

    @property
    def snow_brightness(self) -> Path:
        return self.output_dir / f"{self.tile_id}_snow_brightness.tif"

    @property
    def snow_texture_strength(self) -> Path:
        return self.output_dir / f"{self.tile_id}_snow_texture_strength.tif"

    @property
    def rock_visibility(self) -> Path:
        return self.output_dir / f"{self.tile_id}_rock_visibility.tif"

    @property
    def forest_snow_intensity(self) -> Path:
        return self.output_dir / f"{self.tile_id}_forest_snow_intensity.tif"

    @property
    def road_visibility(self) -> Path:
        return self.output_dir / f"{self.tile_id}_road_visibility.tif"

    @property
    def roof_snow_intensity(self) -> Path:
        return self.output_dir / f"{self.tile_id}_roof_snow_intensity.tif"

    @property
    def ice_probability(self) -> Path:
        return self.output_dir / f"{self.tile_id}_ice_probability.tif"

    @property
    def summer_exposure(self) -> Path:
        return self.output_dir / f"{self.tile_id}_summer_exposure.tif"

    @property
    def landcover_mask(self) -> Path:
        return self.output_dir / f"{self.tile_id}_landcover_mask.tif"

    @property
    def winter_rgb(self) -> Path:
        return self.output_dir / f"{self.tile_id}_winter_rgb.tif"

    @property
    def quality_flags(self) -> Path:
        return self.output_dir / f"{self.tile_id}_quality_flags.tif"

    @property
    def qa_report(self) -> Path:
        return self.output_dir / "qa_report.json"


def tile_paths(config: dict, tile_id: str) -> TilePaths:
    intermediate = Path(config["paths"]["intermediate_root"]) / tile_id
    output = Path(config["paths"]["output_root"]) / tile_id
    intermediate.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    return TilePaths(tile_id=tile_id, intermediate_dir=intermediate, output_dir=output)
