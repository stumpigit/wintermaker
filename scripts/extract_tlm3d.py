#!/usr/bin/env python3
"""Extract clipped swissTLM3D layers for the wintermaker pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from winter_ortho.utils.config import load_config

DEFAULT_SOURCE = PROJECT_ROOT / "data/raw/swisstlm/SWISSTLM3D_2026_LV95_LN02.gpkg"

# swissTLM3D 2026 text labels (field: objektart)
ROAD_OBJEKTART = {
    "3m Strasse",
    "4m Strasse",
    "6m Strasse",
    "8m Strasse",
    "10m Strasse",
    "Autobahn",
    "Autostrasse",
    "Autozug",
    "Dienstzufahrt",
    "Einfahrt",
    "Ausfahrt",
    "Hauptstrasse",
    "Platz",
    "Raststaette",
    "Verbindung",
    "Rampe",
}

PATH_OBJEKTART = {
    "1m Weg",
    "1m Wegfragment",
    "2m Weg",
    "2m Wegfragment",
    "Klettersteig",
    "Markierte Spur",
}

FOREST_OBJEKTART = {"Wald", "Wald offen"}

ROCK_OBJEKTART = {
    "Fels",
    "Fels locker",
    "Felsbloecke",
    "Felsbloecke locker",
    "Lockergestein",
    "Lockergestein locker",
}

WATER_BB_OBJEKTART = {"Stehende Gewaesser", "Fliessgewaesser"}


def _read_layer(
    source: Path,
    layer: str,
    clip_box,
    *,
    where: str | None = None,
) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(source, layer=layer, bbox=clip_box)
    except Exception as exc:
        print(f"  skip {layer}: {exc}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")
    if gdf.empty:
        return gdf
    if where:
        gdf = gdf.query(where, engine="python")
    gdf = gdf[gdf.intersects(clip_box)].copy()
    gdf["geometry"] = gdf.geometry.intersection(clip_box)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf


def _filter_objektart(gdf: gpd.GeoDataFrame, allowed: set[str]) -> gpd.GeoDataFrame:
    if gdf.empty or "objektart" not in gdf.columns:
        return gdf
    return gdf[gdf["objektart"].isin(allowed)].copy()


def _write(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    if gdf.empty:
        gdf = gpd.GeoDataFrame({"objektart": []}, geometry=[], crs="EPSG:2056")
    gdf.to_file(path, driver="GPKG")
    print(f"  wrote {path.name}: {len(gdf)} features")


def extract_tlm3d(
    *,
    source: Path,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    target_crs: str = "EPSG:2056",
) -> dict[str, int]:
    minx, miny, maxx, maxy = bbox
    if maxx <= minx or maxy <= miny:
        raise ValueError(f"Invalid bbox: {bbox}")

    span_km = max((maxx - minx), (maxy - miny)) / 1000.0
    if span_km > 50:
        print(f"Warning: bbox spans ~{span_km:.0f} km — extraction may be slow.")

    clip_box = box(minx, miny, maxx, maxy)
    print(f"Source: {source}")
    print(f"BBox:   {bbox}")
    print(f"Output: {output_dir}")

    counts: dict[str, int] = {}

    print("\nbuildings")
    buildings = _read_layer(source, "tlm_bauten_gebaeude_footprint", clip_box)
    _write(buildings, output_dir / "buildings.gpkg")
    counts["buildings"] = len(buildings)

    print("\nroads")
    streets = _read_layer(source, "tlm_strassen_strasse", clip_box)
    roads = _filter_objektart(streets, ROAD_OBJEKTART)
    _write(roads, output_dir / "roads.gpkg")
    counts["roads"] = len(roads)

    print("\npaths")
    paths = _filter_objektart(streets, PATH_OBJEKTART)
    if not streets.empty and "wanderwege" in streets.columns:
        wanderwege = streets[streets["wanderwege"].notna()]
        paths = pd.concat([paths, wanderwege], ignore_index=True)
        if "uuid" in paths.columns:
            paths = paths.drop_duplicates(subset=["uuid"])
    if not roads.empty and "uuid" in paths.columns and "uuid" in roads.columns:
        paths = paths[~paths["uuid"].isin(roads["uuid"])].copy()
    _write(paths, output_dir / "paths.gpkg")
    counts["paths"] = len(paths)

    print("\nwater")
    water_bb = _filter_objektart(
        _read_layer(source, "tlm_bb_bodenbedeckung", clip_box),
        WATER_BB_OBJEKTART,
    )
    _write(water_bb, output_dir / "water.gpkg")
    counts["water"] = len(water_bb)

    print("\nforest")
    forest = _filter_objektart(
        _read_layer(source, "tlm_bb_bodenbedeckung", clip_box),
        FOREST_OBJEKTART,
    )
    _write(forest, output_dir / "forest.gpkg")
    counts["forest"] = len(forest)

    print("\nsettlement")
    settlement_parts = [
        _read_layer(source, "tlm_areale_nutzungsareal", clip_box),
        _read_layer(source, "tlm_areale_verkehrsareal", clip_box),
        _read_layer(source, "tlm_areale_freizeitareal", clip_box),
    ]
    non_empty = [g for g in settlement_parts if not g.empty]
    if non_empty:
        settlement = gpd.GeoDataFrame(
            pd.concat(non_empty, ignore_index=True), crs=non_empty[0].crs
        )
        if "uuid" in settlement.columns:
            settlement = settlement.drop_duplicates(subset=["uuid"])
    else:
        settlement = gpd.GeoDataFrame(geometry=[], crs=target_crs)
    _write(settlement, output_dir / "settlement.gpkg")
    counts["settlement"] = len(settlement)

    print("\nlandcover")
    landcover = _read_layer(source, "tlm_bb_bodenbedeckung", clip_box)
    _write(landcover, output_dir / "landcover.gpkg")
    counts["landcover"] = len(landcover)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config/default.yaml"),
        help="Path to default.yaml",
    )
    parser.add_argument(
        "--tile-id",
        default="davos_001",
        help="Tile id with bbox in config",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Master swissTLM3D GeoPackage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: paths.tlm3d parent from config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.tile_id not in config.get("tiles", {}):
        raise SystemExit(f"Tile '{args.tile_id}' not found in config")

    bbox = tuple(float(v) for v in config["tiles"][args.tile_id]["bbox"])
    if args.output_dir:
        output_dir = args.output_dir
    else:
        root = Path(args.config).resolve().parent.parent
        output_dir = root / "data/raw/tlm3d"

    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")

    counts = extract_tlm3d(source=args.source, output_dir=output_dir, bbox=bbox)
    print("\nDone:")
    for name, count in counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
