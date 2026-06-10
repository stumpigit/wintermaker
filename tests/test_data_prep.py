from winter_ortho.data_prep.dem import (
    dem_tile_indices,
    resolve_dem_tiles,
    swissalti3d_tile_url,
)
from winter_ortho.data_prep.wmts import (
    parse_extent,
    tile_range_for_bbox,
    zoom_for_resolution,
)


def test_parse_extent():
    assert parse_extent("2782250,1185000,2789000,1190000") == (
        2782250.0,
        1185000.0,
        2789000.0,
        1190000.0,
    )


def test_dem_tile_indices_for_davos_bbox():
    bbox = (2782250.0, 1185000.0, 2789000.0, 1190000.0)
    eastings, northings = dem_tile_indices(bbox)
    assert list(eastings) == list(range(2782, 2789))
    assert list(northings) == list(range(1185, 1190))


def test_swissalti3d_tile_url():
    url = swissalti3d_tile_url(2780, 1183, 2023)
    assert url.endswith("swissalti3d_2023_2780-1183_2_2056_5728.tif")
    assert "data.geo.admin.ch/ch.swisstopo.swissalti3d" in url


def test_resolve_dem_tiles_zimmerwald_uses_available_year():
    bbox = (2602000.0, 1191000.0, 2604000.0, 1193000.0)
    tiles = resolve_dem_tiles(bbox, preferred_year=2023)
    assert len(tiles) == 4
    assert all(tile.year in {2019, 2025} for tile in tiles)
    assert {tile.cell_id for tile in tiles} == {
        "2602-1191",
        "2602-1192",
        "2603-1191",
        "2603-1192",
    }


def test_zoom_for_resolution_2m():
    assert zoom_for_resolution(2.0) == 23


def test_wmts_tile_range_for_davos_bbox_at_2m():
    bbox = (2782250.0, 1185000.0, 2789000.0, 1190000.0)
    zoom = zoom_for_resolution(2.0)
    tile_range = tile_range_for_bbox(bbox, zoom=zoom)
    assert tile_range.zoom == 23
    assert tile_range.col_min == 707
    assert tile_range.col_max == 720
    assert tile_range.row_min == 312
    assert tile_range.row_max == 322
    assert (tile_range.col_max - tile_range.col_min + 1) * (
        tile_range.row_max - tile_range.row_min + 1
    ) == 154
