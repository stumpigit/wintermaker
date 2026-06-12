# wintermaker

Rule-based pipeline to generate synthetic winter orthophotos from summer imagery, swissTLM3D vectors, and a 2 m DEM.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Data layout

Place raw inputs under `data/raw/` (or configure paths in `config/default.yaml`):

```
data/raw/
  orthophoto/     # summer RGB/RGBI GeoTIFF
  dem/            # 2 m DEM GeoTIFF
  tlm3d/          # swissTLM3D GeoPackage or shapefiles per layer
  reference/      # optional winter reference orthophoto
```

Intermediate and output tiles are written to:

```
data/intermediate/tiles/{tile_id}/
data/output/tiles/{tile_id}/
```

For regions prepared automatically (see below), raw inputs live under `data/raw/regions/{name}/` with the same subfolder layout.

## Prepare region data (download + config)

Instead of downloading and wiring paths by hand, `prepare-region` fetches the inputs for a bounding box and writes a ready-to-run region config and rendering profile:

| Source | What is fetched |
|--------|-----------------|
| [SWISSIMAGE](https://www.swisstopo.admin.ch/en/ortho-images-swissimage) WMTS | Summer orthophoto mosaic (`summer_rgb.tif`) |
| [swissALTI3D](https://www.swisstopo.admin.ch/en/height-model-swissalti3d) STAC | 2 m DEM mosaic (`dem.tif`) |
| Local swissTLM3D GeoPackage | Clipped vector layers (`buildings`, `roads`, `paths`, `water`, `forest`, `settlement`, `landcover`) |

**Prerequisite:** place the national swissTLM3D GeoPackage at  
`data/raw/swisstlm/SWISSTLM3D_2026_LV95_LN02.gpkg` (or pass `--tlm-source`).

```bash
winter-ortho prepare-region \
  --name demo_test \
  --extent 2788000,1181000,2792000,1185000
```

Extent is `minx,miny,maxx,maxy` in EPSG:2056 (LV95). The command writes:

```
data/raw/regions/demo_test/
  orthophoto/summer_rgb.tif
  dem/dem.tif
  tlm3d/*.gpkg
config/regions/demo_test.yaml
config/rendering_profiles/demo_test.yaml
```

At the end it prints the pipeline command, e.g.:

```bash
winter-ortho run-all --tile-id demo_test_001 --profile demo_test --config config/regions/demo_test.yaml
```

Useful options:

```bash
# Prefer a specific DEM vintage (falls back per tile via STAC)
winter-ortho prepare-region --name demo_test --extent ... --dem-year 2023

# Override WMTS zoom (default: derived from resolution_m in config)
winter-ortho prepare-region --name demo_test --extent ... --wmts-zoom 26

# Skip individual steps if data is already present
winter-ortho prepare-region --name demo_test --extent ... --skip-ortho
winter-ortho prepare-region --name demo_test --extent ... --skip-dem
winter-ortho prepare-region --name demo_test --extent ... --skip-tlm
```

### Re-download orthophoto at higher resolution

The summer orthophoto is fetched from [SWISSIMAGE WMTS](https://www.swisstopo.admin.ch/en/ortho-images-swissimage) and written to `data/raw/regions/{name}/orthophoto/summer_rgb.tif`. Two settings control the result:

| Setting | Effect |
|---------|--------|
| `resolution_m` in the region config | Target pixel size of the saved GeoTIFF |
| `--wmts-zoom` | WMTS tile matrix level (optional; overrides auto-selection) |

Without `--config`, `prepare-region` reads `resolution_m` from `config/default.yaml`. For an existing region, pass the region config so the intended resolution is used:

```bash
winter-ortho prepare-region \
  --name demo_test \
  --extent 2788000,1181000,2792000,1185000 \
  --config config/regions/demo_test.yaml \
  --skip-dem \
  --skip-tlm
```

`--skip-dem` and `--skip-tlm` keep the existing DEM and vectors; only the orthophoto is re-downloaded.

Available SWISSIMAGE zoom levels (matrix set `2056_28`):

| Zoom | Native resolution | Tiles for demo_test (4×4 km) |
|------|-------------------|------------------------------|
| 25 | 1.0 m | 289 |
| 26 | 0.5 m | 1024 |
| 27 | 0.25 m | 3969 |
| 28 | 0.1 m | ~25 000 (very slow) |

Set the target resolution in `config/regions/{name}.yaml`, e.g. `resolution_m: 0.5` (auto-selects zoom 26) or `resolution_m: 0.25` (zoom 27). To force a specific WMTS level regardless of `resolution_m`:

```bash
winter-ortho prepare-region \
  --name demo_test \
  --extent 2788000,1181000,2792000,1185000 \
  --config config/regions/demo_test.yaml \
  --wmts-zoom 27 \
  --skip-dem \
  --skip-tlm
```

After re-downloading, re-run the pipeline so harmonized rasters and outputs match the new resolution:

```bash
winter-ortho run-all --tile-id demo_test_001 --profile demo_test --config config/regions/demo_test.yaml
```

For the 3D viewer, re-export with a higher texture limit if needed (see [3D viewer](#3d-viewer)).

## Extract swissTLM3D layers (manual)

From the national GeoPackage, clip the pipeline layers for your tile bbox:

```bash
winter-ortho extract-tlm3d --tile-id davos_001
# or
python scripts/extract_tlm3d.py --tile-id davos_001
```

Source: `data/raw/swisstlm/SWISSTLM3D_2026_LV95_LN02.gpkg`  
Output: `data/raw/tlm3d/*.gpkg`

## Configuration

Parameter reference (German):

- [Roadmap](docs/roadmap.md) — project phases (Stage 1 MVP, extensions, Phase 8 ML)
- [Region config](docs/regions.md) — `config/regions/*.yaml` (paths, tile grid, terrain, QA)
- [Rendering profiles](docs/rendering_profiles.md) — `config/rendering_profiles/*.yaml` (snow model, class rules, visual style)

For a new region, prefer `prepare-region` (above) — it generates `config/regions/{name}.yaml` and a matching rendering profile. For the built-in Davos example, edit `config/default.yaml` with your tile bounding box and source paths, then run:

```bash
winter-ortho run-all --tile-id davos_001 --profile davos
```

`run-all` and `run-all-snow` finish with a high-quality viewer export (`--stride 2 --max-texture-dim 16384`). Use `run-all-snow` when harmonize/masks/terrain are already done and you only need to regenerate from snow-surface onwards:

```bash
winter-ortho run-all-snow --tile-id davos_001 --profile davos
```

Individual steps:

```bash
winter-ortho harmonize --tile-id davos_001
winter-ortho masks --tile-id davos_001
winter-ortho terrain --tile-id davos_001
winter-ortho snow --tile-id davos_001 --profile davos
winter-ortho render --tile-id davos_001 --profile davos
winter-ortho qa --tile-id davos_001
```

## 3D viewer

Inspect the winter orthophoto draped on the aligned DEM in a local web viewer (Three.js).

Exported assets are written to `viewer/data/{tile_id}/` (gitignored). Mesh geometry and orthophoto textures are decimated separately for browser memory.

```bash
# Export mesh + textures only
winter-ortho viewer-export --tile-id demo_test_001 --config config/regions/demo_test.yaml

# Export and start the viewer (binds 0.0.0.0:8765, opens http://127.0.0.1:8765 locally)
# Reuses an existing export; does not overwrite quality settings from viewer-export.
winter-ortho viewer --tile-id demo_test_001 --config config/regions/demo_test.yaml
```

Quality tuning (lower `stride` = denser mesh; higher `--max-texture-dim` = sharper textures):

For `demo_test_001` the source rasters are **8000×8000 px** (4000 m tile at 0.5 m/px). Defaults downsample heavily (≈1000 px texture, stride ≈32 mesh). Use explicit settings for a sharp result:

```bash
# Recommended balance for demo_test_001 (full-res texture, ~1 M mesh vertices)
winter-ortho viewer-export --tile-id demo_test_001 --config config/regions/demo_test.yaml \
  --stride 8 --max-texture-dim 8192

# Sharper mesh, still browser-friendly (stride 4 → ~4 M vertices)
winter-ortho viewer-export --tile-id demo_test_001 --config config/regions/demo_test.yaml \
  --stride 4 --max-texture-dim 8192

# Maximum detail (stride 2 → ~16 M vertices, ~900 MB on disk; may be slow in the browser)
winter-ortho viewer-export --tile-id demo_test_001 --config config/regions/demo_test.yaml \
  --stride 2 --max-texture-dim 16384
```

`winter-ortho viewer` **does not re-export** if `viewer/data/{tile_id}/scene.json` already exists. Pass `--re-export` to regenerate (e.g. after pipeline changes).

Export stats (vertex count, texture size) are printed after export and stored in `viewer/data/{tile_id}/scene.json`.

**Controls**

- Left mouse button: rotate
- Right mouse button: pan
- Mouse wheel: zoom
- UI: switch winter/summer texture, switch base DEM vs. snow surface (when exported), adjust vertical exaggeration
- Scene lighting: winter sun (~11:00, low southern sun) with blue sky background

The snow surface mesh is exported automatically when `snow_surface_dem.tif` exists for the tile (after the `snow-surface` pipeline step). Re-run `viewer-export` after regenerating snow layers.

After re-exporting, hard-refresh the browser (`Ctrl+Shift+R`) if the viewer was already open.

## Design

- Deterministic, geometry-preserving rule-based renderer (stage 1)
- All intermediate layers saved as COG GeoTIFF for explainability
- ML refinement is intentionally deferred
