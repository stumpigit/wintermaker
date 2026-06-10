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

## Extract swissTLM3D layers

From the national GeoPackage, clip the pipeline layers for your tile bbox:

```bash
winter-ortho extract-tlm3d --tile-id davos_001
# or
python scripts/extract_tlm3d.py --tile-id davos_001
```

Source: `data/raw/swisstlm/SWISSTLM3D_2026_LV95_LN02.gpkg`  
Output: `data/raw/tlm3d/*.gpkg`

## Configuration

Edit `config/default.yaml` with your Davos test tile bounding box and source paths, then run:

```bash
winter-ortho run-all --tile-id davos_001 --profile davos
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

Exported assets are written to `viewer/data/{tile_id}/` (gitignored). The mesh is decimated for browser memory; the orthophoto texture is exported separately at higher resolution.

```bash
# Export mesh + textures only
winter-ortho viewer-export --tile-id demo_test_001 --config config/regions/demo_test.yaml

# Export and start the viewer (opens http://127.0.0.1:8765)
winter-ortho viewer --tile-id demo_test_001 --config config/regions/demo_test.yaml
```

For a denser mesh (more detail, more memory):

```bash
winter-ortho viewer-export --tile-id demo_test_001 --stride 8
```

**Controls**

- Left mouse button: rotate
- Right mouse button: pan
- Mouse wheel: zoom
- UI: switch winter/summer texture, adjust vertical exaggeration

After re-exporting, hard-refresh the browser (`Ctrl+Shift+R`) if the viewer was already open.

## Design

- Deterministic, geometry-preserving rule-based renderer (stage 1)
- All intermediate layers saved as COG GeoTIFF for explainability
- ML refinement is intentionally deferred
