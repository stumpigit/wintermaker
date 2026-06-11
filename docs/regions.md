# Region-Konfiguration (`config/regions/*.yaml`)

Region-Configs steuern **Datenpfade**, **Kachelgeometrie**, **Terrain-Vorverarbeitung** und **QA-Schwellen**. Sie werden von `prepare-region` erzeugt oder manuell bearbeitet und mit `--config` an die Pipeline übergeben.

Beispiel: `config/regions/demo_test.yaml`

---

## Übersicht

| Abschnitt | Pipeline-Schritt | Zweck |
|-----------|------------------|-------|
| Top-Level (`crs`, `tile_size_m`, `resolution_m`) | harmonize, terrain, snow, render | Raster-Grid und Auflösung |
| `paths` | harmonize, masks | Eingabedaten |
| `tiles` | alle | Bounding Box pro Kachel |
| `harmonize` | harmonize | Resampling beim Ausrichten |
| `terrain` | terrain, render | DEM-Ableitungen und Hillshade |
| `qa` | qa | Automatische Qualitätsprüfungen |

---

## Top-Level

### `crs`

| | |
|---|---|
| **Typ** | String (EPSG-Code) |
| **Standard** | `"EPSG:2056"` |
| **Bereich** | Jedes von Rasterio unterstützte CRS; für Schweizer Daten LV95 (`EPSG:2056`) |
| **Wirkung** | Koordinatensystem für alle ausgerichteten Raster und Vektoren. Bounding Boxes in `tiles` müssen in diesem CRS angegeben sein. |

### `tile_size_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `512` |
| **Bereich** | > 0; typisch 256–4096 |
| **Wirkung** | Referenzgrösse für Kacheln in der Projektstruktur. Die tatsächliche Kachelgrösse ergibt sich aus `tiles.{id}.bbox` und `resolution_m`. |

### `resolution_m`

| | |
|---|---|
| **Typ** | Zahl (Meter pro Pixel) |
| **Standard** | `1` (`default.yaml`), `0.5` (demo_test) |
| **Bereich** | > 0; üblich 0.1–2.0 |
| **Wirkung** | Ziel-Pixelgrösse aller harmonisierten Raster. Bestimmt Kachelbreite/-höhe in Pixeln: `(maxx−minx) / resolution_m`. Beeinflusst auch die automatische WMTS-Zoom-Wahl bei `prepare-region` (z. B. 0.5 m → Zoom 26). Kleinere Werte = schärfere Orthophotos, mehr Speicher und Laufzeit. |

---

## `paths`

Pfade zu Rohdaten und Ausgabeverzeichnissen. Relativ zum Projektroot oder absolut.

### `raw_root`, `intermediate_root`, `output_root`

| | |
|---|---|
| **Typ** | String (Verzeichnispfad) |
| **Standard** | `data/raw`, `data/intermediate/tiles`, `data/output/tiles` |
| **Wirkung** | `raw_root`: Eingangsdaten. `intermediate_root`: Zwischenprodukte pro `{tile_id}`. `output_root`: finales Winter-Orthophoto und QA. |

### `orthophoto`, `dem`

| | |
|---|---|
| **Typ** | String (GeoTIFF-Pfad) |
| **Wirkung** | Sommer-RGB-Orthophoto bzw. Höhenmodell (swissALTI3D, typisch 2 m). Werden in `harmonize` auf das Tile-Grid reprojiziert und resampled. |

### `tlm3d.*`

| | |
|---|---|
| **Typ** | String (GeoPackage-Pfad) pro Layer |
| **Layer** | `buildings`, `roads`, `paths`, `water`, `forest`, `settlement`, `landcover` |
| **Wirkung** | swissTLM3D-Vektordaten für Maskenerzeugung (`masks`). Jeder Layer wird auf das Raster-Grid rasterisiert. |

### `winter_reference`

| | |
|---|---|
| **Typ** | String (GeoTIFF-Pfad) |
| **Wirkung** | Optionales Referenz-Winterbild für visuelle Metriken (nicht für die Regel-Pipeline zwingend). |

---

## `tiles`

Definiert eine oder mehrere Kacheln. Der Schlüssel ist die `tile_id` (z. B. `demo_test_001`).

### `tiles.{tile_id}.bbox`

| | |
|---|---|
| **Typ** | Liste mit 4 Zahlen |
| **Format** | `[minx, miny, maxx, maxy]` in Einheiten von `crs` |
| **Bereich** | `minx < maxx`, `miny < maxy`; für LV95 typisch 2'000'000–2'900'000 (x), 1'000'000–1'400'000 (y) |
| **Wirkung** | Geografischer Ausschnitt der Pipeline. Alle Raster werden exakt auf dieses Rechteck zugeschnitten. Beispiel 4000×4000 m bei `resolution_m: 0.5` → 8000×8000 Pixel. |

---

## `harmonize`

Steuert das Ausrichten von Orthophoto und DEM auf das gemeinsame Tile-Grid (`harmonize`-Schritt).

### `orthophoto_resampling`, `dem_resampling`

| | |
|---|---|
| **Typ** | String |
| **Standard** | `"bilinear"` |
| **Mögliche Werte** | `nearest`, `bilinear`, `cubic`, `cubic_spline`, `lanczos`, `average`, `mode`, `max`, `min`, `med`, `q1`, `q3`, `sum`, `rms` (Rasterio) |
| **Wirkung** | Interpolationsmethode beim Reprojizieren/Resampling. `bilinear` ist ein guter Kompromiss für RGB und DEM. `nearest` für kategorische Daten. |

### `mask_resampling`

| | |
|---|---|
| **Typ** | String |
| **Standard** | `"nearest"` |
| **Wirkung** | In der Config vorgesehen für Masken-Resampling. Aktuell noch nicht im Code verdrahtet — Masken werden direkt beim Rasterisieren erzeugt. |

---

## `terrain`

Parameter für Terrain-Features (`terrain`-Schritt) und für Hillshade im Render-Schritt.

### `generalized_hillshade_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `55` |
| **Bereich** | ≥ 0; typisch 20–100 |
| **Wirkung** | Gauß-Glättung des DEM vor Hillshade-Berechnung. Grössere Werte → Relief auf Gebirgs-/Tal-Massstab statt Meter-Rauheit. Wird im Render-Schritt für kartographisches Schattieren und Makro-Relief genutzt. Skaliert intern mit `resolution_m` zu Pixel-Sigma. |

### `hillshade.azimuth`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `45` (Region), überschrieben durch `rendering.map_shading.sun_azimuth` im Profil |
| **Bereich** | 0–360 |
| **Wirkung** | Azimut der Lichtquelle für Hillshade (0° = Nord, im Uhrzeigersinn). Terrain-Hillshade und generalisierter Hillshade nutzen diese Richtung, sofern das Rendering-Profil keinen eigenen Sonnenstand setzt. |

### `hillshade.altitude`

| | |
|---|---|
| **Typ** | Zahl (Grad über Horizont) |
| **Standard** | `32` |
| **Bereich** | 0–90; typisch 15–45 für Wintersonne |
| **Wirkung** | Sonnenhöhe für Hillshade. Niedrigere Werte → längere Schatten, stärkerer Kontrast. |

### `tpi_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `5` |
| **Bereich** | ≥ 1; typisch 3–15 |
| **Wirkung** | Radius für den Terrain Position Index (TPI): Differenz zwischen Pixelhöhe und lokalem Mittel. Positiv = Kamm/Rücken, negativ = Senke/Tal. Beeinflusst Schneedicken-Modulation (`valley_deposition_factor`, `ridge_scour_factor`) und Fels-Sichtbarkeit. Fenstergrösse = `2 × radius + 1` Pixel. |

### `roughness_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `3` |
| **Bereich** | ≥ 1; typisch 2–8 |
| **Wirkung** | Radius für lokale Höhen-Standardabweichung (Rauheit). Höhere Werte glätten die Rauheit über grössere Nachbarschaft. Wird für Fels-Sichtbarkeit (`rock.slope_visibility_threshold_deg`, `roughness_visibility_threshold`) genutzt. |

### `flow_iterations`

| | |
|---|---|
| **Typ** | Integer |
| **Standard** | `3` |
| **Bereich** | 1–10 (intern gekappt) |
| **Wirkung** | Iterationen für einen Fluss-Akkumulations-Proxy aus Hangneigung. Mehr Iterationen → stärker geglättetes Drainage-Muster. Wird als Terrain-Band gespeichert, aktuell nicht direkt im Snow/Render-Modell verwendet. |

---

## `qa`

Schwellen für automatische Qualitätsprüfungen (`qa`-Schritt). Ein Check gilt als bestanden, wenn der Score die Toleranz erfüllt.

### `building_edge_tolerance`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.35` |
| **Wirkung** | Maximal erlaubter Kantenverlust an Gebäudeumrissen (Canny-Overlap Sommer vs. Winter). Score ≥ `1 − tolerance` → bestanden. Höher = toleranter. |

### `road_brightness_min`

| | |
|---|---|
| **Typ** | Zahl (0–1, normierte Helligkeit) |
| **Standard** | `0.25` |
| **Wirkung** | Minimale mittlere Helligkeit von Strassenpixeln im Winterbild. Verhindert, dass Strassen unter Schnee verschwinden. |

### `water_boundary_tolerance`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.15` |
| **Wirkung** | Maximal erlaubte Helligkeitsabweichung an Gewässergrenzen (Sommer vs. Winter). Score ≥ `1 − tolerance` → bestanden. |

### `forest_boundary_tolerance`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.30` |
| **Wirkung** | Wie `water_boundary_tolerance`, aber für Waldränder. Etwas lockerer, da Winterwald stärker vom Sommerbild abweicht. |

### `hallucination_tolerance`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.40` |
| **Wirkung** | Maximal erlaubte mittlere Pixeländerung ausserhalb geschützter Masken (Gebäude, Wasser). Verhindert zu starke globale Verfärbung. Niedriger = strenger. |

---

## Typischer Workflow

1. Region mit `winter-ortho prepare-region --name … --extent …` anlegen.
2. `resolution_m` und ggf. `bbox` anpassen.
3. Pipeline starten:

```bash
winter-ortho run-all --tile-id demo_test_001 --profile demo_test --config config/regions/demo_test.yaml
```

Rendering-Parameter (Schneedecke, Klassen-Look) liegen im passenden Profil unter `config/rendering_profiles/` — siehe [rendering_profiles.md](rendering_profiles.md).
