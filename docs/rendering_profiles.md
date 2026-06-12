# Rendering-Profile (`config/rendering_profiles/*.yaml`)

Rendering-Profile steuern **Schneemodell**, **Klassenregeln** und **visuelles Erscheinungsbild** des Winter-Orthophotos. Sie werden mit `--profile {name}` geladen (`config/rendering_profiles/{name}.yaml`).

Das Feld `profile` am Dateianfang ist der Profilname und muss zum Dateinamen passen.

---

## Übersicht

| Abschnitt | Pipeline-Schritt | Zweck |
|-----------|------------------|-------|
| `snow_surface` | snow-surface, snow, render | Physische Schneedecke (DEM + Dicke) |
| `elevation`, `aspect` | snow | Höhen- und Expositions-Modifikatoren |
| `open_land`, `forest`, `settlement`, `rock` | snow, render | Landbedeckungsklassen |
| `roads`, `paths`, `buildings`, `water` | snow, render | Infrastruktur und Gewässer |
| `rendering` | render | Farben, Schatten, Relief, Tonwertkorrektur |

Viele Parameter sind **normierte Gewichte** (0–1). Einige Schatten-Parameter (`shadow_boost`, `sun_max`) können > 1 sein.

---

## `snow_surface`

Erzeugt `snow_surface_dem.tif` und `snow_thickness_m.tif`. Aktiviert dickengestützte Schneefraktionen, wenn dieser Block im Profil vorhanden ist.

### `base_snow_height_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `2.0` |
| **Bereich** | ≥ 0; typisch 0.5–10 |
| **Wirkung** | Referenz-Schneehöhe auf ebenem Gelände. Skaliert die gesamte Schneedicke und dient als Nenner für `thickness_fraction` (0–1) in den Klassenregeln. |

### `max_accumulation_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `30` |
| **Bereich** | 0–90; typisch 25–40 |
| **Wirkung** | Maximale Hangneigung, auf der Schnee akkumuliert. Steiler → kein Schneeauftrag auf der Oberfläche; Hang bleibt sichtbar. Steuert auch `accumulation_mask` und Hillshade-Blend im Render. |

### `smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `20` |
| **Bereich** | ≥ 1; typisch 10–80 |
| **Wirkung** | Gauß-Glättung des DEM für die Makro-Schneeoberfläche. Grössere Werte → gleichmässigere Schneedecke über kleine Geländewelligkeiten. |

### `micro_suppression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.0` (aus) |
| **Bereich** | 0–1 |
| **Wirkung** | Maximale Stärke der Mikrorelief-Nivellierung auf flachem Gelände. Senken werden aufgefüllt (`depression_fill`), Kamm-Mikrorelief reduziert (`ridge_micro_retention`). Die tatsächliche Wirkung wird mit dem hangabhängigen Faktor `leveling_full_slope_deg` / `leveling_end_slope_deg` multipliziert — unabhängig vom Landbedeckungstyp. |

### `leveling_full_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `30` |
| **Bereich** | 0–90 |
| **Wirkung** | Bis zu dieser Hangneigung volle DEM-Nivellierung (Muldenfüllung, Mikroglättung). Grossräumige Struktur bleibt über `smoothing_sigma_m` erhalten. |

### `leveling_end_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `max_accumulation_slope_deg` |
| **Bereich** | > `leveling_full_slope_deg` |
| **Wirkung** | Ab dieser Neigung keine Nivellierung mehr — Schneeoberfläche folgt dem Sommer-DEM. Dazwischen (`leveling_full_slope_deg` … `leveling_end_slope_deg`) linearer Übergang. Typisch `35` bei `max_accumulation_slope_deg: 35`. |

### `depression_fill`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.85` |
| **Bereich** | 0–1 |
| **Wirkung** | Anteil, um den negative Mikrorelief-Abweichungen (Senken) zur Schneedecke hin geglättet werden. Nur aktiv wenn `micro_suppression > 0`. |

### `ridge_micro_retention`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.25` |
| **Bereich** | 0–1 |
| **Wirkung** | Verbleibendes positives Mikrorelief auf Kämmen (0 = vollständig abflachen, 1 = unverändert). Nur mit `micro_suppression`. |

### `peak_retention`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `1.0` |
| **Bereich** | 0–1 |
| **Wirkung** | Anteil von Gipfelspitzen über dem geglätteten Makro-DEM, der erhalten bleibt. Nur aktiv wenn `micro_suppression = 0`. 1 = volle Spitzenhöhe, 0 = komplett geglättet. |

### `tpi_smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `0` (aus) |
| **Bereich** | ≥ 0 |
| **Wirkung** | Gauß-Glättung des TPI vor der Dickemodulation. Reduziert Rauschen in Tal-/Kamm-Faktoren. |

### `thickness_smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `0` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Glättung der berechneten Schneedicke vor Auftrag auf die Oberfläche. |

### `surface_post_smooth_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `0` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Nachglättung der finalen Schneeoberfläche, gewichtet mit `accumulation_blend_sigma_m`-Maske (nur auf Akkumulationsflächen). |

### `surface_macro_smooth_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `0` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Zusätzliche Makro-Glättung der Schneeoberfläche (stärker als `surface_post_smooth_sigma_m`). |

### `accumulation_blend_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `0` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Weichzeichnung der Akkumulations-Grenze (Übergang hangab / kein Schnee). Grössere Werte → weichere Übergänge an Steilhängen. |

### `valley_deposition_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.30` |
| **Bereich** | 0–1 |
| **Wirkung** | Zusätzliche Schneedicke in Senken (negativer TPI). 0.3 → bis +30 % über Basisdicke in tiefen Tälern. |

### `ridge_scour_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.50` |
| **Bereich** | 0–1 |
| **Wirkung** | Schneeverlust auf Kämmen (positiver TPI). 0.5 → bis −50 % Basisdicke auf Graten. |

### `windward_aspect_penalty`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.15` |
| **Bereich** | 0–1 |
| **Wirkung** | Zusätzlicher Schneeverlust auf windanliegenden (südwestlichen) Kämmen. Referenzrichtung: 202.5°. Wirkt nur in Kombination mit positivem TPI. |

---

## `elevation`

Modifiziert Schneefraktion nach Höhe (über Meer).

### `reference_m`

| | |
|---|---|
| **Typ** | Zahl (Meter NN) |
| **Standard** | `1500` |
| **Wirkung** | Referenzhöhe: darüber mehr, darunter weniger Schnee in den Klassenregeln. |

### `snow_increase_per_100m`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.03`–`0.04` |
| **Bereich** | typisch 0.01–0.08 |
| **Wirkung** | Additiver Schnee-Boost pro 100 m über `reference_m` (als Faktor auf `snow_fraction`). |

### `max_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.15`–`0.18` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Obergrenze des Höhenmodifikators (symmetrisch nach unten begrenzt auf `−max_boost`). |

---

## `aspect`

Modifiziert Schneefraktion nach Hangausrichtung.

### `south_thinning`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.10`–`0.12` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Reduktion der Schneefraktion auf südorientierten Hängen (mehr Schmelze/Sonneneinstrahlung). |

### `north_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.08`–`0.12` |
| **Bereich** | ≥ 0 |
| **Wirkung** | Erhöhung der Schneefraktion auf nordorientierten Hängen. |

---

## `open_land`

Offenes Gelände (Wiesen, Weiden, alpine Matten).

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard** | `[0.97, 1.0]` |
| **Wirkung** | Ziel-Schneefraktion. Mit Schneedicke: interpoliert zwischen min und max nach `thickness_fraction`; ohne: Höhenmodifikator + fester Anteil. |

### `snow_brightness`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.98` |
| **Wirkung** | Mischung zwischen flacher Schneefarbe und schattiertem Schneelayer. Höher = heller, weniger Kontrast in der Schneeschicht. |

### `snow_texture_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.28` |
| **Wirkung** | Stärke des bandbegrenzten Rauschens auf der Schneedecke. |

### `original_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.0` |
| **Wirkung** | Sichtbarkeit der Sommertextur unterhalb der Schneedecke. 0 = vollständig überdeckt. |

### `max_snow_blend`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `1.0` |
| **Wirkung** | Obergrenze der Überblendung Schnee/Sommer, unabhängig von `snow_fraction`. |

### `hillshade_compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.28`–`0.46` |
| **Wirkung** | Reduziert Hillshade-Dynamik: `0.5 + (hillshade − 0.5) × compression`. Niedriger → flachere, gleichmässigere Schneedecke. |

### `snow_flattening`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.12`–`0.82` |
| **Wirkung** | Reduziert Hillshade-Stärke proportional zur Schneefraktion. Höher → weniger Relief auf verschneiten Flächen. |

### `summer_shade_weight`, `hillshade_shade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.22` / `0.78` (davos), `0.0` / `0.0` (demo_test) |
| **Wirkung** | Gewichtung Sommer-Luminanz vs. Hillshade im kombinierten Schattenfeld. Summe bestimmt die Mischung; beide 0 → nur Hillshade (mit Makro-Anteil). |

### `macro_hillshade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.85`–`0.90` |
| **Wirkung** | Anteil geglättetem (generalisiertem) Hillshade vs. feinem Relief. Höher → grossräumigere Schatten. |

### `cast_shadow_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.18`–`0.28` |
| **Wirkung** | Einfluss extrahierter Sommer-Wurfschatten (Gebäude, Wald) auf das Schattenfeld der Schneedecke. |

### `shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `1.30`–`1.35` |
| **Wirkung** | Verstärkung dunkler Hillshade-Anteile (Schatten) gegenüber Lichtern. > 1 = tiefere Schatten. |

### `highlight_cap`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.45`–`0.55` |
| **Wirkung** | Begrenzung heller Hillshade-Anteile. Niedriger → weniger ausgebrannte Schneehighlights. |

---

## `forest`

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard** | `[0.78, 0.95]` |
| **Wirkung** | Schneefraktion unter dem Kronendach. Mit Schneedicke: skaliert nach `thickness_fraction × canopy_thickness_factor`. |

### `canopy_thickness_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.82` |
| **Wirkung** | Wie stark die physische Schneedicke die Kronen-Schneefraktion beeinflusst. Nur mit `snow_surface`. |

### `forest_snow_intensity`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.94` |
| **Wirkung** | Multiplikator auf `snow_fraction` für die Kronen-Schneeüberblendung. |

### `crown_tophat_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `4` |
| **Bereich** | ≥ 2 |
| **Wirkung** | Radius für White-Top-Hat-Filter zur Kronenerkennung. Grösser → gröbere Kronenstruktur. |

### `green_suppression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.68`–`0.78` |
| **Wirkung** | Entsättigung des Sommergrüns. Höher → grauer Winterwald vor Schneeüberblendung. |

### `winter_luminance_lo`, `winter_luminance_hi`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.18` / `0.46` |
| **Wirkung** | Ziel-Luminanzbereich beim Umsetzen der Sommerschattenstruktur auf Wintertonwerte. |

### `summer_structure_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.62`–`0.82` |
| **Wirkung** | Wie stark die Sommer-Luminanzstruktur (Schatten unter Bäumen) im Winter erhalten bleibt. |

### `max_crown_snow_alpha`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.48`–`0.74` |
| **Wirkung** | Maximale Deckkraft des Schnees auf Kronen. |

### `canopy_blanket`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.50`–`0.58` |
| **Wirkung** | Grundbedeckung der Kronenfläche mit Schnee, unabhängig von Einzelbaumstruktur. |

### `crown_highlight_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.42`–`0.48` |
| **Wirkung** | Schnee-Highlights auf erkannten Kronen (Top-Hat-Signal). |

### `crown_noise_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.04`–`0.05` |
| **Wirkung** | Rausch-Textur auf Kronen-Schnee. |

### `original_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.08` |
| **Wirkung** | Sommertextur in Kronenlücken (max. ~0.14 intern). |

---

## `settlement`

Siedlungsflächen. Fehlt der Block, werden `open_land`-Werte als Fallback genutzt.

Parameter analog zu `open_land`: `snow_fraction`, `snow_brightness`, `snow_texture_strength`, `original_texture_visibility`, `max_snow_blend`, `hillshade_compression`, `snow_flattening`, `summer_shade_weight`, `hillshade_shade_weight`, `macro_hillshade_weight`, `cast_shadow_weight`, `shadow_boost`, `highlight_cap`.

Typische Abweichungen: etwas weniger `max_snow_blend`, höheres `cast_shadow_weight` (Gebäudeschatten in Dörfern).

---

## `rock`

Fels und offener Boden.

### `slope_visibility_threshold_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `38` |
| **Wirkung** | Ab dieser Hangneigung steigt die Fels-Sichtbarkeit. |

### `roughness_visibility_threshold`

| | |
|---|---|
| **Typ** | Zahl (normiert 0–1) |
| **Standard** | `0.52` |
| **Wirkung** | Ab dieser lokalen Rauheit wird Fels sichtbarer. |

### `max_snow_fraction`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.94` |
| **Wirkung** | Maximale Schneefraktion auf Fels (wird durch Sichtbarkeit und Aspekt reduziert). |

### `thickness_burial_radius_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard** | `20`–`22` |
| **Wirkung** | Nachbarschaftsradius für lokale Schneetiefe (Maximum-Filter). Grösserer Radius „vergräbt" kleine Felsvorsprünge unter Nachbarschnee. |

### `thickness_burial_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.82`–`0.85` |
| **Wirkung** | Stärke der Schneebedeckung auf Fels durch lokale Schneedicke. |

### `min_rock_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.18` |
| **Wirkung** | Minimale Fels-Sichtbarkeit auf steilen Hängen. |

### `aspect_south_penalty`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.08`–`0.12` |
| **Wirkung** | Reduktion der Schneefraktion auf südorientiertem Fels. |

### `gentle_slope_max_deg`, `steep_slope_min_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `28`–`32` / `42`–`46` |
| **Wirkung** | Definieren den Übergang flach ↔ steil für sanften Schnee-Boost und Sichtbarkeitskurven. |

### `gentle_snow_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.22`–`0.36` |
| **Wirkung** | Zusätzliche Schneefraktion auf flachem Fels (Snow-Layer). |

### `gentle_render_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.18`–`0.30` |
| **Wirkung** | Zusätzliche Schnee-Überblendung auf flachem Fels im Render-Schritt (Alias für `gentle_snow_boost` im Renderer). |

### `hillshade_compression`, `snow_flattening`, `summer_shade_weight`, `hillshade_shade_weight`, `shadow_boost`, `highlight_cap`

Wie bei `open_land`, angepasst für Fels (typisch höhere `shadow_boost`, niedrigere `snow_flattening`).

### `summer_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.20`–`0.42` |
| **Wirkung** | Erhalt der Sommertextur/Farbe auf sichtbarem Fels. Höher → mehr Sommeranteil auf Felsflächen. |

---

## `roads`

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard** | `[0.2, 0.5]` |
| **Wirkung** | Geringe Schneefraktion — Strassen bleiben erkennbar. |

### `road_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.85` |
| **Wirkung** | Grundsichtbarkeit der Strasse im Winterbild. |

### `summer_line_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.72`–`0.75` |
| **Wirkung** | Nutzung dunkler Sommerstruktur zur Strassenführung. |

### `min_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.38`–`0.42` |
| **Wirkung** | Untergrenze der Strassendeckkraft (unabhängig von Sommerkontrast). |

---

## `paths`

Wanderwege — sollen unter Schnee verschwinden.

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard** | `[0.90, 0.98]` |
| **Wirkung** | Hohe Schneefraktion in den Snow-Layern. |

### `snow_brightness`, `snow_texture_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.94` / `0.42` |
| **Wirkung** | Helligkeit und Textur der Schneedecke auf Wegflächen. |

### `bury_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.90`–`0.98` |
| **Wirkung** | Verstärkung der Schneeüberblendung. 1 ≈ vollständig vergraben. |

### `max_snow_blend`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.82`–`0.98` |
| **Wirkung** | Maximale Deckkraft der Schneedecke über dem Weg. |

---

## `buildings`

### `roof_snow_intensity`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard** | `[0.5, 0.8]` |
| **Wirkung** | Schneeintensität auf Dächern (Mittelwert + Höhenmodifikator in Snow-Layer). |

### `edge_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.95` |
| **Wirkung** | Gebäudekanten aus dem Sommerbild werden überblendet, um geometrische Schärfe zu erhalten. |

### `brighten_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.45`–`0.55` |
| **Wirkung** | Skalierung der Dach-Schneeüberblendung. |

### `wall_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.62`–`0.68` |
| **Wirkung** | Erhalt der Fassadenfarbe/Struktur unterhalb des Dachschnees. |

---

## `water`

### `shore_snow_width_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `3` |
| **Wirkung** | Breite des Uferschnee-Streifens entlang der Wassergrenze. 0 = aus. |

### `shore_snow_intensity`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.4` |
| **Wirkung** | Schneefraktion im Uferstreifen. |

### `water_darken`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.85` |
| **Wirkung** | Multiplikator auf RGB in Gewässerflächen (dunkleres Winterwasser). |

---

## `rendering`

Globale Render-Einstellungen.

### `snow_color`

| | |
|---|---|
| **Typ** | `[R, G, B]` (0–1) |
| **Standard** | `[0.94, 0.96, 0.98]` |
| **Wirkung** | Basis-Schneefarbe für alle Klassen. |

### `road_color`

| | |
|---|---|
| **Typ** | `[R, G, B]` (0–1) |
| **Standard** | `[0.55, 0.58, 0.62]` (wintergrau) |
| **Wirkung** | Zielfarbe für Strassenpixel. |

### `noise_scale_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `8` |
| **Bereich** | ≥ 1 |
| **Wirkung** | Wellenlänge des bandbegrenzten Schneerauschens. Grösser → grobkörnigere Textur. |

### `hillshade_strength_open_land`, `hillshade_strength_rock`, `hillshade_strength_settlement`, `hillshade_strength_forest`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.12`–`0.18` (davos), `0.0` (demo_test) |
| **Wirkung** | Hillshade-Stärke auf der Schneeschicht pro Klasse. 0 = flache Schneefarbe ohne Relief. |

---

## `rendering.map_shading`

Kartographisches Schattieren — letzter Schritt, nach Tonwertkorrektur.

### `sun_azimuth`, `sun_altitude`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard** | `135` / `32` |
| **Wirkung** | Überschreibt `terrain.hillshade` für Render und Relief. Typisch tiefstehende Südsonne (~11:00 Uhr). |

### `south_shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.2`–`1.35` |
| **Wirkung** | Verstärkung von Schatten auf der Lee-Seite (Expositions-Schattierung). |

### `strength`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.0`–`0.50` |
| **Wirkung** | Gesamtstärke der kartographischen Schattierung. 0 = aus. |

### `hillshade_weight`, `aspect_weight`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.72` / `0.28` |
| **Wirkung** | Mischung Hillshade vs. Expositions-Relief im Map-Shading. |

### `compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.88`–`0.90` |
| **Wirkung** | Hillshade-Dynamikreduktion vor Anwendung. |

### `min_snow`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.10`–`0.12` |
| **Wirkung** | Minimale Schneefraktion, ab der Map-Shading wirkt. |

### `shade_min`, `sun_max`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.52`–`0.55` / `1.35`–`1.38` |
| **Wirkung** | Untere und obere Grenze des Luminanz-Multiplikators (`clip(1 + relief, shade_min, sun_max)`). |

### `protect_masks`

| | |
|---|---|
| **Typ** | Liste von Maskennamen |
| **Standard** | `["water_mask"]` |
| **Wirkung** | Klassen, auf denen Map-Shading nicht angewendet wird. |

---

## `rendering.cast_shadows`

Extrahiert und malt Sommer-Wurfschatten (Gebäude, Wald) auf Schnee.

### `reference_sigma_px`, `fine_sigma_px`

| | |
|---|---|
| **Typ** | Zahl (Pixel) |
| **Standard** | `32`–`34` / `6`–`7` |
| **Wirkung** | Gauß-Radien zur Schattenextraktion aus Sommer-Luminanz. Gross = weite Schatten, klein = feine Details. |

### `building_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard** | `10`–`12` |
| **Wirkung** | Dilatationsradius um Gebäude für Schattenverstärkung. |

### `building_boost`, `forest_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `1.6`–`1.75` / `0.0`–`1.25` |
| **Wirkung** | Lokale Verstärkung der Schattenextraktion nahe Gebäuden bzw. im Wald. |

### `base_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.0`–`0.35` |
| **Wirkung** | Grundstärke der Schattenextraktion ausserhalb Gebäude/Wald. |

### `final_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.62`–`0.68` |
| **Wirkung** | Endgültige Stärke beim Auftragen der Wurfschatten auf das Winterbild. |

### `max_darken`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.48`–`0.50` |
| **Wirkung** | Maximale Abdunklung durch Wurfschatten. |

### `min_snow`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.20`–`0.22` |
| **Wirkung** | Minimale Schneefraktion für Wurfschatten-Anwendung. |

---

## `rendering.relief`

Feines Winter-Relief auf Schneeflächen (vor Tonwertkorrektur).

### `hillshade_strength`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard** | `0.10`–`0.26` |
| **Wirkung** | Hillshade-Relief auf Schneeflächen. |

### `compression`, `macro_hillshade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.32`–`0.52` / `0.85` |
| **Wirkung** | Wie bei Klassen-Hillshade — Makro vs. Mikro-Relief. |

### `min_snow`, `shade_min`, `sun_max`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.30` / `0.74`–`0.88` / `1.12`–`1.16` |
| **Wirkung** | Schwellen und Luminanz-Clipping für Relief-Pass. |

### `snow_tint_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.38`–`0.50` |
| **Wirkung** | Schneefarb-Tönung statt neutralem Grau in reliefierten Bereichen. |

### `summer_relief_weight`, `cast_shadow_relief_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard** | `0.0` / `0.0`–`0.55` |
| **Wirkung** | Einbindung Sommer-Luminanz bzw. Wurfschatten ins Relief. |

### `class_weights`

| | |
|---|---|
| **Typ** | Map Klasse → Zahl |
| **Schlüssel** | `open_land`, `settlement`, `rock`, `paths`, `buildings` |
| **Standard** | z. B. `rock: 0.62`, `open_land: 0.14` |
| **Wirkung** | Reliefstärke pro Landbedeckungsklasse. 0 = kein Relief-Pass auf dieser Klasse. |

---

## `rendering.summer_structure`

Abschliessende sommerverankerte Tonwertkorrektur.

### `highlight_rolloff`

| | |
|---|---|
| **Typ** | Boolean |
| **Standard** | `true` |
| **Wirkung** | Aktiviert weiche Highlight-Kompression gegen ausgebrannte Schneeflächen. |

### `highlight_knee`, `highlight_compression`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard** | `0.83`–`0.88` / `0.52` |
| **Bereich** | knee: 0.5–0.98, compression: 0.1–1.0 |
| **Wirkung** | Ab welcher Luminanz und mit welcher Stärke Highlights abgerollt werden. |

### `protect_masks`

| | |
|---|---|
| **Typ** | Liste von Maskennamen |
| **Standard** | `["water_mask"]` |
| **Wirkung** | Von der Tonwertkorrektur ausgeschlossene Klassen. |

### `classes.{mask_name}`

Pro Klasse (z. B. `open_land_mask`, `forest_mask`):

| Parameter | Typ | Standard | Wirkung |
|-----------|-----|----------|---------|
| `lum_lo` | 0–1 | klassenabhängig | Untere Ziel-Luminanz |
| `lum_hi` | 0–1 | klassenabhängig | Obere Ziel-Luminanz |
| `strength` | 0–1 | `0.16`–`0.38` | Mischung Sommer-Luminanzstruktur → Winterzielbereich |

Ohne `classes` werden globale Parameter `global_strength`, `global_lum_lo`, `global_lum_hi` verwendet (falls gesetzt).

---

## Profil abstimmen

1. **Schneehöhe und -verteilung:** `snow_surface` → `winter-ortho snow-surface`
2. **Klassen-Schneefraktionen:** Klassen-Blöcke → `winter-ortho snow`
3. **Visuelles Erscheinungsbild:** `rendering` + Klassen-Render-Parameter → `winter-ortho render`

Region-Parameter (Auflösung, Terrain, QA) liegen in `config/regions/` — siehe [regions.md](regions.md).

```bash
winter-ortho run-all --tile-id demo_test_001 --profile demo_test --config config/regions/demo_test.yaml
```
