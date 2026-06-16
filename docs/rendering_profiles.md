# Rendering-Profile (`config/rendering_profiles/*.yaml`)

Rendering-Profile steuern **Schneemodell**, **Klassenregeln** und **visuelles Erscheinungsbild** des Winter-Orthophotos. Sie werden mit `--profile {name}` geladen (`config/rendering_profiles/{name}.yaml`).

Das Feld `profile` am Dateianfang ist der Profilname und muss zum Dateinamen passen. Die folgende Referenz beschreibt ausschliesslich die Parameter des Profils **`default`** (`config/rendering_profiles/default.yaml`).

**Pipeline-Schritte:** `snow-surface` → `snow` → `render`

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

## `profile`

| | |
|---|---|
| **Typ** | String |
| **Standard (`default`)** | `default` |
| **Verwendet** | Metadaten (Logging) |
| **Wirkung** | Profilname; muss zum Dateinamen passen. |

---

## `snow_surface`

Erzeugt `snow_surface_dem.tif` und `snow_thickness_m.tif`. Aktiviert dickengestützte Schneefraktionen, wenn dieser Block im Profil vorhanden ist.

### `base_snow_height_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `2.0` |
| **Verwendet** | snow-surface, snow |
| **Wirkung** | Referenz-Schneehöhe für volle Schneedecke (`thickness_fraction` = 1). Dient als Nenner in den Klassenregeln. Die tatsächliche Tiefe wird mit `snow_amount` skaliert. |

### `snow_amount`

| | |
|---|---|
| **Typ** | Zahl (0–1+) |
| **Standard (`default`)** | `0.9` |
| **Verwendet** | snow-surface |
| **Wirkung** | Globaler Multiplikator auf die berechnete Schneedicke. `base_snow_height_m: 2.0` und `snow_amount: 0.9` ergeben ca. 1.8 m auf flachem Gelände und `thickness_fraction` ≈ 0.9 — ohne die Referenzhöhe zu ändern. **Nur `base_snow_height_m` zu senken reicht nicht**, weil Dicke und Nenner gleich skaliert werden. |

### `max_accumulation_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `35` |
| **Verwendet** | snow-surface, snow, render |
| **Wirkung** | Maximale Hangneigung, auf der Schnee akkumuliert. Steiler → kein Schneeauftrag auf der Oberfläche; Hang bleibt sichtbar. Steuert auch `accumulation_mask` und Hillshade-Blend im Render. |

### `accumulation_transition_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `20` |
| **Verwendet** | snow-surface |
| **Wirkung** | Breite des weichen Übergangs von voller Akkumulation (`max_accumulation_slope_deg`) zu keinem Schnee auf Steilhängen. Grössere Werte → längere Rampe. |

### `accumulation_edge_feather_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `45` |
| **Verwendet** | snow-surface |
| **Wirkung** | Horizontaler Abstand von Steilhang-Kanten, über den die Schneedecke ausfadet. Verhindert harte Schneekanten an Felswänden. `0` = aus. |

### `leveling_full_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `30` |
| **Verwendet** | snow-surface |
| **Wirkung** | Bis zu dieser Hangneigung volle DEM-Nivellierung (Muldenfüllung, Mikroglättung). Grossräumige Struktur bleibt über `smoothing_sigma_m` erhalten. |

### `leveling_end_slope_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `40` |
| **Verwendet** | snow-surface |
| **Wirkung** | Ab dieser Neigung keine Nivellierung mehr — Schneeoberfläche folgt dem Sommer-DEM. Dazwischen (`leveling_full_slope_deg` … `leveling_end_slope_deg`) weicher Übergang. |

### `smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `90` |
| **Verwendet** | snow-surface |
| **Wirkung** | Gauß-Glättung des DEM für die Makro-Schneeoberfläche. Grössere Werte → gleichmässigere Schneedecke über kleine Geländewelligkeiten. |

### `micro_suppression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.75` |
| **Verwendet** | snow-surface |
| **Wirkung** | Maximale Stärke der Mikrorelief-Nivellierung auf flachem Gelände. Senken werden aufgefüllt (`depression_fill`), Kamm-Mikrorelief reduziert (`ridge_micro_retention`). Die tatsächliche Wirkung wird mit dem hangabhängigen Faktor `leveling_full_slope_deg` / `leveling_end_slope_deg` multipliziert. |

### `depression_fill`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.85` |
| **Verwendet** | snow-surface |
| **Wirkung** | Anteil, um den negative Mikrorelief-Abweichungen (Senken) zur Schneedecke hin geglättet werden. Nur aktiv wenn `micro_suppression > 0`. |

### `ridge_micro_retention`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.12` |
| **Verwendet** | snow-surface |
| **Wirkung** | Verbleibendes positives Mikrorelief auf Kämmen (0 = vollständig abflachen, 1 = unverändert). Nur mit `micro_suppression > 0`. |

### `tpi_smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `70` |
| **Verwendet** | snow-surface |
| **Wirkung** | Gauß-Glättung des TPI vor der Dickemodulation. Reduziert Rauschen in Tal-/Kamm-Faktoren. `0` = aus. |

### `thickness_smoothing_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `35` |
| **Verwendet** | snow-surface |
| **Wirkung** | Glättung der berechneten Schneedicke vor Auftrag auf die Oberfläche. `0` = aus. |

### `surface_post_smooth_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `18` |
| **Verwendet** | snow-surface |
| **Wirkung** | Nachglättung der finalen Schneeoberfläche, gewichtet mit Akkumulationsmaske (nur auf Akkumulationsflächen). `0` = aus. |

### `surface_macro_smooth_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `20` |
| **Verwendet** | snow-surface |
| **Wirkung** | Zusätzliche Makro-Glättung der Schneeoberfläche (stärker als `surface_post_smooth_sigma_m`). `0` = aus. |

### `accumulation_blend_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `15` |
| **Verwendet** | snow-surface |
| **Wirkung** | Glättung der Hangneigung vor der Akkumulations-Gewichtung. Reduziert pixelgenaues Salz-und-Pfeffer-Rauschen an Schneegrenzen. `0` = aus. |

### `cover_transition_sigma_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `42` |
| **Verwendet** | snow-surface |
| **Wirkung** | Räumliche Glättung des Schnee↔Fels-Übergangs (`cover = blend_weight × leveling_weight`). Verhindert sichtbare Abstufungsringe um Felsinseln; grössere Werte → weicherer Übergang. |

### `valley_deposition_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.15` |
| **Verwendet** | snow-surface |
| **Wirkung** | Zusätzliche Schneedicke in Senken (negativer TPI). 0.15 → bis +15 % über Basisdicke in tiefen Tälern. |

### `ridge_scour_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.10` |
| **Verwendet** | snow-surface |
| **Wirkung** | Schneeverlust auf Kämmen (positiver TPI). 0.10 → bis −10 % Basisdicke auf Graten. |

### `windward_aspect_penalty`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.08` |
| **Verwendet** | snow-surface |
| **Wirkung** | Zusätzlicher Schneeverlust auf windanliegenden (südwestlichen) Kämmen. Referenzrichtung: 202.5°. Wirkt nur in Kombination mit positivem TPI. |

---

## `elevation`

Modifiziert Schneefraktion nach Höhe (über Meer).

### `reference_m`

| | |
|---|---|
| **Typ** | Zahl (Meter NN) |
| **Standard (`default`)** | `1560` |
| **Verwendet** | snow |
| **Wirkung** | Referenzhöhe: darüber mehr, darunter weniger Schnee in den Klassenregeln. |

### `snow_increase_per_100m`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.04` |
| **Verwendet** | snow |
| **Wirkung** | Additiver Schnee-Boost pro 100 m über `reference_m` (als Faktor auf `snow_fraction`). |

### `max_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.18` |
| **Verwendet** | snow |
| **Wirkung** | Obergrenze des Höhenmodifikators (symmetrisch nach unten begrenzt auf `−max_boost`). |

---

## `aspect`

Modifiziert Schneefraktion nach Hangausrichtung.

### `south_thinning`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.10` |
| **Verwendet** | snow |
| **Wirkung** | Reduktion der Schneefraktion auf südorientierten Hängen (mehr Schmelze/Sonneneinstrahlung). |

### `north_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.12` |
| **Verwendet** | snow |
| **Wirkung** | Erhöhung der Schneefraktion auf nordorientierten Hängen. |

---

## `open_land`

Offenes Gelände (Wiesen, Weiden, alpine Matten).

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.97, 1.0]` |
| **Verwendet** | snow |
| **Wirkung** | Ziel-Schneefraktion. Mit Schneedicke: interpoliert zwischen min und max nach absoluter Schneehöhe (siehe `full_snow_thickness_m`); ohne: Höhenmodifikator + fester Anteil. Auf Steilhängen kann die Fraktion unter min fallen, bleibt aber über `slope_min_snow_fraction` gedeckelt. |

### `full_snow_thickness_m`

| | |
|---|---|
| **Typ** | Meter |
| **Standard (`default`)** | `0.5` |
| **Verwendet** | snow |
| **Wirkung** | Ab dieser absoluten Schneehöhe gilt volle Schneedecke (`snow_fraction` max) auf offenem Land. In Akkumulationszonen zählt die nominale Blanket-Dicke (`blanket_thickness_m`), auf Steilflanken die geometrische Höhe. |

### `slope_snow_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.28` |
| **Verwendet** | snow |
| **Wirkung** | Master-Regler für Hangreduktion. `0` schaltet Steilhang-Logik vollständig aus. Mit gesetztem `slope_min_snow_fraction`: wirkt unabhängig von `full_snow_thickness_m` (gedeckelte Reduktion). |

### `slope_snow_start_deg`, `slope_snow_end_deg`, `slope_min_snow_scale`

| | |
|---|---|
| **Typ** | Grad / Grad / Zahl (0–1) |
| **Standard (`default`)** | `28` / `48` / `0.35` |
| **Verwendet** | snow |
| **Wirkung** | Definieren die Rampe der Hangreduktion: ab `slope_snow_start_deg` beginnt die Abschwächung, bei `slope_snow_end_deg` erreicht der Skalierungsfaktor `slope_min_snow_scale`. |

### `slope_min_snow_fraction`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.96` |
| **Verwendet** | snow |
| **Wirkung** | Untergrenze der Schneefraktion auf Steilhängen. Verhindert Sommergrün-Durchscheinen, erlaubt aber weniger Schnee als auf flachem Gelände. Wenn gesetzt, werden `slope_texture_visibility`, `protrusion_snow_reduction` und `protrusion_texture_visibility` **nicht** ausgewertet. |

### `slope_min_snow_softness`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.05` |
| **Verwendet** | snow |
| **Wirkung** | Breite des weichen Übergangs zur Untergrenze (`slope_min_snow_fraction`). Höher = sanftere Abstufung statt harter Kante. |

### `slope_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | snow (bedingt) |
| **Wirkung** | Sommerbild-Durchscheinen auf steilen offenen Hängen (`summer_exposure`). **Nur aktiv**, wenn `slope_min_snow_fraction` **nicht** gesetzt ist. Im Profil `default` daher wirkungslos. |

### `deck_depth_cover_floor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.55` |
| **Verwendet** | snow |
| **Wirkung** | Mindest-Deckgewicht für die Tiefen-Gating-Logik: `snow_cover_weight` wird für die Schneedichtenberechnung mindestens auf diesen Wert angehoben. Verhindert, dass dünne Deckzonen als «zu wenig Schnee» interpretiert werden. |

### `deck_snow_fraction_boost`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.85` |
| **Verwendet** | snow |
| **Wirkung** | Hebt die effektive Untergrenze (`slope_min_snow_fraction`) in gut deckenden Zonen an: `effective_min = min_steep + (hi − min_steep) × deck × boost`. |

### `hillshade_deck_floor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.72` |
| **Verwendet** | render |
| **Wirkung** | Mindestgewicht beim Blend zwischen Sommer- und Schnee-Hillshade (`blend_hillshade_for_snow`). Erhält Relief auf teilweise deckenden Flächen. |

### `protrusion_full_m`, `protrusion_strength`, `protrusion_snow_reduction`, `protrusion_texture_visibility`

| | |
|---|---|
| **Typ** | Meter / Zahl (0–1) / Zahl (0–1) / Zahl (0–1) |
| **Standard (`default`)** | `0.5` / `0.85` / `0.90` / `0.75` |
| **Verwendet** | snow (bedingt) |
| **Wirkung** | Wo das Sommer-DEM über `snow_surface_dem` ragt (Geröllfelder, Felsinseln): Schnee wird reduziert und Sommertextur eingeblendet. `protrusion_snow_reduction` und `protrusion_texture_visibility` werden **nur ausgewertet**, wenn `slope_min_snow_fraction` **nicht** gesetzt ist — im Profil `default` daher wirkungslos. `protrusion_full_m` und `protrusion_strength` werden weiterhin für die Protrusions-Erkennung genutzt. |

### `snow_brightness`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.98` |
| **Verwendet** | snow, render |
| **Wirkung** | Mischung zwischen flacher Schneefarbe und schattiertem Schneelayer. Höher = heller, weniger Kontrast in der Schneeschicht. |

### `snow_texture_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.28` |
| **Verwendet** | snow, render |
| **Wirkung** | Stärke des bandbegrenzten Rauschens auf der Schneedecke. |

### `original_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Sichtbarkeit der Sommertextur unterhalb der Schneedecke. `0` = vollständig überdeckt (deaktiviert). |

### `max_snow_blend`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `1.0` |
| **Verwendet** | render |
| **Wirkung** | Obergrenze der Überblendung Schnee/Sommer, unabhängig von `snow_fraction`. |

### `hillshade_compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.46` |
| **Verwendet** | render |
| **Wirkung** | Reduziert Hillshade-Dynamik: `0.5 + (hillshade − 0.5) × compression`. Niedriger → flachere, gleichmässigere Schneedecke. |

### `snow_flattening`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.22` |
| **Verwendet** | render |
| **Wirkung** | Reduziert Hillshade-Stärke proportional zur Schneefraktion. Höher → weniger Relief auf verschneiten Flächen. |

### `summer_shade_weight`, `hillshade_shade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Gewichtung Sommer-Luminanz vs. Hillshade im kombinierten Schattenfeld. Beide `0` → nur Hillshade (mit Makro-Anteil). |

### `macro_hillshade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Anteil geglättetem (generalisiertem) Hillshade vs. feinem Relief. `0` = nur feines Relief. |

### `cast_shadow_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.18` |
| **Verwendet** | render |
| **Wirkung** | Einfluss extrahierter Sommer-Wurfschatten (Gebäude, Wald) auf das Schattenfeld der Schneedecke. |

### `shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `1.30` |
| **Verwendet** | render |
| **Wirkung** | Verstärkung dunkler Hillshade-Anteile (Schatten) gegenüber Lichtern. > 1 = tiefere Schatten. |

### `highlight_cap`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.55` |
| **Verwendet** | render |
| **Wirkung** | Begrenzung heller Hillshade-Anteile. Niedriger → weniger ausgebrannte Schneehighlights. |

---

## `forest`

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.78, 0.95]` |
| **Verwendet** | snow |
| **Wirkung** | Schneefraktion unter dem Kronendach. Mit Schneedicke: skaliert nach `thickness_fraction` (interner Default-Faktor `canopy_thickness_factor = 0.82`, nicht im Profil konfigurierbar). |

### `forest_snow_intensity`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.94` |
| **Verwendet** | snow, render |
| **Wirkung** | Multiplikator auf `snow_fraction` für die Kronen-Schneeüberblendung. |

### `crown_tophat_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard (`default`)** | `4` |
| **Verwendet** | render |
| **Wirkung** | Radius für White-Top-Hat-Filter zur Kronenerkennung. Grösser → gröbere Kronenstruktur. |

### `green_suppression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.68` |
| **Verwendet** | render |
| **Wirkung** | Entsättigung des Sommergrüns. Höher → grauer Winterwald vor Schneeüberblendung. |

### `winter_luminance_lo`, `winter_luminance_hi`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.18` / `0.46` |
| **Verwendet** | render |
| **Wirkung** | Ziel-Luminanzbereich beim Umsetzen der Sommerschattenstruktur auf Wintertonwerte. |

### `summer_structure_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.62` |
| **Verwendet** | render |
| **Wirkung** | Wie stark die Sommer-Luminanzstruktur (Schatten unter Bäumen) im Winter erhalten bleibt. |

### `max_crown_snow_alpha`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.74` |
| **Verwendet** | render |
| **Wirkung** | Maximale Deckkraft des Schnees auf Kronen. |

### `canopy_blanket`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.58` |
| **Verwendet** | render |
| **Wirkung** | Grundbedeckung der Kronenfläche mit Schnee, unabhängig von Einzelbaumstruktur. |

### `crown_highlight_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.48` |
| **Verwendet** | render |
| **Wirkung** | Schnee-Highlights auf erkannten Kronen (Top-Hat-Signal). |

### `original_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.08` |
| **Verwendet** | render |
| **Wirkung** | Sommertextur in Kronenlücken (max. ~0.14 intern). |

### `crown_noise_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.05` |
| **Verwendet** | render |
| **Wirkung** | Rausch-Textur auf Kronen-Schnee. |

---

## `settlement`

Siedlungsflächen. Fehlt der Block, werden `open_land`-Werte als Fallback genutzt.

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.86, 0.97]` |
| **Verwendet** | snow |
| **Wirkung** | Ziel-Schneefraktion mit Höhenmodifikator. |

### `snow_brightness`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.95` |
| **Verwendet** | snow, render |
| **Wirkung** | Helligkeit der Schneeschicht auf Siedlungsflächen. |

### `snow_texture_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.32` |
| **Verwendet** | snow, render |
| **Wirkung** | Stärke des Schneerauschens. |

### `original_texture_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.04` |
| **Verwendet** | render |
| **Wirkung** | Sommertextur-Durchscheinen unter der Schneedecke. |

### `max_snow_blend`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.98` |
| **Verwendet** | render |
| **Wirkung** | Obergrenze der Schneeüberblendung. |

### `hillshade_compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.42` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Dynamikreduktion (wie `open_land`). |

### `snow_flattening`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.12` |
| **Verwendet** | render |
| **Wirkung** | Relief-Reduktion auf verschneiten Siedlungsflächen. |

### `summer_shade_weight`, `hillshade_shade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Mischung Sommer-Luminanz / Hillshade im Schattenfeld. |

### `macro_hillshade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Anteil Makro-Hillshade. `0` = deaktiviert. |

### `cast_shadow_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.38` |
| **Verwendet** | render |
| **Wirkung** | Wurfschatten-Einfluss (höher als `open_land` wegen Gebäudeschatten in Dörfern). |

### `shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `1.24` |
| **Verwendet** | render |
| **Wirkung** | Verstärkung dunkler Schattenanteile. |

### `highlight_cap`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.52` |
| **Verwendet** | render |
| **Wirkung** | Begrenzung heller Highlights. |

---

## `rock`

Fels und offener Boden.

### `slope_visibility_threshold_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `35` |
| **Verwendet** | snow |
| **Wirkung** | Ab dieser Hangneigung steigt die Fels-Sichtbarkeit. |

### `roughness_visibility_threshold`

| | |
|---|---|
| **Typ** | Zahl (normiert 0–1) |
| **Standard (`default`)** | `0.46` |
| **Verwendet** | snow |
| **Wirkung** | Ab dieser lokalen Rauheit wird Fels sichtbarer. |

### `max_snow_fraction`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.94` |
| **Verwendet** | snow |
| **Wirkung** | Maximale Schneefraktion auf Fels (wird durch Sichtbarkeit und Aspekt reduziert). Ab `full_snow_thickness_m` nomineller Blanket-Dicke gilt volle Decke auch auf Felsvorsprüngen. |

### `full_snow_thickness_m`

| | |
|---|---|
| **Typ** | Meter |
| **Standard (`default`)** | `0.5` |
| **Verwendet** | snow |
| **Wirkung** | Ab dieser nominalen Schneehöhe (`blanket_thickness_m`) wird Fels/Geröll wie voll verschneit behandelt — hohe `snow_fraction`, reduzierte `rock_visibility`. |

### `thickness_burial_radius_m`

| | |
|---|---|
| **Typ** | Zahl (Meter) |
| **Standard (`default`)** | `38` |
| **Verwendet** | snow |
| **Wirkung** | Nachbarschaftsradius für lokale Schneetiefe (Maximum-Filter). Grösserer Radius „vergräbt" kleine Felsvorsprünge unter Nachbarschnee. |

### `thickness_burial_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.72` |
| **Verwendet** | snow |
| **Wirkung** | Stärke der Schneebedeckung auf Fels durch lokale Schneedicke. |

### `min_rock_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.24` |
| **Verwendet** | snow, render |
| **Wirkung** | Minimale Fels-Sichtbarkeit auf steilen Hängen. |

### `aspect_south_penalty`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.08` |
| **Verwendet** | snow |
| **Wirkung** | Reduktion der Schneefraktion auf südorientiertem Fels. |

### `gentle_slope_max_deg`, `steep_slope_min_deg`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `28` / `50` |
| **Verwendet** | snow, render |
| **Wirkung** | Definieren den Übergang flach ↔ steil für sanften Schnee-Boost und Sichtbarkeitskurven. |

### `gentle_snow_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.22` |
| **Verwendet** | snow |
| **Wirkung** | Zusätzliche Schneefraktion auf flachem Fels im Snow-Layer. |

### `gentle_render_boost`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.30` |
| **Verwendet** | render |
| **Wirkung** | Zusätzliche Schnee-Überblendung auf flachem Fels im Render-Schritt (`gentle_factor × gentle_render_boost`). Fehlt der Parameter, wird `gentle_snow_boost` als Fallback genutzt. |

### `steep_snow_cap`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.55` |
| **Verwendet** | snow |
| **Wirkung** | Reduziert maximale Schneefraktion auf steilem Fels proportional zur Hangneigung. |

### `hillshade_compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.55` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Dynamikreduktion auf Fels. |

### `snow_flattening`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.24` |
| **Verwendet** | render |
| **Wirkung** | Relief-Reduktion auf verschneitem Fels. |

### `summer_shade_weight`, `hillshade_shade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Mischung Sommer-Luminanz / Hillshade. |

### `shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `1.38` |
| **Verwendet** | render |
| **Wirkung** | Verstärkung dunkler Schattenanteile. |

### `highlight_cap`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.55` |
| **Verwendet** | render |
| **Wirkung** | Begrenzung heller Highlights. |

### `summer_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.28` |
| **Verwendet** | render |
| **Wirkung** | Erhalt der Sommertextur/Farbe auf sichtbarem Fels. Höher → mehr Sommeranteil auf Felsflächen. |

---

## `roads`

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.2, 0.5]` |
| **Verwendet** | snow |
| **Wirkung** | Geringe Schneefraktion — Strassen bleiben erkennbar. |

### `road_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.85` |
| **Verwendet** | snow, render |
| **Wirkung** | Grundsichtbarkeit der Strasse im Winterbild. |

### `summer_line_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.75` |
| **Verwendet** | render |
| **Wirkung** | Nutzung dunkler Sommerstruktur zur Strassenführung. |

### `min_visibility`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.42` |
| **Verwendet** | render |
| **Wirkung** | Untergrenze der Strassendeckkraft (unabhängig von Sommerkontrast). |

---

## `paths`

Wanderwege — sollen unter Schnee verschwinden.

### `snow_fraction`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.90, 0.98]` |
| **Verwendet** | snow |
| **Wirkung** | Hohe Schneefraktion in den Snow-Layern. |

### `snow_brightness`, `snow_texture_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.94` / `0.42` |
| **Verwendet** | snow, render |
| **Wirkung** | Helligkeit und Textur der Schneedecke auf Wegflächen. |

### `bury_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.98` |
| **Verwendet** | render |
| **Wirkung** | Verstärkung der Schneeüberblendung. 1 ≈ vollständig vergraben. |

### `max_snow_blend`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.98` |
| **Verwendet** | render |
| **Wirkung** | Maximale Deckkraft der Schneedecke über dem Weg. |

---

## `buildings`

### `roof_snow_intensity`

| | |
|---|---|
| **Typ** | `[min, max]` (0–1) |
| **Standard (`default`)** | `[0.5, 0.8]` |
| **Verwendet** | snow, render |
| **Wirkung** | Schneeintensität auf Dächern (Mittelwert + Höhenmodifikator in Snow-Layer). |

### `edge_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.95` |
| **Verwendet** | render |
| **Wirkung** | Gebäudekanten aus dem Sommerbild werden überblendet, um geometrische Schärfe zu erhalten. |

### `brighten_factor`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.45` |
| **Verwendet** | render |
| **Wirkung** | Skalierung der Dach-Schneeüberblendung. |

### `wall_preservation`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.68` |
| **Verwendet** | render |
| **Wirkung** | Erhalt der Fassadenfarbe/Struktur unterhalb des Dachschnees. |

---

## `water`

### `shore_snow_width_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard (`default`)** | `3` |
| **Verwendet** | snow |
| **Wirkung** | Breite des Uferschnee-Streifens entlang der Wassergrenze. `0` = aus. |

### `shore_snow_intensity`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.4` |
| **Verwendet** | snow |
| **Wirkung** | Schneefraktion im Uferstreifen. |

### `water_darken`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.85` |
| **Verwendet** | render |
| **Wirkung** | Multiplikator auf RGB in Gewässerflächen (dunkleres Winterwasser). |

---

## `rendering`

Globale Render-Einstellungen.

### `snow_color`

| | |
|---|---|
| **Typ** | `[R, G, B]` (0–1) |
| **Standard (`default`)** | `[0.96, 0.98, 0.99]` |
| **Verwendet** | render |
| **Wirkung** | Basis-Schneefarbe für alle Klassen. |

### `road_color`

| | |
|---|---|
| **Typ** | `[R, G, B]` (0–1) |
| **Standard (`default`)** | `[0.55, 0.58, 0.62]` |
| **Verwendet** | render |
| **Wirkung** | Zielfarbe für Strassenpixel. |

### `noise_scale_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard (`default`)** | `8` |
| **Verwendet** | render |
| **Wirkung** | Wellenlänge des bandbegrenzten Schneerauschens. Grösser → grobkörnigere Textur. |

### `hillshade_strength_open_land`, `hillshade_strength_rock`, `hillshade_strength_settlement`, `hillshade_strength_forest`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.0` / `0.0` / `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Stärke auf der Schneeschicht pro Klasse. `0` = flache Schneefarbe ohne Relief (deaktiviert). |

---

## `rendering.map_shading`

Kartographisches Schattieren — letzter Schritt, nach Tonwertkorrektur.

### `sun_azimuth`, `sun_altitude`

| | |
|---|---|
| **Typ** | Zahl (Grad) |
| **Standard (`default`)** | `135` / `32` |
| **Verwendet** | render |
| **Wirkung** | Überschreibt `terrain.hillshade` für Render und Relief. Typisch tiefstehende Südsonne (~11:00 Uhr). |

### `south_shadow_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.2` |
| **Verwendet** | render |
| **Wirkung** | Verstärkung von Schatten auf der Lee-Seite (Expositions-Schattierung) in Map-Shading und Relief. |

### `strength`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Gesamtstärke der kartographischen Schattierung. `0` = aus. |

### `hillshade_weight`, `aspect_weight`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Mischung Hillshade vs. Expositions-Relief im Map-Shading. Bei `strength: 0` ohne Wirkung. |

### `compression`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.90` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Dynamikreduktion vor Anwendung. |

### `min_snow`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.10` |
| **Verwendet** | render |
| **Wirkung** | Minimale Schneefraktion, ab der Map-Shading wirkt. |

### `shade_min`, `sun_max`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.52` / `1.38` |
| **Verwendet** | render |
| **Wirkung** | Untere und obere Grenze des Luminanz-Multiplikators (`clip(1 + relief, shade_min, sun_max)`). |

### `protect_masks`

| | |
|---|---|
| **Typ** | Liste von Maskennamen |
| **Standard (`default`)** | `["water_mask"]` |
| **Verwendet** | render |
| **Wirkung** | Klassen, auf denen Map-Shading nicht angewendet wird. |

---

## `rendering.cast_shadows`

Extrahiert und malt Sommer-Wurfschatten (Gebäude, Wald) auf Schnee.

### `reference_sigma_px`, `fine_sigma_px`

| | |
|---|---|
| **Typ** | Zahl (Pixel) |
| **Standard (`default`)** | `34` / `7` |
| **Verwendet** | render |
| **Wirkung** | Gauß-Radien zur Schattenextraktion aus Sommer-Luminanz. Gross = weite Schatten, klein = feine Details. |

### `building_radius_px`

| | |
|---|---|
| **Typ** | Integer (Pixel) |
| **Standard (`default`)** | `12` |
| **Verwendet** | render |
| **Wirkung** | Dilatationsradius um Gebäude für Schattenverstärkung. |

### `building_boost`, `forest_boost`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `1.75` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Lokale Verstärkung der Schattenextraktion nahe Gebäuden bzw. im Wald. `forest_boost: 0` = keine Wald-Verstärkung. |

### `base_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Grundstärke der Schattenextraktion ausserhalb Gebäude/Wald. `0` = nur lokale Verstärkung. |

### `final_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.68` |
| **Verwendet** | render |
| **Wirkung** | Endgültige Stärke beim Auftragen der Wurfschatten auf das Winterbild. |

### `max_darken`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.50` |
| **Verwendet** | render |
| **Wirkung** | Maximale Abdunklung durch Wurfschatten. |

### `min_snow`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.20` |
| **Verwendet** | render |
| **Wirkung** | Minimale Schneefraktion für Wurfschatten-Anwendung. |

---

## `rendering.relief`

Feines Winter-Relief auf Schneeflächen (vor Tonwertkorrektur).

### `hillshade_strength`

| | |
|---|---|
| **Typ** | Zahl (≥ 0) |
| **Standard (`default`)** | `0.0` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Relief auf Schneeflächen. `0` = deaktiviert. |

### `compression`, `macro_hillshade_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.52` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Hillshade-Dynamikreduktion und Makro-Anteil. Bei `hillshade_strength: 0` ohne sichtbare Wirkung. |

### `min_snow`, `shade_min`, `sun_max`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.30` / `0.74` / `1.16` |
| **Verwendet** | render |
| **Wirkung** | Schwellen und Luminanz-Clipping für Relief-Pass. |

### `snow_tint_strength`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.38` |
| **Verwendet** | render |
| **Wirkung** | Schneefarb-Tönung statt neutralem Grau in reliefierten Bereichen. |

### `summer_relief_weight`, `cast_shadow_relief_weight`

| | |
|---|---|
| **Typ** | Zahl (0–1) |
| **Standard (`default`)** | `0.0` / `0.0` |
| **Verwendet** | render |
| **Wirkung** | Einbindung Sommer-Luminanz bzw. Wurfschatten ins Relief. `0` = deaktiviert. |

### `class_weights`

| | |
|---|---|
| **Typ** | Map Klasse → Zahl |
| **Schlüssel (`default`)** | `open_land: 0.0`, `settlement: 0.0`, `rock: 0.0`, `paths: 0.0`, `buildings: 0.0` |
| **Verwendet** | render |
| **Wirkung** | Reliefstärke pro Landbedeckungsklasse. Alle `0` → kein klassenspezifisches Relief. |

---

## `rendering.summer_structure`

Abschliessende sommerverankerte Tonwertkorrektur.

### `highlight_rolloff`

| | |
|---|---|
| **Typ** | Boolean |
| **Standard (`default`)** | `true` |
| **Verwendet** | render |
| **Wirkung** | Aktiviert weiche Highlight-Kompression gegen ausgebrannte Schneeflächen. |

### `highlight_knee`, `highlight_compression`

| | |
|---|---|
| **Typ** | Zahl |
| **Standard (`default`)** | `0.88` / `0.52` |
| **Verwendet** | render |
| **Wirkung** | Ab welcher Luminanz und mit welcher Stärke Highlights abgerollt werden. |

### `protect_masks`

| | |
|---|---|
| **Typ** | Liste von Maskennamen |
| **Standard (`default`)** | `["water_mask"]` |
| **Verwendet** | render |
| **Wirkung** | Von der Tonwertkorrektur ausgeschlossene Klassen. |

### `classes.{mask_name}`

Pro Klasse (`open_land_mask`, `forest_mask`, `rock_or_bare_ground_mask`, `settlement_mask`, `path_mask`):

| Parameter | Standard (`default`) | Verwendet | Wirkung |
|-----------|---------------------|-----------|---------|
| `lum_lo` | siehe unten | render | Untere Ziel-Luminanz |
| `lum_hi` | siehe unten | render | Obere Ziel-Luminanz |
| `strength` | siehe unten | render | Mischung Sommer-Luminanzstruktur → Winterzielbereich |

**Standardwerte pro Klasse:**

| Maske | `lum_lo` | `lum_hi` | `strength` |
|-------|----------|----------|------------|
| `open_land_mask` | `0.84` | `0.98` | `0.0` |
| `forest_mask` | `0.34` | `0.68` | `0.36` |
| `rock_or_bare_ground_mask` | `0.72` | `0.94` | `0.16` |
| `settlement_mask` | `0.58` | `0.90` | `0.36` |
| `path_mask` | `0.70` | `0.92` | `0.38` |

`strength: 0.0` bei `open_land_mask` deaktiviert die Tonwertkorrektur auf offenem Land.

---

## Profil abstimmen

1. **Schneehöhe und -verteilung:** `snow_surface` → `winter-ortho snow-surface`
2. **Klassen-Schneefraktionen:** Klassen-Blöcke → `winter-ortho snow`
3. **Visuelles Erscheinungsbild:** `rendering` + Klassen-Render-Parameter → `winter-ortho render`

Region-Parameter (Auflösung, Terrain, QA) liegen in `config/regions/` — siehe [regions.md](regions.md).

```bash
winter-ortho run-all --tile-id finsteraarhorn_001 --profile default --config config/regions/finsteraarhorn.yaml
```
