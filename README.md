# green_project

Measures vegetation "greenness" (productivity) per city sub-sector, per year, from
Copernicus HR-VPP satellite raster tiles, and tracks the trend over time (2017–2023).

Sectors are defined as polygons in a GeoJSON file (one per city: e.g. Sofia, Ruse).
For each year, a raster tile (or pair of tiles) is processed and masked against those
polygons to get per-sector vegetation stats, which are then combined into one final
spreadsheet-style CSV.

## Data source

Raster inputs are HR-VPP Vegetation Productivity tiles (`VPP_<year>_S2_<tile>-010m_V10x_s1_TPROD.tif`)
from the Copernicus WEkEO portal:
https://data.wekeo.copernicus.eu/data?view=dataset&dataset=EO%3AEEA%3ADAT%3ACLMS_HRVPP_VPP

These are **not** currently downloaded automatically — they're expected to already exist
in the local folders referenced by `full_chain.py` / `full_chain_dual_raster.py`
(e.g. `tif_files_sofia/`, `tif_files_sofia_dual/TPROD/`, `tif_files_ruse/HR_VPP/`)
before running the pipeline. Access requires a free WEkEO account with API/HDA credentials.

## Pipeline order

```
 1. Download raster tiles into the expected folder (manual, for now)
 2. full_chain.py            (single tile per year)
      or
    full_chain_dual_raster.py (two adjacent tiles per year, merged)
        -> writes image_0_data.csv ... image_6_data.csv (one per year)
 3. excel_builder.py
        -> combines the per-year CSVs into one final CSV
           (e.g. final_data_sofia_dual.csv, final_data_ruse.csv)
 4. (optional, standalone) bin_to_green.py, centroid_finder.py
        -> visualization / sanity checks on one raster at a time
```

## Scripts

### `full_chain.py`
Main single-tile pipeline. For each year's raster tile:
1. Fills NoData pixels with the mean of nearby valid pixels (`replace_nodata_with_mean`).
2. Thresholds the result to a binary vegetation / non-vegetation image (`grey_to_binary_and_plot`).
3. Masks the binary raster against every sector polygon in the GeoJSON and computes
   pixel counts + mean value per sector (`get_data`).
4. Writes one `image_<i>_data.csv` per year.

Currently configured for the Sofia area, tile `T34TFN`, years 2017–2023, reading from
`tif_files_sofia/`.

### `full_chain_dual_raster.py`
Same pipeline as `full_chain.py`, but for areas that span two adjacent tiles per year
(e.g. Sofia tiles `T34TFN` + `T34TGN`, or Ruse tiles `E55N24` + `E56N24`). Adds a
`merge_rasters()` mosaic step before the fill / binarize / mask steps. Writes its
per-year CSVs into the `sofia_dual/` output folder.

### `excel_builder.py`
Aggregation step. Reads all the per-year `image_<i>_data.csv` files, pivots them into
one row per sector with one column per year (2017–2023), adds `max`, `min`, and a
year-over-year trend `correlation` column, joins in sector names/IDs from the GeoJSON,
and writes the final combined CSV. This is what produced `final_data_sofia_dual.csv`
and `final_data_ruse.csv` already in this repo.

### `bin_to_green.py`
Standalone visualization script. Takes one already-processed binary raster + the
GeoJSON, counts vegetated ("white") pixels per sector, colors each sector a shade of
green based on % vegetation cover, and plots the map. Also writes `white_pixel_counts.csv`.
Not part of the year-by-year loop — run manually on a single raster for a quick look.

### `centroid_finder.py`
Standalone QA/diagnostic script. For one binary raster + the GeoJSON, compares each
polygon's geometric centroid to the centroid of its vegetated pixels, and plots both,
to sanity-check alignment between the GeoJSON polygons and the raster data.

### `fill_data.py` / `grey_to_bin.py`
Early, single-purpose versions of the "fill NoData" and "binarize" steps that were
later folded into `full_chain.py`. Kept for reference/debugging on a single file;
not meant to be run as part of the main pipeline.

### `final_data_ruse.csv` / `final_data_sofia_dual.csv`
Pre-generated **outputs** of the pipeline (from `excel_builder.py`), included in the
repo for reference — not inputs to anything.

## Known rough edges (as of last working session)
- Raster tile downloads are manual — this is the next thing to automate.
- Several paths are hardcoded to local Windows paths (`C:\job\...`) and will need
  updating per machine/environment.
- `bin_to_green.py` and `centroid_finder.py` are standalone tools, not wired into
  the main `full_chain*` loops.
