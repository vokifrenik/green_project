"""
download_vpp.py - download HR-VPP TPROD rasters from WEkEO for the area of a GeoJSON.

How it works (the "order form" idea):
  1. Read the GeoJSON and compute its bounding box (the "where").
  2. For every year from FIRST_YEAR up to the current year, build a query (the "what").
  3. Send it to WEkEO with client.search() -> a list of matching files.
  4. Keep only the files we want (10 m resolution, right year).
  5. Download them into OUT_DIR.

Can be run on its own, or imported by the pipeline:
    from download_vpp import download_all, find_pairs

Usage (with the `green` conda env active, and ~/.hdarc set up):
    python download_vpp.py              # DRY RUN: only lists what it would download
    python download_vpp.py --download   # actually downloads
"""
import argparse
from datetime import date
from pathlib import Path

import geopandas as gpd
from hda import Client

# ---- settings you may want to change -------------------------------------
GEOJSON = "data/uz_26_sofpr_20090000.geojson"  # polygons that define the area
OUT_DIR = "tif_files_sofia"                    # where the .tif files go
FIRST_YEAR = 2017
DATASET_ID = "EO:EEA:DAT:CLMS_HRVPP_VPP"
PRODUCT_TYPE = "TPROD"                         # total productivity
SEASON = "s1"                                  # your filenames use season 1
RESOLUTION = "010m"                            # 10 m pixels (100 m also exists)
# Tiles to keep. Your Sofia dual-raster pipeline used these two.
# Set to None to download every tile that touches the GeoJSON.
TILES = ["T34TFN", "T34TGN"]
# ---------------------------------------------------------------------------


def get_bbox(geojson_path):
    """Bounding box of all polygons as [min_lon, min_lat, max_lon, max_lat]."""
    gdf = gpd.read_file(geojson_path).to_crs(4326)  # WEkEO wants lon/lat (WGS84)
    minx, miny, maxx, maxy = gdf.total_bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def build_query(bbox, year):
    """The 'shopping list' for one year."""
    return {
        "dataset_id": DATASET_ID,
        "productType": PRODUCT_TYPE,
        "productGroupId": SEASON,
        "bbox": bbox,
        "startdate": f"{year}-01-01T00:00:00.000Z",
        "enddate": f"{year}-12-31T23:59:59.999Z",
    }


def wanted(result, year, tiles=TILES):
    """Keep only 10 m files of the given year (and of the chosen tiles)."""
    name = result.get("id", "")
    if tiles is not None and not any(t in name for t in tiles):
        return False
    return RESOLUTION in name and f"VPP_{year}_" in name


def download_all(geojson_path=GEOJSON, out_dir=OUT_DIR, tiles=TILES,
                 first_year=FIRST_YEAR, dry_run=False):
    """Download every wanted file from first_year up to the current year.
    Files already present in out_dir are skipped."""
    bbox = get_bbox(geojson_path)
    print(f"Bounding box (lon/lat): {bbox}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in out.iterdir()]

    client = Client()  # logs in using ~/.hdarc

    for year in range(first_year, date.today().year + 1):
        matches = client.search(build_query(bbox, year))
        keep = [r for r in matches.results if wanted(r, year, tiles)]
        todo = [r for r in keep if not any(r["id"] in f for f in existing)]

        print(f"\n{year}: {len(matches)} results, {len(keep)} match, "
              f"{len(todo)} still to download")
        for r in keep:
            print("   ", r["id"])

        if todo and not dry_run:
            matches.results = todo          # download only what we filtered
            matches.download(download_dir=str(out))

    if dry_run:
        print("\nDry run only - nothing downloaded. Re-run with --download.")


def find_pairs(folder, tile_a, tile_b, first_year=FIRST_YEAR):
    """For each year where BOTH tiles are on disk, return (year, path_a, path_b).
    Used by the dual-raster pipeline, which merges two tiles per year."""
    folder = Path(folder)
    pairs = []
    for year in range(first_year, date.today().year + 1):
        pattern = f"VPP_{year}_S2_{{tile}}-{RESOLUTION}_*_{SEASON}_{PRODUCT_TYPE}.tif"
        a = sorted(folder.glob(pattern.format(tile=tile_a)))
        b = sorted(folder.glob(pattern.format(tile=tile_b)))
        if a and b:
            pairs.append((year, str(a[-1]), str(b[-1])))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="really download (default is a dry run)")
    args = parser.parse_args()
    download_all(dry_run=not args.download)


if __name__ == "__main__":
    main()
