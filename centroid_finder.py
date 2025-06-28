import rasterio
from rasterio.mask import mask
from rasterio.transform import xy
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import numpy as np

# 1. Open your GeoTIFF
raster_path = "C:/job/failove_ot_tate/tezifailovebroi/binary_raster_2.tif"
with rasterio.open(raster_path) as src:
    raster_crs   = src.crs
    raster_data  = src.read(1)
    raster_extent = (
        src.bounds.left, src.bounds.right,
        src.bounds.bottom, src.bounds.top
    )

    # 2. Load your GeoJSON polygons
    vector_path = "C:/job/failove_ot_tate/tezifailovebroi/FUs_Ruse_ID.geojson"
    polygons = gpd.read_file(vector_path)

    # 3. Reproject if needed
    if polygons.crs != raster_crs:
        polygons = polygons.to_crs(raster_crs)

    # 4. Compute centroids and differences
    results = []
    for idx, row in polygons.iterrows():
        # 4a. Compute the polygon's own centroid
        poly_centroid = row.geometry.centroid
        x_p, y_p = poly_centroid.x, poly_centroid.y

        # 4b. Mask the raster to this polygon
        out_img, out_transform = mask(
            src, [row.geometry], crop=True, filled=True, nodata=0
        )
        band = out_img[0]

        # 4c. Threshold to find "white" pixels
        white_mask = band == 1
        if not white_mask.any():
            # no white pixels in this polygon
            continue

        # 4d. Compute center‐of‐mass in array coordinates
        r_c, c_c = center_of_mass(white_mask.astype(np.uint8))

        # 4e. Convert to map coordinates
        x_c, y_c = xy(out_transform, r_c, c_c, offset="center")

        # 4f. Compute difference vector
        dx = x_c - x_p
        dy = y_c - y_p

        results.append({
            "idx":       idx,
            "poly_x":    x_p,
            "poly_y":    y_p,
            "white_x":   x_c,
            "white_y":   y_c,
            "delta_x":   dx,
            "delta_y":   dy
        })

# 5. Print out centroids and differences
print(f"Computed {len(results)} matching centroids:\n")
for r in results:
    print(
        f"Polygon {r['idx']}: "
        f"poly_centroid=({r['poly_x']:.2f}, {r['poly_y']:.2f}), "
        f"white_centroid=({r['white_x']:.2f}, {r['white_y']:.2f}), "
        f"Δ=({r['delta_x']:.2f}, {r['delta_y']:.2f})"
    )

# 6. Plot everything
fig, ax = plt.subplots(figsize=(10, 10))

# background raster
ax.imshow(
    raster_data,
    extent=raster_extent,
    cmap="gray",
    origin="upper"
)

# polygon outlines
polygons.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    linewidth=1,
    zorder=1
)

# unpack our result lists
poly_xs = [r["poly_x"] for r in results]
poly_ys = [r["poly_y"] for r in results]
white_xs = [r["white_x"] for r in results]
white_ys = [r["white_y"] for r in results]

# plot polygon centroids in red
ax.scatter(
    poly_xs, poly_ys,
    c="red", marker="x",
    s=100, linewidths=2,
    zorder=3, label="Polygon Centroid"
)

# plot white‐mass centroids in blue
ax.scatter(
    white_xs, white_ys,
    c="blue", marker="o",
    s=100, edgecolors="white",
    linewidths=1, zorder=4,
    label="White‐pixel Centroid"
)

ax.legend(loc="upper right")
ax.set_title("GeoTIFF + Polygons + Both Centroids")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()
