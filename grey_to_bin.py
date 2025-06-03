import geopandas as gpd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio
from shapely.geometry import box

def grey_to_binary_and_plot(raster_path):
    with rasterio.open(raster_path) as raster:
        # Read the entire raster image
        image = raster.read(1)  # Read the first band

        # Convert to binary based on the threshold of 1000
        binary_image = np.where(image > 1000, 1, 0).astype(np.uint8)

        # Plot the binary image
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Raster Image')
        plt.axis('off')  # Hide the axis
        plt.show()

# Paths to the GeoJSON and raster files
geojson_path = r"C:\job\uz_26_sofpr_20090000.geojson"
raster_path2 = r"C:\job\VPP_2021_S2_T34TFN-010m_V105_s1_TPROD_FILLNODATA.tif"

grey_to_binary_and_plot(raster_path2)
