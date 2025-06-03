import geopandas as gpd
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import rasterio
from shapely.geometry import box

dict = {}


def count_white_pixels_in_sectors(geojson_path, raster_path):
    # Load the vector data
    sectors = gpd.read_file(geojson_path)

    # Load the raster/satellite image
    with rasterio.open(raster_path) as raster:
        # Ensure the CRS of the vector data matches the raster data
        sectors = sectors.to_crs(crs=raster.crs)

        # Get the raster bounding box and convert it to a shapely polygon
        raster_bounds = box(*raster.bounds)

        # Dictionary to store the count of white pixels in each sector
        white_pixel_counts = {}

        # Loop through each polygon in the GeoJSON file
        for index, sector in sectors.iterrows():
            geom = [sector['geometry']]  # Extract the current polygon

            # Check if the polygon intersects with the raster bounds
            if not geom[0].intersects(raster_bounds):
                print(f"Skipping sector {index}, as it doesn't overlap the raster.")
                continue

            try:
                # Mask the raster with the polygon
                masked_image, _ = mask(raster, geom, crop=True, nodata=raster.nodata)
            except ValueError as e:
                print(f"Error masking sector {index}: {e}")
                continue

            # Check if the raster is multi-band, then select the first band
            if len(masked_image.shape) == 3:
                masked_image = masked_image[0]

            # Ensure masked pixels outside the polygon are not counted
            masked_image = np.ma.masked_equal(masked_image, raster.nodata)

            # Debug: print unique values in the masked image
            unique_values = np.unique(masked_image.compressed())
            #print(f"Sector {index} unique values: {unique_values}")

            # Count the white pixels (255 value)
            white_pixel_count = np.sum(masked_image == 1)

            # Store the result in the dictionary
            white_pixel_counts[index] = white_pixel_count

            dict[index] = white_pixel_count

            # Print results for this polygon
            #print(f"Sector {index}: {white_pixel_count} white pixels")

        return white_pixel_counts

def plot_green(geojson_path, raster_path, white_pixel_counts):
    # Count white pixels in each sector
    #white_pixel_counts = count_white_pixels_in_sectors(geojson_path, raster_path)
    sectors = gpd.read_file(geojson_path)
    
    # Calculate the total number of pixels in each sector
    with rasterio.open(raster_path) as raster:
        pixel_area = (raster.res[0] * raster.res[1])  # Area of a single pixel
        for index, sector in sectors.iterrows():
            geom = sector['geometry']
            sector_area = geom.area
            total_pixels = sector_area / pixel_area
            white_pixel_count = white_pixel_counts.get(index, 0)
            white_pixel_percentage = (white_pixel_count / total_pixels) * 100
            
            # Assign shades of green based on the percentage of white pixels
            if white_pixel_percentage >= 80:
                sectors.at[index, 'color'] = '#006400'  # DarkGreen
            elif white_pixel_percentage >= 60:
                sectors.at[index, 'color'] = '#228B22'  # ForestGreen
            elif white_pixel_percentage >= 40:
                sectors.at[index, 'color'] = '#32CD32'  # LimeGreen
            elif white_pixel_percentage >= 20:
                sectors.at[index, 'color'] = '#7CFC00'  # LawnGreen
            elif white_pixel_percentage > 0:
                sectors.at[index, 'color'] = '#ADFF2F'  # GreenYellow
            else:
                sectors.at[index, 'color'] = 'none'

    # Plot the sectors
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    base = sectors.plot(ax=ax, color='none', edgecolor='black')
    sectors[sectors['color'] != 'none'].plot(ax=base, color=sectors['color'])
    plt.title('Sectors Colored by Percentage of White Pixels')
    plt.show()


# Paths to the GeoJSON and raster files
geojson_path = r"C:\job\uz_26_sofpr_20090000.geojson"
raster_path1 = r"C:\job\VPP_2021_S2_T34TFN-010m_V105_s1_TPROD_mask_FILLNODATA.tif"


# Call the function and get the white pixel counts
white_pixel_counts = count_white_pixels_in_sectors(geojson_path, raster_path1)
plot_green(geojson_path, raster_path1, white_pixel_counts)

# Write dict to a csv file
import csv
with open('white_pixel_counts.csv', 'w') as f:
    for key in dict.keys():
        f.write("%s,%s\n"%(key,dict[key]))

        





