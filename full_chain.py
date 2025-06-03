import csv
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from rasterio.mask import mask


## Replace NoData values with the mean of the closest k pixels
def replace_nodata_with_mean(raster_path, k=10):
    with rasterio.open(raster_path) as raster:
        nodata = raster.nodata
        profile = raster.profile

        # Read the entire raster image
        image = raster.read(1).astype(np.float32)
        
        # Create a mask for NoData values
        nodata_mask = image == nodata
        
        # Compute distance to the nearest non-NoData pixel
        distance, indices = distance_transform_edt(nodata_mask, return_indices=True)
        
        # Replace NoData values with the mean of the closest k pixels
        replaced_image = image.copy()
        rows, cols = np.where(nodata_mask)
        
        for row, col in zip(rows, cols):
            # Get the indices of the closest non-NoData pixels
            neighbor_rows = indices[0, row, col]
            neighbor_cols = indices[1, row, col]

            # Extract the k closest valid pixel values
            neighbors = []
            for n_row, n_col in zip(neighbor_rows.flat, neighbor_cols.flat):
                if len(neighbors) < k and not np.isnan(image[n_row, n_col]) and image[n_row, n_col] != nodata:
                    neighbors.append(image[n_row, n_col])
                if len(neighbors) >= k:
                    break
            
            # Calculate the mean of the neighbors
            if neighbors:
                replaced_image[row, col] = np.mean(neighbors)


        # Update the profile to match the new data type
        profile.update(dtype=rasterio.float32, nodata=nodata)

        # Write the modified image to a new file and give it a unique name
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(replaced_image, 1)


## From grey scale to binary 
def grey_to_binary_and_plot(raster_path):
    with rasterio.open(raster_path) as raster:
        # Read the entire raster image
        image = raster.read(1)  # Read the first band

        # Convert to binary based on the threshold of 1000
        binary_image = np.where(image > 1000, 1, 0).astype(np.uint8)

        print("Unique values in the binary image:", np.unique(binary_image))

        # Plot the binary image
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Raster Image')
        plt.axis('off')  # Hide the axis
        plt.show()

        # Save the new binary image to a file
        profile = raster.profile
        profile.update(dtype=rasterio.float64)
        
        with rasterio.open(output_path2, 'w', **profile) as dst:
            dst.write(binary_image, 1)

## Third step
def get_data(geojson_path, output_path2, data):
    # Load the vector data
    sectors = gpd.read_file(geojson_path)

    # Load the raster/satellite image
    with rasterio.open(output_path2) as raster:
        # Ensure the CRS of the vector data matches the raster data
        sectors = sectors.to_crs(crs=raster.crs)

        # Get the raster bounding box and convert it to a shapely polygon
        raster_bounds = box(*raster.bounds)

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
            print(f"Sector {index} unique values: {unique_values}")

            # Count the total pixels
            total_pixel_count = masked_image.count()

            # Count the white pixels (255 value)
            white_pixel_count = np.sum(masked_image == 1)

            # Calculate the mean of the non-masked pixels
            mean_pixel_value = masked_image.mean()

            # Store the results in the dictionary
            data[index] = {
                'total_pixel_count': total_pixel_count,
                'white_pixel_count': white_pixel_count,
                'mean_pixel_value': mean_pixel_value
            }
        
def create_file(data, index):
    with open(f'image_{index}_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sector_id', 'total_pixel_count', 'white_pixel_count', 'mean_pixel_value'])
        for sector_id, values in data.items():
            writer.writerow([sector_id, values['total_pixel_count'], values['white_pixel_count'], values['mean_pixel_value']])


## List of images
folder_name = "tif_files_sofia"
images = [f"{folder_name}/VPP_2017_S2_T34TFN-010m_V101_s1_TPROD.tif",
          f"{folder_name}/VPP_2018_S2_T34TFN-010m_V101_s1_TPROD.tif",
          f"{folder_name}/VPP_2019_S2_T34TFN-010m_V101_s1_TPROD.tif",
          f"{folder_name}/VPP_2020_S2_T34TFN-010m_V101_s1_TPROD.tif",
          f"{folder_name}/VPP_2021_S2_T34TFN-010m_V105_s1_TPROD.tif",
          f"{folder_name}/VPP_2022_S2_T34TFN-010m_V105_s1_TPROD.tif",
          f"{folder_name}/VPP_2023_S2_T34TFN-010m_V105_s1_TPROD.tif"]




'''
raster_path = 'VPP_2021_S2_T34TFN-010m_V105_s1_TPROD.tif'
output_path = 'replaced_nodata_raster.tif'
output_path2 = 'binary_raster.tif'

## step 1
replace_nodata_with_mean(raster_path)

## step 2
grey_to_binary_and_plot(output_path)

## step 3
data = {}
get_data(geojson_path, output_path2, data)
'''
## Set the necessary paths
geojson_path = r"C:\job\sofia\uz_26_sofpr_20090000.geojson"

for i in range(len(images)):
    raster_path = images[i]
    output_path = f'replaced_nodata_raster_{i}.tif'
    output_path2 = f'binary_raster_{i}.tif'

    replace_nodata_with_mean(raster_path)
    grey_to_binary_and_plot(output_path)

    data = {}
    get_data(geojson_path, output_path2, data)

    create_file(data, i)
    


