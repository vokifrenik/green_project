import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

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

        # Write the modified image to a new file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(replaced_image, 1)

    return replaced_image

def plot(image, title, vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()

# Example usage
raster_path = 'VPP_2021_S2_T34TFN-010m_V105_s1_TPROD.tif'
output_path = 'replaced_nodata_raster.tif'
replaced_image = replace_nodata_with_mean(raster_path)
plot(replaced_image, 'Replaced NoData with Mean')

# Save the replaced image to a new tif file

