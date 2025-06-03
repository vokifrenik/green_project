import pandas as pd
import numpy as np
import os
import csv
import geopandas as gpd

# Folder where the CSV files are located
folder_name = 'sofia_dual'

# List of CSV files to read from
files = ['image_0_data.csv', 'image_1_data.csv', 'image_2_data.csv', 'image_3_data.csv', 'image_4_data.csv', 'image_5_data.csv', 'image_6_data.csv']

# Prepend the folder name to each file path
file_paths = [os.path.join(folder_name, file) for file in files]

# Output file name
file_name = 'final_data_sofia_dual.csv'

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Iterate over each file
for i, file_path in enumerate(file_paths):
    if os.path.exists(file_path):
        # Read the current CSV file
        df = pd.read_csv(file_path)
        # Increment the sector_id by 1
        df['sector_id'] = df['sector_id'] + 1
        # Extract only the relevant columns ('sector_id' and 'mean_pixel_value')
        year_column = 'mean_pixel_value'
        df = df[['sector_id', year_column]]
        # Rename the year column to the corresponding year based on file index
        df.columns = ['sector_id', f'{2017 + i}']
        # Merge the data into the combined DataFrame
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='sector_id', how='outer')
    else:
        print(f"File not found: {file_path}")

# Convert all columns to numeric, coercing errors to NaN
combined_df.iloc[:, 1:] = combined_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Calculate the 'max' and 'min' columns
combined_df['max'] = combined_df.iloc[:, 1:].max(axis=1)
combined_df['min'] = combined_df.iloc[:, 1:].min(axis=1)

# Calculate the correlation for each row
def calculate_row_correlation(row):
    # Get the year values, ignoring NaN values
    year_values = row[1:-2].dropna().astype(float)
    if len(year_values) < 2:
        return np.nan
    # Debugging output
    print("Calculating correlation for row:")
    print(year_values)
    # Calculate correlation with a sequence (1, 2, 3, ...)
    return np.corrcoef(year_values, np.arange(len(year_values)))[0, 1]

# Apply the correlation function
combined_df['correlation'] = combined_df.apply(calculate_row_correlation, axis=1)

# Load the GeoJSON file
geojson_path = r"C:\job\sofia\uz_26_sofpr_20090000.geojson"
gdf = gpd.read_file(geojson_path)

# Extract the 'id' and 'new_end' fields from the GeoJSON 'properties'
gdf_clean = gdf[['id', 'new_end']]  # Keep 'id' and 'new_end' from the 'properties'

# If the CSV uses 'sector_id' as the key, but GeoJSON uses 'id', rename 'id' to 'sector_id'
gdf_clean = gdf_clean.rename(columns={'id': 'sector_id'})

# Merge the GeoJSON data (gdf_clean) with the combined DataFrame based on 'sector_id'
combined_df = pd.merge(combined_df, gdf_clean, on='sector_id', how='left')


# Write the final DataFrame to a CSV file
combined_df.to_csv(file_name, index=False, encoding='utf-8')
