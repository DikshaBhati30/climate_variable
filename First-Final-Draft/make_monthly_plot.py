import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def create_monthly_graph(base_path: str, output_image_path: str):
    """
    Loads a master NetCDF file, calculates the monthly climatology,
    and plots the result.
    """
    # Use the filename for the data from 1990-2023. You can change this if needed.
    master_file_name = 'combined_pet_data_2024.nc'
    master_file_path = os.path.join(base_path, master_file_name)

    try:
        print(f"Attempting to open file: {master_file_path}")
        ds = xr.open_dataset(master_file_path, chunks={'time': 366})
    except FileNotFoundError:
        print(f"Error: The master file was not found at '{master_file_path}'")
        print("Please ensure you have successfully created this file.")
        sys.exit(1) # Exit the script if the file isn't found

    # Select the data variable
    pet_data = ds['ReferenceET_PenmanMonteith_FAO56']

    # Group data by month, calculate the mean across all years,
    # and then calculate the mean across the spatial (lat, lon) dimensions.
    print("Calculating monthly averages across all years... (This may take a moment)")
    monthly_avg = pet_data.groupby('time.month').mean(dim=['time', 'lat', 'lon']).load()
    print("Calculation complete.")

    # Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 7))

    months = monthly_avg['month'].values
    avg_values = monthly_avg.values

    ax.plot(months, avg_values, marker='o', linestyle='-', color='#1f77b4', label='Monthly Average')

    # Format the plot for clarity
    ax.set_title('Average Monthly Reference Evapotranspiration (1990-2023)', fontsize=16, pad=20)
    ax.set_ylabel('Average PET ($mm \\ d^{-1}$)', fontsize=12) # Using LaTeX for units
    ax.set_xlabel('Month', fontsize=12)
    ax.set_xticks(months)
    
    # Use month abbreviations for the x-axis labels
    month_names = [pd.to_datetime(str(m), format='%m').strftime('%b') for m in months]
    ax.set_xticklabels(month_names)

    ax.set_xlim(0.5, 12.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the figure
    print(f"Saving plot to {output_image_path}...")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully: {output_image_path}")

if __name__ == "__main__":
    # The script will look for the data in the directory where you run it,
    # or you can provide a full path.
    base_data_path = '/Volumes/SSD/data/sis-agrometeorological-indicators'
    output_file = 'monthly_average_pet_graph.png'
    
    # Check if the base path exists
    if not os.path.isdir(base_data_path):
        print(f"Error: The directory '{base_data_path}' was not found.")
        print("Please update the 'base_data_path' variable in the script.")
    else:
        create_monthly_graph(base_data_path, output_file)