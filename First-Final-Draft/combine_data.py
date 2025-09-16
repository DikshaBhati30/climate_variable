import xarray as xr
import os
import datetime

def combine_daily_netcdf(base_dir: str, year: int) -> xr.Dataset:
    year_dir = os.path.join(base_dir, f'sis-agrometeorological-indicators_{year}_')
    
    if not os.path.exists(year_dir):
        return None

    file_list = sorted([
        os.path.join(year_dir, f)
        for f in os.listdir(year_dir)
        if f.endswith('.nc') and not f.startswith('.')
    ])
    
    if not file_list:
        return None

    datasets = []
    for i, file_path in enumerate(file_list):
        try:
            # Print progress so you can see where it stops
            print(f"Opening file {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
            ds = xr.open_dataset(file_path, engine='netcdf4')
            datasets.append(ds)
        except Exception as e:
            print(f"\n\nERROR: Failed to open or read file: {file_path}")
            print(f"Error message: {e}\n\n")
            return None # Stop processing on error

    print("All files opened successfully. Concatenating...")
    combined_ds = xr.concat(datasets, dim='time')
    return combined_ds
def main():
    # Define your base data path and the range of years
    base_path = '/Volumes/SSD/data/sis-agrometeorological-indicators'
    start_year = 1990
    end_year = 2025

    # The script processes each year one by one
    for year in range(start_year, end_year + 1):
        print(f"\n--- Processing Year {year} ---")
        
        # Combine the daily data for the current year
        combined_data = combine_daily_netcdf(base_path, year)
        
        if combined_data is not None:
            # Define the output path for the combined yearly file
            output_filename = f'combined_pet_data_{year}.nc'
            output_path = os.path.join(base_path, output_filename)
            
            # Save the combined dataset to a new single file
            print(f"Saving combined data for {year} to {output_path}...")
            try:
                # Use zlib=True for compression to save disk space
                combined_data.to_netcdf(output_path, engine='netcdf4', encoding={'ReferenceET_PenmanMonteith_FAO56': {'zlib': True, 'complevel': 4}})
                print(f"Successfully saved {output_filename}")
            except Exception as e:
                print(f"Error saving file for year {year}: {e}")
            
if __name__ == "__main__":
    # Check if the base path exists before running the script
    base_dir_check = '/Volumes/SSD/data/sis-agrometeorological-indicators'
    if not os.path.isdir(base_dir_check):
        print(f"Error: Base directory '{base_dir_check}' not found. Please check your path.")
    else:
        main()