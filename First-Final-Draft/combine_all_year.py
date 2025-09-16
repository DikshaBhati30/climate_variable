import xarray as xr
import os
import glob

def combine_all_yearly_data(base_path: str, output_filename: str):
    """
    Finds all yearly combined NetCDF files and merges them into a single
    master file using Dask for memory-efficient processing.
    """
    # 1. Find all the yearly NetCDF files using a wildcard pattern
    # The pattern looks for any file starting with 'combined_pet_data_' and ending with '.nc'
    search_pattern = os.path.join(base_path, 'combined_pet_data_*.nc')
    yearly_files = sorted(glob.glob(search_pattern))
    
    if not yearly_files:
        print(f"Error: No yearly files found at '{base_path}' with pattern 'combined_pet_data_*.nc'")
        return

    print(f"Found {len(yearly_files)} yearly files to combine.")
    print("First file:", os.path.basename(yearly_files[0]))
    print("Last file:", os.path.basename(yearly_files[-1]))

    # 2. Open all files simultaneously using Dask-powered mfdataset
    # This is the key step for memory efficiency.
    print("\nOpening multiple files with Dask...")
    combined_ds = xr.open_mfdataset(
        yearly_files,
        combine='nested',      # More reliable when files are sorted
        concat_dim='time',     # The dimension to join along
        parallel=True,         # Enable Dask's parallel processing
        engine='netcdf4',
        chunks={'time': 366}   # Process the data in chunks of about one year
    )
    
    print("Dataset structure opened successfully:")
    print(combined_ds)

    # 3. Save the final combined dataset to a new file
    output_path = os.path.join(base_path, output_filename)
    print(f"\nSaving combined data to {output_path}...")
    print("This will take a long time. Please be patient. ⏳")
    
    try:
        # Define compression settings to save space
        encoding = {'ReferenceET_PenmanMonteith_FAO56': {'zlib': True, 'complevel': 5}}
        
        # This triggers the Dask computation (reading, combining, compressing, writing)
        combined_ds.to_netcdf(output_path, engine='netcdf4', encoding=encoding)
        
        print(f"\nSuccessfully saved the final combined file! 🎉")
        print(output_path)
    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")

def main():
    # --- CONFIGURE YOUR PATHS HERE ---
    base_data_path = '/Volumes/SSD/data/sis-agrometeorological-indicators'
    final_output_filename = 'master_pet_data_1990-2025.nc'
    
    combine_all_yearly_data(base_data_path, final_output_filename)

if __name__ == "__main__":
    main()