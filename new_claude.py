import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def mann_kendall_test(data):
    """
    Perform Mann-Kendall trend test on time series data
    Returns: (trend, slope, p_value)
    """
    if np.isnan(data).all() or len(data) < 3:
        return 0, 0, 1
    
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 3:
        return 0, 0, 1
    
    n = len(clean_data)
    
    # Calculate S statistic
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if clean_data[j] > clean_data[i]:
                S += 1
            elif clean_data[j] < clean_data[i]:
                S -= 1
    
    # Calculate variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate Z statistic
    if S > 0:
        Z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_s)
    else:
        Z = 0
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    # Calculate Sen's slope
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i != j:
                slope = (clean_data[j] - clean_data[i]) / (j - i)
                slopes.append(slope)
    
    if slopes:
        sen_slope = np.median(slopes)
    else:
        sen_slope = 0
    
    return Z, sen_slope, p_value

def preprocess_era5_data(ds):
    """
    Preprocess ERA5-Land data with forecast steps
    """
    logging.info("Preprocessing ERA5-Land data...")
    
    # Convert from Kelvin to Celsius
    ds['t2m'] = ds['t2m'] - 273.15
    
    # Handle the forecast step dimension
    if 'step' in ds.dims:
        logging.info(f"Found {len(ds.step)} forecast steps: {ds.step.values}")
        # Take the first step (usually the analysis or shortest forecast)
        ds = ds.isel(step=0)
        logging.info(f"Selected first forecast step")
    
    # Remove unnecessary dimensions first
    if 'surface' in ds.dims:
        ds = ds.isel(surface=0)
        logging.info("Removed surface dimension")
    if 'number' in ds.dims:
        ds = ds.isel(number=0)
        logging.info("Removed number dimension")
    
    # Handle time coordinates more carefully
    logging.info(f"Available coordinates: {list(ds.coords.keys())}")
    
    if 'valid_time' in ds.coords:
        if 'time' in ds.coords:
            # Both exist - use valid_time and drop time
            logging.info("Both time and valid_time found - using valid_time")
            ds = ds.drop_vars('time', errors='ignore')
            time_coord = ds.valid_time
        else:
            time_coord = ds.valid_time
        
        # Create new dataset with valid_time as time
        new_coords = {k: v for k, v in ds.coords.items() if k != 'valid_time'}
        new_coords['time'] = time_coord
        
        # Create new data variables
        new_data_vars = {}
        for var_name, var in ds.data_vars.items():
            if 'valid_time' in var.dims:
                new_dims = [d if d != 'valid_time' else 'time' for d in var.dims]
                new_data_vars[var_name] = (new_dims, var.values, var.attrs)
            else:
                new_data_vars[var_name] = var
        
        # Create new dataset
        ds = xr.Dataset(new_data_vars, coords=new_coords, attrs=ds.attrs)
        logging.info("Successfully converted valid_time to time")
    
    # Sort by time
    ds = ds.sortby('time')
    
    # Convert time to a more standard format if needed
    if hasattr(ds.time.values[0], 'astype'):
        try:
            ds['time'] = pd.to_datetime(ds.time.values)
            logging.info("Converted time to pandas datetime")
        except:
            logging.warning("Could not convert time to pandas datetime")
    
    logging.info(f"Preprocessed data shape: {ds['t2m'].shape}")
    logging.info(f"Final coordinates: {list(ds.coords.keys())}")
    
    try:
        time_min = pd.to_datetime(ds.time.min().values)
        time_max = pd.to_datetime(ds.time.max().values)
        logging.info(f"Time range: {time_min} to {time_max}")
    except:
        logging.info(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    
    return ds

def extract_decadal_seasonal_data(ds, variable='t2m'):
    """
    Extract seasonal averages for each decade from ERA5 dataset
    """
    logging.info("Extracting decadal seasonal data...")
    
    # Convert time to pandas datetime for easier manipulation
    ds['time'] = pd.to_datetime(ds.time.values)
    
    # Define decades
    decades = {
        '1991-2000': (1991, 2000),
        '2001-2010': (2001, 2010), 
        '2011-2020': (2011, 2020),
        '2021-2024': (2021, 2024)  # Partial decade
    }
    
    # Define seasons
    seasons = {
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8], 
        'Autumn': [9, 10, 11],
        'Winter': [12, 1, 2]
    }
    
    decadal_seasonal_data = {}
    
    for decade_name, (start_year, end_year) in decades.items():
        logging.info(f"Processing decade: {decade_name}")
        
        # Filter data for the decade
        decade_mask = (ds['time'].dt.year >= start_year) & (ds['time'].dt.year <= end_year)
        decade_data = ds.where(decade_mask, drop=True)
        
        if len(decade_data.time) == 0:
            logging.warning(f"No data found for decade {decade_name}")
            continue
            
        decade_seasons = {}
        
        for season_name, months in seasons.items():
            # Filter data for the season within this decade
            season_mask = decade_data['time'].dt.month.isin(months)
            season_data = decade_data.where(season_mask, drop=True)
            
            if len(season_data.time) > 0:
                # Group by year and calculate seasonal means for this decade
                seasonal_mean = season_data.groupby('time.year').mean('time')
                decade_seasons[season_name] = seasonal_mean[variable]
                logging.info(f"  {season_name}: {len(seasonal_mean.year)} years")
            else:
                logging.warning(f"  No data found for {season_name} in {decade_name}")
        
        if decade_seasons:
            decadal_seasonal_data[decade_name] = decade_seasons
    
    return decadal_seasonal_data

def calculate_decadal_trend_maps(decadal_seasonal_data):
    """
    Calculate Mann-Kendall trends for each decade and season
    """
    decadal_trend_results = {}
    
    for decade, seasonal_data in decadal_seasonal_data.items():
        logging.info(f"Calculating trends for decade: {decade}")
        decade_results = {}
        
        for season, data in seasonal_data.items():
            logging.info(f"  Processing {season} for {decade}...")
            
            # Initialize result arrays
            ny, nx = data.shape[1], data.shape[2]
            trend_values = np.full((ny, nx), np.nan)
            slope_values = np.full((ny, nx), np.nan)
            p_values = np.full((ny, nx), np.nan)
            
            # Process each grid point
            total_points = ny * nx
            processed = 0
            
            for i in range(ny):
                for j in range(nx):
                    time_series = data.values[:, i, j]
                    if not np.isnan(time_series).all() and len(time_series) >= 3:
                        z_stat, slope, p_val = mann_kendall_test(time_series)
                        trend_values[i, j] = z_stat
                        slope_values[i, j] = slope * 10  # Convert to per decade
                        p_values[i, j] = p_val
                    
                    processed += 1
                    if processed % 500 == 0:
                        logging.info(f"    Processed {processed}/{total_points} grid points")
            
            decade_results[season] = {
                'trend': trend_values,
                'slope': slope_values,
                'p_value': p_values,
                'lat': data.latitude.values,
                'lon': data.longitude.values,
                'years': len(data.year)
            }
            
            # Log some statistics
            significant_points = np.sum(p_values <= 0.05)
            total_valid = np.sum(~np.isnan(trend_values))
            logging.info(f"    {season}: {significant_points}/{total_valid} points with significant trends")
        
        decadal_trend_results[decade] = decade_results
    
    return decadal_trend_results

def create_trend_colormap():
    """
    Create colormap similar to the reference images
    """
    # Colors from blue (cooling) to white (no trend) to red (warming)
    colors = [
        '#000080',  # Dark blue (strong cooling)
        '#0040C0',  # Blue
        '#4080FF',  # Light blue
        '#80C0FF',  # Very light blue
        '#C0E0FF',  # Pale blue
        '#FFFFFF',  # White (no trend)
        '#FFE0C0',  # Pale red
        '#FFC080',  # Light red
        '#FF8040',  # Orange-red
        '#FF4000',  # Red
        '#C00000'   # Dark red (strong warming)
    ]
    
    n_bins = 21  # Number of discrete color levels
    cmap = mcolors.LinearSegmentedColormap.from_list('trend', colors, N=n_bins)
    return cmap

def plot_decadal_seasonal_trends(decadal_trend_results, output_dir='decadal_trend_maps'):
    """
    Create individual trend maps for each decade and season
    """
    Path(output_dir).mkdir(exist_ok=True)
    logging.info(f"Creating decadal trend maps in {output_dir}/")
    
    # Create colormap
    cmap = create_trend_colormap()
    
    # Define trend range for consistent color scaling
    trend_range = (-3, 3)  # Z-score range
    
    for decade, seasonal_results in decadal_trend_results.items():
        logging.info(f"Plotting trends for decade: {decade}")
        
        for season, results in seasonal_results.items():
            logging.info(f"  Creating {season} map for {decade}...")
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10), 
                                  subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Plot trend data
            trend_data = results['trend']
            lons, lats = results['lon'], results['lat']
            
            # Mask non-significant trends (p > 0.05)
            significant_mask = results['p_value'] <= 0.05
            trend_masked = np.where(significant_mask, trend_data, np.nan)
            
            # Create meshgrid for plotting
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Plot the data
            im = ax.pcolormesh(lon_grid, lat_grid, trend_masked,
                              cmap=cmap, vmin=trend_range[0], vmax=trend_range[1],
                              transform=ccrs.PlateCarree(), shading='auto')
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, color='gray')
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.1)
            
            # Set extent based on data
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            ax.set_extent([lon_min-0.1, lon_max+0.1, lat_min-0.1, lat_max+0.1], ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.7, color='gray')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=30)
            cbar.set_label('Mann-Kendall Z-Score\n(Temperature Trend Significance)', fontsize=11, labelpad=15)
            cbar.ax.tick_params(labelsize=9)
            
            # Add title with decade and season
            plt.title(f'{season} Temperature Trends ({decade})\nMann-Kendall Analysis - ERA5-Land Data', 
                     fontsize=14, fontweight='bold', pad=20)
            
            # Add north arrow
            ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                       fontsize=16, fontweight='bold', ha='center', va='center')
            ax.annotate('↑', xy=(0.95, 0.92), xycoords='axes fraction',
                       fontsize=20, ha='center', va='center')
            
            # Add statistics as text
            significant_warming = np.sum((trend_data > 0) & (results['p_value'] <= 0.05))
            significant_cooling = np.sum((trend_data < 0) & (results['p_value'] <= 0.05))
            total_points = np.sum(~np.isnan(trend_data))
            
            stats_text = f"Period: {decade} ({results['years']} years)\n"
            stats_text += f"Significant trends: {significant_warming + significant_cooling}/{total_points} points\n"
            stats_text += f"Warming: {significant_warming}, Cooling: {significant_cooling}"
            
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Save the figure with decade and season in filename
            decade_clean = decade.replace('-', '_')
            output_file = Path(output_dir) / f'{decade_clean}_{season.lower()}_trends.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logging.info(f"  Saved: {output_file}")
    
    # Create summary plot showing all decades for each season
    create_decadal_comparison_plots(decadal_trend_results, output_dir)

def create_decadal_comparison_plots(decadal_trend_results, output_dir):
    """
    Create comparison plots showing all decades for each season
    """
    logging.info("Creating decadal comparison plots...")
    
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    decades = list(decadal_trend_results.keys())
    
    cmap = create_trend_colormap()
    trend_range = (-3, 3)
    
    for season in seasons:
        if not all(season in decadal_trend_results[decade] for decade in decades):
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), 
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        for i, decade in enumerate(decades):
            if i >= len(axes):
                break
                
            ax = axes[i]
            results = decadal_trend_results[decade][season]
            
            # Plot trend data
            trend_data = results['trend']
            lons, lats = results['lon'], results['lat']
            
            # Mask non-significant trends
            significant_mask = results['p_value'] <= 0.05
            trend_masked = np.where(significant_mask, trend_data, np.nan)
            
            # Create meshgrid for plotting
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # Plot the data
            im = ax.pcolormesh(lon_grid, lat_grid, trend_masked,
                              cmap=cmap, vmin=trend_range[0], vmax=trend_range[1],
                              transform=ccrs.PlateCarree(), shading='auto')
            
            # Add geographic features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray')
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.1)
            
            # Set extent
            lon_min, lon_max = lons.min(), lons.max()
            lat_min, lat_max = lats.min(), lats.max()
            ax.set_extent([lon_min-0.05, lon_max+0.05, lat_min-0.05, lat_max+0.05], 
                         ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 8}
            gl.ylabel_style = {'size': 8}
            
            # Add subtitle
            significant_points = np.sum(results['p_value'] <= 0.05)
            total_points = np.sum(~np.isnan(trend_data))
            ax.set_title(f'{decade}\n({results["years"]} years, {significant_points}/{total_points} significant)', 
                        fontsize=11, pad=10)
        
        # Add overall title
        fig.suptitle(f'{season} Temperature Trends by Decade\nMann-Kendall Analysis - ERA5-Land Data', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Mann-Kendall Z-Score', fontsize=12, labelpad=15)
        
        # Save comparison plot
        output_file = Path(output_dir) / f'{season.lower()}_decadal_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logging.info(f"Saved comparison plot: {output_file}")

def process_era5_grib_file(file_path):
    """
    Process ERA5-Land GRIB file with decadal analysis
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logging.info(f"Loading ERA5-Land GRIB file: {file_path}")
    
    try:
        # Load the dataset
        ds = xr.open_dataset(file_path, engine='cfgrib')
        logging.info("Successfully loaded GRIB file")
        
        # Preprocess the data
        ds_processed = preprocess_era5_data(ds)
        
        # Extract decadal seasonal data
        decadal_seasonal_data = extract_decadal_seasonal_data(ds_processed)
        
        # Calculate trends for each decade
        decadal_trend_results = calculate_decadal_trend_maps(decadal_seasonal_data)
        
        # Create plots
        plot_decadal_seasonal_trends(decadal_trend_results)
        
        return decadal_trend_results
        
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

def process_era5_grib_file(file_path):
    """
    Process ERA5-Land GRIB file with decadal analysis
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logging.info(f"Loading ERA5-Land GRIB file: {file_path}")
    
    try:
        # Load the dataset
        ds = xr.open_dataset(file_path, engine='cfgrib')
        logging.info("Successfully loaded GRIB file")
        
        # Preprocess the data
        ds_processed = preprocess_era5_data(ds)
        
        # Extract decadal seasonal data
        decadal_seasonal_data = extract_decadal_seasonal_data(ds_processed)
        
        # Calculate trends for each decade
        decadal_trend_results = calculate_decadal_trend_maps(decadal_seasonal_data)
        
        # Create plots
        plot_decadal_seasonal_trends(decadal_trend_results)
        
        return decadal_trend_results
        
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

def main():
    """
    Main function for ERA5-Land temperature trend analysis
    """
    # Your file path
    file_address = '/Volumes/SSD/data/reanalysis-era5-land-temp/'
    file_name = file_address + 'era5-land_all_years.grib'
    
    try:
        print("="*70)
        print("ERA5-LAND TEMPERATURE TREND ANALYSIS")
        print("Mann-Kendall Test for Seasonal Temperature Trends")
        print("="*70)
        
        results = process_era5_grib_file(file_name)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print("Generated trend maps:")
        print("- spring_trends.png")
        print("- summer_trends.png") 
        print("- autumn_trends.png")
        print("- winter_trends.png")
        
        # Print detailed summary statistics
        print("\n" + "="*70)
        print("DETAILED TREND ANALYSIS RESULTS")
        print("="*70)
        
        for season, data in results.items():
            significant_warming = np.sum((data['trend'] > 0) & (data['p_value'] <= 0.05))
            significant_cooling = np.sum((data['trend'] < 0) & (data['p_value'] <= 0.05))
            total_points = np.sum(~np.isnan(data['trend']))
            non_significant = total_points - significant_warming - significant_cooling
            
            warming_pct = (significant_warming / total_points * 100) if total_points > 0 else 0
            cooling_pct = (significant_cooling / total_points * 100) if total_points > 0 else 0
            
            print(f"\n{season.upper()} SEASON:")
            print(f"  Total grid points analyzed: {total_points}")
            print(f"  Significant warming points: {significant_warming} ({warming_pct:.1f}%)")
            print(f"  Significant cooling points: {significant_cooling} ({cooling_pct:.1f}%)")
            print(f"  No significant trend: {non_significant} ({100-warming_pct-cooling_pct:.1f}%)")
            
            # Calculate average trends for significant points
            warming_mask = (data['trend'] > 0) & (data['p_value'] <= 0.05)
            cooling_mask = (data['trend'] < 0) & (data['p_value'] <= 0.05)
            
            if np.any(warming_mask):
                avg_warming_trend = np.nanmean(data['slope'][warming_mask])
                print(f"  Average warming rate: {avg_warming_trend:.3f}°C/decade")
                
            if np.any(cooling_mask):
                avg_cooling_trend = np.nanmean(data['slope'][cooling_mask])
                print(f"  Average cooling rate: {avg_cooling_trend:.3f}°C/decade")
                
            # Overall average for all significant trends
            significant_mask = data['p_value'] <= 0.05
            if np.any(significant_mask):
                overall_avg_trend = np.nanmean(data['slope'][significant_mask])
                print(f"  Overall average trend: {overall_avg_trend:.3f}°C/decade")
        
        print(f"\nAnalysis region: {data['lat'].min():.2f}°N to {data['lat'].max():.2f}°N, "
              f"{data['lon'].min():.2f}°E to {data['lon'].max():.2f}°E")
        print("Data source: ERA5-Land reanalysis")
        print("Method: Mann-Kendall trend test with Sen's slope estimator")
        print("Significance level: p ≤ 0.05")
                    
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify the file path is correct")
        print("2. Check that the GRIB file is accessible")
        print("3. Ensure all required packages are installed:")
        print("   pip install xarray matplotlib cartopy scipy cfgrib eccodes pandas")

if __name__ == "__main__":
    main()