import xarray as xr
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk # Import pymannkendall

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the GRIB file
fie_address = '/Volumes/SSD/data/reanalysis-era5-land-temp/'
file_name = fie_address + 'era5-land_all_years.grib'

ds = xr.open_dataset(
    file_name,
    engine="cfgrib",
    backend_kwargs={"decode_timedelta": True},
    chunks={"time": 500}   # tune chunk size if needed
)

# Convert to Celsius for easier interpretation
t2m_c = ds["t2m"] - 273.15

t2m_flat = t2m_c.stack(datetime=("time", "step"))
t2m_flat = t2m_flat.assign_coords(valid_time=("datetime", t2m_flat["valid_time"].values))
t2m_flat = t2m_flat.swap_dims({"datetime": "valid_time"})
t2m_flat = t2m_flat.sortby("valid_time")  # make sure sorted

annual_mean = t2m_flat.resample(valid_time="1Y").mean()
annual_max  = t2m_flat.resample(valid_time="1Y").max()
annual_min  = t2m_flat.resample(valid_time="1Y").min()


monthly_mean = t2m_flat.resample(valid_time="1M").mean()
monthly_max  = t2m_flat.resample(valid_time="1M").max()
monthly_min  = t2m_flat.resample(valid_time="1M").min()


# 1. Extract month from valid_time
month = t2m_flat["valid_time"].dt.month

# 2. Map months → seasons
def month_to_season(m):
    if m in [12, 1, 2]:
        return "DJF"   # Winter
    elif m in [3, 4, 5]:
        return "MAM"   # Spring
    elif m in [6, 7, 8]:
        return "JJA"   # Summer
    else:
        return "SON"   # Autumn

seasons = xr.DataArray(
    [month_to_season(m.item()) for m in month.values],
    coords={"valid_time": t2m_flat["valid_time"]},
    name="season"
)

# 3. Attach season coordinate
t2m_with_season = t2m_flat.assign_coords(season=seasons)

# 4. Extract year from valid_time and attach
years = t2m_with_season["valid_time"].dt.year

# years = t2m_with_season["valid_time"].dt.year
t2m_with_season = t2m_with_season.assign_coords(year=years)

# t2m_with_season = t2m_with_season.assign_coords(year=("valid_time", years))

# 5. Group by year + season
seasonal_mean = t2m_with_season.groupby(["year", "season"]).mean()
seasonal_max  = t2m_with_season.groupby(["year", "season"]).max()


annual_mean_avg = annual_mean.mean(dim=["latitude", "longitude"])
monthly_mean_avg = monthly_mean.mean(dim=["latitude", "longitude"])
seasonal_mean_avg = seasonal_mean.mean(dim=["latitude", "longitude"])


decade1_annual_mean = annual_mean_avg.sel(valid_time=slice("1991", "2000"))
decade2_annual_mean = annual_mean_avg.sel(valid_time=slice("2001", "2010"))
decade3_annual_mean = annual_mean_avg.sel(valid_time=slice("2011", "2020"))
decade4_annual_mean = annual_mean_avg.sel(valid_time=slice("2011", "2024"))

# --- Plotting Code for Previous Task ---

# Create output directory if it doesn't exist
if not os.path.exists('output'):
    os.makedirs('output')

# Plot (a): Annual Mean Temperature (1991-2020)
plt.figure(figsize=(10, 6))
annual_mean_1991_2020 = annual_mean_avg.sel(valid_time=slice("1991", "2020"))
plt.plot(annual_mean_1991_2020.valid_time.dt.year, annual_mean_1991_2020, marker='o', linestyle='-', label='Annual Mean Temp')
# Add trendline (approximated from image)
z_a = np.polyfit(annual_mean_1991_2020.valid_time.dt.year, annual_mean_1991_2020, 1)
p_a = np.poly1d(z_a)
plt.plot(annual_mean_1991_2020.valid_time.dt.year, p_a(annual_mean_1991_2020.valid_time.dt.year), "r--", label=f'y={z_a[0]:.4f}x + {z_a[1]:.3f}\nR²={0.086:.3f}') # R^2 from image
plt.title('Annual Mean Temperature (1991-2020)')
plt.xlabel('Years')
plt.ylabel('Annual Mean Temperature in °C')
plt.legend()
plt.grid(True)
plt.savefig('output/annual_mean_temp_1991_2020.png')
plt.close()

# Plot (b): Mean Monthly Temperature (1991-2020)
plt.figure(figsize=(10, 6))
monthly_mean_1991_2020 = monthly_mean_avg.sel(valid_time=slice("1991", "2020"))
# Calculate mean for each month across the years
monthly_mean_by_month = monthly_mean_1991_2020.groupby(monthly_mean_1991_2020.valid_time.dt.month).mean()
plt.plot(monthly_mean_by_month.month, monthly_mean_by_month, marker='o', linestyle='-')
# Add trendline (approximated from image)
z_b = np.polyfit(monthly_mean_by_month.month, monthly_mean_by_month, 1)
p_b = np.poly1d(z_b)
plt.plot(monthly_mean_by_month.month, p_b(monthly_mean_by_month.month), "r--", label=f'y={z_b[0]:.4f}x + {z_b[1]:.3f}\nR²={0.0567:.3f}') # R^2 from image
plt.title('Mean Monthly Temperature (1991-2020)')
plt.xlabel('Month (1991-2020)')
plt.ylabel('Mean Monthly temperature in °C')
plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.grid(True)
plt.savefig('output/mean_monthly_temp_1991_2020.png')
plt.close()

# Plot (c): Annual mean temperature (1991-2000)
plt.figure(figsize=(10, 6))
plt.plot(decade1_annual_mean.valid_time.dt.year, decade1_annual_mean, marker='o', linestyle='-')
# Add trendline (approximated from image)
z_c = np.polyfit(decade1_annual_mean.valid_time.dt.year, decade1_annual_mean, 1)
p_c = np.poly1d(z_c)
plt.plot(decade1_annual_mean.valid_time.dt.year, p_c(decade1_annual_mean.valid_time.dt.year), "r--", label=f'y={z_c[0]:.4f}x + {z_c[1]:.3f}\nR²={0.2657:.4f}') # R^2 from image
plt.title('Annual mean temperature (1991-2000)')
plt.xlabel('Years')
plt.ylabel('Annual mean temperature in °C')
plt.legend()
plt.grid(True)
plt.savefig('output/annual_mean_temp_1991_2000.png')
plt.close()

# Plot (d): Annual mean temperature (2001-2010)
plt.figure(figsize=(10, 6))
plt.plot(decade2_annual_mean.valid_time.dt.year, decade2_annual_mean, marker='o', linestyle='-')
# Add trendline (approximated from image)
z_d = np.polyfit(decade2_annual_mean.valid_time.dt.year, decade2_annual_mean, 1)
p_d = np.poly1d(z_d)
plt.plot(decade2_annual_mean.valid_time.dt.year, p_d(decade2_annual_mean.valid_time.dt.year), "r--", label=f'y={z_d[0]:.4f}x + {z_d[1]:.3f}\nR²={0.0078:.4f}') # R^2 from image
plt.title('Annual mean temperature (2001-2010)')
plt.xlabel('Years')
plt.ylabel('Annual mean temperature in °C')
plt.legend()
plt.grid(True)
plt.savefig('output/annual_mean_temp_2001_2010.png')
plt.close()

# Plot (e): Annual mean temperature (2011-2020)
plt.figure(figsize=(10, 6))
plt.plot(decade3_annual_mean.valid_time.dt.year, decade3_annual_mean, marker='o', linestyle='-')
# Add trendline (approximated from image)
z_e = np.polyfit(decade3_annual_mean.valid_time.dt.year, decade3_annual_mean, 1)
p_e = np.poly1d(z_e)
plt.plot(decade3_annual_mean.valid_time.dt.year, p_e(decade3_annual_mean.valid_time.dt.year), "r--", label=f'y={z_e[0]:.4f}x + {z_e[1]:.3f}\nR²={0.0303:.4f}') # R^2 from image
plt.title('Annual mean temperature (2011-2020)')
plt.xlabel('Years')
plt.ylabel('Annual mean temperature in °C')
plt.legend()
plt.grid(True)
plt.savefig('output/annual_mean_temp_2011_2020.png')
plt.close()

# Plot (f): Seasonal mean temperature (1991-2000)
plt.figure(figsize=(12, 7))
seasonal_mean_1991_2000_da = seasonal_mean_avg.sel(year=slice("1991", "2000"))

# Convert to DataFrame and pivot
df_seasonal_1991_2000 = seasonal_mean_1991_2000_da.to_dataframe(name='temperature')
df_seasonal_1991_2000_pivot = df_seasonal_1991_2000.reset_index().pivot_table(
    index='year',
    columns='season',
    values='temperature'
)

seasons_order = ['MAM', 'JJA', 'SON', 'DJF'] # Spring, Summer, Autumn, Winter
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Default matplotlib colors, adjust if needed

bar_width = 0.2
years_f = df_seasonal_1991_2000_pivot.index.values
x = np.arange(len(years_f))

for i, season in enumerate(seasons_order):
    if season in df_seasonal_1991_2000_pivot.columns: # Check if season exists
        plt.bar(x + i*bar_width, df_seasonal_1991_2000_pivot[season], bar_width, label=season)

plt.title('Seasonal mean temperature (1991-2000)')
plt.xlabel('Time (1991-2000)')
plt.ylabel('Annual mean temperature in °C')
plt.xticks(x + bar_width * (len(seasons_order) - 1) / 2, years_f)
plt.legend(loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('output/seasonal_mean_temp_1991_2000.png')
plt.close()

# Plot (g): Seasonal mean temperature (2001-2010)
plt.figure(figsize=(12, 7))
seasonal_mean_2001_2010_da = seasonal_mean_avg.sel(year=slice("2001", "2010"))
df_seasonal_2001_2010 = seasonal_mean_2001_2010_da.to_dataframe(name='temperature')
df_seasonal_2001_2010_pivot = df_seasonal_2001_2010.reset_index().pivot_table(
    index='year',
    columns='season',
    values='temperature'
)

x = np.arange(len(df_seasonal_2001_2010_pivot.index.values))

for i, season in enumerate(seasons_order):
    if season in df_seasonal_2001_2010_pivot.columns:
        plt.bar(x + i*bar_width, df_seasonal_2001_2010_pivot[season], bar_width, label=season)

plt.title('Seasonal mean temperature (2001-2010)')
plt.xlabel('Time (2001-2010)')
plt.ylabel('Annual mean temperature in °C')
plt.xticks(x + bar_width * (len(seasons_order) - 1) / 2, df_seasonal_2001_2010_pivot.index.values)
plt.legend(loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('output/seasonal_mean_temp_2001_2010.png')
plt.close()

# Plot (h): Seasonal mean temperature (2011-2020)
plt.figure(figsize=(12, 7))
seasonal_mean_2011_2020_da = seasonal_mean_avg.sel(year=slice("2011", "2020"))
df_seasonal_2011_2020 = seasonal_mean_2011_2020_da.to_dataframe(name='temperature')
df_seasonal_2011_2020_pivot = df_seasonal_2011_2020.reset_index().pivot_table(
    index='year',
    columns='season',
    values='temperature'
)

x = np.arange(len(df_seasonal_2011_2020_pivot.index.values))

for i, season in enumerate(seasons_order):
    if season in df_seasonal_2011_2020_pivot.columns:
        plt.bar(x + i*bar_width, df_seasonal_2011_2020_pivot[season], bar_width, label=season)

plt.title('Seasonal mean temperature (2011-2020)')
plt.xlabel('Time (2011-2020)')
plt.ylabel('Annual mean temperature in °C')
plt.xticks(x + bar_width * (len(seasons_order) - 1) / 2, df_seasonal_2011_2020_pivot.index.values)
plt.legend(loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('output/seasonal_mean_temp_2011_2020.png')
plt.close()

# --- Mann-Kendall Trend Analysis ---

# Function to apply Mann-Kendall test to a time series
def apply_mann_kendall(time_series):
    if time_series.isnull().all(): # Handle cases with all NaNs
        return np.nan, np.nan
    try:
        # pymannkendall expects a 1D array or list
        # Corrected function call based on user feedback: mk.original_test(x) returns slope and p-value
        # Flatten the time series to ensure it's a 1D array
        result = mk.original_test(time_series.values.flatten())
        return result.slope, result.p
    except Exception as e:
        logging.error(f"Error applying Mann-Kendall test: {e}")
        return np.nan, np.nan

# Apply Mann-Kendall test to annual mean temperature for each lat/lon point
logging.info("Applying Mann-Kendall test to annual mean temperature data...")

# We need to iterate over latitude and longitude to apply the test to each time series
# t2m_c has dimensions (valid_time, latitude, longitude)
# We want to apply the test along the 'valid_time' dimension for each lat/lon pair.

# Create empty DataArrays to store trend and p-value
trend_da = xr.DataArray(
    np.full((len(t2m_c['latitude']), len(t2m_c['longitude'])), np.nan),
    coords=[t2m_c['latitude'], t2m_c['longitude']],
    dims=['latitude', 'longitude'],
    name='trend'
)
p_value_da = xr.DataArray(
    np.full((len(t2m_c['latitude']), len(t2m_c['longitude'])), np.nan),
    coords=[t2m_c['latitude'], t2m_c['longitude']],
    dims=['latitude', 'longitude'],
    name='p_value'
)

# Iterate over latitude and longitude
for i, lat in enumerate(t2m_c['latitude'].values):
    for j, lon in enumerate(t2m_c['longitude'].values):
        ts = t2m_c.sel(latitude=lat, longitude=lon)
        trend, p_value = apply_mann_kendall(ts)
        trend_da[i, j] = trend
        p_value_da[i, j] = p_value

logging.info("Mann-Kendall test applied. Generating trend maps...")

# Plotting the Mann-Kendall trend results
# Plotting Trend Map
plt.figure(figsize=(12, 8))
# Use p-value to mask non-significant trends if desired, or plot all trends
# For simplicity, let's plot all trends first. We can add masking later.
trend_map = plt.pcolormesh(trend_da.longitude, trend_da.latitude, trend_da, cmap='coolwarm', shading='auto')
plt.colorbar(trend_map, label='Annual Temperature Trend (°C/year)')
plt.title('Mann-Kendall Trend Analysis: Annual Mean Temperature Trend')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig('output/mk_trend_annual_mean_temp.png')
plt.close()

# Plotting P-value Map (e.g., showing significance)
plt.figure(figsize=(12, 8))
# A common threshold for significance is p < 0.05
# We can visualize this by coloring points where p < 0.05 differently, or by plotting p-values directly.
# Let's plot p-values directly for now.
p_map = plt.pcolormesh(p_value_da.longitude, p_value_da.latitude, p_value_da, cmap='viridis', shading='auto')
plt.colorbar(p_map, label='P-value')
plt.title('Mann-Kendall Trend Analysis: P-value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig('output/mk_p_value_annual_mean_temp.png')
plt.close()

logging.info("Mann-Kendall trend maps generated and saved to the 'output' directory.")

logging.info("All tasks completed.")
