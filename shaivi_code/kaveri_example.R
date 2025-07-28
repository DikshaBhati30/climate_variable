# Load required libraries
library(terra)
library(modifiedmk)
library(raster)

# Set working directory
setwd("D:/SHAIVI_THESIS/DATASETS/RAINFALL & TEMP_ERA5/Temp_era5_grib(2014-2024)/Kaveri")

# List all GRIB files
file_list <- list.files(pattern = "*.grib", full.names = TRUE)

# Read all GRIB files as a SpatRaster stack
temp_stack <- rast(file_list) #rast used in terra package for stacking and stack fun used in raster package

# Get the number of layers in the stack
num_layers <- nlyr(temp_stack)
num_layers         #87648 layers


# Create year index based on the number of layers
num_yrs <- num_layers/8760
# Create a sequence of years without worrying about leap years
years <- rep(2014:2023, each = 8760)

# Find positions where leap years should have extra 24 values
leap_years <- c(2016, 2020)
for (leap in leap_years) {
  leap_indices <- which(years == leap)  # Find positions of the leap year
  extra_indices <- tail(leap_indices, 24)  # Pick last 24 values
  years <- append(years, rep(leap, 24), after = max(extra_indices))  # repeat leap year number 24 times each and atlast hihest values get into position
}

# Trim or extend to match num_layers
years <- years[1:num_layers]  # Ensure correct length

# Aggregate hourly data to annual maximum values
annual <- tapp(temp_stack, index = years, fun = max, na.rm = TRUE)

tsfun = function(x,na.rm){
  if(all(is.na(x))){return(NaN)}
  return(pwmk(x)[["Sen's Slope"]])
}
result <- app(annual, fun=tsfun, na.rm = TRUE)

result

plot(result)

# Save the aggregated annual raster
#use "filetype in terra packakge and use "format in raster package
file2 <- "D:/SHAIVI_THESIS/DATASETS/RAINFALL & TEMP_ERA5/Temp_era5_grib(2014-2024)/Kaveri.tif"
writeRaster(result, "D:/SHAIVI_THESIS/DATASETS/RAINFALL & TEMP_ERA5/Temp_era5_grib(2014-2024)/Kaveri.tif", filetype="GTiff", overwrite=TRUE)


