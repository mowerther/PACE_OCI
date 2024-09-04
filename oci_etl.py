import numpy as np
from scipy.spatial.distance import cdist

# Extract bands within a wavelength range
def extract_bands(dataset, prefix, min_wavelength, max_wavelength):
    bands = {}
    for var_name in dataset.variables:
        if var_name.startswith(prefix):
            parts = var_name.split('_')
            if len(parts) > 2:
                try:
                    wavelength = float(parts[-1])
                    if min_wavelength <= wavelength <= max_wavelength:
                        bands[wavelength] = dataset[var_name][:]
                except ValueError:
                    pass  # skip if wavelength can't be converted to float
    return bands

# Find closest pixel
def find_closest_pixel(target_lat, target_lon, lat, lon):
    coords = np.column_stack((lat.ravel(), lon.ravel()))
    target_coord = np.array([[target_lat, target_lon]])
    distances = cdist(coords, target_coord)
    return np.unravel_index(np.argmin(distances), lat.shape)

# Get 3x3 pixel average
def get_3x3_average(data, center_index):
    y, x = center_index
    y_min, y_max = max(0, y-1), min(data.shape[0], y+2)
    x_min, x_max = max(0, x-1), min(data.shape[1], x+2)
    return np.nanmean(data[y_min:y_max, x_min:x_max])

# Extract 3x3 spectral data
def extract_spectral_data(bands, index, use_average=False):
    if use_average:
        return {wave: get_3x3_average(data, index) for wave, data in bands.items()}
    else:
        return {wave: data[index] for wave, data in bands.items()}
