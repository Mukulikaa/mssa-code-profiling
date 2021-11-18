import numpy as np
import xarray as xr

# Step 1: Reading data with xarray. Returns a NumPy array.
# path = test-data/TRMM-GPM_pr_Indian_region_1998.nc

def read_data(path):
    dset = xr.open_dataset(path)
    return dset.p

# Step 2: Convert time x lat x lon data into time x space and vice-versa
# Note: ENH: generalize lat, lon

def stack(data): 
    return data.stack(space=("lat", "lon"))

def unstack(data):
    return data.unstack()

# Step 3: Centring data (Subtract dataset mean from each datapoint)

def centre(data):
    return data - data.mean()
