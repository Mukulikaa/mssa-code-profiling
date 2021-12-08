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

# Step 4: Centered moving mean

def moving_mean(a, nave, dim, opt=0):
    """
    Parameters
    ----------
    a : array_like
        The object to be converted.
    nave : int, required
        Size of window.
    dim : str, required
        The dimension across which the moving mean has to be computed.
    opt : [-1, 0]
        -1 utilizes circular conditions.
        0 (default) handles edge cases by replacing them with NaNs.

    Notes
    -----
    For opt=-1, we backcast and forecast the timeseries to compute the moving
    means for the edge cases.
    """

    if opt == -1:
        n = nave // 2
        if nave % 2 == 1:
            temp = xr.concat((a[-n:], a, a[:n]), dim=dim)
            return temp.rolling(dim={dim:nave}, center=True).mean().dropna(dim)
        else:
            temp = xr.concat((a[-n:], a, a[:(n - 1)]), dim=dim)
            return temp.rolling(dim={dim:nave}, center=True).mean().dropna(dim)
    elif opt == 0:
        return a.rolling(time=nave, center=True).mean()