import xarray as xr
import numpy as np
import pandas as pd
import dask.array
import sys
from importlib import reload

# Custom functions 
sys.path.append('/home/563/ab2313/MJO/functions')
import load_dataset as load
import subphase_calc_functions as subphase_calc
import access_functions as af
import access_correlation_functions as access_corr

# The Pearson-correlation between the ensemble mean of ACCESS-S and AWAP. 
def awap_access_correlation(awap, access):
    
    # Finding the mean across the dimension year
    awap_mean = awap.mean(dim = 'year')
    access_mean = access.mean(dim = 'year')
    
    # Taking the mean from all of the values
    awap_y_sub_mean = awap - awap_mean 
    access_y_sub_mean = access - access_mean 
    
    
    # The covarience: the sum of the product, divfided by the number of samples -1
    cov = (awap_y_sub_mean * access_y_sub_mean).sum(dim = 'year')/(awap_y_sub_mean.count(dim = 'year') - 1)
    
    # the standsard deviation
    awap_std = awap.std(dim = 'year')
    access_std = access.std(dim = 'year')

    # Correlation: cross-covarience divided by product of the standard-deviations
    rho = cov/(awap_std * access_std)

    return rho


# This function calculated the Fisher-z Transform of the rho-values, then calculates the mean of the
# z-values. The mean z-value is then converted back into rho-space.

def fisher_z(data):
    
    # Converting all the data to a z-space
    z_space = 0.5 * np.log((1 + data.precip)/(1- data.precip)).to_dataset(name = 'z_val')

    # Taking the mean across the north of Australia
    z_mean = z_space.mean(dim = ['lat','lon'])

    #Getting the mean of all these values
    rho_mean = np.tanh(z_mean.z_val)
    
    return rho_mean

# This function is a function for the spearman_correlation function
def helper(x, len_a):
    # split x into  two parts: 'a' and 'b'.
    from scipy.stats import spearmanr
    
    
    # Splitting the two concatenated datasets into one data set.
    xa = x[0:len_a]
    xb = x[len_a:]
    
    # If either of the data sets contains only non values
    if np.all(np.isnan(xa)) or np.all(np.isnan(xb)):
        return np.nan
    
    # Removing the points from both dataset that where nan values are located
    idx = np.isfinite(xa) & np.isfinite(xb)
    xa = xa[idx]
    xb = xb[idx]
    
    # Spearmanr doesn't seem to like it when there is less than 4 values
    if len(xa) < 4 or len(xb) < 4:
        return np.nan
    
    spearman_val = spearmanr(xa,xb, nan_policy = 'omit')[0]
    sig = spearmanr(xa,xb, nan_policy = 'omit')[1]
 
    return spearman_val, sig_val

def spearman_correlation(a, b):

    # THis is concating the two different datasets into the one xarry file. They will be split down the middle
    # in the helper functions
    len_a = a.dims['year']
    combined = xr.concat([a,b], dim = 'year').precip

    # This is the axis to apply the function along
    axis =  combined.get_axis_num('year')
    
    spearman_index_meta,spearman_index_meta  = np.apply_along_axis(helper, arr = combined, axis = axis,len_a = len_a)
    
    # Converting the numpy arrays back into xarray files. 
    spearman_index = xr.Dataset({'precip':(('month','phase','lat','lon'), spearman_index_meta)},{
        'month':a.month.values,
        'phase':a.phase.values,
        'lat':a.lat.values,
        'lon':a.lon.values
    })

    
    spearman_sig = xr.Dataset({'precip':(('month','phase','lat','lon'), spearman_sig_meta)},{
        'month':a.month.values,
        'phase':a.phase.values,
        'lat':a.lat.values,
        'lon':a.lon.values
    })

    
    
    return spearman_index,spearman_sig

######################### Reading in Datasets and anomaly calculations


# Import AWAP, ACCESS, RMM in observations and ACCESS and then make sure they are all for the same points in time.
awap, access, rmm_obs, rmm_access = load.load_accessifies_variables2()

# AWAP: Split AWAP into the subphases: enhanced, suppressed, transition and inactive.
awap_split = subphase_calc.split_into_subphase(awap, rmm_obs)

# ACCESS: Split ACCESS into the subphases: enhanced, suppressed, transition and inactive.
access_split = af.access_rmm_split(access, rmm_access)

# Counting the number of days in each rmm subphase in each year. 
rmm_count_obs  = access_general.count_in_rmm_subphase_year_month_resample(rmm_obs)

# AWAP: Calculating the anomalies for the number of days in each MJO phase for each year and each month
# of the wet season. 
count_anom = access_corr.count_month_year_anomalies(awap_split, awap, rmm_count_obs)


# ACCESS: Calculating the anomalies for the number of days in each MJO phase for each year and each month
# of the wet season for each ensemble (_ens) the ensemble med (ens_med). This function includes the 
# counting the days in RMM. 
count_anom_ens, count_anom_ens_med = \
                access_general.access_ensemble_calculation(access_split, access, rmm_access,
                                                           access_corr.count_month_year_anomalies)


######################### Spearman and Pearson Correlations
# Spearman Correlation at each grid cell
spearman_count, sig_count = spearman_correlation(count_anom,count_anom_ens_med)

# Pearsons correlation at each grid cell
rho_count = awap_access_correlation(count_anom,count_anom_ens_med)

# Mean of person correlations using the fisher-z transform. 
rho_mean_count = fisher_z(rho_count)