
import xarray as xr
import numpy as np
import pandas as pd
import dask.array
import datetime as dt
import sys
import warnings
warnings.filterwarnings('ignore')



def group_count(group):
    return group.groupby('time.year').count()
# The percent of days as raindays, for each year, month and phase of the MJO

def count_month_year_anomalies(data_split, data, rmm_count):
    

    
    # The number of raindays for each MJO phase for each month and each year
    data_rain_count = data_split.groupby('time.month').apply(group_count)
    
    # Dividing this by the number of days to get a percent
    data_frac = data_rain_count * 100 /rmm_count

    
    # The climatology of the above
    climatology_rain_count = data.groupby('time.month').apply(group_count)
    
    climatology_frac = climatology_rain_count * 100/rmm_count.sum(dim = 'phase')

    # Converting to an aomly
    anom_count = data_frac/climatology_frac
    
    return anom_count

def group_sum(group):
    return group.groupby('time.year').sum()
# The percent of days as raindays, for each year, month and phase of the MJO

def sum_month_year_anomalies(data_split, data, rmm_count):
    

    
    # The number of raindays for each MJO phase for each month and each year
    data_rain_sum = data_split.groupby('time.month').apply(group_sum)
    
    # Dividing this by the number of days to get a percent
    data_frac = data_rain_sum /rmm_count

    
    # The climatology of the above
    climatology_rain_sum = data.groupby('time.month').apply(group_sum)
    
    climatology_frac = climatology_rain_sum/rmm_count.sum(dim = 'phase')

    # Converting to an aomly
    anom_sum = data_frac/climatology_frac
    
    return anom_sum


def group_mean(group):
    return group.groupby('time.year').mean()
# The percent of days as raindays, for each year, month and phase of the MJO

def mean_month_year_anomalies(data_split, data, rmm_count):
    # Note: rmm_count is not used in this function, however, it is still there as it 
    # needs to be for the access_general_functions.access_ensemble_calculation that will 
    # call this function will need it :)

    
    # The mean values of each rainday for each month and each mjo phase
    data_rain_mean = data_split.groupby('time.month').apply(group_mean)
  

    # The climatology of the above
    climatology_rain_mean = data.groupby('time.month').apply(group_mean)
    

    # Converting to an aomly
    anom_mean = data_rain_mean/climatology_rain_mean
    
    return anom_mean

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Spearman Rank Correlationdd'''

# This function is from Scott, and is for appyling np.apply_along_axis
# for two arrays so that spearman rank correlation fucntion can be performed
def helper(x, len_a):
    # split x into 'a' and 'b' parts
    from scipy.stats import spearmanr

    xa = x[0:len_a]
    xb = x[len_a:]

    if np.all(np.isnan(xa)) or np.all(np.isnan(xb)):
        return np.nan
    
    idx = np.isfinite(xa) & np.isfinite(xb)
    
    xa = xa[idx]
    xb = xb[idx]
    
    if len(xa) < 4 or len(xb) < 4:
        return np.nan
    
    corr = spearmanr(xa,xb, nan_policy = 'omit')[0]
 
    return corr


def helper_2(x, len_a):
    # split x into 'a' and 'b' parts
    from scipy.stats import spearmanr

    xa = x[0:len_a]
    xb = x[len_a:]

    if np.all(np.isnan(xa)) or np.all(np.isnan(xb)):
        return np.nan
    
    idx = np.isfinite(xa) & np.isfinite(xb)
    
    xa = xa[idx]
    xb = xb[idx]
    
    if len(xa) < 4 or len(xb) < 4:
        return np.nan
    
    sig = spearmanr(xa,xb, nan_policy = 'omit')[1]
 
    return sig

def spearman_correlation(a, b):

    # THis is concating the two different datasets into the one xarry file. They will be split down the middle
    # later on
    len_a = a.dims['year']
    combined = xr.concat([a,b], dim = 'year').precip

    axis =  combined.get_axis_num('year')
    
    spearman_index_meta  = np.apply_along_axis(helper, arr = combined, axis = axis,len_a = len_a)
    
    spearman_index = xr.Dataset({'precip':(('month','phase','lat','lon'), spearman_index_meta)},{
        'month':a.month.values,
        'phase':a.phase.values,
        'lat':a.lat.values,
        'lon':a.lon.values
    })

    
    
    spearman_sig_meta  = np.apply_along_axis(helper_2, arr = combined, axis = axis,len_a = len_a)
    
    spearman_sig = xr.Dataset({'precip':(('month','phase','lat','lon'), spearman_sig_meta)},{
        'month':a.month.values,
        'phase':a.phase.values,
        'lat':a.lat.values,
        'lon':a.lon.values
    })

    
    
    return spearman_index,spearman_sig