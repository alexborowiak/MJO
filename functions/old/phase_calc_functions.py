import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import matplotlib.colors as colors
import datetime as dt
from matplotlib.colors import BoundaryNorm
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions all work together to calculate the anomalies in a given phase of the MJO'''



# This function readis in the RMM form the Bureau's website and turns it into an 
# xarrau file. The useragent will need to be changed depending on the computer. Currently it is set to the VDI

def load_rmm():
    
    import urllib
    import io

    url = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'
    headers={'User-Agent':user_agent}
    request=urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    csv = io.StringIO(data.decode('utf-8'))

    rmm_df = pd.read_csv(csv, sep=r'\s+', header=None, skiprows=2,
        usecols=[0,1,2,3,4,5,6,7], names=['year', 'month', 'day','RMM1','RMM2', 'phase', 'amplitude', 'origin'])
    index = pd.to_datetime(rmm_df.loc[:,['year','month','day']])
    rmm_df.index = index

    rmm_xr = rmm_df.loc[:,['RMM1','RMM2', 'phase','amplitude']].to_xarray().rename({'index':'time'})
    
    return rmm_xr





'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




def split_into_1to8(datafile, rmm_xr):
    
    
    
    '''~~~~~~~~~~~~~~~~~~ Inactive Phases'''
    rmm_inact_dates = rmm_xr.where(rmm_xr.amplitude < 1, drop = True).time.values
    datafile_inact = datafile.where(datafile.time.isin(rmm_inact_dates), drop = True)

    '''~~~~~~~~~~~~~~~~~~ Active Phases
    Summary: Looping through all the different RMM phases; getting the dates fro this phase; finding just the rainfall
    in this phase'''
    single_phase = [] # Storage for later concatinating in xarray
    rmm_act = rmm_xr.where(rmm_xr.amplitude >= 1, drop = True) # Only acitve when RMM > 1
    phases = np.arange(1,9) # 8 phases we are looping through
    for phase in phases:
        rmm_single_dates = rmm_act.where(rmm_act.phase == phase, drop = True).time.values # The dates of this phase
        datafile_single = datafile.where(datafile.time.isin(rmm_single_dates), drop = True) # The datafile data in this phase
        single_phase.append(datafile_single) # Appending

    phases = np.append(phases.astype('str'), 'inactive') # The ianctive also needs to be included
    single_phase.append(datafile_inact) 
    # phases = phases.astype(str)


    # Final File
    datafile_RMM_split = xr.concat(single_phase, pd.Index(phases, name = 'phase'))
    
    
    
    return  datafile_RMM_split






'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def  calculate_anomalies_1to8_mean(variable_split, variable):
    phase_mean = variable_split.groupby('phase').mean(dim = 'time')
    overall_mean = variable.mean(dim = 'time')
    anomalies = phase_mean/overall_mean
    
    return phase_mean, anomalies



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def  calculate_anomalies_1to8_percentile(variable_split, variable, q):
    phase_mean = variable_split.groupby('phase').reduce(np.nanpercentile, q = q, dim = 'time')
    overall_mean = variable.reduce(np.nanpercentile, q = q, dim = 'time')
    anomalies = phase_mean/overall_mean
    
    return phase_mean, anomalies

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def calculate_1to8_anomalies_for_variables(variable, anomaly_type = 'mean'):
    
    #Read in RMM
    rmm = load_rmm()
    
    # Split Via RMM
    variable_split = split_into_1to8(variable, rmm)
    
    # Calculate anomalies
    if anomaly_type == 'mean':
        variable_values, variable_anomalies = calculate_anomalies_1to8_mean(variable_split, variable)
    else:
        variable_values, variable_anomalies = calculate_anomalies_1to8_percentile(variable_split, variable, anomaly_type) 
    
    
    return  variable_values, variable_anomalies



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~TREND CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''Each year is a wet season'''

# This function moves the start of the wet season [10, 11, 12] to the next year. This means that
# this year is just the data for one wet season

def wet_season_year(data):
    
    # This is the start of the wet_season, wet want to move it to the next year so that the start of the
    # wet season and the end are both in the one year. This makes it easier for calculatins later on 
    
    data_start = data.where(data.time.dt.month.isin([10,11,12]), drop = True) # The later months of the year
    data_start['time'] = data_start.time + pd.to_timedelta('365day') # moving them forward a year
    
    data_end = data.where(data.time.dt.month.isin([1,2,3]), drop = True) # The end half
    
    print(data_start, data_end)
    total = data_end.combine_first(data_start) # All in one year now :)
    
    return total






sys.path.append('/home/563/ab2313/MJO/functions')
import mystats

# Calculates the trend for each individaul grid cell
def grid_trend(x,t):
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    if np.all(np.isnan(x)):
        return float('nan')
    
    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) & np.isfinite(t) #checking where the nans are for both
    grad = np.polyfit(x[idx],t[idx],1)[0]
    return grad


def calculate_trend(percentile):
    
  
    # The axis number that year is
    axis_num = percentile.get_axis_num('year')
    
    '''Applying trends along each grid cell'''
    percenilte_trend_meta = np.apply_along_axis(grid_trend,axis_num, percentile.values, 
                                                t = percentile.year.values)

    '''Turning into an xarray dataset'''
    trend  = xr.Dataset(
        {'trend':(('phase','lat','lon'), percenilte_trend_meta)},

        {
        'phase':percentile.phase.values, 
         'lat':percentile.lat,
        'lon':percentile.lon}
    )
    
    
    
    return trend






def convert_to_percent_per_decade(percentile, trend):
    
    mean_gridcell = percentile.mean(dim = 'year')
    
    
    return (trend * 10 / mean_gridcell) * 100





def calculate_pvals(percentile, trend):
    year_num = percentile.get_axis_num('year')
    
    trend_pval_meta = np.apply_along_axis(mystats.mann_kendall, year_num, percentile)


    pvals  = xr.Dataset(
        {'pvals':(('phase','lat','lon'), trend_pval_meta)},

        {
        'phase':percentile.phase.values, 
         'lat':percentile.lat,
        'lon':percentile.lon}
    )
    
    
    return pvals




def significant_trend_cacl(data, pvals):
    sig = data.where(np.logical_and(pvals.pvals >= 0 ,pvals.pvals <= 0.05  ))
    

    return sig




def return_alltrendinfo(data,q = 90):
    #Calculated the percentile
    # The percentiles of each year. Maintains MJO splits
    if type(q) == int:
        percentile = data.groupby('time.year').reduce(np.nanpercentile, dim = 'time', q = q)
    elif q == 'mean':
        percentile = data.groupby('time.year').mean(dim = 'time')
    elif q == 'all':
        pass
    
    percentile = percentile.to_array().squeeze()
    print('percentile/mean has been calculated')

#     Calculates the trend
    trend = calculate_trend(percentile)
    
    print('trend has been calculated')
    
    
    # Convertes to percent per decade
    trend_percent = convert_to_percent_per_decade(percentile, trend)
    print('trend has been converted to percent')
    
    # Calculates the significant values
    pvals =  calculate_pvals(percentile, trend)
    trend_sig = significant_trend_cacl(trend, pvals)
    trend_percent_sig = significant_trend_cacl(trend_percent, pvals)
    print('significant points habe been found')
    print('function is complete')

    return percentile, trend, trend_sig, trend_percent, trend_percent_sig
    
    
