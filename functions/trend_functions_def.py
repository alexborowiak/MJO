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
sys.path.append('/home/563/ab2313/MJO/functions')
import subphase_calc_functions as subphase_calc
import mystats








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
    
    total = data_end.combine_first(data_start) # All in one year now :)
    
    return total

'''Counts the number of days in each of the MJO subphase for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_subphase_monthly(rmm):
    
    

    # We are looking at wet season, so will need to be made into wet season data set
    rmm = wet_season_year(rmm)
    
    enhanced = [5,6,7]
    suppressed = [1,2,3]
    transition = [4,8]

    phase_dict =  {'enhanced': enhanced, 'suppressed': suppressed, 'transition': transition}
    single_phase = []
    
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)

    for phase_name, phase_nums in phase_dict.items():

         # Just the data for this single rmm phase
        rmm_single_phase = rmm_act.where(rmm_act.phase.isin(phase_nums))#, drop = True)
         # Resmapling via year, to get the number of days in each phase
        number_per_year = rmm_single_phase.phase.resample(time = 'y').count(dim = 'time')
        # Appending
        single_phase.append(number_per_year.values)



    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
    number_per_year_inact = rmm_inact.phase.resample(time = 'y').count(dim = 'time')

    single_phase.append(number_per_year_inact.values)

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    datafile_RMM_split = xr.Dataset({'number':(('phase','year'), single_phase)},
                                   {'phase':titles,
                                    'year': number_per_year.time.dt.year.values
                                   })
    
    
#     datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return datafile_RMM_split




'''Counts the number of days in each of the MJO subphase for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_subphase(rmm):
    
    # We are looking at wet season, so will need to be made into wet season data set
    rmm = wet_season_year(rmm)
    
    enhanced = [5,6,7]
    suppressed = [1,2,3]
    transition = [4,8]

    phase_dict =  {'enhanced': enhanced, 'suppressed': suppressed, 'transition': transition}
    single_phase = []
    
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)

    for phase_name, phase_nums in phase_dict.items():

         # Just the data for this single rmm phase
        rmm_single_phase = rmm_act.where(rmm_act.phase.isin(phase_nums))#, drop = True)
         # Resmapling via year, to get the number of days in each phase
        number_per_year = rmm_single_phase.phase.resample(time = 'y').count(dim = 'time')
        # Appending
        single_phase.append(number_per_year.values)



    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
    number_per_year_inact = rmm_inact.phase.resample(time = 'y').count(dim = 'time')

    single_phase.append(number_per_year_inact.values)

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    datafile_RMM_split = xr.Dataset({'number':(('phase','year'), single_phase)},
                                   {'phase':titles,
                                    'year': number_per_year.time.dt.year.values
                                   })
    
    
#     datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return datafile_RMM_split



sys.path.append('/home/563/ab2313/MJO/functions')
import mystats



# Calculates the trend for each individaul grid cell
def grid_trend(x,t):
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    if np.all(np.isnan(x)):
        return float('nan')
    
    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) & np.isfinite(t) #checking where the nans are for both
    grad = np.polyfit(t[idx],x[idx],1)[0]
    return grad


#     elif np.sum(x) == 0:
#         return 0



def calculate_trend(data):
    
  
    # The axis number that year is
    axis_num = data.get_axis_num('year')
    
    '''Applying trends along each grid cell'''
    trend_meta = np.apply_along_axis(grid_trend,axis_num,data.values, 
                                                t = data.year.values)

    '''Turning into an xarray dataset'''
    trend  = xr.Dataset(
        {'trend':(('phase','lat','lon'), trend_meta)},

        {
        'phase':data.phase.values, 
         'lat':data.lat,
        'lon':data.lon}
    )
    
    
    
    return trend




def convert_to_percent_per_decade(year_vals, trend):
    
    mean_gridcell = year_vals.mean(dim = 'year')
    
    
    return (trend * 10 / mean_gridcell) * 100





def calculate_pvals(data, trend):
    year_num = data.get_axis_num('year')
    
    trend_pval_meta = np.apply_along_axis(mystats.mann_kendall, year_num, data)


    pvals  = xr.Dataset(
        {'pvals':(('phase','lat','lon'), trend_pval_meta)},

        {
        'phase':data.phase.values, 
         'lat':data.lat,
        'lon':data.lon}
    )
    
    
    return pvals




def significant_trend_cacl(data, pvals):
    sig = data.where(np.logical_and(pvals.pvals >= 0 ,pvals.pvals <= 0.05  ))
    

    return sig


def return_alltrendinfo_custom(data, normalise = 0):


        
    if normalise:
        rmm = subphase_calc.load_rmm()
        subphase_count = count_in_rmm_subphsae(rmm)
        data = (data/subphase_count.number)
        print('data has been normalised')
    
    # Calculates the trend using the apply along axis method
    trend = calculate_trend(data)
    
    print('trend has been calculated')
    
    # Normlaising by the number of days in each phase. This is only need for the count indices.
    # All the others this doesn't make sense

    
    # Convertes to percent per decade. The original datafile must be fed into this
    # in order to gt the mean: (year_vals, trend)
    trend_percent = convert_to_percent_per_decade(data, trend)
    print('trend has been converted to percent')
    
    # Calculates the significant values. The oringonal values are also needed for this 
    pvals =  calculate_pvals(data, trend) #(data, trend)
    trend_sig = significant_trend_cacl(trend, pvals)
    trend_percent_sig = significant_trend_cacl(trend_percent, pvals)
    print('significant points habe been found')
    print('function is complete')

    return trend, trend_sig, trend_percent, trend_percent_sig









def return_alltrendinfo_percentile(data,q = 90, normalise = 0):
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
    
    # Normlaising by the number of days in each phase. This is only need for the count indices.
    # All the others this doesn't make sense
    
    if nomralise:
        rmm = subphase_calc.load_rmm()
        subphase_count = count_in_rmm_subphsae(rmm)
        trend = (trend.trend/subphase_count.number).to_dataset(name = 'trend')
    
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