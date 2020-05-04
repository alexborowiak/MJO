
import xarray as xr
import numpy as np
import pandas as pd
import dask
import miscellaneous as misc

def ordinal(n):
    if 4 <= n <= 20:
        suffix = 'th'
    elif n == 1 or (n % 10) == 1:
        suffix = 'st'
    elif n == 2 or (n % 10) == 2:
        suffix = 'nd'
    elif n == 3 or (n % 10) == 3:
        suffix = 'rd'
    elif n < 100:
        suffix = 'th'
    ord_num = str(n) + suffix
    return ord_num




def group_resample_count(group):
    return group.groupby('time.year').count()


def group_resample_count_2(group):
    rain_count = group.groupby('time.year').count()
    total_count = group.fillna(0).groupby('time.year').count()
    
    return rain_count/total_count

def group_resample_count_percent(group):
    rain = group.resample(time = 'y').groupby('time.month').count(dim = 'time')
    day = group.fillna(0).resample(time = 'y').groupby('time.month').count(dim = 'time')
    
    percent = rain/day
    
    return percent 
# Does not work



# Returns the number of ensembles that AWAP is greater than 
def greater_than(single_ensemble,awap_y):
    return awap_y.where(awap_y.precip > single_ensemble.precip)




def return_awap_greater_access(obs_index, mod_index):
    
    ens_stor = []

    # Looping through all of the ensembles
    for ensemble in mod_index.ensemble.values:\
        
        # Selecting a single ensemble
        mod_single = mod_index.sel(ensemble  = ensemble)

        # Getting rid of all the instances when access is greater than awap
        above = obs_index.where(obs_index.precip >= mod_single.precip)

        ens_stor.append(above)
    
    # This now only includes instances where AWAP is greater than ACCESS for each different ensemble member
    awap_greater = xr.concat(ens_stor, pd.Index(mod_index.ensemble.values, name = 'ensemble'))
    
    return awap_greater



# Creates a histogram of the data
def hist_hist(*args, **kwargs):
    return np.histogram(*args, **kwargs)[0]






# Takes in the split obs and split model and returns the hists for all the weekdays
# 'fucntion': group_resample_count
def weekly_ranks_bins(obs, mod, function):
    
    
    weeks = [np.arange(7,15), np.arange(15, 22), np.arange(22, 29)]

    week_stor_early = []
    week_stor_late = []
    
    for week in weeks:
        
        # Geeting just a single weeks worth of data from all the different months
        obs_sub = obs.where(obs.time.dt.day.isin(week), drop = True)
#         obs_index = function(obs_sub)
        
        
        obs_index = obs_sub.groupby('time.month').apply(function)


        mod_sub = mod.where(mod.time.dt.day.isin(week), drop = True)
#         mod_index = function(mod_sub)
    
        mod_index = mod_sub.groupby('time.month').apply(function)


        # Only reruns a value if AWAP is greater than the ensemble
        awap_greater = return_awap_greater_access(obs_index, mod_index)
#       awap_greate = mod_count.groupby('ensemble').apply(greater_than, awap_y = obs_count)

        # indexing the number of times AWAP is greater than an ensemble
        ranks =  awap_greater.count(dim = 'ensemble')
        # Converting this to a rank
        ranks['precip'] = ranks.precip + 1
    
        # Converting the data into numpy format
        data = ranks.precip.values
        
        
        # Concating along the time dimension, this changes shape from (23, 4, 49, 149)
        # to (69, 4, 49, 149) (time, phase, lat, lon). Now all the time values for three
        # sepeate months can be taken into the one histogram. Now, the histograms have
        # more values
        month_group_early = np.concatenate((data[0], data[1], data[3]), axis = 0)
        month_group_late = np.concatenate((data[3], data[4], data[5]), axis = 0)
        
        
        bins = np.arange(1,13)
        
        
        # Turning the ranks in a relative frequency by apllying the hist function along
        # th yearly dimension
        binned_meta_early = np.apply_along_axis(hist_hist, 0, month_group_early, 
                                          bins = bins, density = True)
        
        week_stor_early.append(binned_meta_early)
        
        
        binned_meta_late = np.apply_along_axis(hist_hist, 0, month_group_late, 
                                  bins = bins, density = True)
        
        week_stor_late.append(binned_meta_late)
        
    # Combining all the binned data into the one xarray file
    binned_data = xr.Dataset({'hist_':(('month','week','bin_','phase','lat','lon'), 
                                       [week_stor_early, week_stor_late])},
                    {
                     'month':['early','late'],
                     'week':['2','3','4'],
                     'bin_':bins[:-1],
                    'phase':ranks.phase.values,
                    'lat':ranks.lat.values,
                    'lon':ranks.lon.values})
        
    return binned_data



def calc_argmax(data):
    argmax = data.hist_.argmax(dim = 'bin_')
    argmax = argmax + 1
    argmax = argmax.to_dataset(name = 'argmax_')
    argmax = misc.apply_masks(argmax)
    return argmax