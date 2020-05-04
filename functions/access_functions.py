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
import glob
warnings.filterwarnings('ignore')
import subphase_calc_functions as subphase_calc
import matplotlib.gridspec as gridspec



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Phase Splitting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def access_rmm_split(access,rmm_access):

    
    
    ensemble_stor = []


    for ensemble in rmm_access.ensemble.values:
        rmm_single = rmm_access.sel(ensemble = ensemble)
        rmm_act = rmm_single.where(rmm_single.amplitude >=1 ,drop = True)

        enhanced = [5,6,7]
        suppressed = [1,2,3]
        transition = [4,8]
        phase_dict =  {'enhanced': enhanced, 'suppressed': suppressed, 'transition': transition}
        phase_stor = []


        for phase in phase_dict.values():

            rmm_phase = rmm_act.where(rmm_act.phase.isin(phase), drop = True)
            rmm_phase_dates = rmm_phase.time

            access_single = access.sel(ensemble = ensemble)
            access_phase = access_single.where(access_single.time.isin(rmm_phase_dates), drop = True)


            phase_stor.append(access_phase)

        rmm_inact = rmm_single.where(rmm_single.amplitude < 1 ,drop = True)
        rmm_inact_dates = rmm_inact.time
        access_inact = access_single.where(access_single.time.isin(rmm_inact_dates), drop = True)
        phase_stor.append(access_inact)

        titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])


        ensemble_xr = xr.concat(phase_stor, pd.Index(titles, name = 'phase')).drop('ensemble')

        ensemble_stor.append(ensemble_xr)

        print(str(ensemble) + ' ', end = '')

    access_split = xr.concat(ensemble_stor, pd.Index(rmm_access.ensemble.values, name = 'ensemble'))


    return access_split
    

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Statistics~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
def spearmanr_function(p1, p2):
    from scipy.stats import spearmanr
    
    # Flattening the values
    p1 = p1.values.flatten()
    p2 = p2.values.flatten()

    # Resolving issues with nans:
    # Removing nanas
    idx = np.isfinite(p1) & np.isfinite(p2)
    p1 = p1[idx]
    p2 = p2[idx]
    
    
    # Replacing all the zeroes in the ocean with nan
    p1 = np.array(p1)
    p1 = np.where(p1 == 0, np.nan,p1)
    p2 = np.array(p2)
    p2 = np.where(p2 == 0, np.nan,p2)
        

    return spearmanr(p1,p2, nan_policy = 'omit')
    
    
    
    
    
    
    
    
    

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Finding Extremes~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
    
    
    





'''Returns the events that are above a specific percentile'''

def find_events_above_q(raw, split, q):
    
    # These are the q-values for each month in each location. The threshold
    print('calculating pvals:', end = '')
    pval = raw.groupby('time.month').reduce(np.nanpercentile, q = q, dim = 'time')
    print('complete')
    
    storage = []
   
    for month in [10,11,12,1,2,3]:

        # qth percentile for each grid cell of that month
        month_pval = pval.sel(month = month)
        print(str(month) + '   ', end = '')
        # These are the events that are just in the month in question
        data_month = split.where(split.time.dt.month == month, drop = True)

        # These are the events above the percentile that we are looking at
        data_pval = data_month.where(data_month >= month_pval, drop = True)
        
    
        storage.append(data_pval)
    print('')
    print('concatinating : ', end = '')
    # Merging everything back together
    above_q = xr.concat(storage, dim = 'time').to_dataset(name  = 'precip').squeeze()
    print('complete')
    return above_q.drop('month')




def unsplit_find_events_above_q(data, q):
    
    # Findint the qth percentile of each month for the dataset
    pval = data.groupby('time.month').reduce(np.nanpercentile, q = q, dim = 'time')
    
    
    storage = []
    # Looping through each month and getting the events above the qth percentile
    for month in [10,11,12,1,2,3]:
        
        # The raw data that is just in the month in question
        data_month = data.where(data.time.dt.month == month, drop = True)
        
        # The qth percentile for the month in question
        pval_month = pval.sel(month =  month)
        
        # The data for this month that is above the percentile
        data_pval = data_month.where(data_month.precip > pval_month.precip, drop = True)
        
        # Storing to be put back into xarray form below
        storage.append(data_pval)
    
    # Back into one continuos xararay file
    above_q = xr.concat(storage, dim = 'time').drop('month')
    
    return above_q


def return_access_extremes(access, q):
    # Looping though all the ensembles and storign output
    total_stor = []
    for ensemble in access.ensemble.values:

        ens_data = access.sel(ensemble = ensemble)
        ens_stor = []


        # This is the threshhold to be considered extreme
        ens_pval = ens_data.groupby('time.month').reduce(np.nanpercentile, q = q, dim = 'time')

        # Rach month is different so need to loop through all the different months.
        # run_num is for storing in numpy array
        for  month in ens_pval.month.values:
            # The rainfall data for just this month
            ens_month_data = ens_data.where(ens_data.time.dt.month == month, drop = True)

            # Threshhold for this month
            ens_pval_month = ens_pval.sel(month = month)

            # The extreme events for this month
            ens_month_extreme = ens_month_data.where(ens_month_data.precip > ens_pval_month.precip, drop = True)

            ens_stor.append(ens_month_extreme)

        # These are all the extreme evens for 
        ens_total = xr.concat(ens_stor, dim = 'time').drop('month')

        total_stor.append(ens_total)

    extreme = xr.concat(total_stor, dim = 'ensemble') 
    
    return extreme


















    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Pattern Correlation Timeseries~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def month_pattern_correlations(obs,model):
    num_cols = len(obs.month.values)
    num_rows = len(obs.phase.values)
    spearman_r  = np.zeros((num_rows,num_cols))
    pval = np.zeros((num_rows,num_cols))

    for row,phase in enumerate(obs.phase.values):
        for col,month in enumerate([10,11,12,1,2,3]):
            
       
            obs_sub = obs.sel(phase = phase, month = month)
            model_sub = model.sel(phase = phase, month = month)

            r = spearmanr_function(obs_sub.precip, model_sub.precip)
            corr = r[0]
            p = r[1]

            spearman_r[row, col] = corr
            pval[row, col] = p





    pattern_correlation = xr.Dataset({'correlation':(['phase','month'], spearman_r),
               'pval':(['phase','month'], pval)},
               {'phase':obs.phase.values,
                'month':[10,11,12,1,2,3],
               }
              )
    
    return pattern_correlation





'''This funtion loops through a list and does a pattern correlation with all of the items in the list with
AWAP. These results are the concatenated and returned'''
def list_pattern_correlation(awap_index, ensemble_list):
    
    single_stor = []
    
    # Looping through all of the ensemble
    for ensemble_single in ensemble_list:
        single_pattern_correlation = month_pattern_correlations(awap_index, ensemble_single)
        single_stor.append(single_pattern_correlation)

    
    ensemble_correlation = xr.concat(single_stor, pd.Index(np.arange(1,12), name = 'ensemble'))
    return ensemble_correlation






