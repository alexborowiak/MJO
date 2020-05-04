
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




'''Counts the number of days in each of the MJO subphase for each wet-season. This is useful for 
normalising all of the count trends'''
'''Counts the number of days in each of the MJO subphase for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_subphase(rmm):
    
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
        number_per_phase = rmm_single_phase.phase.count(dim = 'time')
        # Appending
        single_phase.append(number_per_phase.values)



    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
    number_per_inact = rmm_inact.phase.count(dim = 'time')

    single_phase.append(number_per_inact.values)

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    datafile_RMM_split = xr.Dataset({'number':(('phase'), single_phase)},
                                   {'phase':titles
                                   })


    
    return datafile_RMM_split


'''Ensemble Looper: This function goes through all the ensembles of access-s, split each ensemble, then appliead the function:
function. The split can be either phase or subphase'''
import phase_calc_functions as phase_calc
import subphase_calc_functions as subphase_calc
def ensemble_func_all(access, rmm_access, function, split = subphase_calc.split_into_subphase):
    
    

    raw_stor = []
    anom_stor = []
    ensembles = access.ensemble.values

    for ensemble in ensembles:
        
        # Extracting a single ensemble
        access_single = access.sel(ensemble = ensemble)
        rmm_single = rmm_access.sel(ensemble = ensemble)

        # Splitting access into different phases
        access_split = split(access_single, rmm_single)
    
        # Running the function
        raw, anom = function(access_split, access_single, rmm_single)
        
        # Storing to be concatenated lates
        raw_stor.append(raw)
        anom_stor.append(anom)


    # Storing all the results in a xarray files.
    access_raw_total = xr.concat(raw_stor, dim = 'ensemble')
    access_anom_total = xr.concat(anom_stor, dim = 'ensemble')

    # Taking the mean across all ensembles

    access_raw_mean = access_raw_total.mean(dim = 'ensemble').compute()
    access_anom_mean = access_anom_total.mean(dim = 'ensemble').compute()
    
    
    return access_raw_mean, access_anom_mean, raw_stor, anom_stor


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ Anomalies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''






def count_anomalies_phase_week(data_split, data, rmm):
    
    
    month_groups = [[10,11,12],[1,2,3]]
    week_groups = [np.arange(7,15),np.arange(15,22), np.arange(22,29)]
    
    month_percent = []
    month_anom = []
    
    
    
    for month in month_groups:
        
        week_percent = []
        week_anom = []
        for week in week_groups:
    
    
            # Just for the months in consideration
            data_split_sub = data_split.where(np.logical_and(data_split.time.dt.month.isin(month),
                                                       data_split.time.dt.day.isin(week)
                                                      )
                                              , drop = True)
            
            rmm_sub = rmm.where(np.logical_and(rmm.time.dt.month.isin(month),
                                                       rmm.time.dt.day.isin(week)
                                                      )
                                              , drop = True)
            '''Phase Percent'''
            # Number of rainfall events
            data_split_count = data_split_sub.groupby('phase').count(dim = 'time')

            # Number of days
            rmm_count = count_in_rmm_subphase(rmm_sub)

            # Percent of days as raindays
            percent = (data_split_count.precip * 100/ rmm_count.number ).to_dataset(name = 'precip')


            '''Climatology'''

            data_count = data.where(np.logical_and(data.time.dt.month.isin(month),
                                                       data.time.dt.day.isin(week)
                                                      )
                                              , drop = True).count(dim = 'time')
            
            total_days = rmm_sub.count(dim = 'time').drop('phase').rename({'amplitude':'number'})
            climatology = (data_count.precip * 100/ total_days.number).to_dataset(name = 'precip')


            '''Anomaly'''
            anom = percent/climatology
            
            
            '''Storing'''
            week_percent.append(percent)
            week_anom.append(anom)
        
        # Month Loop
        per_week = xr.concat(week_percent, pd.Index(['2','3','4'], name = 'week'))
        month_percent.append(per_week)
        
        anom_week = xr.concat(week_anom, pd.Index(['2','3','4'], name = 'week'))
        month_anom.append(anom_week)
        
            
    per_month = xr.concat(month_percent, pd.Index(['early','late'], name = 'month'))
    anom_month = xr.concat(month_anom, pd.Index(['early','late'], name = 'month'))

    return per_month,anom_month




def mean_anomalies_phase_week(data_split, data, rmm):
    
    
    month_groups = [[10,11,12],[1,2,3]]
    week_groups = [np.arange(7,15),np.arange(15,22), np.arange(22,29)]
    
    month_int = []
    month_anom = []
    for month in month_groups:
        
        week_int = []
        week_anom = []
        for week in week_groups:
    
    
            # Just for the months in consideration
            data_split_sub = data_split.where(np.logical_and(data_split.time.dt.month.isin(month),
                                                       data_split.time.dt.day.isin(week)
                                                      )
                                              , drop = True)
            


            '''Phase int'''
            # Number of rainfall events
            data_split_mean = data_split_sub.groupby('phase').mean(dim = 'time')



            '''Climatology'''

            climatology = data.where(np.logical_and(data.time.dt.month.isin(month),
                                                       data.time.dt.day.isin(week)
                                                      )
                                              , drop = True).mean(dim = 'time')
            


            '''Anomaly'''
            anom = data_split_mean/climatology
            
            
            '''Storing'''
            week_int.append(data_split_mean)
            week_anom.append(anom)
        
        # Month Loop
        per_week = xr.concat(week_int, pd.Index(['2','3','4'], name = 'week'))
        month_int.append(per_week)
        
        anom_week = xr.concat(week_anom, pd.Index(['2','3','4'], name = 'week'))
        month_anom.append(anom_week)
        
            
    per_month = xr.concat(month_int, pd.Index(['early','late'], name = 'month'))
    anom_month = xr.concat(month_anom, pd.Index(['early','late'], name = 'month'))

    return per_month,anom_month
