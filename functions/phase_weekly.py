
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




def rmm_count_week(rmm_sub):


    rmm_act = rmm_sub.where(rmm_sub.amplitude >= 1, drop = True)

    rmm_count_act = rmm_act.groupby('phase').count(dim = 'time').rename({'amplitude':'number'})
    rmm_count_act['phase'] =  rmm_count_act.phase.values.astype(int).astype(str)

    rmm_inact_count = rmm_sub.where(rmm_sub.amplitude < 1, drop = True).count(dim = 'time')\
    .rename({'phase':'number'}).drop(['amplitude'])
    rmm_inact_count['phase'] = ['inactive']
    rmm_inact_count

    rmm_count = xr.concat([rmm_count_act , rmm_inact_count ], dim = 'phase')
    
    return rmm_count


'''Ensemble Looper: This function goes through all the ensembles of access-s, split each ensemble, then appliead the function:
function. The split can be either phase or subphase'''
import phase_calc_functions as phase_calc
def ensemble_func_all(access, rmm_access, function, split = phase_calc.split_into_1to8):
    
    

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
        raw, anom = function(access_split, access, rmm_single)
        
        # Storing to be concatenated lates
        raw_stor.append(raw)
        anom_stor.append(anom)


    # Storing all the results in a xarray files.
    access_raw_total = xr.concat(raw_stor, dim = 'ensemble')
    access_anom_total = xr.concat(anom_stor, dim = 'ensemble')

    # Taking the mean across all ensembles

    access_raw_mean = access_raw_total.mean(dim = 'ensemble').compute()
    access_anom_mean = access_anom_total.mean(dim = 'ensemble').compute()
    
    
    return access_raw_mean, access_anom_mean, access_raw_total, access_anom_total


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
            rmm_count = rmm_count_week(rmm_sub)

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
