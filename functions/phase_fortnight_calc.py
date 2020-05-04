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
'''~~~~~~~~~~~~~~~~~~~~~~Splitting~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
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


    # Final File
    datafile_RMM_split = xr.concat(single_phase, pd.Index(phases, name = 'phase'))
    
    
    
    return  datafile_RMM_split




'''Counts the number of days in each of the MJO phases for each wet-season. This is useful for 
normalising all of the count trends'''
def rmm_count_phase(rmm_total):
    rmm_stor = []
    months = [10,11,12,1,2,3]


    for month in months:


        rmm = rmm_total.where(rmm_total.time.dt.month == month, drop = True)



        # Counting the number of days in each phase each year, then summing this up.
        rmm_act = rmm.where(rmm.amplitude >=1 , drop = True)
        rmm_act_count = rmm_act.groupby('phase').count(dim = 'time')
        rmm_act_count['phase'] = rmm_act_count.phase.values.astype(int).astype(str)
        rmm_act_count = rmm_act_count.rename({'amplitude':'number'})


        rmm_inact_count = rmm.where(rmm.amplitude < 1, drop = True).count(dim = 'time')\
        .rename({'phase':'number'}).drop('amplitude')
        rmm_inact_count['phase'] = ['inactive']

        rmm_count = xr.concat([rmm_act_count, rmm_inact_count ], dim = 'phase').drop(['RMM1','RMM2'])
        rmm_stor.append(rmm_count)
        
    rmm_total_count = xr.concat(rmm_stor, pd.Index(months, name = 'month'))
    return rmm_total_count


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ Anomalies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def count_and_anomly_month_subphase(data_split_total, data_total, rmm_total, climatology = 0):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    import subphase_calc_functions as subphase_calc
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    count_stor = []
    anom_stor = []
    month_groups = [[10,11,12],[1,2,3]]
    
    rmm_count_total = rmm_count_phase(rmm_total)
    
    for month in month_groups:
    

        # Subset of the data
        data_split = data_split_total.where(data_split_total.time.dt.month.isin(month), drop = True)
        data = data_total.where(data_total.time.dt.month.isin(month), drop = True)
        rmm_count = rmm_count_total.sel(month = month).sum(dim = 'month')
        
    
        #Count the number of events in each MJO phase   
        data_count = data_split.groupby('phase').count(dim = 'time')
    

        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_count_norm = data_count / data_split.groupby('phase').count(dim = 'time', skipna = False)
#         data_count_norm = (data_count.precip * 100/rmm_count.number).to_dataset(name = 'precip')

        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # Total number of rainfall events
        total_count = data.count(dim = 'time')
        
        # Total Number of Days
        total_days = rmm_count.sum(dim = 'phase')
        

        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_count_norm = (total_count.precip * 100/ total_days.number).to_dataset(name = 'precip')


        # Comparing to the climatology
        '''This is the second thing to be stored'''
        count_anomaly = data_count_norm/total_count_norm
    
        
        count_stor.append(data_count_norm)
        anom_stor.append(count_anomaly)
        
    data_count_norm_tot = xr.concat(count_stor, pd.Index(['early','late'], name = 'month'))   
    count_anomaly_tot = xr.concat(anom_stor, pd.Index(['early','late'], name = 'month'))  
#     return data_count_norm, total_count_norm
    
    if climatology:
        return data_count_norm_tot, count_anomaly_tot, total_count_norm
    
    else:
        return data_count_norm_tot, count_anomaly_tot


