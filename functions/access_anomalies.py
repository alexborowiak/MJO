import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import pickle
import matplotlib.colors as colors
import datetime as dt
import pickle
from matplotlib.colors import BoundaryNorm
import glob
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



'''This function is for non extreme rainfall. It counts the number of rainfall events
then divides this by the total number of days (total number of days is calcualted by counting rmm)'''

def count_and_anomly_month_subphase(data_split_total, data_total, rmm_total):
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
    months = [10,11,12,1,2,3]
    
    
    #Count the number of events in each MJO phase
    rmm_count_total = subphase_calc.count_in_rmm_subphase_monthly(rmm_total)
    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        rmm_count = rmm_count_total.sel(month = month)
        
    
    
        #Count the number of events in each MJO phase   
        data_count = data_split.groupby('phase').count(dim = 'time')
    


        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_count_norm = (data_count.precip * 100/rmm_count.number).to_dataset(name = 'precip')

        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # Total number of rainfall events
        total_count = data.count(dim = 'time')
        
        # Total Number of Days
        total_days = rmm_count_total.sel(month = month).sum(dim = 'phase').number
        

        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_count_norm = total_count * 100/ total_days


        # Comparing to the climatology
        '''This is the second thing to be stored'''
        count_anomaly = data_count_norm/total_count_norm
    
        
        count_stor.append(data_count_norm)
        anom_stor.append(count_anomaly)
        
    data_count_norm_tot = xr.concat(count_stor, pd.Index(months, name = 'month'))   
    count_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    return data_count_norm_tot, count_anomaly_tot


'''This function is for non extreme rainfall and extreme rainfall. It sums the number of rainfall events
then divides this by the total number of days (total number of days is calcualted by suming rmm)'''

def sum_and_anomly_month_subphase(data_split_total, data_total, rmm_total):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    import subphase_calc_functions as subphase_calc
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    sum_stor = []
    anom_stor = []
    months = [10,11,12,1,2,3]
    
    
    #sum the number of events in each MJO phase ! THis is a count and not a sum, only called sum due to ctrl + f replace 
    rmm_sum_total = subphase_calc.count_in_rmm_subphase_monthly(rmm_total)
    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        rmm_sum = rmm_sum_total.sel(month = month)
        
    
    
        #sum the number of events in each MJO phase   
        data_sum = data_split.groupby('phase').sum(dim = 'time')
    


        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_sum_norm = (data_sum.precip /rmm_sum.number).to_dataset(name = 'precip')

        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # Total amount of rainfall
        total_sum = data.sum(dim = 'time')
        
        # Total Number of Days
        total_days = rmm_sum_total.sel(month = month).sum(dim = 'phase').number
        

        # The total amount of rainfall / the total number of days (average rainfall per day)
        total_sum_norm = total_sum / total_days


        # Comparing to the climatology
        '''This is the second thing to be stored'''
        sum_anomaly = data_sum_norm/total_sum_norm
    
        
        sum_stor.append(data_sum_norm)
        anom_stor.append(sum_anomaly)
        
    data_sum_norm_tot = xr.concat(sum_stor, pd.Index(months, name = 'month'))   
    sum_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    return data_sum_norm_tot, sum_anomaly_tot



def mean_and_anomly_month_subphase(data_split_total, data_total):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    import subphase_calc_functions as subphase_calc
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    mean_stor = []
    anom_stor = []
    months = [10,11,12,1,2,3]
    

    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        
    
    
        #mean the number of events in each MJO phase   
        data_mean = data_split.groupby('phase').mean(dim = 'time')
    

        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # Total amount of rainfall
        total_mean = data.mean(dim = 'time')
    

        # Comparing to the climatology
        '''This is the second thing to be stored'''
        mean_anomaly = data_mean/total_mean
    
        
        mean_stor.append(data_mean)
        anom_stor.append(mean_anomaly)
        
    data_mean_tot = xr.concat(mean_stor, pd.Index(months, name = 'month'))   
    mean_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    return data_mean_tot, mean_anomaly_tot