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

from importlib import reload
import subphase_calc_functions as subphase_calc




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''







def count_anomalies(raw, split, q):
    
    
    # Thr events above the percentile
    above_q = find_events_above_q(raw, split, q)
    
    # The number of events above the percentile for each phase
    above_q_count = above_q.groupby('time.month').count(dim = 'time')
    
    # The total number of events for each phase
    split_count = split.groupby('time.month').count(dim = 'time')
    
    
    # The anomaly calculation
    per = (100 - q)/100
    anomaly = (above_q_count/ split_count)/per
    
    
    return anomaly






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






def extreme_count_and_anomly_month_subphase(data_split_total, data_total,rain_split_total, rain_total,
                                           climatology = 0):
    
    # data_split_total is the extreme split into different mjo phases
    # data_total is the total extreme rainfall events
    # data_all is all rainfall events (not just the extremes)
    
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
    climatology_stor = []
    months = [10,11,12,1,2,3]
    

    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        rain_split_month = rain_split_total.where(rain_split_total.time.dt.month == month, drop = True)
        rain = rain_total.where(rain_total.time.dt.month == month, drop = True)
        
        '''Geting the total number of rainfall events and then dividing this by the total number of rainfall events
        to get the percent as extremes (should be 10% for the 90th percentile for the climatology by definition)'''
    
        #Count the number of events in each MJO phase   
        data_count = data_split.groupby('phase').count(dim = 'time')
    
        # The total number of rainfall events 
        rain_count = rain_split_month.groupby('phase').count(dim = 'time')
        
        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_count_norm = (data_count.precip * 100/rain_count.precip).to_dataset(name = 'precip')


        # Total number of rainfall events
        total_count = data.count(dim = 'time')
        
        # Total Number of Days
        rain_count = rain.count(dim = 'time')

        # The total number of extreme rainfall events/ the total number of rainfall events (the average percent of rain days)
        # this is the climatology and should be 10% everywhere
        total_count_norm = total_count * 100/ rain_count
        climatology_stor.append(total_count_norm)

        # Comparing to the climatology
        '''This is the second thing to be stored'''
        count_anomaly = data_count_norm/total_count_norm
    
        
        count_stor.append(data_count_norm)
        anom_stor.append(count_anomaly)
        
    data_count_norm_tot = xr.concat(count_stor, pd.Index(months, name = 'month'))   
    count_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    if climatology:
        return data_count_norm_tot, count_anomaly_tot, total_count_norm
    else:
        return data_count_norm_tot, count_anomaly_tot
    
    
    
    
    
    
    
    
#############################################################################################################################






#############################################################################################################################
    

    
    
    
    
def extreme_sum_and_anomly_month_subphase(data_split_total, data_total,rain_split_total, rain_total,
                                           climatology = 0):
    
    # data_split_total is the extreme split into different mjo phases
    # data_total is the total extreme rainfall events
    # data_all is all rainfall events (not just the extremes)
    
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
    climatology_stor = []
    months = [10,11,12,1,2,3]
    

    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        rain_split_month = rain_split_total.where(rain_split_total.time.dt.month == month, drop = True)
        rain = rain_total.where(rain_total.time.dt.month == month, drop = True)
        
        '''Geting the total number of rainfall events and then dividing this by the total number of rainfall events
        to get the percent as extremes (should be 10% for the 90th percentile for the climatology by definition)'''
    
        #sum the number of events in each MJO phase   
        data_sum = data_split.groupby('phase').sum(dim = 'time')
    
        # The total number of rainfall events 
        rain_sum = rain_split_month.groupby('phase').sum(dim = 'time')
        
        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_sum_norm = (data_sum.precip * 100/rain_sum.precip).to_dataset(name = 'precip')


        # Total number of rainfall events
        total_sum = data.sum(dim = 'time')
        
        # Total Number of Days
        rain_sum = rain.sum(dim = 'time')

        # The total number of extreme rainfall events/ the total number of rainfall events (the average percent of rain days)
        # this is the climatology and should be 10% everywhere
        total_sum_norm = total_sum * 100/ rain_sum
        climatology_stor.append(total_sum_norm)

        # Comparing to the climatology
        '''This is the second thing to be stored'''
        sum_anomaly = data_sum_norm/total_sum_norm
    
        
        sum_stor.append(data_sum_norm)
        anom_stor.append(sum_anomaly)
        
    data_sum_norm_tot = xr.concat(sum_stor, pd.Index(months, name = 'month'))   
    sum_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    if climatology:
        return data_sum_norm_tot, sum_anomaly_tot, climatology_stor
    else:
        return data_sum_norm_tot, sum_anomaly_tot
    
    
    
    
    
    
    
    
#############################################################################################################################



####   Intensity


#############################################################################################################################
# Note: This function works for both extremes and all events
def extreme_intensity_and_anomly_month_subphase(data_split_total, data_total,
                                           climatology = 0):
    
    # data_split_total is the extreme split into different mjo phases
    # data_total is the total extreme rainfall events
    # data_all is all rainfall events (not just the extremes)
    
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    import subphase_calc_functions as subphase_calc
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    int_stor = []
    anom_stor = []
    climatology_stor = []
    months = [10,11,12,1,2,3]
    

    
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)

        
        '''Geting the total number of rainfall events and then dividing this by the total number of rainfall events
        to get the percent as extremes (should be 10% for the 90th percentile for the climatology by definition)'''
    
        #Count the number of events in each MJO phase   
        data_int = data_split.groupby('phase').mean(dim = 'time')
    

        # Total number of rainfall events
        total_int = data.mean(dim = 'time')
        

        # The total number of extreme rainfall events/ the total number of rainfall events (the average percent of rain days)
        # this is the climatology and should be 10% everywhere
        climatology_stor.append(total_int)

        # Comparing to the climatology
        '''This is the second thing to be stored'''
        int_anomaly = data_int/total_int
    
        
        int_stor.append(data_int)
        anom_stor.append(int_anomaly)
        
    data_int_tot = xr.concat(int_stor, pd.Index(months, name = 'month'))   
    int_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    if climatology:
        return data_int_tot, int_anomaly_tot, climatology_stor
    else:
        return data_int_tot, int_anomaly_tot
 
