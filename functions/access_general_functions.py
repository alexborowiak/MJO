import xarray as xr
import numpy as np
import pandas as pd
import dask.array
import datetime as dt
import sys
import warnings
warnings.filterwarnings('ignore')






'''------------------- Count and Month RMM Resample'''
# The count anomlies work better when normalised by the number of days in each phase, rather than dividing by data.fillna(0)


def rmm_count_resample(group):
    return group.groupby('time.year').count(dim = 'time')


def count_in_rmm_subphase_year_month_resample(rmm):
    
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
        number_per_year = rmm_single_phase.phase.groupby('time.month').apply(rmm_count_resample)
        # Appending
        single_phase.append(number_per_year)


    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
    number_per_year_inact = rmm_inact.phase.groupby('time.month').apply(rmm_count_resample)

    single_phase.append(number_per_year_inact)

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    phase_resamples = xr.concat(single_phase, pd.Index(titles, name = 'phase')).to_dataset(name = 'precip')

    
    return phase_resamples 




'''~~~~~~~~~~~~~ Ensemble Loop'''

# Loops through and calculates the anaomalies (don't think this will work for extremes)

def access_ensemble_calculation(access_split, access, rmm_access, function):
    
    
    # Calculating the percent for ACCESS-S


    ensembles = access_split.ensemble.values


    rain_stor = []


    for ensemble in ensembles:


        # Getting a single ensemble
        single_access_split = access_split.sel(ensemble = ensemble)
        single_access = access.sel(ensemble  = ensemble)

        single_rmm = rmm_access.sel(ensemble = ensemble)

        # Counting the number of days in each RMM phase
        rmm_count_ens  = count_in_rmm_subphase_year_month_resample(single_rmm)


        # Calculating the anomalies
        anom_ensemble = function(single_access_split, single_access, rmm_count_ens)


        rain_stor.append(anom_ensemble)


    # The anomalies for all the different ensebmel member
    access_rain_anom = xr.concat(rain_stor, pd.Index(ensembles, name = 'ensemble'))


    # The median of all the different ensembles
    access_rain_anom_med = access_rain_anom.median(dim = 'ensemble')
    
    
    return access_rain_anom, access_rain_anom_med