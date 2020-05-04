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
'''~~~~~~~~~~~~~~~~~~~~~~Pattern Correlation Timeseries~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
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
    





def week_pattern_correlations(obs,model):
    month_len = 2 # The number of month
    num_rows = 3 # The number of weeks
    num_cols = 4 # The nubmer of MJO phases


    # All the results are stored in a numpy array of dimension 2 x 3 x 4
    spearman_r = np.zeros((month_len, num_rows, num_cols))
    
    months = obs.month.values
    weeks = obs.week.values
    phases = obs.phase.values
    
    month_num = 0


    
    for month in months:
        row_num = 0
        for week in weeks:
            col_num = 0
            for phase in phases:
                
                obs_sub = obs.sel(phase = phase, month = month, week = week)
                model_sub = model.sel(phase = phase, month = month, week = week)
                

                r = spearmanr_function(obs_sub.precip, model_sub.precip)
                
                
                corr = r[0]
                
                spearman_r[month_num,row_num, col_num] = corr
                
                # Moving to the new row
                col_num += 1
             # Moving to the next column   
            row_num += 1
        month_num += 1
    

    pattern_correlation = xr.Dataset({'correlation':(['month', 'week','phase'], spearman_r)},
               {'month': months,
                'week': weeks,
                'phase':phases
               }
              )
    
    return pattern_correlation





'''This funtion loops through a list and does a pattern correlation with all of the items in the list with
AWAP. These results are the concatenated and returned'''
def list_pattern_correlation(awap_index, ensemble_list):
    
    single_stor = []
    
    # Looping through all of the ensemble
    for ensemble_single in ensemble_list:
        single_pattern_correlation = week_pattern_correlations(awap_index, ensemble_single)
        single_stor.append(single_pattern_correlation)

    
    ensemble_correlation = xr.concat(single_stor, pd.Index(np.arange(1,12), name = 'ensemble'))
    return ensemble_correlation
