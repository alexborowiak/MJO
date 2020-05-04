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
import calendar
import matplotlib.gridspec as gridspec
import sys
sys.path.append('/home/563/ab2313/MJO/functions')
import access_functions as af
from importlib import reload
import miscellaneous as misc



def stacked_patter_correlation_timeseries(data, savename = '', savedir = ''):
    
    fig,ax = plt.subplots(1,1, figsize = (10,5))
    colors = ['g','b','r','purple','orange']

    for phase_num, phase in enumerate(data.phase.values):
        for month_num,month in enumerate(data.month.values):

            x = np.tile(month_num,len(data.correlation.values))
            y = data.sel(month = month, phase = phase).correlation.values

            # phase_num/10 is added so the scatter points don't overlap
            if month == 3:
                ax.scatter(x + phase_num/10,y, label = phase.capitalize(), color = colors[phase_num])
            else:
                ax.scatter(x + phase_num/10,y, color = colors[phase_num])




    leg = plt.legend(loc = 'best', ncol = 4)
    leg.set_title('MJO Phase')


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    ax.set_ylabel('Correlation', size = 13)
    ax.set_xticklabels(np.append([''],[calendar.month_name[month] for month in data.month.values]), size = 10);

    fig.suptitle(savename, fontsize = 18);


    if savedir != '':
        fig.savefig(savedir + savename + '.png', dpi = 500, bbox_inches = 'tight', pad = 0)
        
        
        
def weekly_stacked_patter_correlation_timeseries(data, title = '', savefig = 0, savedir = '', savetitle = ''):
    
    fig = plt.figure(figsize = (10,10))
    colors = ['g','b','r','purple','orange']
    
    
    for month_num,month in enumerate(data.month.values):
        for phase_num, phase in enumerate(data.phase.values):
            for week_num,week in enumerate(data.week.values):
                
                ax = fig.add_subplot(2,1,month_num + 1)
                
                # We want the same point 11 times for each phase so that they are stacked. 
                x = np.tile(week_num,len(data.correlation.values)) 
                y = data.sel(month = month, phase = phase, week = week).correlation.values # The correlation values

                # phase_num/10 is added so the scatter points don't overlap
                # The logic is only for labelling purposes
                if week == '2':
                    ax.scatter(x + phase_num/10,y, label = phase.capitalize(), color = colors[phase_num])
                else:
                    ax.scatter(x + phase_num/10,y, color = colors[phase_num])

                ax.set_ylim(np.min(data.correlation.values) - 0.1, 1)
                ax.set_title(month.capitalize() + ' Wet Season', size = 14)




            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            
            ax.set_ylabel('Correlation', size = 14)
            if month_num == 0:
                ax.set_xticks([])
                ax.set_xticklabels([]);
                ax.spines['bottom'].set_visible(False)
  

            else:
                ax.set_xticks([0,1,2])
                ax.set_xticklabels(data.week.values, size = 14);
                ax.set_xlabel('Lead Time', size = 14)
            
    leg = ax.legend(bbox_to_anchor = (1.3, 1.4), ncol = 1, fontsize = 14)
#     leg.set_title('MJO Phase', fontsize = 14)
    fig.suptitle(title, fontsize = 25);


    if savefig:
        fig.savefig(savedir + save_title + '.png', dpi = 300)