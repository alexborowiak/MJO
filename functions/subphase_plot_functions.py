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
rb = plt.cm.RdBu
bm = plt.cm.Blues
best_blue = '#9bc2d5'
recherche_red = '#fbc4aa'
wondeful_white = '#f8f8f7'
import glob
import pdb
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec



'''Round up: This is for colorbars. Making a neat number for the vmax to go up to.'''
# This function rounds to the nearest 10 multiple (10,100,1000 ect.)
def ceil_10multiple(vmax):
    # This is the order of magnitude of the number
    mag = np.ceil(np.log10(vmax))
    
    # Dividing by 1 less than the magnitude gives as an order 1 number, taking the ceil to get 
    # upwards to a whole number, then making it back as the same order as before
    num = np.ceil(vmax / 10 ** (mag -1)) * 10 ** (mag - 1)
                  
    return num

'''This is for the ticks on the colorbar. Rounding them to the nearest 10 multiple so they loook neater'''
def round_10multiple(array):
    # This is the order of magnitude of the number
    mag = np.ceil(np.log10(array))
    
    # Dividing by 1 less than the magnitude gives as an order 1 number, taking the ceil to get 
    # upwards to a whole number, then making it back as the same order as before
    num = np.round(array / 10 ** (mag -2)) * 10 ** (mag - 2)
               
    # The above calculation screws up 0 and makes it nan
    num = np.nan_to_num(num,0)    
    
    return num


'''This is just plotting the raw values for all the different variables. 
The colorbar is automated by a 99th a 1st percentile and steps inbetween the two'''
def raw_plot(data,savefig = 0 , **kwargs):
    
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mpc
    
    fig = plt.figure(figsize = (10,12))
    gs = gridspec.GridSpec(5,4, hspace = 0.5, wspace = 0, height_ratios = [0.2,1,1,1,1])
    
    # These two percentiles are used compared to max, as accasionally there can 
    # be values that are significantly higher than the rest. 
    vmax = np.round(np.nanpercentile(data, 99))
    vmax = ceil_10multiple(vmax) # Function as specified above (round to nearest neat multiple of 10)
    vmin = np.round(np.nanpercentile(data, 1))
    
    '''Creating a custom colormap between vmin and vmax, that has 8 discrete units to it'''
    levels = np.linspace(vmin, vmax, 8)
    custom_cmap = plt.cm.get_cmap('Blues', len(levels))(np.arange(len(levels)))
    cmap_custom_RdBu = mpc.LinearSegmentedColormap.from_list("MyBlue", custom_cmap,len(levels))


    for plot_num,phase in enumerate(data.phase.values):
        # Main plot features
        phase_data = data.sel(phase = phase)
        ax = fig.add_subplot(gs[plot_num + 1,:], projection  = ccrs.PlateCarree())
        plot = phase_data.plot(ax = ax, add_colorbar = False, cmap = cmap_custom_RdBu, vmax = vmax, vmin = vmin)
        
        # Additional plot features
        ax.set_title(phase.capitalize(), size = 15)
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
    
    
    '''Creating the colorbar'''
    cax = plt.subplot(gs[0,1:3])
    cbar = plt.colorbar(plot, cax = cax, orientation = 'horizontal', ticks = levels, boundaries = levels)
    
    cbar.ax.set_xticklabels(round_10multiple(np.round(levels,0).astype(int)).astype(int), fontsize = 10);
    
    
    '''Looping through all of the key word arguements now'''
    kwarg_keys = kwargs.keys()
    
    if 'cbar_title' in kwarg_keys:
        cbar.ax.set_title(kwargs['cbar_title'], size = 15);
        
    if 'plot_title' in kwarg_keys:
        fig.suptitle(kwargs['plot_title'], fontsize = 20, y = 0.97)
    
    if savefig == 1:
        fig.savefig(kwargs['savedir'] + kwargs['plot_title'] + '.png')
  
    if kwargs['dont_plot'] == 1 :
        plt.close(fig)
    


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY Plots~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
    
def anomalie_cbar_1(vmax, l1 = []):
    import matplotlib as mpl
    
    
    if len(l1) == 0: # THis means I have set my own custom levels for the proble
        if  vmax == 3:
            l1 = np.array([1.25,1.5,1.75,2,2.5,3])
        elif vmax == 2:
            l1 = np.array([1.2,1.4,1.6,1.8,2])
        elif vmax == 1.5:
            l1 = np.array([1.1,1.2,1.3,1.4,1.5])
        elif vmax == 1.1:
            l1 = np.array([1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1])
    
    # The decimal values are the inverse of these values
    l2 = 1/l1 
    
    # Need to order them in the right direction
    l2 = np.flip(l2) 
    
    # Comining everything back together
    levels = np.concatenate((l2,np.array([1]),l1))
    
    # Creating a colorbar with the levels where you want them
    custom_RdBu = plt.cm.get_cmap("RdBu",len(levels))(np.arange(len(levels)))
    
    
    cmap_custom_RdBu = mpl.colors.LinearSegmentedColormap.from_list("RdWtBu", custom_RdBu,len(levels))
    
    return cmap_custom_RdBu, levels


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def anomalie_cbar_2(cax, levels,vmax,pdata, cbar_title):
    
    tick_locations = levels
    if len(tick_locations) > 10: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]


    tick_strings = np.round(tick_locations,2).astype(str)

    cbar = plt.colorbar(pdata, cax = cax,orientation = 'horizontal',
                        ticks = tick_locations)


    
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
def anomalies_plots(datafile,vmax = 3, title = '', cbar_title = '',cbar_num_steps = 10,  savefig = 0 , savedir = '',dont_plot = 0,
                 cmap = plt.cm.Blues, l1 = []):
    
    assert vmax == 2 or vmax == 3 or vmax == 1.5 or vmax == 1.1
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc


    fig = plt.figure(figsize = (6,12))
    gs = gridspec.GridSpec(5,1,hspace = 0.5, wspace = 0, height_ratios = [0.2, 1,1,1,1])
    fig.suptitle(title, fontsize = 20, y = 0.95)
    
    
    phases = datafile.phase.values

    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
        
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1)
    vmin = 1/vmax

    for i,phase in enumerate(phases):
       
        ax = fig.add_subplot(gs[i + 1], projection  = ccrs.PlateCarree())
    
        data = datafile.sel(phase = str(phase))
        
        data = data.fillna(1)
        data = data.where(data > vmin, vmin)
        data = data.where(data < vmax, vmax)
        data = data.where(data != 1, np.nan)
        
        
        pdata = data.plot(ax = ax, cmap = custom_RdBu, 
                             vmin = vmin , vmax = vmax,
                             add_colorbar = False,
                             norm = BoundaryNorm(levels, len(levels)-1)) 

        #Removing the spines of the plot. Cartopy requires different method
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
      
        ax.set_title(str(phase).capitalize(), size = 15)
        
        '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0])
    anomalie_cbar_2(axes,levels,vmax, pdata, cbar_title)
    
    
    if savefig:
        fig.savefig(savedir + title + '.png', dpi = 400)
        print('saving:' + title)
    if dont_plot == 1:
        plt.close(fig)

        
        
        
        
        
        '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
def enso_anomalies_plots(datafile,vmax = 3, title = '', cbar_title = '',cbar_num_steps = 10,  savefig = 0 , savedir = '',dont_plot = 0,
                 cmap = plt.cm.Blues, l1 = []):
    
    assert vmax == 2 or vmax == 3 or vmax == 1.5 or vmax == 1.1
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc


    fig = plt.figure(figsize = (10,12))
    gs = gridspec.GridSpec(5,2,hspace = 0.5, wspace = 0, height_ratios = [0.2, 1,1,1,1])
    fig.suptitle(title, fontsize = 20, y = 0.97)
    
    
    

    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
        
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1)
    vmin = 1/vmax
    
    
    phases = datafile.phase.values
    enso_phases = datafile.nino.values
    col_num = 0
    for nino in enso_phases:
        enso_datafile = datafile.sel(nino = nino)
        for i,phase in enumerate(phases):
            ax = fig.add_subplot(gs[i + 1,col_num], projection  = ccrs.PlateCarree())

            data = enso_datafile.sel(phase = str(phase))
            
            # If this is not here, this will distort the colorbar by putting an extend on, and 1 will
            # no longer be between the two sides of the colorbar
            data = data.fillna(1) #This gets rid of the haze around the plot
            data = data.where(data > vmin, vmin) # If the values are greater than what I want the max to be, replace them with max
            data = data.where(data < vmax, vmax)
            data = data.where(data != 1, np.nan)


            pdata = data.plot(ax = ax, cmap = custom_RdBu, 
                                 vmin = vmin , vmax = vmax,
                                 add_colorbar = False,
                                 norm = BoundaryNorm(levels, len(levels)-1)) 

            #Removing the spines of the plot. Cartopy requires different method
            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
            if col_num == 0:
                ax.annotate(str(phase).capitalize(), (-0.1,0.5), xycoords = 'axes fraction', va = 'center', size = 15,
                           rotation = 90)
           
            # Columns titles
            if i == 0:
                subtitle = str(nino).split()
                if nino == 'el nino':
                    ax.set_title(subtitle[0].capitalize() + ' '  \
                                 +  subtitle[1].capitalize()  + ' Like ' '(' + 'nino3.4 > 0' + ')', size =15)
                elif nino == 'la nina':
                    ax.set_title(subtitle[0].capitalize() + ' '  \
                                 +  subtitle[1].capitalize()  + ' Like ' '(' + 'nino3.4 < 0' + ')', size =15)
            else:
                ax.set_title('')
        col_num = 1
        
        
     
        '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0, :])
    anomalie_cbar_2(axes,levels,vmax, pdata, cbar_title)
    
    
    if savefig:
        
        print(title  + ' has been save')
        fig.savefig(savedir + title + '.png', dpi = 400)
    if dont_plot == 1:
        plt.close(fig)