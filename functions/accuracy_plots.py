
import xarray as xr
import numpy as np
import pandas as pd
import dask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import sys
import miscellaneous as misc
import rank_histograms as rh
from importlib import reload
import calendar


best_blue = '#9bc2d5'
rechere_red = '#fbc4aa'
kinda_white = '#f8f8f7' 



def accuracy_plot(data, vmax = 100, vmin = 0,savetitle = '', savedir = ''):


    fig = plt.figure(figsize = (12,6))
    gs = gridspec.GridSpec(3,2, height_ratios = [0.2,1,1], hspace = 0.4)

    levels = np.arange(vmin, vmax + 10, 10)
    phases = data.phase.values
    cmap  = plt.get_cmap('RdBu', len(levels))
    plt.suptitle(savetitle, fontsize = 15)
    
    for plot_num,phase in enumerate(phases):
        subdata = data.sel(phase = phase)
        ax = fig.add_subplot(gs[plot_num + 2], projection = ccrs.PlateCarree())
#         pdata = subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels)#vmin = vmin, vmax = vmax)
       
        X,Y = np.meshgrid(subdata.lon, subdata.lat)
        pdata = ax.contourf(X,Y,subdata.precip.values, levels = levels, cmap = cmap)
    

        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        ax.set_title(phase.capitalize(), fontsize = 15)
        x0 = subdata.lon.values[0]
        x1 = subdata.lon.values[-1]
        y0 = subdata.lat.values[0]
        y1 = subdata.lat.values[-1]
        
        ax.set_extent([x0,x1,y0,y1 + 1.35])



    axes = plt.subplot(gs[0:2])
    cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
    tick_labels = [str(i) + '%' for i in levels]
    cbar.ax.set_xticklabels(tick_labels);
    cbar.ax.set_title('Proportion Correct', fontsize = 15)

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)   
        
        
        
def accuracy_plot_single_phase_multi_indinice(data_list_top, titles,phase = 'enhanced',
                                              vmax = 100, vmin = 0,savetitle = '', savedir = ''):
    
    data_list = []
    
    for item in data_list_top:
        data_list.append(item.sel(phase = phase))
    
    

    fig = plt.figure(figsize = (12,12))
    
    cols = len(data_list)
    months = data_list[0].month.values
    rows = len(months)
    
    gs = gridspec.GridSpec(rows + 2,cols, height_ratios = [0.2, 0.1] + [1] * rows, hspace = 0.25)
    
    levels = np.arange(vmin, vmax + 10, 10)

    cmap  = plt.get_cmap('RdBu', len(levels))
    plt.suptitle(savetitle, fontsize = 15, y = 0.94)
    
    for row,month in enumerate([10,11,12,1,2,3]):
        
        for col in range(cols):
        
            subdata = data_list[col].sel(month = month)
            ax = fig.add_subplot(gs[row + 2, col], projection = ccrs.PlateCarree())
            pdata = subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels)#vmin = vmin, vmax = vmax)

            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
          
            if col == 0:
                ax.annotate(calendar.month_name[month], xy = (-0.2, 0.5), xycoords = 'axes fraction', 
                            fontsize = 12.5, rotation = 0)
                
            if row == 0:
                ax.set_title(titles[col], fontsize  =12.5)
            else:
                ax.set_title('')



    axes = plt.subplot(gs[0:cols])
    cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
    tick_labels = [str(i) + '%' for i in levels]
    cbar.ax.set_xticklabels(tick_labels);
    cbar.ax.set_title('Proportion Correct', fontsize = 12.5)
    
    # Adding in extra space: not sure why this works, but it does???
    ax = plt.subplot(gs[0:cols])
#     ax.spined
    

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)          