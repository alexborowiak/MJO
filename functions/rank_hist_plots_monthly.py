
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


best_blue = '#9bc2d5'
rechere_red = '#fbc4aa'
kinda_white = '#f8f8f7' 

def rank_maps(single_phase,vmax = 0.6, step = 0.1, save_name = '', savedir  = ''):
    

    import matplotlib.colors as mpc

    single_phase = misc.apply_masks(single_phase)
    
    fig = plt.figure(figsize = (20,8))
    gs = gridspec.GridSpec(4,4, wspace = 0, hspace = 0.2, height_ratios = [0.2,1,1,1])

    fig.suptitle(save_name, fontsize = 25)
    
    ranks = single_phase.rank_.values
    
    for rank in ranks:
        sb = single_phase.sel(rank_ = rank)

        levels = np.arange(0,vmax + step,step)
    #     custom_cmap = plt.cm.get_cmap('Reds', len(levels))(np.arange(len(levels)))
    #     cmap_custom = mpc.LinearSegmentedColormap.from_list("MyBlue", custom_cmap,len(levels))
        cmap_custom = plt.get_cmap('Reds',8)

    #     sb = sb.where(sb < vmax, 0)
        sb = sb.where(sb != 0, np.nan)

        ax = fig.add_subplot(gs[rank + 3], projection = ccrs.PlateCarree())
        pdata = sb.frequency.plot(ax = ax, add_colorbar = False, vmax = vmax, cmap = cmap_custom)

        ax.set_title('')
        ax.coastlines(resolution = '50m')
        ax.annotate(rh.ordinal(rank), xy = (0.5, -0.1), xycoords = 'axes fraction', ha = 'center', size = 15)
        ax.outline_patch.set_visible(False)

        cax = plt.subplot(gs[:4])
        cbar = plt.colorbar(pdata, cax = cax, orientation = 'horizontal', boundaries = levels)
        cbar.ax.set_title('Relative Frequency', size = 15)

        
    if savedir != '':
        fig.savefig(savedir + save_name + '.png')
      
    
    
    
    
    
    
def argmax_map_plot(argmax, savetitle = '', savedir = ''):


    fig = plt.figure(figsize = (12,6))
    gs = gridspec.GridSpec(3,2, height_ratios = [0.2,1,1], hspace = 0.25)


    phases = argmax.phase.values
    cmap  = plt.get_cmap('RdBu', 13)
    plt.suptitle(savetitle, fontsize = 15)
    
    for plot_num,phase in enumerate(phases):
        subdata = argmax.sel(phase = phase)
        ax = fig.add_subplot(gs[plot_num + 2], projection = ccrs.PlateCarree())
        pdata = subdata.argmax_.plot(ax = ax, add_colorbar = False, cmap = 'RdBu',vmin = 1, vmax = 12)

        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        ax.set_title(phase.capitalize())



    axes = plt.subplot(gs[0:2])
    cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',ticks  = np.arange(1,14) + 0.5, boundaries = np.arange(1,14))
    cbar.ax.set_xticks(np.arange(1,14) + 0.5);
    tick_labels = [rh.ordinal(i) for i in np.arange(1,14)]
    cbar.ax.set_xticklabels(tick_labels);
    cbar.ax.set_title('Rank', fontsize = 15)

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)        
        
        
      
    
    
def generate_phase_hist(data_total,  savedir = '', save_name = ''):
    
    fig = plt.figure(figsize = (12,8))
    
    plt.suptitle(save_name, fontsize = 15)
    
    phases = data_total.phase.values
    
    for plot_num,phase in enumerate(phases):
        
        ax = fig.add_subplot(2,2,plot_num + 1)

        data = data_total.sel(phase = phase)
        x = data.rank_.values
        y = data.frequency.values
        plt.bar(x,y, color = best_blue)


        ax.set_title(phase.capitalize())
        ax.set_xticks(data.rank_.values)
        ax.set_xticklabels((np.array(data.rank_.values)).astype(str))



        
        if plot_num == 0 or plot_num == 2:
        
            ax.set_ylabel('Relative Frequency')
            
        if plot_num == 2 or plot_num == 3:
            ax.set_xlabel('Rank')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if savedir != '':
        fig.savefig(savedir + save_name + '.png', dpi = 150)