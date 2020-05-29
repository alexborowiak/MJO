
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

def custom_cmap(levels, add_white = 0, extender = 0, cmap_init = 'RdBu'):
    
    
    import matplotlib.colors as mpc
    
    cmap = plt.cm.RdBu

    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
     # This is the extra amount of discrete colors to make
        # List  of all the colors
    custom_cmap = plt.cm.get_cmap(cmap_init, len(levels) + extender)(np.arange(len(levels) + extender)) 
    if extender: # Chopping of some colors that are to dark to see the stippling
        custom_cmap = custom_cmap[extender:-extender] # CLipping the ends of either side
    
    if add_white:
        upper_mid = np.ceil(len(custom_cmap)/2)
        lower_mid = np.floor(len(custom_cmap)/2)
        white = [1,1,1,1]


        custom_cmap[int(upper_mid)] = white
        custom_cmap[int(lower_mid)] = white
        custom_cmap[int(lower_mid) - 1] = white
    
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels) + 1) 
    # Joingi the colormap back together
    return cmap
    
    

def corr_plot(data, sig_data = '', means = '',   vmax = 1, vmin = -1, 
              sig_size = 2.5, extender = 0, add_white = 0,
              savetitle = '', savedir = ''):

    import matplotlib.colors as mpc
    num_cols = 2
    num_rows = 3
    fig  = plt.figure(figsize = (10 * num_cols,3 * num_rows)) 
    gs = gridspec.GridSpec(3,2, height_ratios = [0.2,1,1], hspace = 0.25)
    phases = data.phase.values    
    
    vmax = 1
#     vmin = np.round(data.min().precip.values,1)
    vmin = -1
    levels = np.arange(vmin, vmax + 0.2, 0.2)


#     cmap  = plt.get_cmap('RdBu', len(levels))
    
    cmap = plt.cm.RdBu
#     levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
     # This is the extra amount of discrete colors to make
        # List  of all the colors
    custom_cmap = plt.cm.get_cmap('RdBu', len(levels) + extender)(np.arange(len(levels) + extender)) 
    if extender: # Chopping of some colors that are to dark to see the stippling
        custom_cmap = custom_cmap[extender:-extender] # CLipping the ends of either side
    
    if add_white:
        upper_mid = np.ceil(len(custom_cmap)/2)
        lower_mid = np.floor(len(custom_cmap)/2)
        white = [1,1,1,1]


        custom_cmap[int(upper_mid)] = white
        custom_cmap[int(lower_mid)] = white
        custom_cmap[int(lower_mid) - 1] = white
    
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) 
    # Joingi the colormap back together
    
    
    
    plt.suptitle(savetitle, fontsize = 15)
    
    for plot_num,phase in enumerate(phases):
        subdata = data.sel(phase = phase)
       
        
        ax = fig.add_subplot(gs[plot_num + 2], projection = ccrs.PlateCarree())
        pdata = subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels)#vmin = vmin, vmax = vmax)

        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        ax.set_title(phase.capitalize(), fontsize = 15)
        
        if type(means) != str:
            mean_phase = np.round(means.sel(phase = phase).values,2)
            bbox_props = dict(boxstyle = 'square',lw = 1, fc = 'white') # This is the properties for the box

            ax.annotate(mean_phase, xy = (0.1, 0.65), xycoords = 'axes fraction', fontsize = 12.5, bbox = bbox_props)
        
        if type(sig_data) != str:
            

            
            sub_sig = sig_data.sel(phase = phase)
            sub_sig = sub_sig.where(sub_sig.precip < 0.05)
            X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

            sig = sub_sig.where(~np.isfinite(sub_sig.precip), 1).precip
            size = np.nan_to_num(sig.values, 0)
            size[::2] = 0
            size[::5] = 0
            size = np.transpose(size)
            size[::2] = 0
            size[::5] = 0
            size = np.transpose(size)
            ax.scatter(X,Y, s = size * sig_size, color = 'k', alpha = 1)




    axes = plt.subplot(gs[0:2])
    cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
    cbar.ax.set_title('Correlation', fontsize = 15)

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)   
        
        
def corr_plot_sig_only(data, sig_data = '', means = '',   vmax = 1, vmin = -1, 
              sig_size = 2.5, extender = 0, add_white = 0,
              savetitle = '', savedir = ''):

    import matplotlib.colors as mpc
    num_cols = 2
    num_rows = 3
    fig  = plt.figure(figsize = (10 * num_cols,3 * num_rows)) 
    gs = gridspec.GridSpec(3,2, height_ratios = [0.2,1,1], hspace = 0.25)
    phases = data.phase.values    
    
    vmax = 1
#     vmin = np.round(data.min().precip.values,1)
    vmin = -1
    levels = np.arange(vmin, vmax + 0.2, 0.2)


#     cmap  = plt.get_cmap('RdBu', len(levels))
    
    cmap = plt.cm.RdBu
#     levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
     # This is the extra amount of discrete colors to make
        # List  of all the colors
    custom_cmap = plt.cm.get_cmap('RdBu', len(levels) + extender)(np.arange(len(levels) + extender)) 
    if extender: # Chopping of some colors that are to dark to see the stippling
        custom_cmap = custom_cmap[extender:-extender] # CLipping the ends of either side
    
    if add_white:
        upper_mid = np.ceil(len(custom_cmap)/2)
        lower_mid = np.floor(len(custom_cmap)/2)
        white = [1,1,1,1]


        custom_cmap[int(upper_mid)] = white
        custom_cmap[int(lower_mid)] = white
        custom_cmap[int(lower_mid) - 1] = white
    
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) 
    # Joingi the colormap back together
    
    
    
    plt.suptitle(savetitle, fontsize = 15)
    
    for plot_num,phase in enumerate(phases):
        subdata = data.sel(phase = phase)
        sub_sig = sig_data.sel(phase = phase)
        subdata = subdata.where(sub_sig.precip < 0.1)
        
        ax = fig.add_subplot(gs[plot_num + 2], projection = ccrs.PlateCarree())
        pdata = subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels)#vmin = vmin, vmax = vmax)

        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        ax.set_title(phase.capitalize(), fontsize = 15)
        
        if type(means) != str:
            mean_phase = np.round(means.sel(phase = phase).values,2)
            bbox_props = dict(boxstyle = 'square',lw = 1, fc = 'white') # This is the properties for the box

            ax.annotate(mean_phase, xy = (0.1, 0.65), xycoords = 'axes fraction', fontsize = 12.5, bbox = bbox_props)





    axes = plt.subplot(gs[0:2])
    cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
    cbar.ax.set_title('Correlation', fontsize = 15)

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)   
                
        
        
def correlation_plot_single_phase_multi_indinice(data_list_top, titles,phase = 'enhanced',
                                                 vmax = 0.8, step = 0.1, add_white = 0,
                                              savetitle = '', savedir = ''):
    
   
    levels = np.arange(vmin, vmax + step, step)
    
    data_list = []
    
    for item in data_list_top:
        data_list.append(item.sel(phase = phase))
    
    

    fig = plt.figure(figsize = (12,12))
    
    cols = len(data_list)
    months = data_list[0].month.values
    rows = len(months)
    
    gs = gridspec.GridSpec(rows + 2,cols, height_ratios = [0.2, 0.1] + [1] * rows, hspace = 0.25)
    

#     cmap  = plt.get_cmap('RdBu', len(levels))
    cmap = custom_cmap(levels, add_white = add_white)
    plt.suptitle(savetitle, fontsize = 15, y = 0.94)
    
    for row,month in enumerate([10,11,12,1,2,3]):
        
        for col in range(cols):
        
            subdata = data_list[col].sel(month = month)
            ax = fig.add_subplot(gs[row + 2, col], projection = ccrs.PlateCarree())
            pdata = subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels, extend = 'neither')#vmin = vmin, vmax = vmax)

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
    cbar.ax.set_title('Correlation', fontsize = 12.5)
    
    # Adding in extra space: not sure why this works, but it does???
    ax = plt.subplot(gs[0:cols])
#     ax.spined
    

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 150)     

        
        
def correlation_plot_single_phase_multi_indinice_sig_only(data_list_top, sig_list_top, titles,phase = 'enhanced',
                                                 vmax = 0.8, step = 0.1, add_white = 0, cmap_init = 'RdBu',
                                              savetitle = '', savedir = ''):
    
    
        
    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)

    
    data_list = []
    sig_list = []
    
    for item1, item2 in zip(data_list_top,sig_list_top):
        data_list.append(item1.sel(phase = phase))
        sig_list.append(item2.sel(phase = phase))
    
    

    fig = plt.figure(figsize = (20,20))
    
    cols = len(data_list)
    months = data_list[0].month.values
    rows = len(months)
    
    gs = gridspec.GridSpec(rows + 2,cols, height_ratios = [0.2, 0.1] + [1] * rows, hspace = 0.1, wspace = 0.1)
    

#     cmap  = plt.get_cmap('RdBu', len(levels))
    cmap = custom_cmap(levels, add_white = add_white, cmap_init = cmap_init)
    plt.suptitle(savetitle, fontsize = 15, y = 0.94)
    
    for row,month in enumerate([10,11,12,1,2,3]):
        
        for col in range(cols):
        
            subdata = data_list[col].sel(month = month)
            sub_sig = sig_list[col].sel(month = month)
            
            subdata = subdata.where(sub_sig.precip < 0.1)
            
            
            ax = fig.add_subplot(gs[row + 2, col], projection = ccrs.PlateCarree())
            subdata.precip.plot(ax = ax, add_colorbar = False, cmap = cmap, levels = levels, extend = 'neither')
            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
          
            if col == 0:
                ax.annotate(calendar.month_name[month], xy = (-0.12, 0), xycoords = 'axes fraction', 
                            fontsize = 25, rotation = 90)
                
            if row == 0:
                ax.set_title(titles[col], fontsize  = 25)
            else:
                ax.set_title('')



    axes = plt.subplot(gs[0:cols])
    import matplotlib as mpl
    cbar = mpl.colorbar.ColorbarBase( axes,cmap = cmap, orientation = 'horizontal',
                    ticks  = levels, boundaries = levels)
    cbar.ax.tick_params(labelsize = 15)
    cbar.ax.set_title('Correlation', fontsize = 15)
    
    
   
    # Adding in extra space: not sure why this works, but it does???
    ax = plt.subplot(gs[0:cols])
#     ax.spined
    

    if savedir != '':
        fig.savefig(savedir + savetitle + '.png', dpi = 600, bbox_inces = 'tight', pad = 0)     