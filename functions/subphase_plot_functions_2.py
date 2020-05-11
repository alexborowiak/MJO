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



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def anomalie_cbar_1(vmax, l1 = [], add_white = 0, extender = 0):
    import matplotlib as mpl
    
    
    if len(l1) == 0: # THis means I have set my own custom levels for the proble
        if vmax == 5:
            l1 = np.array([1.5,2,2.5,3,3.5,4, 4.5,5])
        if vmax == 4:
            l1 = np.array([1.25,1.5,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4])
        elif  vmax == 3:
            l1 = np.array([1.25,1.5,1.75,2,2.5,3])
        elif vmax == 3.2:
            l1 = np.array([1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2])
        elif vmax == 2:
            l1 = np.array([1.2,1.4,1.6,1.8,2])
        elif vmax == 2.1:
            l1 = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2, 2.1])
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
    # extener will clip off the darker colours
    custom_RdBu = plt.cm.get_cmap('RdBu', len(levels) + extender)(np.arange(len(levels) + extender)) 
#     custom_RdBu = plt.cm.get_cmap("RdBu",len(levels))(np.arange(len(levels) ))

    if extender: # Chopping of some colors that are to dark to see the stippling
        custom_RdBu = custom_RdBu[extender:-extender] # CLipping the ends of either side
    
#     '''Adding White into the middle'''
#     # Find the middle of the color bar
    if add_white:
        upper_mid = np.ceil(len(custom_RdBu)/2)
        lower_mid = np.floor(len(custom_RdBu)/2)
        white = [1,1,1,1]

        custom_RdBu[int(upper_mid)] = white
        custom_RdBu[int(lower_mid)] = white
        custom_RdBu[int(lower_mid) - 1] = white
    
    
    cmap_custom_RdBu = mpl.colors.LinearSegmentedColormap.from_list("RdWtBu", custom_RdBu,len(levels))
    
    return cmap_custom_RdBu, levels


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def anomalie_cbar_2(axes, levels,vmax,pdata, cbar_title):
    
    tick_locations = levels
#     if len(tick_locations) > 10: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    tick_strings = np.round(tick_locations,2).astype(str)

    cbar = plt.colorbar(pdata, cax = axes,orientation = 'horizontal', extend = 'neither')
    
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    
    
 
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    
    
# Colorbar for just normal indice values
def normal_colorbar_horizontal(axes, total_plot,plot_max, num_cbar_steps, cbar1_title):
    
    
    # x,y, width, height
#     ticks = np.arange(0, plot_max + step, step)
    ticks = np.linspace(0, plot_max, num_cbar_steps + 1)
    cbar = plt.colorbar(total_plot, cax = axes, orientation = 'horizontal',drawedges = True, ticks = ticks)

    cbar.ax.set_title(cbar1_title, fontsize = 15)  
    
    

    
  
    
# This is based upon the origonal subphase plot    
def anomalies_plots_stippled(datafile,vmax = 3, title = '', cbar_title = '',cbar_num_steps = 10, savedir = '',
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
    
    
    datafile = misc.remove_outside_point(datafile, vmax - 0.01, vmin + 0.01)

    for i,phase in enumerate(phases):
       
        ax = fig.add_subplot(gs[i + 1], projection  = ccrs.PlateCarree())
    
        data = datafile.sel(phase = str(phase))
       
        
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
    
    
    if savedir != '':
        fig.savefig(savedir + title + '.png', dpi = 400)
        print('saving:' + title)

    
    
    

    
    
    
# This is a version for plotting all points or just the significant points only. No stipling
'''Phase: This is the comparison plot for PHASES'''
def anomalies_plots_plain(awap, month = False,
                    l1 = [], vmax = 3,add_white = 0, figsize = (8,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc
    

    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
    
    
    # Plot set up
    phases = awap.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,1, height_ratios=[0.2,0.1] + len(phases) * [1])
    gs.update(hspace=0.4)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = misc.remove_outside_point(awap, vmax, vmin)
    awap = misc.apply_masks(awap)
    
    # Initalising the row and column placements

    row_num = 0
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:

        awap_phase = awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2], projection  = ccrs.PlateCarree())
        total_plot = awap_phase.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')

        ax.set_title(phase.capitalize(), size = subsize, pad = subpad)
       
  
    
        row_num += 1
 
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
#     axer = fig.add_subplot(gs[1,:], zorder = -100)
#     axer.spines['top'].set_visible(False)
#     axer.spines['bottom'].set_visible(False)    
#     axer.spines['left'].set_visible(False)
#     axer.spines['right'].set_visible(False)
#     axer.set_yticklabels('')
#     axer.set_yticks([])
#     axer.set_xticklabels('')
#     axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')    
    
    
    
    
# This is based upon the origonal subphase plot    
def anomalies_plots_stippled(datafile,vmax = 3, title = '', cbar_title = '',cbar_num_steps = 10, savedir = '',
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
    
    
    datafile = misc.remove_outside_point(datafile, vmax, vmin)

    for i,phase in enumerate(phases):
       
        ax = fig.add_subplot(gs[i + 1], projection  = ccrs.PlateCarree())
    
        data = datafile.sel(phase = str(phase))
       
        
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
    
    
    if savedir != '':
        fig.savefig(savedir + title + '.png', dpi = 400)
        print('saving:' + title)

    
    
    

    # Removign the points outside of a range that is too big
def max_min_filter(data, vmax, vmin):
    data = data.where(data < vmax, vmax - 0.01)
    data = data.where(data > vmin, vmin + 0.01)
    
    from miscellaneous import apply_masks
    data = apply_masks(data)
    
    return data
    
    
# This is a stripped back version of the access_plot version
'''Phase: This is the comparison plot for PHASES'''
'''!!!!!!!!!!!! Function of Choice !!!!!!!!!!!!!!!!!'''
def anomalies_plots_stippled_2(awap,sig_awap = '', month = False,
                    l1 = [], vmax = 3, add_white = 0, extender = 0, sig_size = 2.5,# Cbar proporties
                     figsize = (8,12), cbar_title = '', save_name  = '', # Formatting
                               savedir = '' ):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc
    

    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
    
    
    # Plot set up
    phases = awap.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,1, height_ratios=[0.2,0.1] + len(phases) * [1])
    gs.update(hspace=0.4)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white, extender)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = max_min_filter(awap, vmax,vmin)
    
    # Initalising the row and column placements

    row_num = 0
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:

        awap_phase = awap.sel(phase = phase)
        sub_sig = sig_awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2], projection  = ccrs.PlateCarree())
        X,Y = np.meshgrid(awap_phase.lon.values, awap_phase.lat.values)
        total_plot = ax.contourf(X,Y, awap_phase.values, levels = levels, cmap = custom_RdBu,norm = BoundaryNorm(levels, len(levels) - 1))
#         total_plot = awap_phase.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
#                                 norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')

        ax.set_title(phase.capitalize(), size = subsize, pad = subpad)
       
    
        #Significant Points
        
        # plt.pcolor() needs an array for X and Y positions, generate with numpy.meshgrid
        X, Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

        sig = sub_sig.where(~np.isfinite(sub_sig), 1)
        size = np.nan_to_num(sig.values, 0)
        size[::2] = 0
        size[::5] = 0
        size = np.transpose(size)
        size[::2] = 0
        size[::5] = 0
        size = np.transpose(size)
        plt.scatter(X,Y, s = size * sig_size, color = 'k', alpha = 0.8)
    
        row_num += 1
 
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')

    
  

# This is a stripped back version of the access_plot version
'''Phase: This is the comparison plot for PHASES'''
'''!!!!!!!!!!!! Function of Choice !!!!!!!!!!!!!!!!!'''
def anomalies_plots_era5(awap, month = False,
                    l1 = [], vmax = 3, add_white = 0, extender = 0, sig_size = 2.5,# Cbar proporties
                     figsize = (8,12), cbar_title = '', save_name  = '', # Formatting
                               savedir = '' ):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc
    

    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
    
    
    # Plot set up
    phases = awap.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,1, height_ratios=[0.2,0.1] + len(phases) * [1])
    gs.update(hspace=0.4)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white, extender)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = max_min_filter(awap, vmax,vmin)
    
    # Initalising the row and column placements

    row_num = 0
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:

        awap_phase = awap.sel(phase = phase)

        
        '''~~~~~~~~~~~~~~~  AWAP'''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2], projection  = ccrs.PlateCarree())
#         X,Y = np.meshgrid(awap_phase.lon.values, awap_phase.lat.values)
#         total_plot = ax.contourf(X,Y, awap_phase.values, levels = levels, cmap = custom_RdBu,norm = BoundaryNorm(levels, len(levels) - 1))
        total_plot = awap_phase.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')

        ax.set_title(phase.capitalize(), size = subsize, pad = subpad)
       
   
    
        row_num += 1
 
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')

    
  

    
# This is a stripped back version of the access_plot version
'''Phase: This is the comparison plot for PHASES'''
def enso_anomalies_plots_stippled(awap,sig_awap = '', month = False,
                    l1 = [], vmax = 3,add_white = 0, figsize = (10,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc
    misc = reload(misc)

    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
    
    
    # Plot set up
    phases = awap.phase.values
    enso_phases = awap.nino.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,2, height_ratios=[0.2,0.1] + len(phases) * [1])
    gs.update(hspace=0.1, wspace = 0)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = misc.remove_outside_point(awap, vmax, vmin)
    awap = misc.apply_masks(awap)
    
    # Initalising the row and column placements

    row_num = 0
    first_row_plots = row_num

    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:
        column = 0
        for nino in enso_phases:

            awap_phase = awap.sel(phase = phase, nino = nino)

            '''~~~~~~~~~~~~~~~  AWAP'''
            # The + 2 is due to the extra plot to make space between the first plot and the colorbar
            ax = fig.add_subplot(gs[row_num + 2, column], projection  = ccrs.PlateCarree())
            total_plot = awap_phase.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                    norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)

            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
            
            
            if row_num == 0 and column == 0:
                ax.set_title('El Nino Like (Nino3.4 > 0)', size = subsize, pad = subpad)
            elif row_num == 0 and column == 1:
                ax.set_title('La Nina Like (Nino3.4 < 0)', size = subsize, pad = subpad)
            else:
                ax.set_title('')
                
            if column == 0:
                ax.annotate(str(phase).capitalize(), (-0.1,0.5), xycoords = 'axes fraction', va = 'center', size = 15,
                           rotation = 90)

            
            if type(sig_awap) != str:
            #Significant Points

                sub_sig = sig_awap.sel(phase = phase, nino = nino)

                # plt.pcolor() needs an array for X and Y positions, generate with numpy.meshgrid
                X, Y = np.meshgrid(sub_sig.lon, sub_sig.lat)


                sig = sub_sig.where(~np.isfinite(sub_sig), 1)
                size = np.nan_to_num(sig.values, 0)
                plt.scatter(X,Y, s = size/8, color = 'k', alpha = 0.8)

            column += 1

        row_num += 1
 

    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0:2])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')
    
    
    
    
    
    
# All months and all extreme indices for a single month of the MJO.
def anomalies_plots_sigle_phase_all_month(count, int_, prop, phase = 'enhanced',
                    l1 = [], vmax = 3,add_white = 0, figsize = (8,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    
    count = count.sel(phase = phase)
    int_ = int_.sel(phase = phase)
    prop = prop.sel(phase = phase)
    
    # Plot set up
    months  = [10,11,12,1,2,3]
    num_rows = len(months)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,3, height_ratios=[0.2,0.1] + len(months) * [1])
    gs.update(hspace=0.1, wspace = 0)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    count = misc.remove_outside_point(count, vmax, vmin)
    count = misc.apply_masks(count)
    
    int_ = misc.remove_outside_point(int_, vmax, vmin)
    int_  = misc.apply_masks(int_ )
    prop = misc.remove_outside_point(prop, vmax, vmin)
    prop = misc.apply_masks(prop)
    
    # Initalising the row and column placements

    row_num = 0

    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for month in months:

        cm = count.sel(month = month)
        im = int_.sel(month = month)
        pm = prop.sel(month = month)
        col_num = 0
        
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        total_plot = cm.precip.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        

        
        if row_num == 0:
            ax.set_title('Extreme Frequency', size = 25)
        else:
            ax.set_title('')
            
        if col_num == 0:
            ax.annotate(calendar.month_name[month], xy = (-0.12, 0.5), xycoords = 'axes fraction', fontsize = 25, rotation = 90)
            
        
        col_num += 1        
        
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        total_plot = im.precip.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == 0:
            ax.set_title('Extreme Intensity', size = 25)
        else:
            ax.set_title('')
        
        col_num += 1
        
                
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        total_plot = pm.precip.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == 0:
            ax.set_title('Extreme Proportion', size = 25)
        else:
            ax.set_title('')
        
        
  
  
    
        row_num += 1
 
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
#     axer = fig.add_subplot(gs[1,:], zorder = -100)
#     axer.spines['top'].set_visible(False)
#     axer.spines['bottom'].set_visible(False)    
#     axer.spines['left'].set_visible(False)
#     axer.spines['right'].set_visible(False)
#     axer.set_yticklabels('')
#     axer.set_yticks([])
#     axer.set_xticklabels('')
#     axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,0:4])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 25)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')

  
# All months and all extreme indices for a single month of the MJO.
def anomalies_plots_sigle_phase_all_month_different_cbar(count, int_, prop, phase = 'enhanced',
                    l1 = [], vmax1 = 3, vmax2 = 3, vmax3 =3,add_white = 0, figsize = (8,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    
    count = count.sel(phase = phase)
    int_ = int_.sel(phase = phase)
    prop = prop.sel(phase = phase)
    
    # Plot set up
    months  = [10,11,12,1,2,3]
    num_rows = len(months)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,3, height_ratios=[0.2,0.1] + len(months) * [1])
    gs.update(hspace=0.1, wspace = 0.1)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
#     plt.suptitle(save_name, fontsize = 30,  y = 1)
   
    
    custom_RdBu1, levels1 = anomalie_cbar_1(vmax1, l1, add_white)
    vmin1 = 1/vmax1
    custom_RdBu2, levels2 = anomalie_cbar_1(vmax2, l1, add_white)
    vmin2 = 1/vmax2
    custom_RdBu3, levels3 = anomalie_cbar_1(vmax3, l1, add_white)
    vmin3 = 1/vmax3
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    count = misc.remove_outside_point(count, vmax1, vmin1)
    count = misc.apply_masks(count)
    
    int_ = misc.remove_outside_point(int_, vmax2, vmin2)
    int_  = misc.apply_masks(int_)
    prop = misc.remove_outside_point(prop, vmax3, vmin3)
    prop = misc.apply_masks(prop)
    
    # Initalising the row and column placements

    row_num = 0

    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for month in months:

        cm = count.sel(month = month)
        im = int_.sel(month = month)
        pm = prop.sel(month = month)
        col_num = 0
        
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        count_plot = cm.precip.plot(ax = ax,vmax = vmax1, vmin = vmin1,cmap = custom_RdBu1, 
                                norm = BoundaryNorm(levels1, len(levels1) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        

        
        if row_num == 0:
            ax.set_title('Extreme Frequency', size = 25)
        else:
            ax.set_title('')
            
        if col_num == 0:
            ax.annotate(calendar.month_name[month], xy = (-0.12, 0.5), xycoords = 'axes fraction', fontsize = 25, rotation = 90)
            
        
        col_num += 1        
        
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        int_plot = im.precip.plot(ax = ax,vmax = vmax2, vmin = vmin2,cmap = custom_RdBu2, 
                                norm = BoundaryNorm(levels2, len(levels2) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == 0:
            ax.set_title('Extreme Intensity', size = 25)
        else:
            ax.set_title('')
        
        col_num += 1
        
                
        '''~~~~~~~~~~~~~~~  '''
        # The + 2 is due to the extra plot to make space between the first plot and the colorbar
        ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
        prop_plot = pm.precip.plot(ax = ax,vmax = vmax3, vmin = vmin3,cmap = custom_RdBu3, 
                                norm = BoundaryNorm(levels3, len(levels3) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == 0:
            ax.set_title('Extreme Proportion', size = 25)
        else:
            ax.set_title('')
        
        
  
  
    
        row_num += 1
 
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,0])
    cbar_helper(axes, levels1, count_plot, cbar_title)
    axes = plt.subplot(gs[0,1])
    cbar_helper(axes, levels2, int_plot, cbar_title)
    axes = plt.subplot(gs[0,2])
    cbar_helper(axes, levels3, prop_plot, cbar_title)

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 300, bbox_inches = 'tight', pad = 0)
        print(save_name + ' has been saved')    
        

def cbar_helper(axes, levels, total_plot, cbar_title):
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 25)
        
        
        
        
        
 # All months and all extreme indices for a single month of the MJO.
def anomalies_plots_sigle_phase_all_month_era5(count,
                    l1 = [], vmax = 3,add_white = 0, figsize = (8,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    
    
    # Plot set up
    months  = [10,11,12,1,2,3]
    phases = count.phase.values
    num_rows = 6
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(6 + 2,len(phases), height_ratios=[0.2,0.1] + len(months) * [1])
    gs.update(hspace=0.1, wspace = 0)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 0.93)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    count = misc.remove_outside_point(count, vmax, vmin)
    count = misc.apply_masks(count)
    

    row_num = 0

    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for month in months: # These are the rows
        col_num = 0
        
        for phase in phases: # These are the columns

            cm = count.sel(month = month, phase = phase)
   
 

            '''~~~~~~~~~~~~~~~  '''
            # The + 2 is due to the extra plot to make space between the first plot and the colorbar
            ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
            total_plot = cm.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                    norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)

            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
        

        
            if row_num == 0:
                ax.set_title(phase.capitalize(), size = 25)
            else:
                ax.set_title('')

            if col_num == 0:
                ax.annotate(calendar.month_name[month], xy = (-0.12, 0), xycoords = 'axes fraction', fontsize = 25, rotation = 90)


            col_num += 1
        row_num += 1
            
           


    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,0:4])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 12) 
    cbar.ax.set_title(cbar_title,size = 25)


    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 500, bbox_inches = 'tight', pad = 0 )
        print(save_name + ' has been saved')

        

# THis is for the wind data, it delates half of the points to make the vecotr field no so dense
def delete_subset(data):
    data = data[::2]
    data = data[::2]    
    data = np.transpose(data)
    data = data[::2]
    data = data[::2]
    data = np.transpose(data)
    
    return data



# For wind fields, to thin them out
def quiver_values(u_tot, v_tot):
    u = delete_subset(u_tot.values)
    v = delete_subset(v_tot.values)
    
    lat = u_tot.lat.values[::2][::2]
    lon = u_tot.lon.values[::2][::2]

    
    return lat,lon, u, v

        
 # All months and all extreme indices for a single month of the MJO.
def anomalies_plots_sigle_phase_all_month_era5_wind(count, uwind, vwind,
                    l1 = [], vmax = 3,add_white = 0, figsize = (8,12), 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    
    
    # Plot set up
    months  = [10,11,12,1,2,3]
    phases = count.phase.values
    num_rows = 6
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(6 + 2,len(phases), height_ratios=[0.2,0.1] + len(months) * [1])
    gs.update(hspace=0.1, wspace = 0.1)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 0.93)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    count = misc.remove_outside_point(count, vmax, vmin)
    count = misc.apply_masks(count)
    

    row_num = 0

    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for month in months: # These are the rows
        col_num = 0
        
        for phase in phases: # These are the columns

            cm = count.sel(month = month, phase = phase)
            
            u = uwind.sel(month = month, phase = phase).u
            v = vwind.sel(month = month, phase = phase).v
            lat, lon, u_plot, v_plot = quiver_values(u,v)
            '''~~~~~~~~~~~~~~~  '''
            # The + 2 is due to the extra plot to make space between the first plot and the colorbar
            ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())

            total_plot = cm.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                    norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
            
            
            ax.quiver(lon, lat, u_plot,v_plot)
            ax.set_extent([110, 155, -20,-10])
            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
        

        
            if row_num == 0:
                ax.set_title(phase.capitalize(), size = 25)
            else:
                ax.set_title('')

            if col_num == 0:
                ax.annotate(calendar.month_name[month], xy = (-0.12, 0.2), xycoords = 'axes fraction', fontsize = 25, rotation = 90)


            col_num += 1
        row_num += 1
            
           


    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,0:4])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 12) 
    cbar.ax.set_title(cbar_title,size = 25)


    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 500, bbox_inches = 'tight', pad = 0 )
        print(save_name + ' has been saved')

    
 # Raw values plot
def raw_values_plots_all_phase_month(count,
                    step = 10,add_white = 0, figsize = (8,12), vmax = '',vmin = '', 
                    cbar_title = '', savedir = '' , save_name  = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    
    
    # Plot set up
    months  = [10,11,12,1,2,3]
    phases = count.phase.values
    num_rows = 6
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(6 + 2,len(phases), height_ratios=[0.2,0.1] + len(months) * [1])
    gs.update(hspace=0.1, wspace = 0)
    
    

    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    plt.suptitle(save_name, fontsize = 30,  y = 0.93)
    
    if vmax == '':
        vmax = np.nanpercentile(count, 99.9)
    if vmin == '':
        vmin = np.nanpercentile(count, 0.1)
    levels = np.arange(vmin, vmax + step, step)
    
    cmap = plt.get_cmap('Blues', len(levels))
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    count = misc.remove_outside_point(count, vmax, vmin)
    count = misc.apply_masks(count)
    

    row_num = 0

    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for month in months: # These are the rows
        col_num = 0
        
        for phase in phases: # These are the columns

            cm = count.sel(month = month, phase = phase)
   
 

            '''~~~~~~~~~~~~~~~  '''
            # The + 2 is due to the extra plot to make space between the first plot and the colorbar
            ax = fig.add_subplot(gs[row_num + 2, col_num], projection  = ccrs.PlateCarree())
            total_plot = cm.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = cmap, 
                                    levels = levels,add_colorbar = False)

            ax.outline_patch.set_visible(False)
            ax.coastlines(resolution = '50m')
        

        
            if row_num == 0:
                ax.set_title(phase.capitalize(), size = 25)
            else:
                ax.set_title('')

            if col_num == 0:
                ax.annotate(calendar.month_name[month], xy = (-0.12, 0.5), xycoords = 'axes fraction', fontsize = 25, rotation = 90)


            col_num += 1
        row_num += 1
            
           


    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,0:4])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

#     if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
#         tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 25)
    

    if savedir != '':
        fig.savefig(savedir + save_name  + '.png', dpi = 500, bbox_inches = 'tight', pad = 0 )
        print(save_name + ' has been saved')

       