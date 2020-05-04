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

def anomalie_cbar_1(vmax, l1 = [], add_white = 0):
    import matplotlib as mpl
    
    
    if len(l1) == 0: # THis means I have set my own custom levels for the proble
        if vmax == 5:
            l1 = np.array([1.5,2,2.5,3,3.5,4, 4.5,5])
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
    custom_RdBu = plt.cm.get_cmap("RdBu",len(levels))(np.arange(len(levels) ))
    
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


# In order for the spearman correlatin to be accuate all the zeroes where the ocean are need to be removed
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


def correlation_calc_and_boxannotate(awap, access,ax, corr_y_val, corr_x_val = 0.03):
    

    corr = spearmanr_function(awap.precip, access.precip)
        
    bbox_props = dict(boxstyle = 'square',lw = 1, fc = 'white') # This is the properties for the box
    corr_text = 'Correlation: ' + str(np.round(corr[0],2))
    '''Adding in if this corrleation is signivicant or not'''
    
        
    annotation = ax.annotate(corr_text,xy=(corr_x_val , corr_y_val), fontsize = 12 , 
                xycoords='axes fraction', textcoords='offset points',
                horizontalalignment='left', verticalalignment='top', bbox = bbox_props)
    
    
    return annotation
    
   








'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~SUBPHASES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''





'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Raw Values Comparison'''



'''~~~~~~~~~~~~~~~~~~~~~~Subphase Anomaly Plots'''
        

    
    
def comparison_plot(awap, access, month = False,
                    rain_type = '', add_white = 0,
                    cbar1_title = '', cbar2_title = '',
                    l1 = [], vmax = 3,
                    plot_max = 6000, num_cbar_steps = 10, 
                    savefig = 0, dont_plot = 0, savedir = '' , save_name = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import calendar
    import miscellaneous as misc
    
    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
        access = access.sel(month = month)
    
    # Plot set up
    phases = access.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = (20,12))
    gs = gridspec.GridSpec(num_rows + 2,3, hspace = 0.4, height_ratios=[0.2,0.6] + len(phases) * [1])
    gs.update(wspace=0.025, hspace=0.0)
    
    
    corr_y_val = 0.95 # This is the initial hieght of the text for the y correlation
    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    
    title = rain_type
    plt.suptitle(title, fontsize = 30,  y = 1)
    
    

    
    # This is the colormap for the raw valued plots
    cmap = plt.cm.get_cmap('Blues',num_cbar_steps)
   
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
  
    comparison = access.precip/awap.precip
    
    # Remving points outside of vmin and vmax so colorbar doesn't extend and misalign
    comparison = misc.remove_outside_point(comparison, vmax - 0.1, vmin + 0.1)
    

    # Initalising the row and column placements
    col_num = 0
    row_num = 2
    
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:
        access_phase = access.sel(phase = phase)
        awap_phase = awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        awap_phase.precip.plot(ax = ax,vmax = plot_max,cmap = cmap, add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == first_row_plots:
            ax.set_title('AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
        
        # Row labels
        if col_num == 0:
            ax.annotate(phase.capitalize(), rotation = 90,xy=(-0.05,0.2), fontsize = fontsize, 
            xycoords='axes fraction', textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom')
            
        
        
        
        col_num += 1

        '''~~~~~~~~~~~~~~~  ACCESS-S'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        total_plot = access_phase.precip.plot(ax = ax,vmax = plot_max, cmap = cmap, add_colorbar = False)
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S', size = fontsize, pad = subpad)
        else:
            ax.set_title('')
            
        col_num += 1

        
        
        '''~~~~~~~~~~~~~~~  ACCESS-S/AWAP'''
        '''This needs a custom cololr bar'''
        
        
        ax = fig.add_subplot(gs[row_num, col_num] ,projection  = ccrs.PlateCarree())
        
        # How access comapres to awap
        comparison_phase = comparison.sel(phase = phase)
        

        
        comp_plot = comparison_phase.plot(ax = ax, cmap = custom_RdBu,vmin = vmin, vmax = vmax, add_colorbar = False
                        ,norm = BoundaryNorm(levels, len(levels)-1))
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S/AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
    
        '''~~~~~~~~~~~~~~~  Pattern Correlation'''
        # Returned is the shifted down corr_values and the annotation.
        # Passed in must be the data; the axis to put it on; and the y-height
        annotation = correlation_calc_and_boxannotate(awap_phase, access_phase,ax, corr_y_val) 
    
        # Go back to first coulm, increase the row number
        col_num = 0
        row_num += 1
        
        
        
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
    axer = fig.add_subplot(gs[1,:], zorder = -100)
    axer.spines['top'].set_visible(False)
    axer.spines['bottom'].set_visible(False)    
    axer.spines['left'].set_visible(False)
    axer.spines['right'].set_visible(False)
    axer.set_yticklabels('')
    axer.set_yticks([])
    axer.set_xticklabels('')
    axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    
    axes = plt.subplot(gs[0,:2])
    # This is the colorbar for the first two plots
    normal_colorbar_horizontal(axes, total_plot,plot_max, num_cbar_steps, cbar1_title)
    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,2])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

    if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]
        
    cbar = plt.colorbar(comp_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks =  tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar2_title,size = 15)
    

    '''~~~~~~~~~~~~~~~  Seperatores y-driection'''
    ax.plot([0,0], [0,4.5], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    ax.plot([0.01,0.01], [0,4.5], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    ax.plot([-1,-1], [0,4.5], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    

    if savefig:
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name  + ' has been saved')
    if dont_plot:
        plt.close(fig)
    




'''Phase: This is the comparison plot for PHASES'''
def anomaly_plot_subphases(awap, access, month = False,
                    l1 = [], vmax = 3,add_white = 0,
                    figsize = (20,12), 
                    cbar_title = '', savename = '',
                          savedir = ''):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc
    

    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
        access = access.sel(month = month)
    
    
    # Plot set up
    phases = access.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,2, hspace = 0.4, height_ratios=[0.2,0.6] + len(phases) * [1])
    gs.update(wspace=0.025, hspace=0.0)
    
    
    corr_y_val = 0.65 # This is the initial hieght of the text for the y correlation
    corr_x_val = -0.3
    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles

    plt.suptitle(savename, fontsize = 30,  y = 1)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = misc.remove_outside_point(awap, vmax - 0.01, vmin + 0.01)
    access = misc.remove_outside_point(access, vmax - 0.01, vmin + 0.01)
    

    # Initalising the row and column placements
    col_num = 0
    row_num = 2
    
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:
        access_phase = access.sel(phase = phase)
        awap_phase = awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        awap_phase.precip.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == first_row_plots:
            ax.set_title('AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
        
        # Row labels
        if col_num == 0:
            # Inacactive is longer so we want roation

            ax.annotate(phase.capitalize(), rotation = 90,xy=(-0.05,0.2), fontsize = fontsize, 
            xycoords='axes fraction', textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom')


        
        
        col_num += 1

        '''~~~~~~~~~~~~~~~  ACCESS-S'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        total_plot = access_phase.precip.plot(ax = ax,vmax = vmax, vmin = vmin, cmap = custom_RdBu,
                                              norm = BoundaryNorm(levels, len(levels) - 1), add_colorbar = False)
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S', size = fontsize, pad = subpad)
        else:
            ax.set_title('')
            
        col_num += 1

        
       
    
        '''~~~~~~~~~~~~~~~  Pattern Correlation'''
        # Returned is the shifted down corr_values and the annotation.
        # Passed in must be the data; the axis to put it on; and the y-height

        annotation = correlation_calc_and_boxannotate(awap_phase, access_phase,ax, corr_y_val, corr_x_val) 
    
        # Go back to first coulm, increase the row number
        col_num = 0
        row_num += 1
        
        
        
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
    axer = fig.add_subplot(gs[1,:], zorder = -100)
    axer.spines['top'].set_visible(False)
    axer.spines['bottom'].set_visible(False)    
    axer.spines['left'].set_visible(False)
    axer.spines['right'].set_visible(False)
    axer.set_yticklabels('')
    axer.set_yticks([])
    axer.set_xticklabels('')
    axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,:])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

    if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savedir != '':
        fig.savefig(savedir + savename + '.png', dpi = 500, bbox_inces = 'tight', pad = 0)
    
      
   

'''~~~~~~~~~~~~~~~~~~~~~~Pattern Correlation Timeseries'''

    
    
# This function plots the pattern correlation as a timeseries for all month for all
# the differey subphases of the MJO
    
    
def timeseries_pattern_correlation_plot(pattern_correlation, custom = 1,
                                       x_coords = [-0.2, 0.1, -0.2, -0.2, -0.2],
                                       y_coords = [0.99, 0.65, 0.73, 0.81, 0.955],
                                       savedir = '', savename  = ''):
    
    
    import calendar
    
    fig = plt.figure(figsize = (10,5))
    gs = gridspec.GridSpec(1,1)
    ax = fig.add_subplot(gs[0])
    
    # Colrs for all the differnt lines
    colors = ['g','b','r','purple','orange']

    for plot_num,phase in enumerate(pattern_correlation.phase.values):

        x = pattern_correlation.sel(phase = phase ).month.values
        y = pattern_correlation.sel(phase = phase ).correlation.values
        ax.plot(y, marker = 'o', color = colors[plot_num], label = phase.capitalize())



    # Annotating all of the lines (though this looked better than a color bar)
    phases = pattern_correlation.phase.values  
    if custom:
        for phase, color, x_pos,y_pos in zip(phases,colors,x_coords, y_coords):

            ax.annotate(phase.capitalize(), xy = (x_pos,y_pos), color = color);
    else:
        plt.legend()


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    ax.set_ylabel('Correlation', size = 13)
    ax.set_xticklabels(np.append([''],[calendar.month_name[month] for month in x]), size = 10);

    fig.suptitle(savename, fontsize = 18);


    if savedir != '':
        fig.savefig(savedir + savename  + '.png', dpi = 300, bbox_inches = 'tight', pad = 0)
        
        
        
   
        
 
       
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~PHASES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

'''Phase: This is the anomaly plot for PHASES'''
    
    
def comparison_plot_phases(awap, access, month = False,
                    rain_type = '', add_white = 0,
                    cbar1_title = '', cbar2_title = '',
                    l1 = [], vmax = 3,
                    plot_max = 6000, num_cbar_steps = 10,
                    figsize = (20,25),
                    savefig = 0, dont_plot = 0, savedir = '', save_name = '' ):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import calendar
    import miscellaneous as misc
    
    # Subsetting the month
    if month:
        awap = awap.sel(month = month)
        access = access.sel(month = month)
    
    
    # Applying filters
    awap = misc.apply_masks(awap)
    access = misc.apply_masks(access)
    
    # Plot set up
    phases = access.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,3, hspace = 0.4, height_ratios=[0.2,0.6] + len(phases) * [1])
    gs.update(wspace=0.025, hspace=0.0)
    
    
    corr_y_val = 0.95 # This is the initial hieght of the text for the y correlation
    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    
    title = ' ' +  rain_type
    plt.suptitle(title, fontsize = 30,  y = 0.97)
    
    

    
    # This is the colormap for the raw valued plots
    cmap = plt.cm.get_cmap('Blues',num_cbar_steps)
   
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
  
    comparison = access.precip/awap.precip
    
    # Remving points outside of vmin and vmax so colorbar doesn't extend and misalign
    comparison = misc.remove_outside_point(comparison, vmax - 0.1, vmin + 0.1)
    

    # Initalising the row and column placements
    col_num = 0
    row_num = 2
    
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:
        access_phase = access.sel(phase = phase)
        awap_phase = awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        awap_phase.precip.plot(ax = ax,vmax = plot_max,cmap = cmap, add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == first_row_plots:
            ax.set_title('AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
        
        # Row labels
        if col_num == 0:
            ax.annotate(phase.capitalize(), rotation = 90,xy=(-0.05,0.2), fontsize = fontsize, 
            xycoords='axes fraction', textcoords='offset points',
            horizontalalignment='left', verticalalignment='bottom')
            
        
        
        
        col_num += 1

        '''~~~~~~~~~~~~~~~  ACCESS-S'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        total_plot = access_phase.precip.plot(ax = ax,vmax = plot_max, cmap = cmap, add_colorbar = False)
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S', size = fontsize, pad = subpad)
        else:
            ax.set_title('')
            
        col_num += 1

        
        
        '''~~~~~~~~~~~~~~~  ACCESS-S/AWAP'''
        '''This needs a custom cololr bar'''
        
        
        ax = fig.add_subplot(gs[row_num, col_num] ,projection  = ccrs.PlateCarree())
        
        # How access comapres to awap
        comparison_phase = comparison.sel(phase = phase)
        

        
        comp_plot = comparison_phase.plot(ax = ax, cmap = custom_RdBu,vmin = vmin, vmax = vmax, add_colorbar = False
                        ,norm = BoundaryNorm(levels, len(levels)-1))
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S/AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
    
        '''~~~~~~~~~~~~~~~  Pattern Correlation'''
        # Returned is the shifted down corr_values and the annotation.
        # Passed in must be the data; the axis to put it on; and the y-height
        annotation = correlation_calc_and_boxannotate(awap_phase, access_phase,ax, corr_y_val) 
    
        # Go back to first coulm, increase the row number
        col_num = 0
        row_num += 1
        
        
        
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
    axer = fig.add_subplot(gs[1,:], zorder = -100)
    axer.spines['top'].set_visible(False)
    axer.spines['bottom'].set_visible(False)    
    axer.spines['left'].set_visible(False)
    axer.spines['right'].set_visible(False)
    axer.set_yticklabels('')
    axer.set_yticks([])
    axer.set_xticklabels('')
    axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    
    axes = plt.subplot(gs[0,:2])
    # This is the colorbar for the first two plots
    normal_colorbar_horizontal(axes, total_plot,plot_max, num_cbar_steps, cbar1_title)
    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,2])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

    if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]
        
    cbar = plt.colorbar(comp_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks =  tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar2_title,size = 15)
    

    '''~~~~~~~~~~~~~~~  Seperatores y-driection'''
    y_size = len(phase) + 2.5
    ax.plot([0,0], [0,y_size ], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    ax.plot([0.01,0.01], [0,y_size ], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    ax.plot([-1,-1], [0,y_size], color = 'black', linestyle = ':', transform =ax.transAxes, clip_on = False)
    

    if savefig:
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name + ' has been saved')
    if dont_plot:
        plt.close(fig)
    
    
    
    
    






'''Phase: This is the anomaly plot for PHASES'''
def anom_plot_phases(awap, access, month = False,
                    rain_type = '',
                    l1 = [], vmax = 3,add_white = 0,
                    figsize = (20,12), cbar_title = '',
                    savefig = 0, dont_plot = 0, savedir = '' , save_name = ''): 
    
    assert vmax == 2 or vmax == 5 or vmax == 3 or vmax == 1.5 or vmax == 1.1 or vmax == 2.1
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as mic
    

    # Subsetting the month
    if month:
        print(month)
        awap = awap.sel(month = month)
        access = access.sel(month = month)
    
    
    # Plot set up
    phases = access.phase.values
    num_rows = len(phases)
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(num_rows + 2,2, hspace = 0.4, height_ratios=[0.2,0.6] + len(phases) * [1])
    gs.update(wspace=0.025, hspace=0.0)
    
    
    corr_y_val = 0.65 # This is the initial hieght of the text for the y correlation
    corr_x_val = -0.1
    fontsize = 15 # Size of the row labels
    subsize = 15 # Size of the column labels
    subpad = 20 # The distance in which the column labels appear from the plot

    
    # Titles
    title = rain_type
    plt.suptitle(title, fontsize = 30,  y = 0.97)
    
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
    
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax
      
    # Removing the points above and below a certain threshold. Function deals with the problems with doing
    # this with nan values
    awap = misc.remove_outside_point(awap, vmax, vmin)
    access = misc.remove_outside_point(access, vmax, vmin)
    

    # Initalising the row and column placements
    col_num = 0
    row_num = 2
    
    first_row_plots = row_num
    
    # Looping through the phases, going through columns and then through rows in the loop (e.g looping through model,
    # then phase
    for phase in phases:
        access_phase = access.sel(phase = phase)
        awap_phase = awap.sel(phase = phase)
        
        '''~~~~~~~~~~~~~~~  AWAP'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        awap_phase.precip.plot(ax = ax,vmax = vmax, vmin = vmin,cmap = custom_RdBu, 
                                norm = BoundaryNorm(levels, len(levels) - 1),add_colorbar = False)
         
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        if row_num == first_row_plots:
            ax.set_title('AWAP', size = subsize, pad = subpad)
        else:
            ax.set_title('')
        
        # Row labels
        if col_num == 0:
            # Inacactive is longer so we want roation
            if phase == 'inactive':
                ax.annotate(phase.capitalize(), rotation = 90,xy=(-0.05,0.2), fontsize = fontsize, 
                xycoords='axes fraction', textcoords='offset points',
                horizontalalignment='left', verticalalignment='bottom')
            # Other phese with no roation as they are just a number
            else:
                ax.annotate(phase, rotation = 0,xy=(-0.05,0.2), fontsize = fontsize, 
                xycoords='axes fraction', textcoords='offset points',
                horizontalalignment='left', verticalalignment='bottom')
            
        
        
        
        col_num += 1

        '''~~~~~~~~~~~~~~~  ACCESS-S'''
        ax = fig.add_subplot(gs[row_num, col_num], projection  = ccrs.PlateCarree())
        total_plot = access_phase.precip.plot(ax = ax,vmax = vmax, vmin = vmin, cmap = custom_RdBu,
                                              norm = BoundaryNorm(levels, len(levels) - 1), add_colorbar = False)
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if row_num == first_row_plots:
            ax.set_title('ACCESS-S', size = fontsize, pad = subpad)
        else:
            ax.set_title('')
            
        col_num += 1

        
       
    
        '''~~~~~~~~~~~~~~~  Pattern Correlation'''
        # Returned is the shifted down corr_values and the annotation.
        # Passed in must be the data; the axis to put it on; and the y-height

        annotation = correlation_calc_and_boxannotate(awap_phase, access_phase,ax, corr_y_val, corr_x_val) 
    
        # Go back to first coulm, increase the row number
        col_num = 0
        row_num += 1
        
        
        
    '''~~~~~~~~~~~~~~~  Seperator between cbar and top plot'''    
    axer = fig.add_subplot(gs[1,:], zorder = -100)
    axer.spines['top'].set_visible(False)
    axer.spines['bottom'].set_visible(False)    
    axer.spines['left'].set_visible(False)
    axer.spines['right'].set_visible(False)
    axer.set_yticklabels('')
    axer.set_yticks([])
    axer.set_xticklabels('')
    axer.set_xticks([])
    
    '''~~~~~~~~~~~~~~~  Colorbars'''
    

    
    # The colorbar for the final plot comparing the two
    axes = plt.subplot(gs[0,:])
    
    tick_locations = levels[1:-1] # Not including the start and end points so I can add > and < symbols

    if len(tick_locations) > 20: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]


    cbar = plt.colorbar(total_plot, cax = axes , extend = 'neither', orientation = 'horizontal', ticks = tick_locations)
    
    tick_strings = np.round(tick_locations,2).astype(str)
    tick_strings[0] = '<' + tick_strings[0]
    tick_strings[-1] = '<' + tick_strings[-1] 
    cbar.ax.set_xticklabels(tick_strings, fontsize = 10) 
    cbar.ax.set_title(cbar_title,size = 15)
    

    if savefig:
        fig.savefig(savedir + save_name  + '.png', dpi = 300)
        print(save_name  + ' has been saved')
    if dont_plot:
        plt.close(fig)
    
    
    
    


