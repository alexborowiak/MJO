import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import matplotlib.gridspec as gridspec
import calendar
import matplotlib as mpl

def pattern_corr_state(data_total_awap, data_total_access): # Pattern correlations for all and states
    import access_functions as af
    #Lons for All,WA, NT and QLD respectivly
    states = [slice(110,160),slice(110,129),slice(130,138), slice(138, 160)] 
        
    state_stor = []    

    for state in states:
        data_awap = data_total_awap.sel(lon = state)
        data_access = data_total_access.sel(lon = state)
        
        pattern_corr = af.month_pattern_correlations(data_awap, data_access)
        
        state_stor.append(pattern_corr)
    state_corr = xr.concat(state_stor, pd.Index(['All','WA','NT','QLD'], name = 'state'))
    return state_corr.transpose()








'''~~~~~~~~~~~~~~~~~~~~~~~ PLOTS ~~~~~~~~~~~'''
def custom_cmap(levels, add_white = 0, extender = 0):
    
    
    import matplotlib.colors as mpc
    
    cmap = plt.cm.RdBu

    
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
    
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels) + 1) 
    # Joingi the colormap back together
    
    
    return cmap

def corr_heatmap_state_group(data_tot, month_reverse = 0
                 , vmax = 0.5,step = 0.1, add_white = 0,
                 savename = '', savedir = ''):
    
    
    # Construction the colormap
    

    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)

    # The custom cmap is overridden, as the colours didn't matchthe values, and there where some errors with the add_white component
    # Nevertheless, it has been left in incase I need it in the future
    cmap = custom_cmap(levels, add_white = add_white)
#     cmap = plt.cm.get_cmap('RdBu', len(levels) + 1)

    fig = plt.figure(figsize = (15, 12))

    gs = gridspec.GridSpec(3,2, hspace = 0.7, wspace = 0.4, height_ratios = [0.15,1,1])


    fig.suptitle(savename)

    state_id = {'All':'All States','NT':'Northern Territory','QLD':'Queensland','WA':'Western Australia'}

    for plot_num, state in enumerate(data_tot.state.values):

        ax = fig.add_subplot(gs[plot_num + 2])


        data = data_tot.sel(state = state)
        plot_vals = data.values

        # Making sure the months are in the correct order: not 1,2,3,10,11,12
        if month_reverse:
            plot_vals = np.concatenate([plot_vals[:,3:], plot_vals[:,0:3]], axis = 1)

        # ax.imshow(data, cmap = 'RdBu', vmin = -1, vmax = 1)

        sns.heatmap(plot_vals.round(2), ax = ax,
                    cmap = cmap, vmax = vmax, vmin = vmin, cbar = False, 
                   annot = True, linewidths = 1, linecolor = 'k')

        ax.set_yticklabels([i.capitalize() for i in data.phase.values], rotation = 360);
        month_names = [calendar.month_name[i] for i in data.month.values]
        if month_reverse:
            month_names= np.concatenate([month_names[3:], month_names[0:3]])
        ax.set_xticklabels(month_names, rotation = 45);

        ax.set_title(state_id[state], fontsize = 15, pad = 5)

    # 

#     ax = fig.add_subplot(gs[0])
#     pdata = ax.imshow(np.transpose(data_tot.values), vmin = vmin, vmax = vmax, cmap = cmap)


    axes = plt.subplot(gs[0:2])
    cbar = mpl.colorbar.ColorbarBase( axes,cmap = cmap, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
    cbar.ax.set_title('Correlation', fontsize = 15);
    
    if savedir != '':
        fig.savefig(savedir + savename + '.png', dpi = 500, pad = 0, bbox_inches = 'tight')



def corr_heatmap_phase_group(data_tot, month_reverse = 0
                 , vmax = 0.5,step = 0.1, add_white = 0,
                 savename = '', savedir = ''):
    
    
    # Construction the colormap
    

    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)
    # The custom cmap is overridden, as the colours didn't matchthe values, and there where some errors with the add_white component
    # Nevertheless, it has been left in incase I need it in the future
    cmap = custom_cmap(levels, add_white = add_white)
#     cmap = plt.cm.get_cmap('RdBu', len(levels) + 1)

    fig = plt.figure(figsize = (15, 12))
    gs = gridspec.GridSpec(3,2, hspace = 0.7, wspace = 0.4, height_ratios = [0.15,1,1])


    fig.suptitle(savename)

    state_id = {'All':'All States','WA':'Western Australia','NT':'Northern Territory','QLD':'Queensland'}

    for plot_num, phase in enumerate(data_tot.phase.values):

        ax = fig.add_subplot(gs[plot_num + 2])


        data = data_tot.sel(phase = phase)
        plot_vals = data.values

        # Making sure the months are in the correct order: not 1,2,3,10,11,12
        if month_reverse:
            plot_vals = np.concatenate([plot_vals[:,3:], plot_vals[:,0:3]], axis = 1)

        sns.heatmap(plot_vals.round(2), ax = ax,
                    cmap = cmap, vmax = vmax, vmin = vmin, cbar = False, 
                   annot = True, linewidths = 1, linecolor = 'k')
        

        ax.set_yticklabels(list(state_id.values()), rotation = 360);
        month_names = [calendar.month_name[i] for i in data.month.values]
        if month_reverse:
            month_names= np.concatenate([month_names[3:], month_names[0:3]])
        ax.set_xticklabels(month_names, rotation = 45);

        ax.set_title(phase.capitalize(), fontsize = 15, pad = 5)

    # 

#     ax = fig.add_subplot(gs[0])
#     pdata = ax.imshow(np.transpose(data_tot.values), vmin = vmin, vmax = vmax + step, cmap = cmap)


    axes = plt.subplot(gs[0:2])
    cbar = mpl.colorbar.ColorbarBase( axes,cmap = cmap, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
#     cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
#                         ticks  = levels, boundaries = levels)
    cbar.ax.set_title('Correlation', fontsize = 15);
    
    if savedir != '':
        fig.savefig(savedir + savename + '.png', dpi = 500, pad = 0, bbox_inches = 'tight')
        
        
        
        
        

def corr_heatmap_multi_index(data_tot, titles = [],month_reverse = 0
                 , vmax = 0.5,step = 0.1, add_white = 0,
                 savename = '', savedir = ''):
    
    
    # Construction the colormap
    

    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)
    # The custom cmap is overridden, as the colours didn't matchthe values, and there where some errors with the add_white component
    # Nevertheless, it has been left in incase I need it in the future
    cmap = custom_cmap(levels, add_white = add_white)
#     cmap = plt.cm.get_cmap('RdBu', len(levels) + 1)

    fig = plt.figure(figsize = (8
, 15))
    gs = gridspec.GridSpec(4,1, hspace = 0.7,  height_ratios = [0.15,1,1,1])


    fig.suptitle(savename)


    for plot_num in range(len(data_tot)):

        ax = fig.add_subplot(gs[plot_num + 1])


        data = data_tot[plot_num]
        plot_vals = data.values

        # Making sure the months are in the correct order: not 1,2,3,10,11,12
        if month_reverse:
            plot_vals = np.concatenate([plot_vals[:,3:], plot_vals[:,0:3]], axis = 1)

        sns.heatmap(plot_vals.round(2), ax = ax,
                    cmap = cmap, vmax = vmax, vmin = vmin, cbar = False, 
                   annot = True, linewidths = 1, linecolor = 'k')

        ax.set_yticklabels([i.capitalize() for i in data.phase.values], rotation = 360);
        month_names = [calendar.month_name[i] for i in data.month.values]
        if month_reverse:
            month_names= np.concatenate([month_names[3:], month_names[0:3]])
        ax.set_xticklabels(month_names, rotation = 45);

        if len(titles):
            ax.set_title(titles[plot_num], fontsize = 15, pad = 5)

    # 

#     ax = fig.add_subplot(gs[0])
#     pdata = ax.imshow(np.transpose(data_tot.values), vmin = vmin, vmax = vmax + step, cmap = cmap)


    axes = plt.subplot(gs[0])
    cbar = mpl.colorbar.ColorbarBase( axes,cmap = cmap, orientation = 'horizontal',
                        ticks  = levels, boundaries = levels)
#     cbar = plt.colorbar(pdata, cax = axes, orientation = 'horizontal',
#                         ticks  = levels, boundaries = levels)
    cbar.ax.set_title('Correlation', fontsize = 15);
    
    if savedir != '':
        fig.savefig(savedir + savename + '.png', dpi = 500, pad = 0, bbox_inches = 'tight')