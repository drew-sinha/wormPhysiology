# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:08:05 2016

@author: Willie
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.stats

import analyzeHealth.selectData as selectData
import analyzeHealth.computeStatistics as computeStatistics
import graphingFigures.plotFigures as plotFigures

def canned_confirmation(subfigures_list, health_data, adult_df, label_points, health_name):
    (absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter) = subfigures_list
    
    # Make bins of lifespans.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    my_adultspans = selectData.get_adultspans(adult_df)/24  
    my_cohorts = life_cohorts

    # Prepare geometry data.
    geometry_dict = computeStatistics.one_d_geometries(adult_df, health_data)
    mean_start = np.mean(geometry_dict['start'])            
    
    # Do the plots. 
    cohort_traces(absolute_trace, health_data, adult_df, the_title = health_name + ' Over Time', the_xlabel = 'Days of Adult Life', the_ylabel = health_name, x_normed = False, skip_conversion = True)
    cohort_scatters(start_scatter, my_adultspans, geometry_dict['start'], adult_df, the_title = 'Start', the_xlabel = 'Days of Adult Lifespan', the_ylabel = health_name + ' at Start of Adulthood', label_coordinates = label_points[0])
    cohort_scatters(rate_scatter, my_adultspans, geometry_dict['rate'], adult_df, the_title = 'Rate', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Rate of Decrease of ' + health_name, label_coordinates = label_points[1], polyfit_degree = 2)
    cohort_scatters(end_scatter, my_adultspans, geometry_dict['end'], adult_df, the_title = 'End', the_xlabel = 'Days of Adult Lifespan', the_ylabel = health_name + ' at Death', label_coordinates = label_points[2], polyfit_degree = 2)

    # Plot the traces and scatter for absolute inflection.
    cohort_traces(variable_trace, health_data, adult_df, the_title = health_name + ' Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = health_name, x_normed = True, skip_conversion = True)
    variable_trace.set_ylim([0, 1.1*mean_start])
    cohort_scatters(variable_scatter, my_adultspans, geometry_dict['absolute_inflection'], adult_df, the_title = 'Absolute Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Average Deviation (' + health_name + ')', label_coordinates = label_points[3])

    # Plot the traces and scatter for relative inflection.
    cohort_traces(relative_trace, health_data, adult_df, the_title = 'Relative ' + health_name + ' Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = 'Relative ' + health_name, x_normed = True, y_normed = True, zero_to_one = False)
    cohort_scatters(relative_scatter, my_adultspans, geometry_dict['self_inflection'], adult_df, the_title = 'Relative Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Relative Deviation (' + health_name + ')', label_coordinates = label_points[4]) 
    return

def color_image(my_subfigure, my_image, the_title = None, the_xlabel = None, the_ylabel = None):
    '''
    Plots a color figure on a subfigure.
    '''
    # Properly display the image.
    image_dtype = my_image.dtype
    dtype_max = np.iinfo(image_dtype).max
    desired_max = np.iinfo('uint8').max
    display_image = (my_image.astype('float64')*desired_max/dtype_max).astype('uint8')
    my_subfigure.axis('off')
    my_subfigure.imshow(np.swapaxes(display_image, 0, 1))

    # Label the subplot.
    my_subfigure.set_title(the_title)
    my_subfigure.set_xlabel(the_xlabel)
    my_subfigure.set_ylabel(the_ylabel)
    return my_subfigure

def measurements_sketch(my_subfigure, adult_df):
    '''
    Plot two simple measurement trajectories to illustrate change over time.
    '''
    # Plot data for movement and autofluorescence.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    my_subfigure.plot(*selectData.cohort_trace(adult_df, life_cohorts[3], 'bulk_movement'), color = 'blue', linewidth = 2)  
    my_subfigure.plot(*selectData.cohort_trace(adult_df, life_cohorts[3], 'intensity_80'), color = 'red', linewidth = 2)    

    # Label the figure.
    my_subfigure.annotate('Autofluorescence', xy = (0.5, -0.5), xycoords = 'data')
    my_subfigure.annotate('Movement', xy = (3.5, 1), xycoords = 'data')
    my_subfigure.set_ylabel('Physiological Measurements')
    my_subfigure.set_xlabel('Age')
    my_subfigure.set_title('Measure Physiological Parameters')
    plotFigures.remove_box(my_subfigure)
    return my_subfigure

def cohort_traces(my_subfigure, a_variable, adult_df, only_worms = None, make_labels=True, bin_width_days=2,bin_mode='day', cohorts_to_use=[], stop_with_death=True, **kwargs):
    '''
    Make cohort traces for a_variable.
    '''
    the_title = kwargs.get('the_title', None)
    the_xlabel = kwargs.get('the_xlabel', None)
    the_ylabel = kwargs.get('the_ylabel', None)
    line_style = kwargs.get('line_style', '-')
    line_color = kwargs.get('line_color', None)
    
    x_normed = kwargs.get('x_normed', None)
    y_normed = kwargs.get('y_normed', None)
    skip_conversion = kwargs.get('skip_conversion', False)
    zero_to_one = kwargs.get('zero_to_one',False)

    # Make bins of lifespans.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = bin_width_days,bin_mode=bin_mode)
    if len(cohorts_to_use) >0:
        life_cohorts = [life_cohorts[c_idx] for c_idx in cohorts_to_use]
        bin_lifes = [bin_lifes[c_idx] for c_idx in cohorts_to_use]
        my_bins = [my_bins[c_idx] for c_idx in cohorts_to_use]
        my_colors = [my_colors[c_idx] for c_idx in cohorts_to_use]

    # Exclude worms.    
    if only_worms != None:
        life_cohorts = [[a_worm for a_worm in a_cohort if a_worm in only_worms] for a_cohort in life_cohorts]
    else:
        pass
    my_cohorts = life_cohorts

    # Plot the actual stuff.
    if y_normed:
        if type(a_variable) == type(''):
            mean_start = np.mean(adult_df.mloc(adult_df.worms, [a_variable], ['0.0'])[:, 0, 0])
            (mean_start, my_unit, fancy_name) = adult_df.display_variables(mean_start, a_variable)
        else:
            mean_start = np.mean(a_variable[:, 0, 0])
    for i in range(0, len(my_cohorts)):
        if len(my_cohorts[i]) > 0:
            # Figure out the cohort data.
            a_cohort = my_cohorts[i]
            if type(a_variable) == type(''):
                cohort_data = adult_df.mloc(adult_df.worms, [a_variable])[a_cohort, 0, :]
                variable_name = a_variable
            else:
                cohort_data = a_variable[a_cohort, 0, :]
                variable_name = 'health'
            
            cohort_data = cohort_data[~np.isnan(cohort_data).all(axis = 1)]
            if stop_with_death: # Suppress data after the first individual in a cohort dies
                cohort_data = np.mean(cohort_data, axis = 0)
            else:
                cohort_data = np.nanmean(cohort_data, axis=0)
            if not skip_conversion:
                (cohort_data, my_unit, fancy_name) = adult_df.display_variables(cohort_data, variable_name)
            else:
                (whatever, my_unit, fancy_name) = adult_df.display_variables(cohort_data, variable_name)
            cohort_data = cohort_data[~np.isnan(cohort_data)]
            if y_normed:
                if zero_to_one:
                    cohort_data = cohort_data - cohort_data[0]
                    cohort_data = cohort_data/cohort_data[-1]
                else:
                    cohort_data = cohort_data - cohort_data[-1]
                    cohort_data = cohort_data/cohort_data[0]

            # Figure out the cohort ages.           
            cohort_ages = adult_df.ages[:cohort_data.shape[0]]          
            if x_normed:
                cohort_ages = cohort_ages/np.max(cohort_ages)
            my_subfigure.plot(cohort_ages, cohort_data, color = my_colors[i] if line_color is None else line_color, linewidth = 2, linestyle=line_style)

    # Label the subplot.
    if the_title == None:
        the_title = fancy_name + ' Over Time'
    if make_labels: my_subfigure.set_title(the_title)
    if make_labels: my_subfigure.set_xlabel(the_xlabel)
    if the_ylabel == None:
        the_ylabel = my_unit
    if make_labels: my_subfigure.set_ylabel(the_ylabel) 
    return my_subfigure

def cohort_scatters(my_subfigure, xdata, ydata, adult_df, the_title = None, the_xlabel = None, the_ylabel = None, label_coordinates = (0, 0), no_cohorts_color = None, polyfit_degree = 1, only_worms = None, make_labels=True,bin_width_days=2,bin_mode='day',plot_trenddata=True,**scatter_kws):
    '''
    Make colorful scatterplots by cohort.
    '''
    # Set up some needed data.
    if no_cohorts_color == None:
        (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = bin_width_days,bin_mode=bin_mode)    
    else:
        life_cohorts = [list(range(xdata.shape[0]))]
        my_colors = [no_cohorts_color]

    # Exclude worms.
    if only_worms != None:
        life_cohorts = [[a_worm for a_worm in a_cohort if a_worm in only_worms] for a_cohort in life_cohorts]
    else:
        pass        


    # Make the actual plot. 
    for i in range(0, len(life_cohorts)):
        a_cohort = life_cohorts[i]
        my_subfigure.scatter(xdata[a_cohort], ydata[a_cohort], color = my_colors[i],**scatter_kws)    

    # Label the plot.
    if make_labels:
        my_subfigure.set_title(the_title)
        my_subfigure.set_xlabel(the_xlabel)
        my_subfigure.set_ylabel(the_ylabel) 
    my_subfigure.set_aspect('auto')

    # Fit a polynomial for a trendline and for r^2.
    p_array = np.polyfit(xdata, ydata, polyfit_degree)
    my_estimator = np.array([p_array[-i]*xdata**(i-1) for i in range(1, len(p_array)+1)])
    my_estimator = my_estimator.sum(axis = 0)
    xrange = np.linspace(np.min(xdata), np.max(xdata), 200)
    my_trendline = np.array([p_array[-i]*xrange**(i-1) for i in range(1, len(p_array)+1)])
    my_trendline = my_trendline.sum(axis = 0)
    #if plot_trenddata: 
    #~ if no_cohorts_color is not None and all(my_colors[0] == [0,0,0]):
        #~ my_subfigure.plot(xrange, my_trendline, color = 'yellow')
    #~ else:
        #~ my_subfigure.plot(xrange, my_trendline, color = 'black')

    label_string = '$r^2 = ' + ('%.3f' % computeStatistics.quick_pearson(my_estimator, ydata)) + '$'
    if plot_trenddata: my_subfigure.annotate(label_string, label_coordinates, textcoords = 'data', size = 10)   
    return my_subfigure

def health_sketch(my_subfigure, adult_df):
    '''
    Plot prognosis trajectory to illustrate change over time.
    '''
    # Plot data for prognosis.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    my_subfigure.plot(*selectData.cohort_trace(adult_df, life_cohorts[3], 'health'), color = 'black', linewidth = 2)    

    # Label the figure.
    my_subfigure.set_ylabel('Prognosis (Time to Death)')
    my_subfigure.set_xlabel('Age')
    my_subfigure.set_title('Make Prognosis')
    plotFigures.remove_box(my_subfigure)
    return my_subfigure
    
def survival_lifespan(survival_plot, lifespans_plot, adult_df, make_labels=True, cohort_info=None):
    '''
    Make a survival curve and lifespan distribution with my cohorts highlighted.
    
    cohort_info - Packeted tuple/list of information on cohorts (i.e. from selectData.adult_cohort_bins) for plotting cohort data
    '''
    # Get lifespan data.    
    lifespans = selectData.get_adultspans(adult_df)/24
    life_histogram = np.histogram(lifespans, density = True, bins = 1000)
    life_times = life_histogram[1]
    cumulative_death = life_histogram[0]/np.sum(life_histogram[0])
    cumulative_death = np.cumsum(cumulative_death)
    cumulative_life = 1 - cumulative_death

    # Insert a point to start with zero days post hatch.
    cumulative_life = np.insert(cumulative_life, 0, 1)
    cumulative_life = np.insert(cumulative_life, 0, 1)
    life_times = np.insert(life_times, 0, 0)

    # Figure out where to start each line.
    if cohort_info is None:
        (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    else:
        (life_cohorts, bin_lifes, my_bins, my_colors) = cohort_info
    cohort_lifes = np.array([lifespans[a_cohort] for a_cohort in life_cohorts])
    cohort_mins = [np.min(cohort_life) for cohort_life in cohort_lifes]
    cohort_mins = [np.argmin(np.abs(life_times - cohort_min)) for cohort_min in cohort_mins]    
    
    # Plot the actual curve.    
    #survival_plot.plot(life_times[:cohort_mins[0] + 1], cumulative_life[:cohort_mins[0] + 1], linewidth = 2, color = 'black')
    survival_plot.plot(life_times[:cohort_mins[0] + 1], cumulative_life[:cohort_mins[0] + 1], linewidth = 2, color = my_colors[0])
    for i in range(0, len(cohort_mins) - 1):
        survival_plot.plot(life_times[cohort_mins[i]: cohort_mins[i+1] + 1], cumulative_life[cohort_mins[i]: cohort_mins[i+1] + 1], linewidth = 2, color = my_colors[i])
    survival_plot.plot(life_times[cohort_mins[-1]:], cumulative_life[cohort_mins[-1]:], linewidth = 2, color = my_colors[-1])

    # Label my survival curve.
    if make_labels: survival_plot.set_ylabel('% Surviving')
    if make_labels: survival_plot.set_xlabel('Days Post-Maturity')
    if make_labels: survival_plot.set_title('spe-9(hc88) Adult Survival in Corrals')
    survival_plot.set_aspect('auto')
    survival_plot.set_ylim([0.0, 1.1])  
    #survival_plot.set_xlim([0, 17])
    survival_plot.set_xlim([0,lifespans.max()//1+1])
    
    # Plot histogram.
    bin_width = 2
    for i in range(0, len(cohort_lifes)):
        my_range = (np.floor(np.min(cohort_lifes[i])), np.ceil(np.max(cohort_lifes[i])))
        (n, bins, patches) = lifespans_plot.hist(cohort_lifes[i], range = my_range, bins = 2/bin_width, normed = False, facecolor = my_colors[i], alpha = 0.75, linewidth = 0)

    # Plot smoothed kde density curve.
    kde_density = scipy.stats.gaussian_kde(lifespans)
    #my_xrange = np.linspace(0, 17, 200)
    my_xrange = np.linspace(0, lifespans.max()//1+1, 200)
    kde_density._compute_covariance()
    lifespans_plot.plot(my_xrange, kde_density(my_xrange)*len(lifespans)*bin_width, color = 'black', linewidth = 1) 
    
    # Label the lifespan distribution.  
    if make_labels: lifespans_plot.set_ylabel('Number of Worms')
    if make_labels: lifespans_plot.set_xlabel('Days Post-Maturity')
    lifespans_plot.set_ylim([0, 300])   
    if make_labels: lifespans_plot.set_title('spe-9(hc88) Adult Lifespans in Corrals')
    return (survival_plot, lifespans_plot)
    
def explain_traces(highlight_life, histogram_high, histogram_low, illustrate_traces, adult_df):
    '''
    Explain how the average traces are made.
    '''
    # Get lifespan data.    
    lifespans = selectData.get_adultspans(adult_df)/24
    life_histogram = np.histogram(lifespans, density = True, bins = 1000)
    life_times = life_histogram[1]
    cumulative_death = life_histogram[0]/np.sum(life_histogram[0])
    cumulative_death = np.cumsum(cumulative_death)
    cumulative_life = 1 - cumulative_death

    # Insert a point to start with zero days post hatch.
    cumulative_life = np.insert(cumulative_life, 0, 1)
    cumulative_life = np.insert(cumulative_life, 0, 1)
    life_times = np.insert(life_times, 0, 0)

    # Figure out where to start each line.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    fade_colors = [(a_color + 1)/2 for a_color in my_colors]
    cohort_lifes = np.array([lifespans[a_cohort] for a_cohort in life_cohorts])
    cohort_mins = [np.min(cohort_life) for cohort_life in cohort_lifes]
    cohort_mins = [np.argmin(np.abs(life_times - cohort_min)) for cohort_min in cohort_mins]    
        
    # Plot histogram.
    life_colors = list(fade_colors)
    life_colors[1] = my_colors[1]
    life_colors[5] = my_colors[5]
    bin_width = 2
    for i in range(0, len(cohort_lifes)):
        my_range = (np.floor(np.min(cohort_lifes[i])), np.ceil(np.max(cohort_lifes[i])))
        (n, bins, patches) = highlight_life.hist(cohort_lifes[i], range = my_range, bins = 2/bin_width, normed = False, facecolor = life_colors[i], alpha = 0.75, linewidth = 0)

    # Plot smoothed kde density curve.
    kde_density = scipy.stats.gaussian_kde(lifespans)
    my_xrange = np.linspace(0, 17, 200)
    kde_density._compute_covariance()
    highlight_life.plot(my_xrange, kde_density(my_xrange)*len(lifespans)*bin_width, color = 'black', linewidth = 1) 

    # Label the lifespan distribution.  
    highlight_life.set_ylabel('Number of Worms')
    highlight_life.set_xlabel('Days Post-Maturity')
    highlight_life.set_ylim([0, 300])   
    highlight_life.set_title('spe-9(hc88) Adult Lifespans in Corrals')

    # Prepare lists for texture histograms.
    histogram_cohorts_times = [(1, '3.0'), (5, '3.0')]
    histogram_plots = [histogram_low, histogram_high]

    # Plot texture distributions.   
    mean_values = []
    for k in range(0, 2):
        texture_data = adult_df.mloc(adult_df.worms, ['health'], [histogram_cohorts_times[k][1]])[life_cohorts[histogram_cohorts_times[k][0]], 0, 0]
        (texture_data, my_unit, fancy_name) = adult_df.display_variables(texture_data, 'health')
        (n, bins, patches) = histogram_plots[k].hist(texture_data, normed = False, facecolor = fade_colors[histogram_cohorts_times[k][0]], alpha = 0.75, linewidth = 0, orientation = 'horizontal')

        # Label the texture distribution.   
        histogram_plots[k].set_xlabel('Number of Worms')
        histogram_plots[k].set_ylabel('Measurement Value at Day ' + str(int(float(histogram_cohorts_times[k][1]))))
        histogram_plots[k].set_ylim([0, 11])    
        histogram_plots[k].set_xlim([0, 27])    
        histogram_plots[k].axhline(y = np.mean(texture_data), xmin = 0, xmax = 1, color = my_colors[histogram_cohorts_times[k][0]], linewidth = 1, clip_on = False)
        mean_values.append(np.mean(texture_data))

    histogram_plots[1].set_title('Long-Lived Cohort')
    histogram_plots[0].set_title('Short-Lived Cohort')

    # Prepare some lists.
    trace_plots = [illustrate_traces]
    cohort_indices = [1, 5]
    
    # Plot the actual stuff.
    a_var = 'health'
    for k in range(0, 1):
        a_cohort = life_cohorts[cohort_indices[k]]
        other_cohort = life_cohorts[cohort_indices[k-1]]
    
        # Prepare data.
        this_cohort_data = adult_df.mloc(adult_df.worms, [a_var])[a_cohort, 0, :]
        this_cohort_data = this_cohort_data[~np.isnan(this_cohort_data).all(axis = 1)]
        this_cohort_data = np.mean(this_cohort_data, axis = 0)
        (this_cohort_data, my_unit, fancy_name) = adult_df.display_variables(this_cohort_data, a_var)
    
        # Prepare more data.
        other_cohort_data = adult_df.mloc(adult_df.worms, [a_var])[other_cohort, 0, :]
        other_cohort_data = other_cohort_data[~np.isnan(other_cohort_data).all(axis = 1)]
        other_cohort_data = np.mean(other_cohort_data, axis = 0)
        (other_cohort_data, my_unit, fancy_name) = adult_df.display_variables(other_cohort_data, a_var)
    
        # Plot traces.
        trace_plots[k].plot(adult_df.ages[:this_cohort_data.shape[0]], this_cohort_data, color = my_colors[cohort_indices[k]], zorder = 1, linewidth = 2)
        trace_plots[k].plot(adult_df.ages[:other_cohort_data.shape[0]], other_cohort_data, color = my_colors[cohort_indices[k-1]], zorder = 1, linewidth = 2)

        # Label.
        trace_plots[k].set_title('Physiology Over Time')
        trace_plots[k].set_ylabel('Measurement Value')  
        trace_plots[k].set_ylim([0, 11])    
        trace_plots[k].set_xlim([0, 14])    
        trace_plots[k].set_xlabel('Age (Days Post-Maturity)')

    # Draw dots.
    trace_plots[0].scatter([3], [mean_values[0]], zorder = 2, s = 100, color = my_colors[1])
    trace_plots[0].scatter([3], [mean_values[1]], zorder = 2, s = 100, color = my_colors[5])

    # Draw lines.
    histogram_plots[0].axhline(y = mean_values[0], xmin = 1, xmax = 1.1, color = my_colors[1], linewidth = 1, clip_on = False)
    trace_plots[0].axhline(y = mean_values[0], xmin = 0, xmax = 0.2, color = fade_colors[1], linewidth = 1, clip_on = False, linestyle = 'dashed')
    
    histogram_plots[1].axhline(y = mean_values[1], xmin = 1, xmax = 1.1, color = my_colors[5], linewidth = 1, clip_on = False)
    histogram_plots[0].axhline(y = mean_values[1], xmin = 0, xmax = 1.1, color = fade_colors[5], linewidth = 1, clip_on = False, linestyle = 'dashed')
    trace_plots[0].axhline(y = mean_values[1], xmin = 0, xmax = 0.2, color = fade_colors[5], linewidth = 1, clip_on = False, linestyle = 'dashed')
    return (highlight_life, histogram_high, histogram_low, illustrate_traces)

def absolute_hypotheses(start_plot, rate_plot, end_plot, adult_df):
    '''
    Make a sketch to illustrate the possible differences between long- and short-lived worms.
    '''
    # Plot the "start" hypothesis.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)   
    for i in range(0, len(life_cohorts)):
        xrange = np.linspace(0, bin_lifes[i], 200)
        yrange = bin_lifes[i] - xrange
        start_plot.plot(xrange, yrange, color = my_colors[i], linewidth = 2)
    
    # Label the "start" plot.
    start_plot.set_title('Start Hypothesis')
    start_plot.set_aspect('auto')
    start_plot.set_xlabel('Days of Adulthood')
    start_plot.set_ylabel('Prognosis (Remaining Days)')
    
    # Plot the "rate" hypothesis.
    overall_mean = np.mean(selectData.get_adultspans(adult_df))/24
    for i in range(0, len(life_cohorts)):
        xrange = np.linspace(0, bin_lifes[i], 200)
        yrange = overall_mean - (overall_mean/bin_lifes[i])*xrange
        rate_plot.plot(xrange, yrange, color = my_colors[i], linewidth = 2)

    # Label the "rate" plot.
    rate_plot.set_title('Rate Hypothesis')
    rate_plot.set_aspect('auto')
    rate_plot.set_xlabel('Days of Adulthood')
    rate_plot.set_ylabel('Prognosis (Remaining Days)')
    
    # Set up some needed data.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)   
    cohort_max = np.max(bin_lifes)

    # Plot the "end" hypothesis.
    for i in range(0, len(life_cohorts)):
        xrange = np.linspace(0, bin_lifes[i], 200)
        yrange = cohort_max  - xrange + i/3.5 - 6/3.5
        end_plot.plot(xrange, yrange, color = my_colors[i], linewidth = 2)
        
    # Label the "end" plot.
    end_plot.set_title('End Hypothesis')
    end_plot.set_aspect('auto')
    end_plot.set_xlabel('Days of Adulthood')
    end_plot.set_ylabel('Prognosis (Remaining Days)')
    return (start_plot, rate_plot, end_plot)

def relative_hypotheses(deviation_plot, measured_deviation, computed_deviation, rescale_plot, negative_plot, positive_plot, adult_df):
    '''
    Make a sketch to illustrate the possible differences between long- and short-lived worms in terms of inflection.
    '''
    # Prepare some data.
    my_colors = np.array([[1, 0, 0], [0, 0, 0], [0, 0.5, 0]])
    fade_colors = [(a_color + 3)/4 for a_color in my_colors]
    measuring_colors = np.array(my_colors)
    measuring_colors[1] = fade_colors[1]

    # Set up points to fit to.
    x_start = np.zeros(3)
    x_end = np.empty(3)
    x_end[:] = np.sqrt(2)
    x_mid = np.empty(3)
    x_mid[:] = (np.sqrt(2))/2
    y_start = np.zeros(3)
    y_end = np.zeros(7)
    y_mid = (np.arange(-1, 2)/10*3)[::-1]
    
    # Prepare a rotation matrix.
    an_angle = np.pi/4
    rotation_matrix = np.array([[np.cos(an_angle), -np.sin(an_angle)], [np.sin(an_angle), np.cos(an_angle)]])
    
    # Prepare my "start" data for normalization.
    geometry_dict = computeStatistics.one_d_geometries(adult_df, 'health')
    start_data = geometry_dict['start']
    mean_start = np.mean(start_data)        

    # Plot the "inflection" idea.
    high_number = np.mean(selectData.get_adultspans(adult_df)/24)
    start_health = mean_start
    for i in range(0, len(my_colors)):
        # Fit a polynomial to some points.
        j = -(i+1)
        x_fit = np.array([x_start[j], x_mid[j], x_end[j]])
        y_fit = np.array([y_start[j], y_mid[j], y_end[j]])
        (x2, x1, x0) = np.polyfit(x_fit, y_fit, deg = 2)
        xrange = np.linspace(0, np.sqrt(2), 200)
        yrange = x0 + x1*xrange + x2*(xrange**2)
        my_points = np.array((xrange, yrange)).transpose()
        
        # Rotate the points.
        rotated_points = np.dot(my_points, rotation_matrix.transpose())
        xrange = rotated_points[:, 0]
        yrange = rotated_points[:, 1]*start_health
        xrange = 1 - xrange
        xrange = xrange*high_number
        deviation_plot.plot(xrange, yrange, color = my_colors[i], linewidth = 2)
        if i != 0:
            measured_deviation.plot(xrange, yrange, color = measuring_colors[i], linewidth = 2)
        if i == 1:
            neutral_yrange = yrange
        if i == 2:
            good_yrange = yrange
    
    arrow_lengths = [1.40, 2.75, 3.62, 3.73]
    for i in [8, 6, 4, 2]:
        measured_deviation.arrow(i, neutral_yrange[((10-i)//2)*40], 0, arrow_lengths[(i//2) - 1], head_width = 0.2, head_length = 0.2, fc = 'k', ec = 'k', linewidth = 2, color = 'black')  
        computed_deviation.scatter(1, arrow_lengths[(i//2) - 1], color = 'black', marker = 'o', s = 40)     
    
    xrange = np.linspace(0.5, 1.5, 200)
    yrange = np.linspace(np.mean(arrow_lengths), np.mean(arrow_lengths), 200)
    computed_deviation.plot(xrange, yrange, color = 'black')
    computed_deviation.tick_params(axis = 'x', which = 'both', bottom = 'off', labelbottom = 'off')

    # Label the "Inflection" plots.
    deviation_plot.set_title('Deviation Illustration')
    measured_deviation.set_title('Measuring Deviation')
    computed_deviation.set_title('Computing Average Deviation')
    computed_deviation.set_ylabel('Deviation (Days)')
    computed_deviation.set_ylim([-1.5, 4.5])
    computed_deviation.set_xlim([0, 2])
    my_plots = [deviation_plot, measured_deviation]
    for i in range(0, 2):
        my_plots[i].set_ylim([0, mean_start*1.1])   
        my_plots[i].set_xlim([0, 12])       
        my_plots[i].set_xlabel('Days of Adult Life')
        my_plots[i].set_ylabel('Prognosis (Remaining Days)')

    # Prepare some data.
    (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)

    # Plot the "rescale" hypothesis.
    for i in range(0, len(life_cohorts)):
        xrange = np.linspace(0, 1 + i/50 - 6/50, 200)
        yrange = 1 - xrange + i/50 - 6/50
        rescale_plot.plot(xrange, yrange*mean_start, color = my_colors[i], linewidth = 2)
        
    # Label the "rescale" plot.
    rescale_plot.set_title('Rescale Hypothesis')
    rescale_plot.set_aspect('auto')
    rescale_plot.set_xlabel('Fractional Adult Lifespan')
    rescale_plot.set_ylabel('Prognosis (Remaining Days)')
    rescale_plot.set_ylim([0, mean_start*1.1])  
    rescale_plot.set_xlim([0, 1.1])     

    # Set up points to fit to.
    x_start = np.zeros(7)
    x_end = np.empty(7)
    x_end[:] = np.sqrt(2)
    x_mid = np.empty(7)
    x_mid[:] = (np.sqrt(2))/2
    y_start = np.zeros(7)
    y_end = np.zeros(7)
    y_mid = np.arange(-3, 4)/10
    
    # Prepare a rotation matrix.
    an_angle = np.pi/4
    rotation_matrix = np.array([[np.cos(an_angle), -np.sin(an_angle)], [np.sin(an_angle), np.cos(an_angle)]])
    
    # Plot the "negative" hypothesis.
    for i in range(0, len(life_cohorts)):
        # Fit a polynomial to some points.
        j = -(i+1)
        x_fit = np.array([x_start[j], x_mid[j], x_end[j]])
        y_fit = np.array([y_start[j], y_mid[j], y_end[j]])
        (x2, x1, x0) = np.polyfit(x_fit, y_fit, deg = 2)
        xrange = np.linspace(0, np.sqrt(2), 200)
        yrange = x0 + x1*xrange + x2*(xrange**2)
        my_points = np.array((xrange, yrange)).transpose()
        
        # Rotate the points.
        rotated_points = np.dot(my_points, rotation_matrix.transpose())
        xrange = rotated_points[:, 0]
        yrange = rotated_points[:, 1]
        xrange = 1 - xrange
        negative_plot.plot(xrange, yrange*mean_start, color = my_colors[i], linewidth = 2)
    
    # Label the "negative" plot.
    negative_plot.set_title('Negative Hypothesis')
    negative_plot.set_aspect('auto')
    negative_plot.set_xlabel('Fractional Adult Lifespan')
    negative_plot.set_ylabel('Prognosis (Remaining Days)')
    negative_plot.set_ylim([0, 1.1*mean_start]) 
    negative_plot.set_xlim([0, 1.1])        
    
    # Plot the "positive" hypothesis.
    for i in range(0, len(life_cohorts)):
        # Fit a polynomial to some points.
        x_fit = np.array([x_start[i], x_mid[i], x_end[i]])
        y_fit = np.array([y_start[i], y_mid[i], y_end[i]])
        (x2, x1, x0) = np.polyfit(x_fit, y_fit, deg = 2)
        xrange = np.linspace(0, np.sqrt(2), 200)
        yrange = x0 + x1*xrange + x2*(xrange**2)
        my_points = np.array((xrange, yrange)).transpose()
        
        # Rotate the points.
        rotated_points = np.dot(my_points, rotation_matrix.transpose())
        xrange = rotated_points[:, 0]
        yrange = rotated_points[:, 1]
        xrange = 1 - xrange
        positive_plot.plot(xrange, yrange*mean_start, color = my_colors[i], linewidth = 2)
    
    # Label the "positive" plot.
    positive_plot.set_title('Positive Hypothesis')
    positive_plot.set_aspect('auto')
    positive_plot.set_xlabel('Fractional Adult Lifespan')
    positive_plot.set_ylabel('Prognosis (Remaining Days)')
    positive_plot.set_ylim([0, 1.1*mean_start]) 
    positive_plot.set_xlim([0, 1.1])        
    return (deviation_plot, measured_deviation, computed_deviation, rescale_plot, negative_plot, positive_plot)

def table_from_dataframe(subfigure_plot, pandas_df):
    '''
    Make a table in matplotlib from a pandas dataframe.
    '''
    # Make the table nicely.
    subfigure_plot.axis('off')
    subfigure_plot.xaxis.set_visible(False)
    subfigure_plot.yaxis.set_visible(False)
    my_table = subfigure_plot.table(colWidths = [0.2]*len(list(pandas_df.columns)), cellText = np.array(pandas_df), rowLabels = pandas_df.index, colLabels = pandas_df.columns, loc = 'center', cellLoc = 'center')
    my_table.set_fontsize(14)
    return subfigure_plot

def show_spans(health_traces, health_traces_normed, health_spans, health_spans_normed, adult_df, relative_time = 0.5, a_var = 'health',make_labels=True,cohort_info=None, my_cutoff=None,bar_plot_mode='original',stop_with_death=True):
    '''
    Illustrate how spans work.
    ''' 
    # Make bins of lifespans.
    if cohort_info is None:
        (life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
    else:
        (life_cohorts, bin_lifes, my_bins, my_colors) = cohort_info
    fade_colors = my_colors + ((1 - my_colors)/2)
    my_cohorts = life_cohorts

    # Plot the absolute time trace.
    cohort_transitions = []
    minimum_life_cohorts = []
    flat_data = np.ndarray.flatten(adult_df.mloc(adult_df.worms, [a_var]))
    flat_data = flat_data[~np.isnan(flat_data)]
    if my_cutoff is None:
        my_cutoff = np.percentile(flat_data, (relative_time)*100)
        my_cutoff = adult_df.display_variables(my_cutoff, a_var)[0]
    my_median = np.median(flat_data)
    my_median = adult_df.display_variables(my_median, a_var)[0]
    for i in range(0, len(my_cohorts)):
        if len(my_cohorts[i]) > 0:
            a_cohort = my_cohorts[i]
            cohort_data = adult_df.mloc(adult_df.worms, [a_var])[a_cohort, 0, :]
            #print(cohort_data.size)
            cohort_data = cohort_data[~np.isnan(cohort_data).all(axis = 1)]
            if not stop_with_death:
                cohort_data = np.nanmean(cohort_data,axis=0)
            else:
                cohort_data = np.mean(cohort_data, axis = 0)
            (cohort_data, my_unit, fancy_name) = adult_df.display_variables(cohort_data, a_var)
            cohort_ages = np.array(adult_df.ages[:cohort_data.shape[0]])            

            healthy_mask = (cohort_data > my_cutoff)
            first_unhealthy_index = healthy_mask.argmin()
            unhealthy_mask = (cohort_data < my_cutoff)          
            unhealthy_mask[first_unhealthy_index - 1] = True
            minimum_life_cohorts.append(cohort_ages[unhealthy_mask][-1])

            if healthy_mask.all():
                cohort_transitions.append(cohort_ages[-1])                  
            else:
                cohort_transitions.append(cohort_ages[first_unhealthy_index])                                   
                        
            health_traces.plot(cohort_ages[healthy_mask], cohort_data[healthy_mask], color = my_colors[i], linewidth = 2)
            health_traces.plot(cohort_ages[unhealthy_mask], cohort_data[unhealthy_mask], color = fade_colors[i], linewidth = 2)

    if make_labels: health_traces.set_title(fancy_name + ' Over Time')
    if make_labels: health_traces.set_ylabel('Prognosis (Remaining Days)')  
    if make_labels: health_traces.set_xlabel('Age (Days Post-Maturity)')    
    health_traces.set_ylim([0, my_median*2.0])  
    
    # Prepare my "start" data for normalization.
    geometry_dict = computeStatistics.one_d_geometries(adult_df, 'health')
    (start_data, my_unit, fancy_name) = adult_df.display_variables(geometry_dict['start'], 'health')

    # Plot the relative time trace.
    normed_cohort_transitions = []
    for i in range(0, len(my_cohorts)):
        if len(my_cohorts[i]) > 0:
            a_cohort = my_cohorts[i]
            cohort_data = adult_df.mloc(adult_df.worms, [a_var])[a_cohort, 0, :]
            if not stop_with_death:
                cohort_data = np.nanmean(cohort_data,axis=0)
            else:
                cohort_data = np.mean(cohort_data, axis = 0)
            cohort_data = cohort_data[~np.isnan(cohort_data)]
            (cohort_data, my_unit, fancy_name) = adult_df.display_variables(cohort_data, a_var)
            cohort_data = cohort_data
            cohort_ages = adult_df.ages[:cohort_data.shape[0]]
            cohort_ages = cohort_ages/np.max(cohort_ages)
            
            healthy_mask = (cohort_data > my_cutoff)
            first_unhealthy_index = healthy_mask.argmin()
            unhealthy_mask = (cohort_data < my_cutoff)          
            unhealthy_mask[first_unhealthy_index - 1] = True        

            if healthy_mask.all():
                normed_cohort_transitions.append(cohort_ages[-1])                   
            else:
                normed_cohort_transitions.append(cohort_ages[first_unhealthy_index])                    
            
            health_traces_normed.plot(cohort_ages[healthy_mask], cohort_data[healthy_mask], color = my_colors[i], linewidth = 2)
            health_traces_normed.plot(cohort_ages[unhealthy_mask], cohort_data[unhealthy_mask], color = fade_colors[i], linewidth = 2)

    if make_labels: health_traces_normed.set_title(fancy_name + ' Over Normalized Time')
    if make_labels: health_traces_normed.set_xlabel('Fractional Adult Lifespan')
    if make_labels: health_traces_normed.set_ylabel('Prognosis (Remaining Days)')
    health_traces_normed.set_ylim([0, my_median*2.0])   

    # Plot my horizontal lines. 
    max_adultspans = selectData.get_adultspans(adult_df).max()/24
    xrange = np.linspace(0, max_adultspans, 200)
    yrange = np.empty(200)
    yrange[:] = my_cutoff
    xrange_normed = np.linspace(0, 1, 200)
    yrange_normed = np.empty(200)
    yrange_normed[:] = my_cutoff
    health_traces.plot(xrange, yrange, color = 'black', linestyle = '--')
    health_traces_normed.plot(xrange_normed, yrange_normed, color = 'black', linestyle = '--')

    # Plot bars.
    offset = len([my_obj for my_obj in health_spans.get_children() if type(my_obj) is matplotlib.patches.Rectangle][:-1])/2
    for i in range(0, len(my_cohorts)):
        if bar_plot_mode is 'original':
            health_spans.barh(my_bins[i][0] + 0.1, minimum_life_cohorts[i], color = fade_colors[i], height = 1.8)
            health_spans.barh(my_bins[i][0] + 0.1, cohort_transitions[i], color = my_colors[i], height = 1.8)
        #health_spans.barh(my_bins[i][0] + 0.1, minimum_life_cohorts[i], color = fade_colors[i], height = 0.9*(my_bins[i][1]-my_bins[i][0]))
        #health_spans.barh(my_bins[i][0] + 0.1, cohort_transitions[i], color = my_colors[i], height = 0.9*(my_bins[i][1]-my_bins[i][0]))
        elif bar_plot_mode is 'uniform':
            health_spans.barh(2*(i+offset), minimum_life_cohorts[i], color = fade_colors[i], height =2)
            health_spans.barh(2*(i+offset), cohort_transitions[i], color = my_colors[i], height = 2)
    if make_labels: health_spans.set_title('Absolute Healthspans')
    if make_labels: health_spans.set_xlabel('Days of Adult Life')
    #health_spans.set_xlim([0, max_adultspans]) # Did I modify this previously....?
    if make_labels: health_spans.set_ylabel('Adult Lifespan Cohorts')

    # Plot relative bars.   
    offset = len([my_obj for my_obj in health_spans_normed.get_children() if type(my_obj) is matplotlib.patches.Rectangle][:-1])/2
    #print(offset)
    for i in range(0, len(my_cohorts)):
        if bar_plot_mode is 'original':
            health_spans_normed.barh(my_bins[i][0] + 0.1, 1, color = fade_colors[i], height = 1.8)
            health_spans_normed.barh(my_bins[i][0] + 0.1, normed_cohort_transitions[i], color = my_colors[i], height = 1.8)
        elif bar_plot_mode is 'uniform':
            health_spans_normed.barh(2*(i+offset), 1, color = fade_colors[i], height = 2)
            health_spans_normed.barh(2*(i+offset), normed_cohort_transitions[i], color = my_colors[i], height = 2)
    if make_labels: health_spans_normed.set_title('Relative Healthspans')
    if make_labels: health_spans_normed.set_xlabel('Fractional Adult Lifespan')
    if make_labels: health_spans_normed.set_ylabel('Adult Lifespan Cohorts')
    return (health_traces, health_traces_normed, health_spans, health_spans_normed)

def main():
    return


if __name__ == "__main__":
    main()
