# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:55:27 2016

@author: Willie
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
if sys.platform == 'win32':
	import statsmodels.api
import scipy.stats

import freeimage
from zplib.image import colorize as zplib_image_colorize

import analyzeHealth.computeStatistics as computeStatistics
import analyzeHealth.characterizeTrajectories as characterizeTrajectories
import analyzeHealth.selectData as selectData
import basicOperations.imageOperations as imageOperations

def subfigure_label(plots_list):
	'''
	Label all plots in plots_list with subfigure labels.
	'''
	my_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'	
	for i in range(0, len(plots_list)):	
		plots_list[i].annotate(my_alphabet[i], xy = (0, 1), xycoords = 'axes fraction')
	return plots_list

def remove_box(figure_part):
	figure_part.spines['top'].set_visible(False)
	figure_part.spines['right'].set_visible(False)
	figure_part.tick_params(
	    axis = 'x',          # changes apply to the x-axis
	    which = 'both',      # both major and minor ticks are affected
	    bottom = 'off',      # ticks along the bottom edge are off/on
	    top = 'off',         # ticks along the top edge are off/on
		labelbottom = 'off' # labels along the bottom edge are off/on
	)
	figure_part.tick_params(
	    axis = 'y',          # changes apply to the x-axis
	    which = 'both',      # both major and minor ticks are affected
	    left = 'off',      # ticks along the left edge are off/on
	    right = 'off',         # ticks along the right edge are off/on
		labelleft = 'off' # labels along the bottom edge are off/on
	)	
	return

def remove_whitespace(a_file):
	'''
	Clean up unnecessary whitespace in the image a_file.
	'''
	if a_file[-len('.png'):] == '.png':
		pass
	else:
		return

	my_image = freeimage.read(a_file)
	y_colors = np.min(my_image, axis = 0)
	x_colors = np.min(my_image, axis = 1)
	
	y_white = (y_colors == [255 for i in range(my_image.shape[2])]).all(axis = 1)	
	x_white = (x_colors == [255 for i in range(my_image.shape[2])]).all(axis = 1)

	(first_x, last_x) = (np.argmin(x_white), x_white.shape[0] - np.argmin(x_white[::-1]))
	(first_y, last_y) = (np.argmin(y_white), y_white.shape[0] - np.argmin(y_white[::-1]))

	new_image = my_image[first_x:last_x, first_y:last_y, :]
	freeimage.write(new_image, a_file)
	return
	

def consistent_subgrid_coordinates(total_size, lower_left_corner, my_width = 1, my_height = 1):
	'''
	Let's me place subplots based on Cartesian coordinates, where the cartesian origin is the lower left corner of my overall plot.
	'''
	upper_left_corner = (lower_left_corner[0], lower_left_corner[1] + my_height - 1)
	subplot_part = plt.subplot2grid((total_size[1], total_size[0]), (total_size[1] - upper_left_corner[1] - 1, upper_left_corner[0]), colspan = my_width, rowspan = my_height)
	return subplot_part
	
def pick_fate_worms(adult_df, directory_bolus, pre_death = '1.0', gallery_number = 10):
	'''
	Pick worms to display in fates_gallery.
	'''
	permuted_worms = np.random.permutation(np.array(adult_df.worms))
	random_worms = [[] for i in range(0, len(permuted_worms))]
	hours_before = pre_death.split('.')
	hours_before = float(hours_before[0])*24 + float(hours_before[1])*3
	for i in range(0, len(random_worms)):
		a_worm = permuted_worms[i]
		pre_death_index = np.nanargmin(np.abs(adult_df.mloc(worms = [a_worm], measures = ['ghost_age'])[0, 0, :] + hours_before))
		pre_death_time = adult_df.times[pre_death_index]
		real_time = selectData.closest_real_time(adult_df, a_worm, pre_death_time, egg_mode = True)
		full_worm = str(a_worm + '/' + real_time)
		random_worms[i] = full_worm

	random_worm_chunks = [np.reshape(random_worms[12*i:12*i + 12], (3, 4)) for i in range(0, gallery_number)]
	[fates_gallery(adult_df, directory_bolus, random_worm_chunks[i], i) for i in range(0, len(random_worm_chunks))]

	worm_images = sorted(random_worms)
	worm_images = [directory_bolus.working_directory + os.path.sep + an_image.split('/')[0] + os.path.sep + an_image.split('/')[1] + ' bf.png' for an_image in worm_images]

	return worm_images

def fates_gallery(adult_df, directory_bolus, random_worms, chunk):
	'''	
	Make a gallery of three randomly selected worms from each life cohort at a_time.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 12)
	plots_grid = [[consistent_subgrid_coordinates((12, 12), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 4)] for j in range(0, 3)]	
	labels_grid = [[consistent_subgrid_coordinates((12, 12), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 4)] for j in range(0, 3)]
	
	# Get images.
	closest_times = [[a_worm.split('/')[1] for a_worm in random_worms_group] for random_worms_group in random_worms]
	cohort_images = [[imageOperations.get_worm(directory_bolus, random_worms[i][j].split('/')[0], closest_times[i][j], box_size = 500) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Clean up images and wrap them in colored borders.
	images_max = np.max(np.array(cohort_images))
	dtype_max = 2**16-1
	cohort_images = [[(cohort_images[i][j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]		
	cohort_images = [[imageOperations.border_box(cohort_images[i][j], border_color = [0, 0, 0], border_width = 500//15) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Make the figure itself.
	for i in range(0, len(plots_grid)):
		for j in range(0, len(plots_grid[0])):
			# Display my image.
			plots_grid[i][j].axis('off')
			plots_grid[i][j].imshow((cohort_images[i][j]/2**8).astype('uint8'))
			
			# Label it.
			labels_grid[i][j].axis('off')			
			labels_grid[i][j].annotate(random_worms[i][j].split('/')[0], xy = (0.5, 0.5), ha = 'center', va = 'center')
	
	# Label the top and save the figure.
	my_figure.suptitle('Fates Gallery #' + str(chunk), fontsize = 15)
	plt.savefig(adult_df.save_directory + os.path.sep + 'fates' + '_' + str(chunk) + '_gallery.png', dpi = 300)
	return

def healthgero_span(complete_df, span_cohorts = 'life_cohorts', divided_spans = None, life_normed = False):
	'''
	Make a figure of both average health and gerospans for a number of different variables, with worms grouped by life cohorts.
	'''
	if divided_spans == None:
		divided_spans = computeStatistics.divide_spans(complete_df)
	(span_variables, span_cutoffs) = divided_spans
	
	if span_cohorts == 'life_cohorts':
		(span_cohorts, bin_lifes, my_bins, my_colors) = selectData.life_cohort_bins(complete_df, my_worms = complete_df.worms, bin_width_days = 2)
	elif span_cohorts in complete_df.worms:
		pass

	bin_dictionary = {}
	for i in range(0, len(bin_lifes)):
		a_bin = bin_lifes[i]
		bin_string = ('%0.2f' % a_bin).zfill(5)
		bin_dictionary[bin_string] = span_cohorts[i]
	if '00nan' in bin_dictionary.keys():
		bin_dictionary.pop('00nan')
	average_transitions = {a_bin: np.mean(span_cutoffs[:, bin_dictionary[a_bin]], axis = 1)/8 for a_bin in bin_dictionary.keys()}
	
	key_list = sorted(list(average_transitions.keys()))
	for j in range(0, len(span_variables)):
		(my_figure, my_axes) = plt.subplots()
		for i in range(0, len(key_list)):
			a_key = key_list[i]
			life_checkpoints = np.array([average_transitions[a_key][j], float(a_key)])
			if life_normed:
				life_checkpoints = life_checkpoints/life_checkpoints[-1]
			my_axes.barh(i, life_checkpoints[0], color = 'g')
			my_axes.barh(i, life_checkpoints[1] - life_checkpoints[0], left = life_checkpoints[0],  color = 'b')
		my_axes.set_title(span_variables[j])
	return

def healthgero_transitions(complete_df, span_cohorts = 'life_cohorts', divided_spans = None, life_normed = False):
	'''
	Make a figure of both health-gero transitions for a number of different variables, with worms grouped by life cohorts.	
	'''
	if divided_spans == None:
		divided_spans = computeStatistics.divide_spans(complete_df)
	(span_variables, span_cutoffs) = divided_spans
	
	if span_cohorts == 'life_cohorts':
		(span_cohorts, bin_lifes, my_bins, my_colors) = computeStatistics.life_cohort_bins(complete_df, my_worms = complete_df.worms, bin_width_days = 2)
	elif span_cohorts in complete_df.worms:
		pass

	bin_dictionary = {}
	for i in range(0, len(bin_lifes)):
		a_bin = bin_lifes[i]
		bin_string = ('%0.2f' % a_bin).zfill(5)
		bin_dictionary[bin_string] = span_cohorts[i]
	if '00nan' in bin_dictionary.keys():
		bin_dictionary.pop('00nan')
	average_transitions = {a_bin: np.mean(span_cutoffs[:, bin_dictionary[a_bin]], axis = 1)/8 for a_bin in bin_dictionary.keys()}
	
	if life_normed:
		for a_key in average_transitions.keys():
			average_transitions[a_key] = average_transitions[a_key]/float(a_key)
	
	key_list = sorted(list(average_transitions.keys()))
	color_list = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
	(my_figure, my_axes) = plt.subplots()
	for i in range(0, len(key_list)):
		a_key = key_list[i]
		key_number = float(a_key)
		for j in range(0, len(span_variables)):
			my_axes.scatter(average_transitions[a_key][j], key_number, color = color_list[j])
			if life_normed:
				my_axes.scatter(1, key_number, color = 'black')
			else:				
				my_axes.scatter(key_number, key_number, color = 'black')
		my_axes.axhline(key_number, color = 'black')
		my_axes.set_title(span_variables[j])
	print(np.array([span_variables, color_list]).transpose())
	print('black is death')
	return

def health_heatmap(complete_df, visualization = None, supplied_projection = None):
	'''
	Make a heatmap of health states.
	'''
	my_measures = complete_df.key_measures

	if visualization == None:
		x = np.ndarray.flatten(complete_df.mloc(measures = [my_measures[0]]))
		y = np.ndarray.flatten(complete_df.mloc(measures = [my_measures[1]]))
	elif visualization == 'PCA':
		my_data = np.vstack([complete_df.mloc(measures = my_measures, times = [a_time])[:, :, 0] for a_time in complete_df.times])
		my_data = computeStatistics.project_PCA(my_data, supplied_projection)
		x = np.ndarray.flatten(my_data[:, 0])
		y = np.ndarray.flatten(my_data[:, 1])
	xy_array = np.array([x, y]).transpose()
	xy_array = xy_array[~np.isnan(xy_array).any(axis = 1)]
	x = xy_array[:, 0]
	y = xy_array[:, 1]

	my_worms = complete_df.worms
	(life_cohorts, bin_lifes, my_bins, my_colors) = computeStatistics.life_cohort_bins(complete_df, my_worms = my_worms, bin_width_days = 4)
	
	cohort_PC_trajectories = []
	color_list = []
	for i in range(0, len(life_cohorts)):
		if len(life_cohorts[i]) > 0:
			a_cohort = life_cohorts[i]
			cohort_data = complete_df.mloc(complete_df.worms, my_measures)[a_cohort, :, :]
			cohort_data = np.mean(cohort_data, axis = 0)
			cohort_data = cohort_data.transpose()
			cohort_data = cohort_data[~np.isnan(cohort_data).any(axis = 1)]
			cohort_PC_trajectory = computeStatistics.project_PCA(cohort_data, supplied_projection)
			cohort_PC_trajectories.append(cohort_PC_trajectory)
			color_list.append(my_colors[i])
			plt.plot(cohort_PC_trajectory[:, 0], cohort_PC_trajectory[:, 1], color = my_colors[i])
	
	(heatmap, xedges, yedges) = np.histogram2d(x, y, bins=30)
	plt.hist2d(x, y, bins = 30)
	plt.show()
	return

def show_distribution(complete_df, a_variable, my_times = None):
	'''
	Show the distribution of a_variable at my_times (pooled) 
	'''
	if my_times == None:
		my_times = complete_df.times
	my_data = np.ndarray.flatten(complete_df.mloc(measures = [a_variable], times = my_times))*complete_df.stds[complete_df.measure_indices[a_variable]] + complete_df.means[complete_df.measure_indices[a_variable]]
	if a_variable == 'total_size':
		my_data = 1.7*my_data
	my_data = my_data[~np.isnan(my_data)]
	my_data = pd.Series(my_data)
	my_data.plot(kind = 'kde')
	return

def check_sizes(complete_df, directory_bolus, my_sizes):
	'''
	Compare the sizes of my worms to those of Zach Pincus's 2011 data.
	'''
	pincus_df = characterizeTrajectories.read_pincus(directory_bolus)
	pincus_size_index = pincus_df.measures.index('area')
	pincus_sizes = np.ndarray.flatten(pincus_df.mloc(measures = ['area']))*pincus_df.stds[pincus_size_index] + pincus_df.means[pincus_size_index]
	pincus_sizes = pd.Series(pincus_sizes[~np.isnan(pincus_sizes)])
	my_size_index = complete_df.measures.index('adjusted_size')
	all_my_sizes = (np.ndarray.flatten(complete_df.mloc(measures = ['adjusted_size']))*complete_df.stds[my_size_index] + complete_df.means[my_size_index])*1.7
	all_my_sizes = pd.Series(all_my_sizes[~np.isnan(all_my_sizes)])
	plt.figure()
	all_my_sizes.plot('kde')	
	pincus_sizes.plot('kde')	
	(my_sizes['H_Size']*1.7).plot('kde')
	(my_sizes['F_Size']*1.7).plot('kde')
	return

def survival_curve(complete_df):
	'''
	Generates a survival curve.
	'''	
	lifespans = selectData.get_lifespans(complete_df)/24
	life_histogram = np.histogram(lifespans, density = True, bins = 1000)
	cumulative_death = life_histogram[0]/np.sum(life_histogram[0])
	cumulative_death = np.cumsum(cumulative_death)
	cumulative_life = 1 - cumulative_death
	plt.plot(life_histogram[1][1:], cumulative_life)
	plt.ylabel('% Surviving')
	plt.xlabel('Days Post-Hatch')
	plt.title('spe-9(hc88) Survival in Corrals')
	return

def single_time_scatter(complete_df, a_variable, dependent_variable, a_time):
	'''
	Make a scatter plot of a_variable vs. dependent_variable at a_time.
	'''
	my_data = complete_df.mloc(measures = [a_variable, dependent_variable], times = [a_time])[:, :, 0]
	plt.scatter(my_data[:, 0], my_data[:, 1])
	plt.xlabel(a_variable)
	plt.ylabel(dependent_variable)
	return

def geometry_scatters(complete_df, my_variable):
	'''
	Make scatter plots of geometric differences between trajectories.
	'''
	# Calculate my geometry stuff.
	curve_data = computeStatistics.one_d_geometries(complete_df, my_variable)

	# Self-Inflection
	plt.figure()
	plt.scatter(curve_data['self_inflection'], curve_data['lifespan'])
	plt.ylabel('Lifespan')	
	plt.xlabel('Self-Inflection')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(curve_data['self_inflection'], curve_data['lifespan']))[:5], (0, 0))
	plt.title('Self-Inflection vs. Lifespan')

	# Absolute-Inflection
	plt.figure()
	plt.scatter(curve_data['absolute_inflection'], curve_data['lifespan'])
	plt.ylabel('Lifespan')	
	plt.xlabel('Absolute-Inflection')
	plt.title('Absolute-Inflection vs. Lifespan')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(curve_data['absolute_inflection'], curve_data['lifespan']))[:5], (0, 0))

	# Start
	plt.figure()
	plt.ylabel('Lifespan')	
	plt.xlabel('Starting Health (Predicted Days of Life)')
	plt.scatter(curve_data['start'], curve_data['lifespan'])
	plt.title('Starting Health vs. Lifespan')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(curve_data['start'], curve_data['lifespan']))[:5], (7, 2))
	
	# End
	plt.figure()
	plt.ylabel('Lifespan')	
	plt.xlabel('Ending Health (Predicted Days of Life)')
	plt.scatter(curve_data['end'], curve_data['lifespan'])
	plt.title('Ending Health vs. Lifespan')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(curve_data['end'], curve_data['lifespan']))[:5], (2, 2))

	# Rate
	plt.figure()
	plt.ylabel('Lifespan')	
	plt.xlabel('Average Decline Rate')
	plt.scatter(curve_data['rate'], curve_data['lifespan'])
	plt.title('Decline Rate vs. Lifespan')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(curve_data['rate'], curve_data['lifespan']))[:5], (0, 0))
	
	# Return my data in a dataframe.
	curve_variables = list(sorted(curve_data.keys()))
	curve_df = pd.DataFrame(index = complete_df.worms, columns = curve_variables)
	for a_variable in curve_variables:
		curve_df.loc[:, a_variable] = curve_data[a_variable]
	return curve_df


def cohort_traces(complete_df, general_parameters = {}, mode_parameters = {}):
	'''
	Show the average traces over time of various variables.
	
	general_parameters: {
		'chosen_worms': list of worms to plot,
		'cohort_mode': how to group worms together,
			'lifespan' group worms by their lifespans
			'individual' don't group them, plot individuals
		'my_variables': which variables to plot. Defaults to complete_df.key_measures, but can also be passed an array or a list of strings.
	}
	
	mode_parameters: {
		'bin_width_days': number of days spanned by each bin when 'cohort_mode' is 'lifespan',
		'color_mode': how to color in the traces,
			'life': use Zach Pincus's color spectrum to code cohorts by lifespan
			'highlight': highlight one individual in red while others are plotted in gray; requries that 'highlight_individual' be set
			None: highlight one individual in red while others are plotted in white (invisible); requries that 'highlight_individual' be set
		
		
	}
	'''
	# If a group of worms is not specified, use all worms.
	if 'chosen_worms' not in general_parameters.keys():
		general_parameters['chosen_worms'] = complete_df.worms
	# If a cohort_mode is not specified, use 'lifespan'.
	if 'cohort_mode' not in general_parameters.keys():
		general_parameters['cohort_mode'] = 'lifespan'
	# If my_variables is not specified, use complete_df.key_measures.
	if 'my_variables' not in general_parameters.keys():
		general_parameters['my_variables'] = complete_df.key_measures
	# If combine_cohort is not specified, use 'loess'.
	if 'combine_cohort' not in general_parameters.keys():
		general_parameters['combine_cohort'] = 'mean'
	# If x_normed' not specified, set it to False.
	if 'x_normed' not in general_parameters.keys():
		general_parameters['x_normed'] = False
	# If y_normed' not specified, set it to False.
	if 'y_normed' not in general_parameters.keys():
		general_parameters['y_normed'] = False
	
	# Process my data according to general_parameters['cohort_mode'].
	if general_parameters['cohort_mode'] == 'lifespan': 
		# Set up default values for mode_parameters.		
		if 'bin_width_days' not in mode_parameters.keys():
			mode_parameters['bin_width_days'] = 2		
		if 'color_mode' not in mode_parameters.keys():
			mode_parameters['color_mode'] = 'life'
		# Make bins of lifespans.		
		if complete_df.adult_only:			
			(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(complete_df, my_worms = general_parameters['chosen_worms'], bin_width_days = mode_parameters['bin_width_days'])
		else:
			(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.life_cohort_bins(complete_df, my_worms = general_parameters['chosen_worms'], bin_width_days = mode_parameters['bin_width_days'])
		
		my_cohorts = [life_cohorts[i] for i in range(0, len(life_cohorts)) if len(life_cohorts[i]) > 0]	
	# Just plot individuals.
	elif general_parameters['cohort_mode'] == 'individual':
		# Set up default values for mode_parameters.		
		if 'highlight_individual' not in mode_parameters.keys():
			mode_parameters['highlight_individual'] = None
		if 'color_mode' not in mode_parameters.keys():
			mode_parameters['color_mode'] = 'life'
		# Select my cohort.
		my_cohorts = [[complete_df.worm_indices[a_worm]] for a_worm in general_parameters['chosen_worms']]

	# Figure out coloring.
	if mode_parameters['color_mode'] == 'life':
		if complete_df.adult_only:
			lifespans = selectData.get_adultspans(complete_df)
		else:
			lifespans = selectData.get_lifespans(complete_df)
		scaled_adult = np.array(lifespans)
		scaled_adult = scaled_adult - np.min(scaled_adult)
		scaled_adult = scaled_adult/np.max(scaled_adult)
		scaled_adult = np.array(scaled_adult)
		cohort_lifes = np.array([np.mean(scaled_adult[my_cohorts[i]]) for i in range(0, len(my_cohorts))])
		my_colors = zplib_image_colorize.color_map(cohort_lifes)
		my_colors = my_colors/255
	elif mode_parameters['color_mode'] == 'gray':
		highlight_index = complete_df.worm_indices[mode_parameters['highlight_individual']]
		my_colors = np.zeros(len(my_cohorts)).astype('str')
		my_colors[:] = 'gray'
		my_colors[highlight_index] = 'red'
		(my_cohorts[-1], my_cohorts[highlight_index]) = (my_cohorts[highlight_index], my_cohorts[-1])
		(my_colors[-1], my_colors[highlight_index]) = (my_colors[highlight_index], my_colors[-1])
	elif mode_parameters['color_mode'] == None:
		highlight_index = complete_df.worm_indices[mode_parameters['highlight_individual']]
		my_colors = np.zeros(len(my_cohorts)).astype('str')
		my_colors[:] = 'white'
		my_colors[highlight_index] = 'red'
		(my_cohorts[-1], my_cohorts[highlight_index]) = (my_cohorts[highlight_index], my_cohorts[-1])
		(my_colors[-1], my_colors[highlight_index]) = (my_colors[highlight_index], my_colors[-1])

	# Prepare my data for plotting.
	plot_data = {}
	plot_ages = {}
	fancy_names = {}
	my_units = {}
	for a_var in general_parameters['my_variables']:
		plot_data[a_var] = []
		plot_ages[a_var] = []
		for i in range(0, len(my_cohorts)):
			a_cohort = my_cohorts[i]
			# Get the raw data for my cohort.		
			if a_var in complete_df.measures:
				cohort_data = complete_df.mloc(measures = [a_var])[a_cohort, 0, :]
			else:
				cohort_data = complete_df.extra_data[a_var][a_cohort, :]
			# Combine the cohort data into a single representative trajectory.				
			if general_parameters['combine_cohort'] == 'mean':
				cohort_data = np.mean(cohort_data, axis = 0)
				cohort_data = cohort_data[~np.isnan(cohort_data)]
				cohort_ages = np.array(complete_df.ages[:cohort_data.shape[0]])
			elif general_parameters['combine_cohort'] == 'loess':
				flat_data = np.ndarray.flatten(cohort_data)
				cohort_ages = complete_df.mloc(complete_df.worms, ['age'])[a_cohort, 0, :]
				flat_ages = np.ndarray.flatten(cohort_ages)
				cohort_data = statsmodels.api.nonparametric.lowess(flat_data, flat_ages, frac=0.1)
				cohort_ages = cohort_data[:, 0]
				cohort_data = cohort_data[:, 1]
			
			# Clean up the data and store it for use later.
			(cohort_data, my_unit, fancy_name) = complete_df.display_variables(cohort_data, a_var)
			if np.max(cohort_data) == 0:
				print(a_cohort)
			if general_parameters['x_normed']:
				cohort_ages = cohort_ages/np.max(cohort_ages)
			if general_parameters['y_normed']:
				cohort_data = cohort_data - np.min(cohort_data)
				cohort_data = cohort_data/np.max(np.max(cohort_data))
			plot_data[a_var].append(cohort_data)
			plot_ages[a_var].append(cohort_ages)
			fancy_names[a_var] = fancy_name
			my_units[a_var] = my_unit

	# Plot the actual stuff.
	for a_var in general_parameters['my_variables']:
		plt.figure()
		for i in range(0, len(my_cohorts)):
			plt.plot(plot_ages[a_var][i], plot_data[a_var][i], color = my_colors[i])
		plt.title(fancy_names[a_var] + ' Over Time')
		plt.ylabel(my_units[a_var])	
		if general_parameters['x_normed']:
			if complete_df.adult_only:
				plt.xlabel('Fraction of Adult Lifespan')
			else:
				plt.xlabel('Fraction of Lifespan')			
		else:
			if complete_df.adult_only:
				plt.xlabel('Adult Age (Days Post-Maturity)')
			else:
				plt.xlabel('Age (Days Post-Hatch)')
		plt.savefig(complete_df.save_directory + os.path.sep + a_var + '_trace.png')
	return

def lifespan_predictivity(complete_df, variables = None):
	'''
	Show how well variables predict lifespan over time.
	'''
	if variables == None:
		variables = complete_df.key_measures
	lifespans = computeStatistics.get_lifespans(complete_df)
	for a_var in variables:
		my_data = complete_df.mloc(complete_df.worms, [a_var])
		my_derivatives = computeStatistics.differentiate(my_data)
		my_integrals = computeStatistics.integrate(my_data)
		my_correlations = np.zeros((3, my_data.shape[2]))
		for i in range(0, my_data.shape[2]):
			my_correlations[0, i] = computeStatistics.quick_pearson(lifespans, my_derivatives[:, 0, i])
			my_correlations[1, i] = computeStatistics.quick_pearson(lifespans, my_data[:, 0, i])
			my_correlations[2, i] = computeStatistics.quick_pearson(lifespans, my_integrals[:, 0, i])
		plt.figure()
		ddt = plt.plot(complete_df.ages[:my_data.shape[2]], my_correlations[0, :], label = 'Derivative')[0]
		fdt = plt.plot(complete_df.ages[:my_data.shape[2]], my_correlations[1, :], label = 'Present Value')[0]
		sdt = plt.plot(complete_df.ages[:my_data.shape[2]], my_correlations[2, :], label = 'Integral')[0]
		plt.title(complete_df.display_names(a_var) + ' Lifespan Correlation Over Time')
		plt.xlabel('Age (Days Post-Hatch)')
		plt.ylabel('Pearson r^2')		
		plt.legend(handles = [ddt, fdt, sdt])
		plt.savefig(complete_df.save_directory + os.path.sep + a_var + '_predictable_lifespan.png')
	return
	
def regression_scatters(complete_df, independent_variables = None, dependent_variable = 'ghost_age', my_times = None):
	'''
	Check regression r^2 values for independent_variables against dependent_variable. This assumes that dependent variable is a time in hours.
	'''
	if my_times == None:
		my_times = complete_df.times
	if independent_variables == None:
		independent_variables = complete_df.key_measures
	
	flat_dependent = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable], times = my_times))/24
	dependent_name = complete_df.display_names(dependent_variable)
	regression_dict = {}
	for a_var in independent_variables:
		flat_var = np.ndarray.flatten(complete_df.mloc(measures = [a_var], times = my_times))
		(flat_var, my_unit, fancy_name) = complete_df.display_variables(flat_var, a_var)
		together_var = np.array((flat_var, flat_dependent)).transpose()
		together_var = together_var[~np.isnan(together_var).any(axis=1)]
		plt.figure()
		plt.title(fancy_name + ' vs. ' + dependent_name)
		plt.xlabel(my_unit)
		plt.ylabel('Days')		
		plt.scatter(together_var[:, 0], together_var[:, 1])
		plt.savefig(complete_df.save_directory + os.path.sep + a_var + 'scatter.png')
	return regression_dict	

def regression_time_progression(complete_df, independent_variables = None, dependent_variable = 'ghost_age'):
	'''
	Makes plots of how much dependent_variable is predictable from independent_variables over time.
	'''
	# Set up some variables I need.
	if independent_variables == None:
		independent_variables = list(complete_df.key_measures)
		independent_variables.append('age')
	independent_indices = [complete_df.measure_indices[a_measure] for a_measure in independent_variables]
	regression_array = np.zeros((len(independent_indices), complete_df.data.shape[2]))
	dependent_name = complete_df.display_names(dependent_variable)
	
	# Actually calculate the r^2 values.
	for i in range(0, complete_df.data.shape[2]):
		life_array = complete_df.mloc(measures = [dependent_variable])[:, 0, i]
		for j in range(0, len(independent_indices)):
			variable_array = complete_df.data[:, independent_indices[j], i]
			my_correlation = computeStatistics.quick_pearson(variable_array, life_array)
			regression_array[j, i] = my_correlation

	# Plot and save my stuff.
	for i in range(0, len(independent_indices)):
		plt.figure()
		plt.plot(complete_df.ages, regression_array[i, :])
		fancy_name = complete_df.display_names(independent_variables[i])
		plt.title(fancy_name + ' vs. ' + dependent_name)
		plt.xlabel('Days')		
		plt.ylabel('Pearson r^2')
		plt.savefig(complete_df.save_directory + os.path.sep + fancy_name + '-' + dependent_variable + '_regression.png')
	return

def multiple_regression_progression_cohorts(complete_df, independent_variables = None, dependent_variable = 'ghost_age'):
	'''
	Makes a plot of the multiple regression r^2 over time.
	'''
	# Set up some variables I need.
	my_times = complete_df.times
	if independent_variables == None:
		independent_variables = complete_df.key_measures
	my_worms = complete_df.worms
	(life_cohorts, bin_lifes, my_bins, my_colors) = computeStatistics.life_cohort_bins(complete_df, my_worms = my_worms, bin_width_days = 4)
	
	flat_dependent = np.ndarray.flatten(complete_df.mloc(measures = [dependent_variable], times = my_times))/24
	all_flats = []
	for a_var in independent_variables:
		flat_var = np.ndarray.flatten(complete_df.mloc(measures = [a_var], times = my_times))
		all_flats.append(flat_var)
	all_flats = np.array(all_flats).transpose()

	keep_indices = ~np.isnan(all_flats).any(axis = 1)
	all_flats = all_flats[keep_indices]
	flat_dependent = flat_dependent[keep_indices]

	(predicted_result, dependent_variable, multiple_regression_weights, my_intercept) = computeStatistics.multiple_regression(all_flats, flat_dependent)

	
	# Plot the actual stuff.
	plt.figure()
	for i in range(0, len(life_cohorts)):
		if len(life_cohorts[i]) > 0:
			full_data = []
			full_ages = []
			for j in range(0, len(complete_df.times)):
				a_cohort = life_cohorts[i]
				cohort_data = complete_df.mloc(complete_df.worms, measures = independent_variables, times = [complete_df.times[j]])[a_cohort, :, 0]
				cohort_data = my_intercept + np.dot(cohort_data, multiple_regression_weights)
				cohort_ages = complete_df.mloc(complete_df.worms, ['age'], [complete_df.times[j]])[a_cohort, 0, 0]
				full_data.append(cohort_data)
				full_ages.append(cohort_ages)
			full_data = np.ndarray.flatten(np.array(full_data))
			full_ages = np.ndarray.flatten(np.array(full_ages))

			cohort_data = statsmodels.api.nonparametric.lowess(full_data, full_ages, frac=0.1)
			cohort_ages = cohort_data[:, 0]
			cohort_data = cohort_data[:, 1]
			plt.plot(cohort_ages, cohort_data, color = my_colors[i])
	plt.savefig(complete_df.save_directory + os.path.sep + a_var + '_trace.png')
	
	

	# Plot and save my stuff.
	plt.title('Multiple Regression Predicted Life Remaining')
	plt.xlabel('Days')		
	plt.ylabel('Pearson r^2')
	plt.savefig(complete_df.save_directory + os.path.sep + 'multiple_regression.png')
	return

def multiple_regression_progression(complete_df, independent_variables = None, dependent_variable = 'ghost_age'):
	'''
	Makes a plot of the multiple regression r^2 over time.
	'''
	# Set up some variables I need.
	if independent_variables == None:
		independent_variables = complete_df.key_measures
	independent_indices = [complete_df.measure_indices[a_measure] for a_measure in independent_variables]
	regression_array = np.zeros((complete_df.data.shape[2]))
	
	# Actually calculate the r^2 values.
	for i in range(15, complete_df.data.shape[2]):
		print(i)
		life_array = complete_df.mloc(measures = [dependent_variable])[:, 0, i]
		variable_array = complete_df.data[:, independent_indices, i]
		my_correlation = computeStatistics.quick_multiple_pearson(variable_array, life_array)
		regression_array[i] = my_correlation

	# Plot and save my stuff.
	plt.figure()
	plt.plot(complete_df.ages, regression_array)
	plt.title('Multiple Regression r^2 Over Time')
	plt.xlabel('Days')		
	plt.ylabel('Pearson r^2')
	plt.savefig(complete_df.save_directory + os.path.sep + 'multiple_regression.png')
	return

def cohort_PCtraces(complete_df, trajectory_PCA, bin_width_days = 2, my_variables = None, my_worms = None):
	'''
	Show the average traces over time of various variables.
	'''
	# Make bins of lifespans.
	if my_worms == None:
		my_worms = complete_df.worms
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.life_cohort_bins(complete_df, my_worms = my_worms, bin_width_days = bin_width_days)
	
	# Plot the actual stuff.
	if my_variables == None:
		my_variables = trajectory_PCA.measures
	plt.figure()
	cohort_PC_trajectories = []
	color_list = []
	for i in range(0, len(life_cohorts)):
		if len(life_cohorts[i]) > 0:
			a_cohort = life_cohorts[i]
			cohort_data = complete_df.mloc(complete_df.worms, my_variables)[a_cohort, :, :]
			cohort_data = np.mean(cohort_data, axis = 0)
			cohort_data = cohort_data.transpose()
			cohort_data = cohort_data[~np.isnan(cohort_data).any(axis = 1)]
			cohort_PC_trajectory = computeStatistics.project_PCA(cohort_data, trajectory_PCA)
			cohort_PC_trajectories.append(cohort_PC_trajectory)
			color_list.append(my_colors[i])
			plt.plot(cohort_PC_trajectory[:, 0], cohort_PC_trajectory[:, 1], color = my_colors[i])
	plt.title('PC Trajectories of Lifespan Cohorts')
	plt.xlabel('PC1')
	plt.ylabel('PC2')	
	plt.savefig(complete_df.save_directory + os.path.sep + 'PC_cohort_trace.png')
	show_velocity(cohort_PC_trajectories, color_list, complete_df)
	show_acceleration(cohort_PC_trajectories, color_list, complete_df)
	return cohort_PC_trajectories

def show_velocity(pc_trajectories, colors, complete_df):
	'''
	Show a crude velocity.
	'''
	plt.figure()
	my_velocities = [computeStatistics.differentiate_trajectory(a_trajectory) for a_trajectory in pc_trajectories]
	for i in range(0, len(my_velocities)):
		plt.plot(my_velocities[i][:, 0], my_velocities[i][:, 1], color = colors[i])
	plt.title('PC Velocities of Lifespan Cohorts')
	plt.xlabel('PC1')
	plt.ylabel('PC2')	
	plt.savefig(complete_df.save_directory + os.path.sep + 'dPC_cohort_trace.png')
	return

def show_acceleration(pc_trajectories, colors, complete_df):
	'''
	Show a crude acceleration.
	'''
	plt.figure()
	my_accelerations = [computeStatistics.differentiate_trajectory(computeStatistics.differentiate_trajectory(a_trajectory)) for a_trajectory in pc_trajectories]
	for i in range(0, len(my_accelerations)):
		plt.plot(my_accelerations[i][:, 0], my_accelerations[i][:, 1], color = colors[i])	
	plt.title('PC Accelerations of Lifespan Cohorts')
	plt.xlabel('PC1')
	plt.ylabel('PC2')	
	plt.savefig(complete_df.save_directory + os.path.sep + 'ddPC_cohort_trace.png')
	return

def development_vs_adultspan(complete_df):
	'''
	Check developmental timing vs. length of adult life.
	'''
	my_frame = pd.DataFrame(index = complete_df.worms, columns = ['developspan', 'adultspan'])
	for a_worm in complete_df.worms:
		age_data = complete_df.mloc([a_worm], ['age', 'egg_age', 'ghost_age'])[0, :, 0]
		my_frame.loc[a_worm, 'adultspan'] = np.abs(age_data[2] - age_data[1])/24
		my_frame.loc[a_worm, 'developspan'] = np.abs(age_data[1] - age_data[0])/24
		if my_frame.loc[a_worm, 'developspan'] < 0:
			raise BaseException(a_worm)
	my_frame.loc[:, 'lifespan'] = my_frame.loc[:, 'developspan'] + my_frame.loc[:, 'adultspan'] 
	my_frame = my_frame.astype('float')

	plt.figure()
	plt.scatter(my_frame.loc[:, 'developspan'], my_frame.loc[:, 'adultspan'])
	plt.title('Development vs. Adultspan')
	plt.xlabel('Days of Development')
	plt.ylabel('Days of Adult Life')	
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(my_frame.loc[:, 'developspan'], my_frame.loc[:, 'adultspan']))[:5], (2, 2))	
	plt.savefig(complete_df.save_directory + os.path.sep + 'development_vs_adultspan.png')
		
	plt.figure()
	plt.scatter(my_frame.loc[:, 'developspan'], my_frame.loc[:, 'lifespan'])
	plt.title('Development vs. Lifespan')
	plt.xlabel('Days of Development')
	plt.ylabel('Days of Life')	
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(my_frame.loc[:, 'developspan'], my_frame.loc[:, 'lifespan']))[:5], (2, 2))	
	plt.savefig(complete_df.save_directory + os.path.sep + 'development_vs_lifespan.png')

	plt.figure()
	plt.scatter(my_frame.loc[:, 'adultspan'], my_frame.loc[:, 'lifespan'])
	plt.title('Adultspan vs. Lifespan')
	plt.xlabel('Days of Adult Life')	
	plt.ylabel('Days of Lifespan')
	plt.annotate('r^2 = ' + str(computeStatistics.quick_pearson(my_frame.loc[:, 'adultspan'], my_frame.loc[:, 'lifespan']))[:5], (2, 2))	
	plt.savefig(complete_df.save_directory + os.path.sep + 'adultspan_vs_lifespan.png')
	
	return my_frame.astype('float')

def visualize_trajectory_differences(pc_trajectories, complete_df, mode = 'Extreme-Normalized'):
	'''
	Visualize trajectory differences as I normalized and close the distance.
	'''
	if mode.split('-')[0] == 'Extreme':
		short_lived = pc_trajectories[0]
		long_lived = pc_trajectories[-1]
	if mode.split('-')[1] == 'Normalized':
		short_lived = characterizeTrajectories.resample(short_lived, long_lived.shape[0])
	if mode.split('-')[1] == 'Truncated':
		long_lived = long_lived[:short_lived.shape[0], :]
		
	(variation_dictionary, trajectory_a, trajectory_b_list) = characterizeTrajectories.separate_variation_sources(long_lived, short_lived)
	title_list = ['Original Difference', 'Start-Normalized', 'Rate, Start-Normalized', 'Rate, Start, Angle-Normalized']
	variation_names = ['Nothing', 'Start', 'Rate', 'Direction']
	variation_list = [variation_dictionary[a_key] for a_key in ['Start', 'Rate', 'Direction']]
	variation_list.insert(0, 0)
	for i in range(0, len(trajectory_b_list)):
		plt.figure()
		plt.plot(trajectory_a[:, 0], trajectory_a[:, 1])
		plt.plot(trajectory_b_list[i][:, 0], trajectory_b_list[i][:, 1])
		plt.annotate(str(100*variation_list[i])[:5] + '% Variation from ' + variation_names[i], (0, 0))
		plt.title(title_list[i])
		plt.xlabel('PC1')
		plt.ylabel('PC2')	
		plt.savefig(complete_df.save_directory + os.path.sep + title_list[i] + '.png')
	return

def reconstruction_error(complete_df, cohort_mode = True):
	'''
	Make some bar graphs of reconstruction error vs. number of dimensions for different manifold learning methods.
	'''
	dimensionality_df = pd.DataFrame(index = [1, 2, 3, 4, 5], columns = ['isomap', 'modified_locally_linear_embedding', 'hessian_locally_linear_embedding', 'ltsa_locally_linear_embedding', 'locally_linear_embedding', 'mds'])
	for a_method in dimensionality_df.columns:
		for a_dimension in dimensionality_df.index:
			
			dimensionality_df.loc[a_dimension, a_method] = computeStatistics.check_dimensionality(complete_df, a_method, a_dimension, cohort_mode = cohort_mode).willie_error

	for a_method in dimensionality_df.columns:
		(my_figure, my_axes) = plt.subplots()
		my_axes.set_title(a_method + ' reconstruction error')
		for a_dimension in dimensionality_df.index:
			my_axes.bar(a_dimension, dimensionality_df.loc[a_dimension, a_method], color = 'blue')
	plt.show()
	return dimensionality_df

# Set up a color scheme.
viridis_data = [
	[0.267004, 0.004874, 0.329415],
	[0.26851, 0.009605, 0.335427],
	[0.269944, 0.014625, 0.341379],
	[0.271305, 0.019942, 0.347269],
	[0.272594, 0.025563, 0.353093],
	[0.273809, 0.031497, 0.358853],
	[0.274952, 0.037752, 0.364543],
	[0.276022, 0.044167, 0.370164],
	[0.277018, 0.050344, 0.375715],
	[0.277941, 0.056324, 0.381191],
	[0.278791, 0.062145, 0.386592],
	[0.279566, 0.067836, 0.391917],
	[0.280267, 0.073417, 0.397163],
	[0.280894, 0.078907, 0.402329],
	[0.281446, 0.08432, 0.407414],
	[0.281924, 0.089666, 0.412415],
	[0.282327, 0.094955, 0.417331],
	[0.282656, 0.100196, 0.42216],
	[0.28291, 0.105393, 0.426902],
	[0.283091, 0.110553, 0.431554],
	[0.283197, 0.11568, 0.436115],
	[0.283229, 0.120777, 0.440584],
	[0.283187, 0.125848, 0.44496],
	[0.283072, 0.130895, 0.449241],
	[0.282884, 0.13592, 0.453427],
	[0.282623, 0.140926, 0.457517],
	[0.28229, 0.145912, 0.46151],
	[0.281887, 0.150881, 0.465405],
	[0.281412, 0.155834, 0.469201],
	[0.280868, 0.160771, 0.472899],
	[0.280255, 0.165693, 0.476498],
	[0.279574, 0.170599, 0.479997],
	[0.278826, 0.17549, 0.483397],
	[0.278012, 0.180367, 0.486697],
	[0.277134, 0.185228, 0.489898],
	[0.276194, 0.190074, 0.493001],
	[0.275191, 0.194905, 0.496005],
	[0.274128, 0.199721, 0.498911],
	[0.273006, 0.20452, 0.501721],
	[0.271828, 0.209303, 0.504434],
	[0.270595, 0.214069, 0.507052],
	[0.269308, 0.218818, 0.509577],
	[0.267968, 0.223549, 0.512008],
	[0.26658, 0.228262, 0.514349],
	[0.265145, 0.232956, 0.516599],
	[0.263663, 0.237631, 0.518762],
	[0.262138, 0.242286, 0.520837],
	[0.260571, 0.246922, 0.522828],
	[0.258965, 0.251537, 0.524736],
	[0.257322, 0.25613, 0.526563],
	[0.255645, 0.260703, 0.528312],
	[0.253935, 0.265254, 0.529983],
	[0.252194, 0.269783, 0.531579],
	[0.250425, 0.27429, 0.533103],
	[0.248629, 0.278775, 0.534556],
	[0.246811, 0.283237, 0.535941],
	[0.244972, 0.287675, 0.53726],
	[0.243113, 0.292092, 0.538516],
	[0.241237, 0.296485, 0.539709],
	[0.239346, 0.300855, 0.540844],
	[0.237441, 0.305202, 0.541921],
	[0.235526, 0.309527, 0.542944],
	[0.233603, 0.313828, 0.543914],
	[0.231674, 0.318106, 0.544834],
	[0.229739, 0.322361, 0.545706],
	[0.227802, 0.326594, 0.546532],
	[0.225863, 0.330805, 0.547314],
	[0.223925, 0.334994, 0.548053],
	[0.221989, 0.339161, 0.548752],
	[0.220057, 0.343307, 0.549413],
	[0.21813, 0.347432, 0.550038],
	[0.21621, 0.351535, 0.550627],
	[0.214298, 0.355619, 0.551184],
	[0.212395, 0.359683, 0.55171],
	[0.210503, 0.363727, 0.552206],
	[0.208623, 0.367752, 0.552675],
	[0.206756, 0.371758, 0.553117],
	[0.204903, 0.375746, 0.553533],
	[0.203063, 0.379716, 0.553925],
	[0.201239, 0.38367, 0.554294],
	[0.19943, 0.387607, 0.554642],
	[0.197636, 0.391528, 0.554969],
	[0.19586, 0.395433, 0.555276],
	[0.1941, 0.399323, 0.555565],
	[0.192357, 0.403199, 0.555836],
	[0.190631, 0.407061, 0.556089],
	[0.188923, 0.41091, 0.556326],
	[0.187231, 0.414746, 0.556547],
	[0.185556, 0.41857, 0.556753],
	[0.183898, 0.422383, 0.556944],
	[0.182256, 0.426184, 0.55712],
	[0.180629, 0.429975, 0.557282],
	[0.179019, 0.433756, 0.55743],
	[0.177423, 0.437527, 0.557565],
	[0.175841, 0.44129, 0.557685],
	[0.174274, 0.445044, 0.557792],
	[0.172719, 0.448791, 0.557885],
	[0.171176, 0.45253, 0.557965],
	[0.169646, 0.456262, 0.55803],
	[0.168126, 0.459988, 0.558082],
	[0.166617, 0.463708, 0.558119],
	[0.165117, 0.467423, 0.558141],
	[0.163625, 0.471133, 0.558148],
	[0.162142, 0.474838, 0.55814],
	[0.160665, 0.47854, 0.558115],
	[0.159194, 0.482237, 0.558073],
	[0.157729, 0.485932, 0.558013],
	[0.15627, 0.489624, 0.557936],
	[0.154815, 0.493313, 0.55784],
	[0.153364, 0.497, 0.557724],
	[0.151918, 0.500685, 0.557587],
	[0.150476, 0.504369, 0.55743],
	[0.149039, 0.508051, 0.55725],
	[0.147607, 0.511733, 0.557049],
	[0.14618, 0.515413, 0.556823],
	[0.144759, 0.519093, 0.556572],
	[0.143343, 0.522773, 0.556295],
	[0.141935, 0.526453, 0.555991],
	[0.140536, 0.530132, 0.555659],
	[0.139147, 0.533812, 0.555298],
	[0.13777, 0.537492, 0.554906],
	[0.136408, 0.541173, 0.554483],
	[0.135066, 0.544853, 0.554029],
	[0.133743, 0.548535, 0.553541],
	[0.132444, 0.552216, 0.553018],
	[0.131172, 0.555899, 0.552459],
	[0.129933, 0.559582, 0.551864],
	[0.128729, 0.563265, 0.551229],
	[0.127568, 0.566949, 0.550556],
	[0.126453, 0.570633, 0.549841],
	[0.125394, 0.574318, 0.549086],
	[0.124395, 0.578002, 0.548287],
	[0.123463, 0.581687, 0.547445],
	[0.122606, 0.585371, 0.546557],
	[0.121831, 0.589055, 0.545623],
	[0.121148, 0.592739, 0.544641],
	[0.120565, 0.596422, 0.543611],
	[0.120092, 0.600104, 0.54253],
	[0.119738, 0.603785, 0.5414],
	[0.119512, 0.607464, 0.540218],
	[0.119423, 0.611141, 0.538982],
	[0.119483, 0.614817, 0.537692],
	[0.119699, 0.61849, 0.536347],
	[0.120081, 0.622161, 0.534946],
	[0.120638, 0.625828, 0.533488],
	[0.12138, 0.629492, 0.531973],
	[0.122312, 0.633153, 0.530398],
	[0.123444, 0.636809, 0.528763],
	[0.12478, 0.640461, 0.527068],
	[0.126326, 0.644107, 0.525311],
	[0.128087, 0.647749, 0.523491],
	[0.130067, 0.651384, 0.521608],
	[0.132268, 0.655014, 0.519661],
	[0.134692, 0.658636, 0.517649],
	[0.137339, 0.662252, 0.515571],
	[0.14021, 0.665859, 0.513427],
	[0.143303, 0.669459, 0.511215],
	[0.146616, 0.67305, 0.508936],
	[0.150148, 0.676631, 0.506589],
	[0.153894, 0.680203, 0.504172],
	[0.157851, 0.683765, 0.501686],
	[0.162016, 0.687316, 0.499129],
	[0.166383, 0.690856, 0.496502],
	[0.170948, 0.694384, 0.493803],
	[0.175707, 0.6979, 0.491033],
	[0.180653, 0.701402, 0.488189],
	[0.185783, 0.704891, 0.485273],
	[0.19109, 0.708366, 0.482284],
	[0.196571, 0.711827, 0.479221],
	[0.202219, 0.715272, 0.476084],
	[0.20803, 0.718701, 0.472873],
	[0.214, 0.722114, 0.469588],
	[0.220124, 0.725509, 0.466226],
	[0.226397, 0.728888, 0.462789],
	[0.232815, 0.732247, 0.459277],
	[0.239374, 0.735588, 0.455688],
	[0.24607, 0.73891, 0.452024],
	[0.252899, 0.742211, 0.448284],
	[0.259857, 0.745492, 0.444467],
	[0.266941, 0.748751, 0.440573],
	[0.274149, 0.751988, 0.436601],
	[0.281477, 0.755203, 0.432552],
	[0.288921, 0.758394, 0.428426],
	[0.296479, 0.761561, 0.424223],
	[0.304148, 0.764704, 0.419943],
	[0.311925, 0.767822, 0.415586],
	[0.319809, 0.770914, 0.411152],
	[0.327796, 0.77398, 0.40664],
	[0.335885, 0.777018, 0.402049],
	[0.344074, 0.780029, 0.397381],
	[0.35236, 0.783011, 0.392636],
	[0.360741, 0.785964, 0.387814],
	[0.369214, 0.788888, 0.382914],
	[0.377779, 0.791781, 0.377939],
	[0.386433, 0.794644, 0.372886],
	[0.395174, 0.797475, 0.367757],
	[0.404001, 0.800275, 0.362552],
	[0.412913, 0.803041, 0.357269],
	[0.421908, 0.805774, 0.35191],
	[0.430983, 0.808473, 0.346476],
	[0.440137, 0.811138, 0.340967],
	[0.449368, 0.813768, 0.335384],
	[0.458674, 0.816363, 0.329727],
	[0.468053, 0.818921, 0.323998],
	[0.477504, 0.821444, 0.318195],
	[0.487026, 0.823929, 0.312321],
	[0.496615, 0.826376, 0.306377],
	[0.506271, 0.828786, 0.300362],
	[0.515992, 0.831158, 0.294279],
	[0.525776, 0.833491, 0.288127],
	[0.535621, 0.835785, 0.281908],
	[0.545524, 0.838039, 0.275626],
	[0.555484, 0.840254, 0.269281],
	[0.565498, 0.84243, 0.262877],
	[0.575563, 0.844566, 0.256415],
	[0.585678, 0.846661, 0.249897],
	[0.595839, 0.848717, 0.243329],
	[0.606045, 0.850733, 0.236712],
	[0.616293, 0.852709, 0.230052],
	[0.626579, 0.854645, 0.223353],
	[0.636902, 0.856542, 0.21662],
	[0.647257, 0.8584, 0.209861],
	[0.657642, 0.860219, 0.203082],
	[0.668054, 0.861999, 0.196293],
	[0.678489, 0.863742, 0.189503],
	[0.688944, 0.865448, 0.182725],
	[0.699415, 0.867117, 0.175971],
	[0.709898, 0.868751, 0.169257],
	[0.720391, 0.87035, 0.162603],
	[0.730889, 0.871916, 0.156029],
	[0.741388, 0.873449, 0.149561],
	[0.751884, 0.874951, 0.143228],
	[0.762373, 0.876424, 0.137064],
	[0.772852, 0.877868, 0.131109],
	[0.783315, 0.879285, 0.125405],
	[0.79376, 0.880678, 0.120005],
	[0.804182, 0.882046, 0.114965],
	[0.814576, 0.883393, 0.110347],
	[0.82494, 0.88472, 0.106217],
	[0.83527, 0.886029, 0.102646],
	[0.845561, 0.887322, 0.099702],
	[0.85581, 0.888601, 0.097452],
	[0.866013, 0.889868, 0.095953],
	[0.876168, 0.891125, 0.09525],
	[0.886271, 0.892374, 0.095374],
	[0.89632, 0.893616, 0.096335],
	[0.906311, 0.894855, 0.098125],
	[0.916242, 0.896091, 0.100717],
	[0.926106, 0.89733, 0.104071],
	[0.935904, 0.89857, 0.108131],
	[0.945636, 0.899815, 0.112838],
	[0.9553, 0.901065, 0.118128],
	[0.964894, 0.902323, 0.123941],
	[0.974417, 0.90359, 0.130215],
	[0.983868, 0.904867, 0.136897],
	[0.993248, 0.906157, 0.143936]
]	

def main():
	return

if __name__ == "__main__":
	main()
