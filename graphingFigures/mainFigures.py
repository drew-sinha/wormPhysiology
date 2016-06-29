# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:57:16 2016

@author: Willie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import os
import sys
if sys.platform == 'win32':
	import statsmodels.api
import inspect	
import matplotlib

import freeimage

import basicOperations.imageOperations as imageOperations
import analyzeHealth.computeStatistics as computeStatistics
import analyzeHealth.characterizeTrajectories as characterizeTrajectories
import analyzeHealth.selectData as selectData
import graphingFigures.supplementFigures as supplementFigures
import graphingFigures.plotFigures as plotFigures
import graphingFigures.cannedFigures as cannedFigures


matplotlib.rcParams['svg.fonttype'] = 'none'
figure_ending = '.eps'

def make_all_figures(adult_df):
	'''
	Make all the final figures for the paper.
	'''
	experimental_overview(adult_df)
	cohorts_overview(adult_df)
	absolute_time(adult_df)
	relative_time(adult_df)
	spans_analysis(adult_df)
	conclusions_sketch(adult_df)
	return


def experimental_overview(adult_df):
	'''
	Make a figure describing the experimental workflow.
	'''
	# Set up my plots.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 12)	
	corral_step = plotFigures.consistent_subgrid_coordinates((2, 3), (0, 2), my_width = 1, my_height = 1)
	acquisition_step = plotFigures.consistent_subgrid_coordinates((2, 3), (1, 2), my_width = 1, my_height = 1)
	measurement_step = plotFigures.consistent_subgrid_coordinates((2, 3), (0, 1), my_width = 1, my_height = 1)
	health_step = plotFigures.consistent_subgrid_coordinates((2, 3), (1, 1), my_width = 1, my_height = 1)
	unhealthy_worm = plotFigures.consistent_subgrid_coordinates((2, 3), (0, 0), my_width = 1, my_height = 1)
	healthy_worm = plotFigures.consistent_subgrid_coordinates((2, 3), (1, 0), my_width = 1, my_height = 1)

	# Label plots properly.
	plotFigures.subfigure_label([corral_step, acquisition_step, measurement_step, health_step, unhealthy_worm, healthy_worm])

	# Prepare a list of individuals ordered by health.
	all_healths = list(selectData.rank_worms(adult_df, 'health', None, return_all = True)[1])
	human_images = []
	for a_health in all_healths:
		the_worm = ' '.join(a_health.split(' ')[:-1])
		the_time = a_health.split(' ')[-1]
		if os.path.isfile(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Human Training Data' + os.path.sep + the_worm + os.path.sep + the_time + ' bf.png'):
			human_images.append(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Human Training Data' + os.path.sep + the_worm + os.path.sep + the_time + ' bf.png')

	# Prepare worm images.
	worms_list = [
		r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Human Training Data\2016.02.20 spe-9 10B 55\2016-02-24t0052 bf.png',
		r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Human Training Data\2016.02.20 spe-9 10B 55\2016-02-27t0133 bf.png',
		r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016 spe-9 Human Training Data\2016.02.20 spe-9 10B 55\2016-03-02t0056 bf.png',
	]
	worms_list.extend([human_images[-1], human_images[4]])
	print([human_images[-1], human_images[4]])
	worm_images = [freeimage.read(a_worm) for a_worm in worms_list]
	worm_masks = [freeimage.read(a_worm.replace('bf.png', 'hmask.png')).astype('bool') for a_worm in worms_list]
	white_worms = [a_worm[2:-2, 2:-2] for a_worm in imageOperations.white_worms(worm_images, worm_masks, box_size = 500)]
	
	# Cut out some borders for the individual worm images.
	white_worms[4] = np.swapaxes(white_worms[4], 0, 1)
	white_worms[3] = white_worms[3][200:-200, 200:-200]
	white_worms[4] = white_worms[4][200:-200, 200:-200]

	# Show the corrals step.
	cannedFigures.color_image(corral_step, freeimage.read(r'C:\Google Drive\Aging Research\WormAgingMechanics\written\Resources\corral_diagram.png'), the_title = 'Isolate Individual Animals')
	
	# Show the acquisitions step.
	series_worms = np.concatenate(white_worms[:3], axis = 0)
	cannedFigures.color_image(acquisition_step, series_worms, the_title = 'Acquire and Process Longitudinal Images')
	acquisition_step.arrow(700, 500, 300, 0, head_width = 50, head_length = 50, fc = 'k', ec = 'k')	
	acquisition_step.arrow(1700, 500, 300, 0, head_width = 50, head_length = 50, fc = 'k', ec = 'k')	

	# Show a sketch of some measurements.
	cannedFigures.measurements_sketch(measurement_step, adult_df)
	
	# Show a sketch of health.
	cannedFigures.health_sketch(health_step, adult_df)

	# Show an example unhealthy worm.
	cannedFigures.color_image(unhealthy_worm, white_worms[3], the_title = 'Poor Prognosis Example')

	# Show an example healthy worm.
	cannedFigures.color_image(healthy_worm, white_worms[4], the_title = 'Good Prognosis Example')
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def cohorts_overview(adult_df):
	'''
	Make a figure describing the lifespan cohorts and our hypotheses.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	survival_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 6), my_width = 2, my_height = 2)
	lifespans_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 6), my_width = 2, my_height = 2)
	
	highlight_life = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	histogram_high = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 1, my_height = 2)
	histogram_low = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 4), my_width = 1, my_height = 2)
	illustrate_traces = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)
	movement_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 2), my_width = 2, my_height = 2)
	autofluorescence_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 2), my_width = 2, my_height = 2)
	texture_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 2), my_width = 2, my_height = 2)
	eggs_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	size_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.8)

	# Label plots properly.
	plotFigures.subfigure_label([survival_plot, lifespans_plot, highlight_life, histogram_high, histogram_low, illustrate_traces, movement_plot, autofluorescence_plot, texture_plot, eggs_plot, size_plot])

	# Plot survival curves and lifespan distribution.
	cannedFigures.survival_lifespan(survival_plot, lifespans_plot, adult_df)

	# Plot cohort trace explanation.
	cannedFigures.explain_traces(highlight_life, histogram_high, histogram_low, illustrate_traces, adult_df)
	
	cannedFigures.cohort_traces(movement_plot, 'bulk_movement', adult_df, the_xlabel = 'Age (Days Post-Maturity)')
	cannedFigures.cohort_traces(autofluorescence_plot, 'intensity_80', adult_df, the_xlabel = 'Age (Days Post-Maturity)')
	cannedFigures.cohort_traces(texture_plot, 'life_texture', adult_df, the_xlabel = 'Age (Days Post-Maturity)')
	cannedFigures.cohort_traces(eggs_plot, 'cumulative_eggs', adult_df, the_xlabel = 'Age (Days Post-Maturity)')
	cannedFigures.cohort_traces(size_plot, 'adjusted_size', adult_df, the_xlabel = 'Age (Days Post-Maturity)')
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def absolute_time(adult_df):
	'''
	Make a figure describing the absolute time results.
	'''
	# Set up the figure.	
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 12)
	start_plot = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 2), my_width = 1, my_height = 1)
	rate_plot = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 2), my_width = 1, my_height = 1)
	end_plot = plotFigures.consistent_subgrid_coordinates((3, 3), (2, 2), my_width = 1, my_height = 1)
	variable_trace = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 1), my_width = 1, my_height = 1)
	start_scatter = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 0), my_width = 1, my_height = 1)
	rate_scatter = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 0), my_width = 1, my_height = 1)
	end_scatter = plotFigures.consistent_subgrid_coordinates((3, 3), (2, 0), my_width = 1, my_height = 1)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.2, hspace = 0.3)
	
	# Label plots properly.
	plotFigures.subfigure_label([start_plot, rate_plot, end_plot, variable_trace, start_scatter, rate_scatter, end_scatter])
	
	# Sketch hypotheses in absolute time.
	cannedFigures.absolute_hypotheses(start_plot, rate_plot, end_plot, adult_df)

	# Plot the actual data.
	cannedFigures.cohort_traces(variable_trace, 'health', adult_df, the_title = 'Prognosis Over Absolute Time', the_xlabel = 'Age (Days Post-Maturity)', the_ylabel = 'Prognosis (Remaining Days)')

	# Set up some needed data.
	my_adultspans = selectData.get_adultspans(adult_df)/24
	geometry_dict = computeStatistics.one_d_geometries(adult_df, 'health')
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)	

	# Prepare my geometry data.
#	(start_data, my_unit, fancy_name) = adult_df.display_variables(geometry_dict['start'], 'health')
#	(end_data, my_unit, fancy_name) = adult_df.display_variables(geometry_dict['end'], 'health')
	start_data = geometry_dict['start']
	end_data = geometry_dict['end']
	rate_data = (start_data - end_data)/my_adultspans

	# Plot the "start" scatter.
	cannedFigures.cohort_scatters(start_scatter, my_adultspans, start_data, adult_df, the_title = 'Start', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Starting Prognosis (Remaining Days)', label_coordinates = (4, 7.5))
	cannedFigures.cohort_scatters(rate_scatter, my_adultspans, rate_data, adult_df, the_title = 'Rate', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Aging Rate (Dimensionless)', label_coordinates = (10, 1.5), polyfit_degree = 2)
	cannedFigures.cohort_scatters(end_scatter, my_adultspans, end_data, adult_df, the_title = 'End', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Ending Prognosis (Remaining Days)', label_coordinates = (0.5, 1), polyfit_degree = 2)
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def relative_time(adult_df):
	'''
	Make a figure describing the relative time results.
	'''
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	deviation_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 6), my_width = 2, my_height = 2)
	measured_deviation = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 6), my_width = 2, my_height = 2) 
	computed_deviation = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 6), my_width = 2, my_height = 2)
	rescale_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	negative_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)	
	positive_plot = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 2, my_height = 2)
	variable_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 2), my_width = 2, my_height = 2)
	variable_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 2), my_width = 2, my_height = 2)
	relative_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	relative_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.7)
	
	# Label plots properly.
	plotFigures.subfigure_label([deviation_plot, measured_deviation, computed_deviation, rescale_plot, positive_plot, negative_plot, variable_trace, variable_scatter, relative_trace, relative_scatter])

	# Plot the hypotheses.
	cannedFigures.relative_hypotheses(deviation_plot, measured_deviation, computed_deviation, rescale_plot, negative_plot, positive_plot, adult_df)
	
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	my_cohorts = life_cohorts

	# Prepare my "inflection" data.	
	geometry_dict = computeStatistics.one_d_geometries(adult_df, 'health')
	start_data = geometry_dict['start']
	mean_start = np.mean(start_data)			
	inflection_data = geometry_dict['absolute_inflection']
	relative_inflection = geometry_dict['self_inflection']

	# Plot the traces and scatter for absolute inflection.
	cannedFigures.cohort_traces(variable_trace, 'health', adult_df, the_title = 'Prognosis Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = 'Prognosis (Remaining Days)', x_normed = True)
	variable_trace.set_ylim([0, 1.1*mean_start])
	cannedFigures.cohort_scatters(variable_scatter, my_adultspans, inflection_data, adult_df, the_title = 'Absolute Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Average Deviation (Days)', label_coordinates = (12, 2.5))

	# Plot the traces and scatter for relative inflection.
	cannedFigures.cohort_traces(relative_trace, 'health', adult_df, the_title = 'Relative Prognosis Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = 'Relative Prognosis (Fractional Remaining Life)', x_normed = True, y_normed = True)
	relative_trace.set_ylim([-0.1, 1.1])
	cannedFigures.cohort_scatters(relative_scatter, my_adultspans, relative_inflection, adult_df, the_title = 'Relative Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Average Deviation (Relative Prognosis)', label_coordinates = (4, -0.4))
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return

def spans_analysis(adult_df, exclude_confirmation = True):
	'''
	Make a figure describing the spans results.
	'''
	# Set up figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 12)
	
	health_traces = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 2), my_width = 1, my_height = 1)
	health_traces_normed = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 2), my_width = 1, my_height = 1)
	health_spans = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 1), my_width = 1, my_height = 1)
	health_spans_normed = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 1), my_width = 1, my_height = 1)
	my_plot = plotFigures.consistent_subgrid_coordinates((3, 3), (2, 2), my_width = 1, my_height = 1)
	stroustrup_plot = plotFigures.consistent_subgrid_coordinates((3, 3), (2, 1), my_width = 1, my_height = 1)
	health_life = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 0), my_width = 1, my_height = 1)
	gero_life = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 0), my_width = 1, my_height = 1)
	health_gero = plotFigures.consistent_subgrid_coordinates((3, 3), (2, 0), my_width = 1, my_height = 1)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)

	# Label plots properly.
	plotFigures.subfigure_label([health_traces, health_traces_normed, my_plot, health_spans, health_spans_normed, stroustrup_plot, health_life, gero_life, health_gero])

	# Show spans.
	cannedFigures.show_spans(health_traces, health_traces_normed, health_spans, health_spans_normed, adult_df)

	# Prepare data for validation scatters.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)


	adultspans = pd.Series(selectData.get_adultspans(adult_df)/24)
	movehealthspans = np.array(computeStatistics.get_spans(adult_df, 'bulk_movement', 'overall_time', 0.85, reverse_direction = False))/24
	relative_movegerospans = pd.Series((adultspans - movehealthspans)/adultspans)
	ordered_life = adultspans.order()
	my_num = relative_movegerospans.shape[0]//5
	print('MYNUM', my_num)
	low_life = ordered_life[:my_num]
	high_life = ordered_life[-my_num:]
	
	# Plot my validation scatter.	
	for i in range(0, len(life_cohorts)):
		plot_low_worms = [a_worm for a_worm in low_life.index if a_worm in life_cohorts[i]]
		plot_high_worms = [a_worm for a_worm in high_life.index if a_worm in life_cohorts[i]]
		
		my_plot.scatter(np.zeros(len(plot_low_worms)) + np.random.uniform(-1, 1, len(plot_low_worms))/10, relative_movegerospans.loc[plot_low_worms], color = my_colors[i])
		my_plot.scatter(np.ones(len(plot_high_worms)) + np.random.uniform(-1, 1, len(plot_high_worms))/10, relative_movegerospans.loc[plot_high_worms], color = my_colors[i])
		
		
	scipy.stats.ttest_ind(relative_movegerospans.loc[low_life.index], relative_movegerospans.loc[high_life.index])
	my_plot.set_xticks([0, 1])
	my_plot.set_xticklabels(['Shortest-Lived Quartile', 'Longest-Lived Quartile'])
	my_plot.set_xlim([-0.5, 1.5])
	my_plot.set_ylabel('Fractional Movement Gerospan')
	my_plot.set_title('Isolated Individuals')

	lowxrange = np.linspace(-0.25, 0.25, 200)
	lowyrange = np.array([relative_movegerospans.loc[low_life.index].mean() for i in range(0, 200)])
	highxrange = np.linspace(0.75, 1.25, 200)
	highyrange = np.array([relative_movegerospans.loc[high_life.index].mean() for i in range(0, 200)])
	my_plot.plot(lowxrange, lowyrange, color = 'black')
	my_plot.plot(highxrange, highyrange, color = 'black')
#	my_plot.annotate('p = ' + '%.3f' % scipy.stats.ttest_ind(relative_movegerospans.loc[low_life.index], relative_movegerospans.loc[high_life.index]).pvalue, xy = (0.35, 0.6), xycoords = 'data')
	my_plot.annotate('$p < 0.001$', xy = (0.35, 0.6), xycoords = 'data')

		
	# Read in Stroustrup data.	
	mutant_data = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.05.11 Stroustrup Movement Data\final_spe6_fer16_data_for_zach.csv')
	uncensored_mutants = np.invert(mutant_data.loc[:, 'Censored'].values.astype('bool'))
	mutant_gerospans = (mutant_data.loc[:, 'Duration Not Fast Moving (d)']/mutant_data.loc[:, 'Age at Death (d) Raw'])[uncensored_mutants]
	mutant_lifespans = mutant_data.loc[:, 'Age at Death (d) Raw'][uncensored_mutants]

	# Plot Nick validation scatter.
	ordered_life = mutant_lifespans.order()
	my_num = ordered_life.shape[0]//4
	print('MYNUM', my_num)
	low_life = ordered_life[:my_num]
	high_life = ordered_life[-my_num:]
	stroustrup_plot.scatter(np.zeros(my_num) + np.random.uniform(-1, 1, my_num)/10, mutant_gerospans.loc[low_life.index], color = 'green')
	stroustrup_plot.scatter(np.ones(my_num) + np.random.uniform(-1, 1, my_num)/10, mutant_gerospans.loc[high_life.index], color = 'green')
	stroustrup_plot.annotate('$p = ' + ('%.3f' % scipy.stats.ttest_ind(mutant_gerospans.loc[low_life.index], mutant_gerospans.loc[high_life.index]).pvalue) + '$', xy = (0.35, 0.3), xycoords = 'data')
	stroustrup_plot.set_xticks([0, 1])
	stroustrup_plot.set_xticklabels(['Shortest-Lived Quartile', 'Longest-Lived Quartile'])
	stroustrup_plot.set_ylabel('Fractional Movement Gerospan')
	stroustrup_plot.set_xlim([-0.5, 1.5])
	stroustrup_plot.set_ylim([-0.05, 0.45])
	stroustrup_plot.set_title('Standard NGM Plates')

	lowxrange = np.linspace(-0.25, 0.25, 200)
	lowyrange = np.array([mutant_gerospans.loc[low_life.index].mean() for i in range(0, 200)])
	highxrange = np.linspace(0.75, 1.25, 200)
	highyrange = np.array([mutant_gerospans.loc[high_life.index].mean() for i in range(0, 200)])
	stroustrup_plot.plot(lowxrange, lowyrange, color = 'black')
	stroustrup_plot.plot(highxrange, highyrange, color = 'black')

#	print(mutant_gerospans.loc[low_life.index].mean())
#	print(mutant_gerospans.loc[high_life.index].mean())


	# Prepare data for spans scatters.
	healthspans = np.array(computeStatistics.get_spans(adult_df, 'health', 'overall_time', 0.5,  reverse_direction = True))/24
	gerospans = adultspans - healthspans

	print(np.std(healthspans))
	print(np.std(gerospans))


	# Plot spans scatters.
	cannedFigures.cohort_scatters(health_life, adultspans, healthspans, adult_df, the_title = 'Healthspan vs. Adult Lifespan', the_xlabel = 'Adult Lifespan (Days)', the_ylabel = 'Healthspan (Days)', label_coordinates = (3, 8))
	health_life.set_xlim([0, 18])
	health_life.set_ylim([-1, 12])
	cannedFigures.cohort_scatters(gero_life, adultspans, gerospans, adult_df, the_title = 'Gerospan vs. Adult Lifespan', the_xlabel = 'Adult Lifespan (Days)', the_ylabel = 'Gerospan (Days)', label_coordinates = (4, 6))
	gero_life.set_xlim([0, 18])
	gero_life.set_ylim([-1, 12])

	# Plot KDE histogram for variability.
	kde_density = scipy.stats.gaussian_kde(healthspans)
	my_xrange = np.linspace(0, 17, 200)
	kde_density._compute_covariance()
	health_gero.plot(my_xrange, kde_density(my_xrange), color = 'black', linewidth = 2)	
	kde_density = scipy.stats.gaussian_kde(gerospans)
	my_xrange = np.linspace(0, 17, 200)
	kde_density._compute_covariance()
	health_gero.plot(my_xrange, kde_density(my_xrange), color = [0.5, 0.5, 0.5], linewidth = 2)	
	health_gero.set_title('Healthspan and Gerospan Variability')	
	health_gero.set_xlim([0, 12])
	health_gero.set_ylim([0, 0.35])
	health_gero.set_ylabel('Density')
	health_gero.set_xlabel('Days')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return
	
def conclusions_sketch(adult_df):
	'''
	Make a figure describing our conclusions.
	'''
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	conclusion_sketch = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
		
	# Prepare some data.
	(life_cohorts, bin_lifes, my_bins, life_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_colors = np.array([[1, 0, 0], [0, 0, 0], [0, 0.5, 0]])
	fade_colors = [[1, 1, 1], (my_colors[1] + 3)/4, [1, 1, 1]]
	my_colors = np.array([life_colors[-2], [0, 0, 0], life_colors[1]])

	# Set up points to fit to.
	x_start = np.zeros(3)
	x_end = np.empty(3)
	x_end[:] = np.sqrt(2)
	x_mid = np.empty(3)
	x_mid[:] = (np.sqrt(2))/2
	y_start = np.zeros(3)
	y_end = np.zeros(7)
	y_mid = (np.arange(-1, 2)/10)[::-1]
	
	# Prepare a rotation matrix.
	an_angle = np.pi/4
	rotation_matrix = np.array([[np.cos(an_angle), -np.sin(an_angle)], [np.sin(an_angle), np.cos(an_angle)]])
	
	# Plot the conclusions.
	low_number = 4
	high_number = 12
	start_health = 10
	together_low_colors = [fade_colors[0], fade_colors[1], my_colors[2]]
	together_high_colors = [my_colors[0], fade_colors[1], fade_colors[2]]
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
		yrange = rotated_points[:, 1]
		xrange = 1 - xrange
		conclusion_sketch.plot(xrange*low_number, yrange*(start_health-1.5)+1.5, color = together_low_colors[i], linewidth = 2)
		conclusion_sketch.plot(xrange*high_number, yrange*start_health, color = together_high_colors[i], linewidth = 2)
	conclusion_sketch.set_xlim([0, 13])
	conclusion_sketch.set_ylim([0, 11])
	conclusion_sketch.set_xlabel('Adult Age (Days)')
	conclusion_sketch.set_title('Physiological Aging of Long vs. Short-Lived Individuals')
	conclusion_sketch.set_ylabel('Prognosis (Days Remaining)')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return


def main():
	return

if __name__ == "__main__":
	main()
