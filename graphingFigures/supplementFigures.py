# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:57:00 2016

@author: Willie
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
if sys.platform == 'win32':
	import statsmodels.api
import scipy.stats
import json
import inspect
import matplotlib

import freeimage
	
import basicOperations.imageOperations as imageOperations
import analyzeHealth.computeStatistics as computeStatistics
import analyzeHealth.characterizeTrajectories as characterizeTrajectories
import analyzeHealth.selectData as selectData
import measurePhysiology.organizeData as organizeData
import graphingFigures.plotFigures as plotFigures
import graphingFigures.cannedFigures as cannedFigures

matplotlib.rcParams['svg.fonttype'] = 'none'
figure_ending = '.eps'


def make_all_supplement(adult_df, directory_bolus):
	'''
	Make all supplementary figures for the paper.
	'''
	if not os.path.isdir(directory_bolus.working_directory):
		development_vs_adulthood(adult_df)
	
	if os.path.isdir(directory_bolus.working_directory):
		cohort_gallery(adult_df, directory_bolus, '2.0')
		manual_mottled_gallery(adult_df, directory_bolus, '2.0')
	if not os.path.isdir(directory_bolus.working_directory):
		mottled_longevity(adult_df)
		mottled_over_time(adult_df)
		non_mottled_health_traces(adult_df)
		plate_sizes(adult_df)
		time_consistency(adult_df)
		experimental_consistency(adult_df)
		linear_inflection_confirmation(adult_df)
		linear_inflection_table(adult_df)
		age_inflection_confirmation(adult_df)
		define_survival(adult_df)
		survival_inflection_confirmation(adult_df)
		show_spans(adult_df, 0.25)
		show_spans(adult_df, 0.375)
		show_spans(adult_df, 0.625)
		show_spans(adult_df, 0.75)
		quality_life(adult_df)
		geometry_table(adult_df)
		correlation_table_categories(adult_df)
		correlation_table_raw(adult_df)
	if os.path.isdir(directory_bolus.working_directory):
		fates_gallery(adult_df, directory_bolus)
	if not os.path.isdir(directory_bolus.working_directory):
		fate_lifes(adult_df)
	if os.path.isdir(directory_bolus.working_directory):
		worm_finding_gallery(adult_df, directory_bolus)
		worm_finding(adult_df, directory_bolus)		
		measure_gallery(adult_df, directory_bolus, 'adjusted_size', '2.0')
		measure_gallery(adult_df, directory_bolus, 'adjusted_size', '5.0')
		measure_gallery(adult_df, directory_bolus, 'adjusted_size', '8.0')
		measure_gallery(adult_df, directory_bolus, 'life_texture', '2.0')
		measure_gallery(adult_df, directory_bolus, 'life_texture', '5.0')
		measure_gallery(adult_df, directory_bolus, 'life_texture', '8.0')
		measure_gallery(adult_df, directory_bolus, 'intensity_80', '7.0')
		measure_gallery(adult_df, directory_bolus, 'intensity_80', '8.0')
		measure_gallery(adult_df, directory_bolus, 'intensity_80', '9.0')
		measure_gallery(adult_df, directory_bolus, 'bulk_movement', '2.0')
		measure_gallery(adult_df, directory_bolus, 'bulk_movement', '5.0')
		measure_gallery(adult_df, directory_bolus, 'bulk_movement', '8.0')
		measure_gallery(adult_df, directory_bolus, 'visible_eggs', '1.0')
		measure_gallery(adult_df, directory_bolus, 'visible_eggs', '2.0')
		measure_gallery(adult_df, directory_bolus, 'visible_eggs', '3.0')
	if not os.path.isdir(directory_bolus.working_directory):
		svr_traces(adult_df)
		plasticity_threshold(adult_df, 0.4)		
		plasticity_threshold(adult_df, 0.6)		
	return


def plasticity_threshold(adult_df, health_duration):
	'''
	'''
	# Set up figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	health_gero = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
#	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)	

	adultspans = pd.Series(selectData.get_adultspans(adult_df)/24)
	healthspans = np.array(computeStatistics.get_spans(adult_df, 'health', 'overall_time', health_duration,  reverse_direction = True))/24
	gerospans = adultspans - healthspans

	healthspans = healthspans/np.mean(healthspans)
	gerospans = gerospans/np.mean(gerospans)

	print(np.std(healthspans))
	print(np.std(gerospans))

	# Plot KDE histogram for variability.
	kde_density = scipy.stats.gaussian_kde(healthspans)
	my_xrange = np.linspace(-3, 5, 200)
	kde_density._compute_covariance()
	health_gero.plot(my_xrange, kde_density(my_xrange), color = 'black', linewidth = 2)	
	kde_density = scipy.stats.gaussian_kde(gerospans)
	my_xrange = np.linspace(-3, 5, 200)
	kde_density._compute_covariance()
	health_gero.plot(my_xrange, kde_density(my_xrange), color = [0.5, 0.5, 0.5], linewidth = 2)	
	health_gero.set_title('Healthspan and Gerospan Variability, ' + str(int(health_duration*100)) + '% Threshold')	
	health_gero.set_xlim([-1, 3])
	health_gero.set_xticks([1])
	health_gero.set_xticklabels(['Mean'])
#	health_gero.set_ylim([0, 0.35])
	health_gero.set_ylabel('Density')
	health_gero.set_xlabel('Length of Span (Arbitrary Units)')
#	health_gero.set_tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
	# Save out my figure.
	graph_name = '%.2f' % health_duration + '_plasticity'
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)		
	return

def linear_inflection_confirmation(adult_df):
	'''
	Re-create inflection traces just using multiple linear regression.
	'''
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	my_cohorts = life_cohorts

	# Prepare my health variable.
	health_data = computeStatistics.multiple_regression_combine(adult_df, ['intensity_80', 'adjusted_size', 'adjusted_size_rate', 'life_texture', 'cumulative_eggs', 'cumulative_eggs_rate', 'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'], dependent_variable = 'ghost_age')[0]
	health_data = health_data*(-1/24)


	# Set up the figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	absolute_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 6), my_width = 2, my_height = 2)
	start_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	rate_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 2, my_height = 2)
	end_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)
	variable_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 2), my_width = 2, my_height = 2)
	variable_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 2), my_width = 2, my_height = 2)
	relative_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	relative_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.7)
	
	# Label plots properly.
	plotFigures.subfigure_label([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter])

	cannedFigures.canned_confirmation([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter], health_data, adult_df, label_points = [(2, 12), (10, 2), (1, 0), (12, 2.2), (4, -0.5)], health_name = 'Linear Prognosis (Days)')

	rate_scatter.set_ylim([-0.5, 2.5])

	end_scatter.set_ylim([-2, 8])
	
	# Make an overall title.
	plt.suptitle('Prognosis from Linear Regression', fontsize = 15)	
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return

def linear_inflection_table(adult_df):
	'''
	Plot the weights for multiple linear regression.
	'''
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 4)
	weights_table = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
	
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	my_cohorts = life_cohorts

	# Prepare my health variable.
	raw_measures = ['intensity_80', 'adjusted_size', 'adjusted_size_rate', 'life_texture', 'cumulative_eggs', 'cumulative_eggs_rate', 'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate']
	display_categories = ['Autofluorescence', 'Body Size', 'Body Size', 'Texture', 'Reproductive', 'Reproductive', 'Movement', 'Movement', 'Movement', 'Movement']
	(health_data, health_weights) = computeStatistics.multiple_regression_combine(adult_df, raw_measures, dependent_variable = 'ghost_age')
	display_weights = ['%.2f' % (a_weight*(-1/24)) for a_weight in health_weights]	
	display_measures = []
	display_units = []
#	display_weights = []
	for i in range(0, len(raw_measures)):
		a_measure = raw_measures[i]
		(display_weight, display_unit, display_measure) = adult_df.display_variables(health_weights[i]/24, a_measure)
		display_measures.append(display_measure)
#		display_units.append(display_unit)
		display_units.append('Days per Standard Deviation')
#		display_weights.append(display_weight)

	table_data = pd.DataFrame(index = display_measures, columns = ['Weight', 'Units', 'Category'])
	table_data.loc[:, 'Weight'] = display_weights
	table_data.loc[:, 'Units'] = display_units
	table_data.loc[:, 'Category'] = display_categories

	cannedFigures.table_from_dataframe(weights_table, table_data)	
	
	# Make an overall title.
	plt.suptitle('Linear Regression Weights', fontsize = 15)	
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return

def age_inflection_confirmation(adult_df, health_data = None):
	'''
	Re-create inflection traces just using multiple linear regression.
	'''	
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	

	# Prepare my health variable.
	if health_data == None:
		(variable_data, health_data, life_data) = computeStatistics.svr_data(adult_df, ['intensity_80', 'adjusted_size', 'adjusted_size_rate', 'life_texture', 'cumulative_eggs', 'cumulative_eggs_rate', 'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'], dependent_variable = 'age')

	health_data = np.expand_dims(health_data, 1)
	health_data = health_data*(-1/24)
	health_data = health_data - np.nanmin(health_data)	
	health_data = health_data/np.mean(health_data[:, :, 0])	
	
	# Set up the figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	absolute_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 6), my_width = 2, my_height = 2)
	start_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	rate_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 2, my_height = 2)
	end_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)
	variable_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 2), my_width = 2, my_height = 2)
	variable_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 2), my_width = 2, my_height = 2)
	relative_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	relative_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.7)
	
	# Label plots properly.
	plotFigures.subfigure_label([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter])

	cannedFigures.canned_confirmation([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter], health_data, adult_df, label_points = [(2, 0.85), (10, 0.15), (1, 0), (4, -0.2), (4, -0.4)], health_name = 'Youthfulness Index')
	
	# Make an overall title.
	plt.suptitle('Youthfulness Index', fontsize = 15)	

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return

def define_survival(adult_df, x_day_survival = 3):
	'''
	'''
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	survival_definition = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)

	(middle_healths, mortality_rates) = computeStatistics.health_to_mortality(adult_df, x_day_survival = x_day_survival)
	survival_rates = 1 - np.array(mortality_rates)
#	flat_data = np.ndarray.flatten(cohort_data)
#	cohort_ages = complete_df.mloc(complete_df.worms, ['age'])[a_cohort, 0, :]
#	flat_ages = np.ndarray.flatten(cohort_ages)
	(xdata, ydata) = (np.array(middle_healths), np.array(survival_rates))
	# Fit a polynomial for a trendline and for r^2.
	p_array = np.polyfit(xdata, ydata, 6)
	my_estimator = np.array([p_array[-i]*xdata**(i-1) for i in range(1, len(p_array)+1)])
	my_estimator = my_estimator.sum(axis = 0)
	xrange = np.linspace(np.min(xdata), np.max(xdata), 200)
	
	def health_to_survival(health_data):
		return np.array([p_array[-i]*health_data**(i-1) for i in range(1, len(p_array)+1)]).sum(axis = 0)
	
	
	my_trendline = health_to_survival(xrange)
#	my_trendline = my_trendline

	survival_definition.plot(xrange, my_trendline, color = 'black')
	survival_definition.scatter(xdata, ydata, color = 'black')
	survival_definition.set_xlabel('Prognosis (Remaining Days)')
	survival_definition.set_ylabel(str(x_day_survival) + '-Day Survival (%)')
	survival_definition.set_title('Prognosis to Survival Conversion')
#	cohort_ages = cohort_data[:, 0]
#	cohort_data = cohort_data[:, 1]

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return health_to_survival


def survival_inflection_confirmation(adult_df):
	'''
	Re-create inflection traces just using survival in place of prognosis.
	'''
	# Prepare my health variable.
	health_data = adult_df.mloc(measures = ['health'])
	(health_data, my_unit, fancy_name) = adult_df.display_variables(health_data, 'health')
	health_to_survival = define_survival(adult_df)
	survival_data = health_to_survival(health_data)
	health_data = survival_data
	
	# Set up the figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	absolute_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 6), my_width = 2, my_height = 2)
	start_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	rate_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 2, my_height = 2)
	end_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)
	variable_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 2), my_width = 2, my_height = 2)
	variable_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 2), my_width = 2, my_height = 2)
	relative_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	relative_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.7)
	
	# Label plots properly.
	plotFigures.subfigure_label([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter])

	cannedFigures.canned_confirmation([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter], health_data, adult_df, label_points = [(2, 0.92), (10, 0.15), (1, 0), (12, 0.4), (4, -0.2)], health_name = 'Predicted Survival Rate')
	
	# Make an overall title.
	plt.suptitle('Predicted Survival Rates', fontsize = 15)	
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return
	
def measure_gallery(adult_df, directory_bolus, a_measure, a_time, refresh_figure = False):
	'''
	Make a gallery of sample images for a_measure at a_time.
	'''
	gallery_chosen = {
		'life_texture;2.0': ['2016.03.14 spe-9 14 148', '2016.03.14 spe-9 14 137', '2016.03.25 spe-9 15A 60', '2016.03.14 spe-9 14 141', '2016.02.16 spe-9 9 021', '2016.02.26 spe-9 11B 036', '2016.02.26 spe-9 11C 109', '2016.03.04 spe-9 13A 11', '2016.02.26 spe-9 11D 143', '2016.03.31 spe-9 16 048', '2016.02.26 spe-9 11D 120', '2016.03.14 spe-9 14 055'],
		'life_texture;5.0': ['2016.02.20 spe-9 10A 19', '2016.03.14 spe-9 14 192', '2016.03.25 spe-9 15B 014', '2016.03.25 spe-9 15B 084', '2016.03.25 spe-9 15B 096', '2016.03.25 spe-9 15B 046', '2016.03.31 spe-9 16 070', '2016.03.25 spe-9 15A 95', '2016.03.31 spe-9 16 135', '2016.03.25 spe-9 15B 016', '2016.03.31 spe-9 16 100', '2016.03.25 spe-9 15A 74'],
		'life_texture;8.0': ['2016.03.25 spe-9 15B 050', '2016.03.25 spe-9 15B 065', '2016.03.25 spe-9 15B 066', '2016.03.31 spe-9 16 129', '2016.03.25 spe-9 15A 93', '2016.03.14 spe-9 14 094', '2016.03.14 spe-9 14 125', '2016.03.14 spe-9 14 061', '2016.03.14 spe-9 14 132', '2016.03.31 spe-9 16 104', '2016.02.26 spe-9 11C 079', '2016.03.25 spe-9 15A 01'],
		'intensity_80;7.0': ['2016.02.26 spe-9 11D 154', '2016.03.31 spe-9 16 017', '2016.03.25 spe-9 15A 27', '2016.02.26 spe-9 11A 14', '2016.03.31 spe-9 16 192', '2016.03.31 spe-9 16 140', '2016.03.25 spe-9 15B 080', '2016.03.14 spe-9 14 119', '2016.02.26 spe-9 11C 065', '2016.03.14 spe-9 14 187', '2016.02.20 spe-9 10A 02', '2016.03.31 spe-9 16 036'],
		'intensity_80;8.0': ['2016.03.25 spe-9 15A 86', '2016.03.31 spe-9 16 194', '2016.03.31 spe-9 16 203', '2016.02.26 spe-9 11D 154', '2016.03.25 spe-9 15B 005', '2016.02.20 spe-9 10B 53', '2016.02.26 spe-9 11C 063', '2016.03.31 spe-9 16 046', '2016.03.14 spe-9 14 192', '2016.02.16 spe-9 9 065', '2016.02.20 spe-9 10A 16', '2016.03.04 spe-9 13B 31'],
		'intensity_80;9.0': ['2016.03.14 spe-9 14 135', '2016.03.25 spe-9 15A 40', '2016.03.31 spe-9 16 123', '2016.03.25 spe-9 15A 65', '2016.02.20 spe-9 10B 53', '2016.03.31 spe-9 16 080', '2016.03.31 spe-9 16 200', '2016.03.25 spe-9 15B 044', '2016.03.25 spe-9 15B 096', '2016.02.26 spe-9 11D 145', '2016.03.25 spe-9 15B 078', '2016.02.26 spe-9 11A 09'],
		'adjusted_size;2.0': ['2016.02.26 spe-9 11A 18', '2016.03.25 spe-9 15B 108', '2016.03.31 spe-9 16 170', '2016.03.25 spe-9 15A 03', '2016.02.26 spe-9 11C 101', '2016.03.14 spe-9 14 100', '2016.03.31 spe-9 16 158', '2016.02.16 spe-9 9 054', '2016.03.14 spe-9 14 107', '2016.02.29 spe-9 12B 39', '2016.03.04 spe-9 13B 43', '2016.03.14 spe-9 14 159'],
		'adjusted_size;5.0': ['2016.02.26 spe-9 11D 113', '2016.02.29 spe-9 12A 11', '2016.03.31 spe-9 16 121', '2016.02.29 spe-9 12A 03', '2016.02.29 spe-9 12A 05', '2016.03.04 spe-9 13A 12', '2016.03.25 spe-9 15B 058', '2016.03.25 spe-9 15A 33', '2016.03.14 spe-9 14 189', '2016.03.04 spe-9 13A 13', '2016.03.14 spe-9 14 167', '2016.03.31 spe-9 16 014'],
		'adjusted_size;8.0': ['2016.02.26 spe-9 11D 117', '2016.02.29 spe-9 12A 11', '2016.02.26 spe-9 11D 137', '2016.02.26 spe-9 11D 152', '2016.03.31 spe-9 16 107', '2016.03.14 spe-9 14 047', '2016.03.25 spe-9 15B 050', '2016.03.14 spe-9 14 043', '2016.02.20 spe-9 10A 20', '2016.03.25 spe-9 15A 73', '2016.03.31 spe-9 16 169', '2016.02.29 spe-9 12B 28'],
		'visible_eggs;1.0': ['2016.03.14 spe-9 14 188', '2016.03.14 spe-9 14 163', '2016.03.14 spe-9 14 119', '2016.02.26 spe-9 11A 21', '2016.03.04 spe-9 13A 19', '2016.03.31 spe-9 16 178', '2016.03.31 spe-9 16 049', '2016.03.14 spe-9 14 049', '2016.03.04 spe-9 13C 59', '2016.03.31 spe-9 16 020', '2016.02.26 spe-9 11A 16', '2016.02.26 spe-9 11A 04'],
		'visible_eggs;2.0': ['2016.03.25 spe-9 15B 046', '2016.02.29 spe-9 12B 43', '2016.02.26 spe-9 11D 143', '2016.02.26 spe-9 11C 072', '2016.02.26 spe-9 11C 096', '2016.03.25 spe-9 15B 092', '2016.03.14 spe-9 14 072', '2016.03.04 spe-9 13C 57', '2016.03.25 spe-9 15A 60', '2016.03.31 spe-9 16 029', '2016.03.25 spe-9 15B 077', '2016.03.25 spe-9 15B 051'],
		'visible_eggs;3.0': ['2016.03.25 spe-9 15B 027', '2016.02.26 spe-9 11D 134', '2016.03.04 spe-9 13C 74', '2016.03.31 spe-9 16 107', '2016.03.25 spe-9 15B 008', '2016.03.25 spe-9 15B 044', '2016.03.25 spe-9 15B 023', '2016.03.31 spe-9 16 036', '2016.02.20 spe-9 10A 16', '2016.02.29 spe-9 12B 34', '2016.03.14 spe-9 14 097', '2016.03.25 spe-9 15A 69'],
		'bulk_movement;2.0': ['2016.03.14 spe-9 14 160', '2016.03.31 spe-9 16 158', '2016.02.26 spe-9 11A 21', '2016.02.20 spe-9 10A 25', '2016.03.31 spe-9 16 072', '2016.03.25 spe-9 15B 087', '2016.03.31 spe-9 16 027', '2016.03.31 spe-9 16 084', '2016.02.20 spe-9 10B 42', '2016.02.26 spe-9 11A 00', '2016.03.31 spe-9 16 190', '2016.02.26 spe-9 11B 015'],
		'bulk_movement;5.0': ['2016.03.25 spe-9 15B 115', '2016.02.26 spe-9 11D 149', '2016.03.31 spe-9 16 193', '2016.03.14 spe-9 14 044', '2016.03.31 spe-9 16 129', '2016.03.31 spe-9 16 122', '2016.03.14 spe-9 14 106', '2016.03.31 spe-9 16 084', '2016.03.14 spe-9 14 070', '2016.03.25 spe-9 15A 40', '2016.03.14 spe-9 14 066', '2016.02.29 spe-9 12A 22'],
		'bulk_movement;8.0': ['2016.03.25 spe-9 15B 083', '2016.02.26 spe-9 11C 093', '2016.02.26 spe-9 11B 052', '2016.02.29 spe-9 12A 18', '2016.03.14 spe-9 14 098', '2016.03.14 spe-9 14 081', '2016.03.04 spe-9 13C 69', '2016.02.26 spe-9 11B 017', '2016.03.14 spe-9 14 092', '2016.03.14 spe-9 14 192', '2016.03.25 spe-9 15B 037', '2016.03.14 spe-9 14 071']
	}
		
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 12)
	plots_grid = [[plotFigures.consistent_subgrid_coordinates((12, 12), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 4)] for j in range(0, 3)]	
	labels_grid = [[plotFigures.consistent_subgrid_coordinates((12, 12), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 4)] for j in range(0, 3)]
	
	# Get my data.
	if refresh_figure:
		given_worms = []
	else:
		given_worms = gallery_chosen[a_measure + ';' + a_time]
	(tercile_images, tercile_choices) = selectData.validate_measure(adult_df, directory_bolus, a_measure, a_time, given_worms = given_worms)

	display_list = [[' '.join(b.split(' ')[:-1]) for b in a] for a in tercile_choices]
	display_string = '[' + str(display_list).replace('[', '').replace(']', '') + ']'
	print(display_string)

	# Make the figure itself.
	for i in range(0, len(plots_grid)):
		for j in range(0, len(plots_grid[0])):
			# Display my image.
			plots_grid[i][j].axis('off')
			plots_grid[i][j].imshow((tercile_images[i][j]/2**8).astype('uint8'))
			
			# Label it.
			labels_grid[i][j].axis('off')			
			labels_grid[i][j].annotate(' '.join(tercile_choices[i][j].split(' ')[:-1]), xy = (0.5, 0.5), ha = 'center', va = 'center')
	
	# Label the top and save the figure.
	actual_day = a_time.split('.')
	actual_day = int(actual_day[0]) + int(actual_day[1])*(1/8)
	my_figure.suptitle(adult_df.display_names(a_measure) + ' at Day ' + str(actual_day)[:5], fontsize = 15)

	# Save out my figure.
	graph_name = a_measure + '_' + a_time + '_gallery'
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return 

def manual_mottled_gallery(adult_df, directory_bolus, a_time, refresh_worms = False):
	'''
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 8)
	plots_grid = [[plotFigures.consistent_subgrid_coordinates((12, 8), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 4)] for j in range(0, 2)]	
	labels_grid = [[plotFigures.consistent_subgrid_coordinates((12, 8), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 4)] for j in range(0, 2)]

	mottled_mask = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.24 Mottled Annotations\manual_mottled.tsv', sep = '\t').loc[:, 'Mottled?'].values

	worm_array = np.array(adult_df.worms)
	if refresh_worms:
		normal_worms = np.random.choice(worm_array[~mottled_mask], 4, replace = False)
		mottled_worms = np.random.choice(worm_array[mottled_mask], 4, replace = False)
	else:
		normal_worms = ['2016.03.25 spe-9 15B 074', '2016.02.29 spe-9 12A 19', '2016.03.25 spe-9 15B 033', '2016.02.26 spe-9 11A 22']
		mottled_worms = ['2016.03.25 spe-9 15B 098', '2016.03.14 spe-9 14 130', '2016.02.26 spe-9 11C 099', '2016.03.31 spe-9 16 055']

	normal_times = [selectData.closest_real_time(adult_df, a_worm, a_time, egg_mode = True) for a_worm in normal_worms]
	mottled_times = [selectData.closest_real_time(adult_df, a_worm, a_time, egg_mode = True) for a_worm in mottled_worms]


	normal_images = [imageOperations.get_worm(directory_bolus, normal_worms[i], normal_times[i], box_size = 500, get_mode = 'worm') for i in range(0, 4)]
	mottled_images = [imageOperations.get_worm(directory_bolus, mottled_worms[i], mottled_times[i], box_size = 500, get_mode = 'worm') for i in range(0, 4)]

	images_max = np.max(np.array([normal_images, mottled_images]))
	dtype_max = 2**16-1
	normal_images = [(normal_images[j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, 4)]
	mottled_images = [(mottled_images[j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, 4)]


	normal_images = [imageOperations.border_box(normal_images[j], border_color = [65535, 65535, 65535], border_width = 500//15) for j in range(0, 4)]
	mottled_images = [imageOperations.border_box(mottled_images[j], border_color =[0, 0, 0], border_width = 500//15) for j in range(0, 4)]


	# Make the figure itself.
	for j in range(0, len(plots_grid[0])):
		# Display my images.
		plots_grid[1][j].axis('off')
		plots_grid[1][j].imshow((normal_images[j]/2**8).astype('uint8'))
		plots_grid[0][j].axis('off')
		plots_grid[0][j].imshow((mottled_images[j]/2**8).astype('uint8'))
		
		# Label them.
		labels_grid[1][j].axis('off')			
		labels_grid[1][j].annotate(normal_worms[j], xy = (0.5, 0.5), ha = 'center', va = 'center')
		labels_grid[0][j].axis('off')			
		labels_grid[0][j].annotate(mottled_worms[j], xy = (0.5, 0.5), ha = 'center', va = 'center')

	# Label the top and save the figure.
	actual_day = a_time.split('.')
	actual_day = int(actual_day[0]) + int(actual_day[1])*(1/8)
	my_figure.suptitle('Normal and Mottled Worms' + ' at Day ' + str(actual_day)[:5], fontsize = 15)


	# Save out my figure.
	graph_name = 'mottled' + '_' + a_time + '_gallery'
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def svr_traces(adult_df):
	'''
	Show the average traces over time of various health variables.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 8)
	variable1_traces = plotFigures.consistent_subgrid_coordinates((6, 4), (0, 0), my_width = 2, my_height = 2)
	variable2_traces = plotFigures.consistent_subgrid_coordinates((6, 4), (0, 2), my_width = 2, my_height = 2)
	variable3_traces = plotFigures.consistent_subgrid_coordinates((6, 4), (2, 0), my_width = 2, my_height = 2)
	variable4_traces = plotFigures.consistent_subgrid_coordinates((6, 4), (2, 2), my_width = 2, my_height = 2)
	variable5_traces = plotFigures.consistent_subgrid_coordinates((6, 4), (4, 1), my_width = 2, my_height = 2)
	my_subplots = [variable1_traces, variable2_traces, variable3_traces, variable4_traces, variable5_traces]
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.5, hspace = 0.7)

	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_cohorts = life_cohorts

	# Plot the actual stuff.
	my_variables = ['autofluorescence', 'eggs', 'movement', 'size', 'texture']
	for j in range(len(my_variables)):
		a_var = my_variables[j]
		for i in range(0, len(my_cohorts)):
			if len(my_cohorts[i]) > 0:
				a_cohort = my_cohorts[i]
				cohort_data = adult_df.mloc(adult_df.worms, [a_var])[a_cohort, 0, :]
				cohort_data = cohort_data[~np.isnan(cohort_data).all(axis = 1)]
				cohort_data = np.mean(cohort_data, axis = 0)
				(cohort_data, my_unit, fancy_name) = adult_df.display_variables(cohort_data, a_var)
				my_subplots[j].plot(adult_df.ages[:cohort_data.shape[0]], cohort_data, color = my_colors[i], linewidth = 2)
		my_subplots[j].set_title(fancy_name + ' Over Time')
		my_subplots[j].set_ylabel(my_unit)	
		my_subplots[j].set_xlabel('Age (Days Post-Maturity)')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)
	return 

def quality_life(adult_df):
	'''
	Plot quality-adjusted life-years of life vs. lifespan.
	'''
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 12)
	plots_grid = [plotFigures.consistent_subgrid_coordinates((1, 3), (0, i), my_width = 1, my_height = 1) for i in range(0, 3)]
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)

	cutoffs = (0, 3, 6)
	adultspans = selectData.get_adultspans(adult_df)	
	
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)	
	
	qalys = []
	for i in range(0, len(cutoffs)):	
		qaly = []
		for a_worm in adult_df.worms:
			worm_data = adult_df.mloc(worms = [a_worm], measures = ['health'])[0, 0, :]
			worm_data = worm_data[~np.isnan(worm_data)]
			(worm_data, my_unit, fancy_name) = adult_df.display_variables(worm_data, 'health')
			qaly.append(np.sum(worm_data - cutoffs[i])/8)

		qalys.append(np.array(qaly))
	
	for i in range(0, 3):
		for j in range(0, len(life_cohorts)):
			a_cohort = life_cohorts[j]
#			start_plot.scatter(my_adultspans, start_data, )	
		
			plots_grid[i].scatter(adultspans[a_cohort]/24, qalys[2-i][a_cohort], color = my_colors[j])
			plots_grid[i].set_xlabel('Days of Adult Life')
			plots_grid[i].set_ylabel('Quality-Adjusted Life Days')
			plots_grid[i].set_title('Zero Quality = ' + str(cutoffs[2-i]) + ' Predicted Days of Life Remaining')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return 

def cohort_gallery(adult_df, directory_bolus, a_time, refresh_choices = False):
	'''	
	Make a gallery of three randomly selected worms from each life cohort at a_time.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(21, 12)
	plots_grid = [[plotFigures.consistent_subgrid_coordinates((21, 12), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 7)] for j in range(0, 3)]	
	labels_grid = [[plotFigures.consistent_subgrid_coordinates((21, 12), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 7)] for j in range(0, 3)]
	
	# Get images.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)	
	if refresh_choices:
		random_worms = np.array([np.array(adult_df.worms)[np.random.choice(life_cohort, 3, replace = False)] for life_cohort in life_cohorts]).transpose()	
	else:
		random_worms = np.flipud(np.array([
			['2016.03.04 spe-9 13B 44', '2016.02.26 spe-9 11C 071', '2016.03.25 spe-9 15B 075'],
			['2016.03.04 spe-9 13C 70', '2016.03.25 spe-9 15A 74', '2016.03.14 spe-9 14 130'],
			['2016.03.25 spe-9 15B 070', '2016.03.04 spe-9 13A 19', '2016.03.14 spe-9 14 174'],
			['2016.03.25 spe-9 15A 41', '2016.03.31 spe-9 16 055', '2016.02.26 spe-9 11B 051'],
			['2016.03.14 spe-9 14 128', '2016.03.25 spe-9 15B 027', '2016.03.14 spe-9 14 101'],
			['2016.03.31 spe-9 16 184', '2016.02.26 spe-9 11B 029', '2016.03.31 spe-9 16 145'],
			['2016.03.14 spe-9 14 167', '2016.02.26 spe-9 11C 072', '2016.03.25 spe-9 15B 038']
		]).transpose())
	closest_times = [[selectData.closest_real_time(adult_df, a_worm, a_time, egg_mode = True) for a_worm in random_worms_group] for random_worms_group in random_worms]
	cohort_images = [[imageOperations.get_worm(directory_bolus, random_worms[i][j], closest_times[i][j], box_size = 500) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Clean up images and wrap them in colored borders.
	images_max = np.max(np.array(cohort_images))
	dtype_max = 2**16-1
	cohort_images = [[(cohort_images[i][j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]		
	cohort_images = [[imageOperations.border_box(cohort_images[i][j], border_color = my_colors[j]*2**16, border_width = 500//15) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Make the figure itself.
	for i in range(0, len(plots_grid)):
		for j in range(0, len(plots_grid[0])):
			# Display my image.
			plots_grid[i][j].axis('off')
			plots_grid[i][j].imshow((cohort_images[i][j]/2**8).astype('uint8'))
			
			# Label it.
			labels_grid[i][j].axis('off')			
			labels_grid[i][j].annotate(random_worms[i][j], xy = (0.5, 0.5), ha = 'center', va = 'center')
	
	# Label the top and save the figure.
	actual_day = a_time.split('.')
	actual_day = int(actual_day[0]) + int(actual_day[1])*(1/8)
	my_figure.suptitle('Lifespan Cohorts' + ' at Day ' + str(actual_day)[:5], fontsize = 15)

	# Save out my figure.
	graph_name = 'cohort' + '_' + a_time + '_gallery'
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def fate_lifes(adult_df):
	'''
	'''
	fate_frame = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.29 Fate Categories\2016.04.29 Fates Annotation.tsv', sep = '\t')
	fate_dict = {'g':'Gonad', 'f':'Food Packing', 'c':'Clear', 'p':'Pressurized', 'n':'Normal'}	
	
	g_mask = (fate_frame.loc[:, 'Fates'] == 'g').values
	f_mask = (fate_frame.loc[:, 'Fates'] == 'f').values
	c_mask = (fate_frame.loc[:, 'Fates'] == 'c').values
	p_mask = (fate_frame.loc[:, 'Fates'] == 'p').values
#	n_mask = (fate_frame.loc[:, 'Fates'] == 'n').values
	my_masks = {'g':g_mask, 'f':f_mask, 'c':c_mask, 'p':p_mask}#, 'n':n_mask}
	
	my_figure = plt.figure()
	adultspans = selectData.get_adultspans(adult_df)/24
	my_figure.set_size_inches(18, 7)
	life_histogram = plotFigures.consistent_subgrid_coordinates((3, 3), (1, 1), my_width = 1, my_height = 2)
	fates_table = plotFigures.consistent_subgrid_coordinates((3, 3), (0, 0), my_width = 3, my_height = 1)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.6)


	mask_keys = sorted(list(my_masks.keys()))

	
	# Plot smoothed kde density curve for overall population.
	for a_category in my_masks.keys():	
		kde_density = scipy.stats.gaussian_kde(adultspans[my_masks[a_category]])
		my_xrange = np.linspace(0, 18, 200)
		kde_density._compute_covariance()
		life_histogram.plot(my_xrange, kde_density(my_xrange), linewidth = 2, label = fate_dict[a_category])			
#		life_histogram.hist(adultspans[my_masks[a_category]])

	life_histogram.legend(loc = 'top left')
	life_histogram.set_xlabel('Days of Adult Lifespan')
	life_histogram.set_ylabel('Normalized Density')
	life_histogram.set_title('Lifespan Distributions of Distinct Death Morphologies')


#	fates_table = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
	
	my_data = {fate_dict[a_key]: adultspans[my_masks[a_key]] for a_key in mask_keys}
	
	table_data = pd.DataFrame([], index = [fate_dict[a_key] for a_key in mask_keys], columns = [fate_dict[a_key] for a_key in mask_keys])
	for group_a in table_data.index:
		for group_b in table_data.columns:
			table_data.loc[group_a, group_b] = '%.3f' % scipy.stats.ks_2samp(my_data[group_a], my_data[group_b])[1]
		
	# Make the table nicely.
	fates_table.axis('off')
	fates_table.xaxis.set_visible(False)
	fates_table.yaxis.set_visible(False)
	my_table = fates_table.table(colWidths = [0.08]*5, cellText = np.array(table_data), rowLabels = [fate_dict[a_key] + ' (' + str(int(np.sum(my_masks[a_key]))) + ')' for a_key in mask_keys], colLabels = table_data.columns, loc = 'center', cellLoc = 'center', rowLoc = 'right')
	my_table.set_fontsize(14)
	fates_table.set_title('p-values for Kolmogorov-Smirnov Tests')	
	plt.tight_layout()

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return	

def fates_gallery(adult_df, directory_bolus, refresh_choices = False):
	'''
	Make a gallery of worms based on their fate.
	'''
	
	fate_frame = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.29 Fate Categories\2016.04.29 Fates Annotation.tsv', sep = '\t')
	fate_dict = {'g':'Gonad', 'f':'Food Packing', 'c':'Clear', 'p':'Pressurized', 'n':'Normal'}	
	
	g_mask = (fate_frame.loc[:, 'Fates'] == 'g').values
	f_mask = (fate_frame.loc[:, 'Fates'] == 'f').values
	c_mask = (fate_frame.loc[:, 'Fates'] == 'c').values
	p_mask = (fate_frame.loc[:, 'Fates'] == 'p').values
#	n_mask = (fate_frame.loc[:, 'Fates'] == 'n').values
	my_masks = {'g':g_mask, 'f':f_mask, 'c':c_mask, 'p':p_mask}

	mask_keys = sorted(list(my_masks.keys()))
	if refresh_choices:
		permuted_worms = [np.random.permutation(np.array(adult_df.worms)[my_masks[mask_keys[i]]]) for i in range(0, len(mask_keys))]
		chosen_worms = [random_ordering[:4] for random_ordering in permuted_worms]
	else:
		chosen_worms = [['2016.02.20 spe-9 10A 19', '2016.02.26 spe-9 11C 081', '2016.02.29 spe-9 12B 36', '2016.03.04 spe-9 13B 25'], ['2016.03.14 spe-9 14 147', '2016.03.31 spe-9 16 018', '2016.03.04 spe-9 13A 20', '2016.03.14 spe-9 14 108'], ['2016.03.04 spe-9 13B 42', '2016.03.25 spe-9 15B 118', '2016.03.14 spe-9 14 057', '2016.02.29 spe-9 12B 34'], ['2016.03.04 spe-9 13C 70', '2016.02.26 spe-9 11B 061', '2016.02.16 spe-9 9 096', '2016.02.20 spe-9 10B 46']]

	full_chosen_worms = [[[] for j in range(0, len(chosen_worms[i]))] for i in range(0, len(chosen_worms))]
	pre_death = '1.0'
	hours_before = pre_death.split('.')
	hours_before = float(hours_before[0])*24 + float(hours_before[1])*3
	for j in range(0, len(mask_keys)):
		for i in range(0, len(chosen_worms)):
			a_worm = chosen_worms[j][i]
			pre_death_index = np.nanargmin(np.abs(adult_df.mloc(worms = [a_worm], measures = ['ghost_age'])[0, 0, :] + hours_before))
			pre_death_time = adult_df.times[pre_death_index]
			real_time = selectData.closest_real_time(adult_df, a_worm, pre_death_time, egg_mode = True)
			full_worm = str(a_worm + '/' + real_time)
			full_chosen_worms[j][i] = full_worm

	print(full_chosen_worms)
	

#	raise BaseException('')
	
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 16)
	plots_grid = [[plotFigures.consistent_subgrid_coordinates((12, 16), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 4)] for j in range(0, 4)]	
	labels_grid = [[plotFigures.consistent_subgrid_coordinates((12, 16), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 4)] for j in range(0, 4)]
	
	# Get images.
	closest_times = [[a_worm.split('/')[1] for a_worm in random_worms_group] for random_worms_group in full_chosen_worms]
	cohort_images = [[imageOperations.get_worm(directory_bolus, full_chosen_worms[i][j].split('/')[0], closest_times[i][j], box_size = 500) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Clean up images and wrap them in colored borders.
	images_max = np.max(np.array(cohort_images))
	dtype_max = 2**16-1
	cohort_images = [[(cohort_images[i][j].astype('float64')/images_max*dtype_max).astype('uint16') for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]		
	cohort_images = [[imageOperations.border_box(cohort_images[i][j], border_color = [0, 0, 0], border_width = 500//15) for j in range(0, len(plots_grid[0]))] for i in range(0, len(plots_grid))]	

	# Make the figure itself.
	for i in range(0, len(plots_grid)):
		plots_grid[i][0].set_ylabel(fate_dict[mask_keys[i]])
		for j in range(0, len(plots_grid[0])):
			# Display my image.
#			plots_grid[i][j].axis('off')
			plots_grid[i][j].set_frame_on(False)
			plots_grid[i][j].set_xticks([])
			plots_grid[i][j].set_yticks([])
#			plots_grid[i][j].set_xlabels([])
#			plots_grid[i][j].set_xlabels([])
			plots_grid[i][j].imshow((cohort_images[i][j]/2**8).astype('uint8'))
			
			# Label it.
			labels_grid[i][j].axis('off')			
			labels_grid[i][j].annotate(full_chosen_worms[i][j].split('/')[0], xy = (0.5, 0.5), ha = 'center', va = 'center')
	
	
	
	# Label the top and save the figure.
	my_figure.suptitle('Fates Gallery', fontsize = 15)
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def mottled_longevity(adult_df):
	'''
	Some plots about being mottled/long-lived.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	life_histogram = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)

	adultspans = selectData.get_adultspans(adult_df)/24
	mottled_mask = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.24 Mottled Annotations\manual_mottled.tsv', sep = '\t').loc[:, 'Mottled?'].values
	
	# Plot smoothed kde density curve for overall population.
	kde_density = scipy.stats.gaussian_kde(adultspans)
	my_xrange = np.linspace(0, 17, 200)
	kde_density._compute_covariance()
	life_histogram.plot(my_xrange, kde_density(my_xrange)*len(adultspans)/len(adultspans), color = 'blue', linewidth = 2, label = 'Overall')		
	
	# Plot smoothed kde density curve for mottled worms.
	kde_density = scipy.stats.gaussian_kde(adultspans[mottled_mask])
	my_xrange = np.linspace(0, 17, 200)
	kde_density._compute_covariance()
	life_histogram.plot(my_xrange, kde_density(my_xrange)*len(adultspans[mottled_mask])/len(adultspans), color = 'red', linewidth = 2, label = 'Mottled')		

	# Plot smoothed kde density curve for non-mottled worms.
	kde_density = scipy.stats.gaussian_kde(adultspans[~mottled_mask])
	my_xrange = np.linspace(0, 17, 200)
	kde_density._compute_covariance()
	life_histogram.plot(my_xrange, kde_density(my_xrange)*len(adultspans[~mottled_mask])/len(adultspans), color = 'green', linewidth = 2, label = 'Non-Mottled')		

	# Label and save the plot.
	life_histogram.legend(loc = 'upper left')
	life_histogram.set_title('Lifespans of Mottled and Non-Mottled Subpopulations')
	life_histogram.set_xlabel('Days of Adult Lifespan')
	life_histogram.set_ylabel('Density')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def mottled_over_time(adult_df):
	'''
	Some plots about being mottled/small over time.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 12)
	size_distribution = [plotFigures.consistent_subgrid_coordinates((1, 3), (0, 2-i), my_width = 1, my_height = 1) for i in range(3)]
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)
	
	my_times = ['2.0', '6.0', '10.0']
	mottled_mask = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.24 Mottled Annotations\manual_mottled.tsv', sep = '\t').loc[:, 'Mottled?'].values
	
	
	for i in range(0, len(my_times)):
		a_time = my_times[i]
		current_sizes = adult_df.mloc(measures = ['adjusted_size'], times = [a_time])[:, 0, 0]
		(current_sizes, my_unit, fancy_name) = adult_df.display_variables(current_sizes, 'adjusted_size')
		mottled_sizes = current_sizes[mottled_mask]
		nonmottled_sizes = current_sizes[~mottled_mask]
		
		current_sizes = current_sizes[~np.isnan(current_sizes)]
		mottled_sizes = mottled_sizes[~np.isnan(mottled_sizes)]
		nonmottled_sizes = nonmottled_sizes[~np.isnan(nonmottled_sizes)]
	
		# Plot smoothed kde density curve for overall population.
		kde_density = scipy.stats.gaussian_kde(current_sizes)
		my_xrange = np.linspace(0, 0.12, 200)
		kde_density._compute_covariance()
		size_distribution[i].plot(my_xrange, kde_density(my_xrange)*len(current_sizes)/len(current_sizes), color = 'blue', linewidth = 2, label = 'Overall')		
		
		# Plot smoothed kde density curve for mottled worms.
		kde_density = scipy.stats.gaussian_kde(mottled_sizes)
		my_xrange = np.linspace(0, 0.12, 200)
		kde_density._compute_covariance()
		size_distribution[i].plot(my_xrange, kde_density(my_xrange)*len(mottled_sizes)/len(current_sizes), color = 'red', linewidth = 2, label = 'Mottled')		
	
		# Plot smoothed kde density curve for non-mottled worms.
		kde_density = scipy.stats.gaussian_kde(nonmottled_sizes)
		my_xrange = np.linspace(0, 0.12, 200)
		kde_density._compute_covariance()
		size_distribution[i].plot(my_xrange, kde_density(my_xrange)*len(nonmottled_sizes)/len(current_sizes), color = 'green', linewidth = 2, label = 'Non-Mottled')		
	
		# Label and save the plot.
		size_distribution[i].legend(loc = 'upper left')
		size_distribution[i].set_title('Areas of Mottled and Non-Mottled Subpopulations at Day ' + my_times[i])
		size_distribution[i].set_xlabel('mm^2')
		size_distribution[i].set_ylabel('Density')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return


def plate_sizes(adult_df):
	'''
	Check consistency of lifespans over time.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	plate_sizes = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
	
	# Plot KDEs of slides.
	the_sizes = imageOperations.plate_sizes_12_day()
	kde_density = scipy.stats.gaussian_kde(the_sizes)
	my_xrange = np.linspace(0, 0.16, 200)
	plate_sizes.plot(my_xrange, kde_density(my_xrange), color = 'black', linewidth = 2)		
	plate_sizes.set_ylabel('Density')
	plate_sizes.set_xlim([0, 0.16])
	plate_sizes.set_xlabel('Cross-Sectional Area (mm^2)')
	plate_sizes.set_title('Day 12 Adult Size Distributions on NGM Plates')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def time_consistency(adult_df):
	'''
	Check consistency of lifespans over time.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(6, 4)
	slide_lifespans = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)
	
	# Read in my metadata.	
	metadata_dir = r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.27 Complete spe-9 Metadata'	
	my_files = os.listdir(metadata_dir)
	metadata_dict = {}
	for a_file in my_files:
		with open(metadata_dir + os.path.sep + a_file, 'r') as read_file:
			metadata_dict['.'.join(a_file.split('.')[:-1])] = json.loads(read_file.read())
	
	# Fill in the worm positions.	
	worm_list = adult_df.worms
	worm_locations = np.empty((len(worm_list), 3))	
	worm_locations[:] = np.nan	
	for i in range(0, len(worm_list)):
		a_worm = worm_list[i]
		my_experiment = ' '.join(a_worm.split(' ')[:-1])
		my_number = a_worm.split(' ')[-1]
		worm_locations[i, :] = metadata_dict[my_experiment]['positions'][my_number]

	# These indices are not inclusive. They include the left bound but not the right one.
	slide_indices = [
		(0, 7),
		(7, 46),
		(46, 155),
		(155, 242),
		(242, 373),
		(373, 566),
		(566, 734)			
	]
	
	# Get lifespans and select colors for each slide.
	adultspans = selectData.get_adultspans(adult_df)/24
	colors = np.array_split(plotFigures.viridis_data, len(slide_indices))
	
	# Plot KDEs of slides.
	for i in range(0, len(slide_indices)):
		a_slide = slide_indices[i]
		slide_lifes = adultspans[a_slide[0]:a_slide[1]]
		kde_density = scipy.stats.gaussian_kde(slide_lifes)
		my_xrange = np.linspace(0, 17, 200)
		slide_lifespans.plot(my_xrange, kde_density(my_xrange)*len(slide_lifes)/len(adultspans), color = colors[i][0], linewidth = 2, label = 'Non-Mottled')		
		slide_lifespans.set_ylabel('Density')
		slide_lifespans.set_xlabel('Days of Adult Life')
		slide_lifespans.set_title('Adult Lifespan Distributions by Experiment Day')

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def experimental_consistency(adult_df, time_mode = False):
	'''
	A few plots to illustrate lack of confounding factors.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 8)
	slide_lifespans = plotFigures.consistent_subgrid_coordinates((3, 2), (0, 1), my_width = 1, my_height = 1)
	great_lawn_size = plotFigures.consistent_subgrid_coordinates((3, 2), (1, 1), my_width = 1, my_height = 1)
	relative_position = plotFigures.consistent_subgrid_coordinates((3, 2), (2, 1), my_width = 1, my_height = 1)
	x_position = plotFigures.consistent_subgrid_coordinates((3, 2), (0, 0), my_width = 1, my_height = 1)
	y_position = plotFigures.consistent_subgrid_coordinates((3, 2), (1, 0), my_width = 1, my_height = 1)
	z_position = plotFigures.consistent_subgrid_coordinates((3, 2), (2, 0), my_width = 1, my_height = 1)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)
	
	# Read in my metadata.	
	metadata_dir = r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.27 Complete spe-9 Metadata'	
	my_files = os.listdir(metadata_dir)
	metadata_dict = {}
	for a_file in my_files:
		with open(metadata_dir + os.path.sep + a_file, 'r') as read_file:
			metadata_dict['.'.join(a_file.split('.')[:-1])] = json.loads(read_file.read())
	
	# Fill in the worm positions.	
	worm_list = adult_df.worms
	worm_locations = np.empty((len(worm_list), 3))	
	worm_locations[:] = np.nan	
	for i in range(0, len(worm_list)):
		a_worm = worm_list[i]
		my_experiment = ' '.join(a_worm.split(' ')[:-1])
		my_number = a_worm.split(' ')[-1]
		worm_locations[i, :] = metadata_dict[my_experiment]['positions'][my_number]

	# These indices are not inclusive. They include the left bound but not the right one.
	slide_indices = [ 
		(0, 2),
		(2, 7),
		(7, 28),
		(28, 46),
		(46, 66),
		(66, 98),
		(98, 155),
		(155, 190),
		(190, 242),
		(242, 296),
		(296, 373),
		(373, 460),
		(460, 543),
		(543, 566),
		(566, 643),
		(643, 734)
	]
	
	if time_mode:
		slide_indices = [
			(0, 7),
			(7, 46),
			(46, 155),
			(155, 242),
			(242, 373),
			(373, 566),
			(566, 734)			
		]
	
	# Get lifespans and select colors for each slide.
	adultspans = selectData.get_adultspans(adult_df)/24
	colors = np.array_split(plotFigures.viridis_data, len(slide_indices))
	
	# Plot KDEs of slides.
	for i in range(0, len(slide_indices)):
		a_slide = slide_indices[i]
		slide_lifes = adultspans[a_slide[0]:a_slide[1]]
		kde_density = scipy.stats.gaussian_kde(slide_lifes)
		my_xrange = np.linspace(0, 17, 200)
		slide_lifespans.plot(my_xrange, kde_density(my_xrange)*len(slide_lifes)/len(adultspans), color = colors[i][0], linewidth = 2, label = 'Non-Mottled')		
		slide_lifespans.set_ylabel('Density')
		slide_lifespans.set_xlabel('Days of Adult Life')
		slide_lifespans.set_title('Adult Lifespan Distributions by Slide')

	# Get great lawn data.	
	great_lawn_areas = adult_df.mloc(measures = ['great_lawn_area'], times = ['0.0'])[:, 0, 0]
	(great_lawn_areas, y_units, great_lawn_string) = adult_df.display_variables(great_lawn_areas, 'great_lawn_area')
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(adultspans, great_lawn_areas)[0]**2)
	
	# Plot and label my stuff.
	great_lawn_size.scatter(adultspans, great_lawn_areas)
	great_lawn_size.set_ylabel(y_units)
	great_lawn_size.set_xlabel('Days of Adult Life')
	great_lawn_size.set_title('Lifespan vs. ' + great_lawn_string)
	great_lawn_size.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (2, 3.5))
	
	# Compute distances to center of slide.	
	relative_distances = np.empty(len(adultspans))
	relative_distances[:] = np.nan	
	for a_slide in slide_indices:
		slide_center = np.mean(worm_locations[a_slide[0]:a_slide[1], :], axis = 0)
		for i in range(a_slide[0], a_slide[1]):
			relative_distances[i] = np.linalg.norm(slide_center - worm_locations[i], 2)

	# Plot and label my stuff.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(relative_distances, great_lawn_areas)[0]**2)	
	relative_position.scatter(adultspans, relative_distances)
	relative_position.set_ylabel('mm')
	relative_position.set_xlabel('Days of Adult Life')
	relative_position.set_title('Lifespan vs. ' + 'Distance from Center of Slide')
	relative_position.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (2, -1))

	# Plot and label my stuff.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(worm_locations[:, 0], adultspans)[0]**2)	
	x_position.scatter(adultspans, worm_locations[:, 0])
	x_position.set_ylabel('Position (mm from arbitrary zero point on stage)')
	x_position.set_xlabel('Days of Adult Life')
	x_position.set_title('Lifespan vs. ' + 'X Location')
	x_position.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (2, 10))

	# Plot and label my stuff.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(worm_locations[:, 1], adultspans)[0]**2)	
	y_position.scatter(adultspans, worm_locations[:, 1])
	y_position.set_ylabel('Position (mm from arbitrary zero point on stage)')
	y_position.set_xlabel('Days of Adult Life')
	y_position.set_title('Lifespan vs. ' + 'Y Location')
	y_position.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (2, -1))
	
	# Plot and label my stuff.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(worm_locations[:, 2], adultspans)[0]**2)	
	z_position.scatter(adultspans, worm_locations[:, 2])
	z_position.set_ylabel('Position (mm from arbitrary zero point on stage)')
	z_position.set_xlabel('Days of Adult Life')
	z_position.set_title('Lifespan vs. ' + 'Z Location')
	z_position.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (1, 23.8))
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def worm_finding(adult_df, directory_bolus, size_df = None):
	'''
	Some plots to validate worm-finding and image analysis.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 4)
	worm_size = plotFigures.consistent_subgrid_coordinates((3, 1), (0, 0), my_width = 1, my_height = 1)
	worm_x = plotFigures.consistent_subgrid_coordinates((3, 1), (1, 0), my_width = 1, my_height = 1)
	worm_y = plotFigures.consistent_subgrid_coordinates((3, 1), (2, 0), my_width = 1, my_height = 1)

	# Get my data.
	if size_df is None:
		size_df = organizeData.validate_generated_masks(directory_bolus.working_directory, directory_bolus.human_directory)

	# Adjust the size to proper units.
	(m, b) = (0.69133050735497115, 8501.9616379946274)
	old_size = size_df.loc[:, 'F_Size'].values
	new_size = (m*old_size) + b
	(units_fsize, my_unit, fancy_name) = adult_df.display_variables(new_size, 'adjusted_size')
	(units_hsize, my_unit, fancy_name) = adult_df.display_variables(size_df.loc[:, 'H_Size'].values, 'adjusted_size')
	
	# Actually plot the size.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(list(size_df.loc[:, 'F_Size']), list(size_df.loc[:, 'H_Size']))[0]**2)	
	worm_size.scatter(units_fsize, units_hsize)
	worm_size.set_ylabel('Manually Measured Size (mm^2)')
	worm_size.set_xlabel('Automatically Measured Size (mm^2)')
	worm_size.set_title('Worm Size Image Segmentation Validation')
	worm_size.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (800, 200))
	
	# Actually plot the x position.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(list(size_df.loc[:, 'F_CentroidX']), list(size_df.loc[:, 'H_CentroidX']))[0]**2)	
	worm_x.scatter(list(size_df.loc[:, 'F_CentroidX']), list(size_df.loc[:, 'H_CentroidX']))
	worm_x.set_ylabel('Manually Measured X Centroid (pixel)')
	worm_x.set_xlabel('Automatically Measured X Centroid (pixel)')
	worm_x.set_title('Worm X Location Image Segmentation Validation')
	worm_x.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (1500, 1000))
		
	# Actually plot the y position.
	my_rsquared = '%.3f' % (scipy.stats.pearsonr(list(size_df.loc[:, 'F_CentroidY']), list(size_df.loc[:, 'H_CentroidY']))[0]**2)	
	worm_y.scatter(list(size_df.loc[:, 'F_CentroidY']), list(size_df.loc[:, 'H_CentroidY']))
	worm_y.set_ylabel('Manually Measured Y Centroid (pixel)')
	worm_y.set_xlabel('Automatically Measured Y Centroid (pixel)')
	worm_y.set_title('Worm Y Location Image Segmentation Validation')
	worm_y.annotate('r^2 = ' + str(my_rsquared)[:5], xy = (1500, 1000))
	
	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def show_spans(adult_df, relative_time = 0.5):
	'''
	Change up the cutoff for spans.
	'''
	# Set up figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 8)
	health_traces = plotFigures.consistent_subgrid_coordinates((2, 2), (0, 1), my_width = 1, my_height = 1)
	health_traces_normed = plotFigures.consistent_subgrid_coordinates((2, 2), (1, 1), my_width = 1, my_height = 1)
	health_spans = plotFigures.consistent_subgrid_coordinates((2, 2), (0, 0), my_width = 1, my_height = 1)
	health_spans_normed = plotFigures.consistent_subgrid_coordinates((2, 2), (1, 0), my_width = 1, my_height = 1)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = None, hspace = 0.3)

	# Label plots properly.
	plotFigures.subfigure_label([health_traces, health_traces_normed, health_spans, health_spans_normed])

	# Show spans.
	cannedFigures.show_spans(health_traces, health_traces_normed, health_spans, health_spans_normed, adult_df, relative_time = relative_time)

	# Save out my figure.
	graph_name = '%.2f' % relative_time + '_spans'
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)		
	return


def non_mottled_health_traces(adult_df):
	'''
	Plot a trace for overall health.
	'''
	# Set up the figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 16)
	absolute_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 6), my_width = 2, my_height = 2)
	start_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (0, 4), my_width = 2, my_height = 2)
	rate_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (2, 4), my_width = 2, my_height = 2)
	end_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (4, 4), my_width = 2, my_height = 2)
	variable_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 2), my_width = 2, my_height = 2)
	variable_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 2), my_width = 2, my_height = 2)
	relative_trace = plotFigures.consistent_subgrid_coordinates((6, 8), (1, 0), my_width = 2, my_height = 2)
	relative_scatter = plotFigures.consistent_subgrid_coordinates((6, 8), (3, 0), my_width = 2, my_height = 2)
	plt.subplots_adjust(left = None, bottom = None, right = None, top = None, wspace = 0.4, hspace = 0.7)
	
	# Label plots properly.
	plotFigures.subfigure_label([absolute_trace, start_scatter, rate_scatter, end_scatter, variable_trace, variable_scatter, relative_trace, relative_scatter])

	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_cohorts = life_cohorts
	mottled_mask = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.24 Mottled Annotations\manual_mottled.tsv', sep = '\t').loc[:, 'Mottled?'].values
	non_mottled_worms = np.array(adult_df.worms)[np.invert(mottled_mask)]	
	non_mottled_indices = [adult_df.worm_indices[a_worm] for a_worm in non_mottled_worms]

	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	


	# Prepare my "inflection" data.	
	geometry_dict = computeStatistics.one_d_geometries(adult_df, 'health')
#	(start_data, my_unit, fancy_name) = adult_df.display_variables(geometry_dict['start'], 'health')
	mean_start = np.mean(geometry_dict['start'])		
	
	inflection_data = geometry_dict['absolute_inflection']
	relative_inflection = geometry_dict['self_inflection']

	health_name = 'Prognosis (Remaining Days)'

	cannedFigures.cohort_traces(absolute_trace, 'health', adult_df, the_title = health_name + ' Over Time', the_xlabel = 'Days of Adult Life', the_ylabel = health_name, x_normed = False, skip_conversion = False, only_worms = non_mottled_indices)

	cannedFigures.cohort_scatters(start_scatter, my_adultspans, geometry_dict['start'], adult_df, the_title = 'Start', the_xlabel = 'Days of Adult Lifespan', the_ylabel = health_name + ' at Start of Adulthood', label_coordinates = (2, 7), only_worms = non_mottled_indices)
	cannedFigures.cohort_scatters(rate_scatter, my_adultspans, geometry_dict['rate'], adult_df, the_title = 'Rate', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Rate of Decrease of ' + health_name, label_coordinates = (8, 1.5), polyfit_degree = 2, only_worms = non_mottled_indices)
	cannedFigures.cohort_scatters(end_scatter, my_adultspans, geometry_dict['end'], adult_df, the_title = 'End', the_xlabel = 'Days of Adult Lifespan', the_ylabel = health_name + ' at Death', label_coordinates = (1, 1), polyfit_degree = 2, only_worms = non_mottled_indices)





	# Plot the traces and scatter for absolute inflection.
	cannedFigures.cohort_traces(variable_trace, 'health', adult_df, the_title = 'Prognosis Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = 'Prognosis (Days)', x_normed = True, skip_conversion = False, only_worms = non_mottled_indices)
	variable_trace.set_ylim([0, 1.1*mean_start])
	cannedFigures.cohort_scatters(variable_scatter, my_adultspans, inflection_data, adult_df, the_title = 'Absolute Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Average Deviation (Days)', label_coordinates = (12, 2), only_worms = non_mottled_indices)

	# Plot the traces and scatter for relative inflection.
	cannedFigures.cohort_traces(relative_trace, 'health', adult_df, the_title = 'Relative Prognosis Over Normalized Time', the_xlabel = 'Fractional Adult Lifespan', the_ylabel = 'Relative Prognosis (Fractional Life)', x_normed = True, y_normed = True, only_worms = non_mottled_indices)
	relative_trace.set_ylim([-0.1, 1.1])
	cannedFigures.cohort_scatters(relative_scatter, my_adultspans, relative_inflection, adult_df, the_title = 'Relative Deviation', the_xlabel = 'Days of Adult Lifespan', the_ylabel = 'Average Deviation (Relative Prognosis)', label_coordinates = (4, -0.4), only_worms = non_mottled_indices)
#	relative_scatter.set_ylim([-0.7, 0.7])
	
	# Make an overall title.
	plt.suptitle('Prognosis Deviation Excluding Mottled Subpopulation', fontsize = 15)	

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

	
def worm_finding_gallery(adult_df, directory_bolus, size_df = None, refresh_choices = False):
	'''
	A gallery of worm mask comparisons.
	'''	
	
	if size_df is None:
		size_df = organizeData.validate_generated_masks(directory_bolus.working_directory, directory_bolus.human_directory)
	
	
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(9, 8)
	plots_grid = [[plotFigures.consistent_subgrid_coordinates((9, 8), (3*i, 4*j + 1), my_width = 3, my_height = 3) for i in range(0, 3)] for j in range(0, 2)]	
	labels_grid = [[plotFigures.consistent_subgrid_coordinates((9, 8), (3*i, 4*j), my_width = 3, my_height = 1) for i in range(0, 3)] for j in range(0, 2)]
	
	# Get images.
	if refresh_choices:
		selected_worms = np.random.choice(size_df.index, size = 3, replace = True)
	else:
		selected_worms = ['2016.03.25 spe-9 15B 036/2016-03-30t1717', '2016.03.14 spe-9 14 146/2016-03-25t0333', '2016.03.25 spe-9 15B 046/2016-04-04t0044']

	machine_mask_locations = [directory_bolus.working_directory + os.path.sep + a_worm.split('/')[0] + os.path.sep + a_worm.split('/')[1] + ' mask.png' for a_worm in selected_worms]
	human_mask_locations = [directory_bolus.human_directory + os.path.sep + a_worm.split('/')[0] + os.path.sep + a_worm.split('/')[1] + ' hmask.png' for a_worm in selected_worms]

	machine_masks = []
	human_masks = []
	for i in range(0, len(human_mask_locations)):
		machine_mask = freeimage.read(machine_mask_locations[i]).astype('bool')
		machine_maskmask = imageOperations.bound_box(machine_mask, box_size = 500)[0]
		machine_mask = machine_mask[machine_maskmask].astype('uint16')
		machine_mask[machine_mask > 0] = -1
		machine_masks.append(np.reshape(machine_mask, (1000, 1000)))

		human_mask = freeimage.read(human_mask_locations[i]).astype('bool')
		human_maskmask = imageOperations.bound_box(human_mask, box_size = 500)[0]
		human_mask = human_mask[human_maskmask].astype('uint16')
		human_mask[human_mask > 0] = -1
		human_masks.append(np.reshape(human_mask, (1000, 1000)))

	# Clean up images and wrap them in colored borders.
	machine_masks = [imageOperations.border_box(machine_masks[j], border_color = [0, 0, 0], border_width = 1) for j in range(0, len(machine_masks))]
	human_masks = [imageOperations.border_box(human_masks[j], border_color = [0, 0, 0], border_width = 1) for j in range(0, len(human_masks))]
	
	# Make the figure itself.
	for j in range(0, len(plots_grid[0])):
		# Display my image.
		plots_grid[0][j].axis('off')
		plots_grid[0][j].imshow(machine_masks[j].astype('bool'))
		plots_grid[1][j].axis('off')
		plots_grid[1][j].imshow(human_masks[j].astype('bool'))

		# Label it.
		labels_grid[0][j].axis('off')			
		labels_grid[0][j].annotate('Automated', xy = (0.5, 0.75), ha = 'center', va = 'center')
		labels_grid[0][j].annotate(selected_worms[j].split('/')[0], xy = (0.5, 0.50), ha = 'center', va = 'center')
		labels_grid[0][j].annotate(selected_worms[j].split('/')[1], xy = (0.5, 0.25), ha = 'center', va = 'center')
		labels_grid[1][j].axis('off')			
		labels_grid[1][j].annotate('Manual', xy = (0.5, 0.75), ha = 'center', va = 'center')
		labels_grid[1][j].annotate(selected_worms[j].split('/')[0], xy = (0.5, 0.50), ha = 'center', va = 'center')
		labels_grid[1][j].annotate(selected_worms[j].split('/')[1], xy = (0.5, 0.25), ha = 'center', va = 'center')
	
	# Label the top and save the figure.
	my_figure.suptitle('Automated and Manually Detected Worm Locations', fontsize = 15)

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return	

def development_vs_adulthood(adult_df):
	'''
	Make scatterplots relating development, adult lifespan, and overall lifespan.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(18, 4)
	development_adult = plotFigures.consistent_subgrid_coordinates((3, 1), (0, 0), my_width = 1, my_height = 1)
	development_life = plotFigures.consistent_subgrid_coordinates((3, 1), (1, 0), my_width = 1, my_height = 1)
	adult_life = plotFigures.consistent_subgrid_coordinates((3, 1), (2, 0), my_width = 1, my_height = 1)
	
	# Set up some needed data.
	my_adultspans = selectData.get_adultspans(adult_df)/24
	my_lifespans = selectData.get_lifespans(adult_df)/24
	my_developspans = my_lifespans - my_adultspans
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)	
	

	# Plot my scatters.
	for i in range(0, len(life_cohorts)):
		a_cohort = life_cohorts[i]
		development_adult.scatter(my_adultspans[a_cohort], my_developspans[a_cohort], color = my_colors[i])	
		development_life.scatter(my_lifespans[a_cohort], my_developspans[a_cohort], color = my_colors[i])	
		adult_life.scatter(my_adultspans[a_cohort], my_lifespans[a_cohort], color = my_colors[i])	

	# Label the plots.
	development_adult.set_title('Development vs. Adult Lifespan')
	development_adult.set_aspect('auto')
	development_adult.set_xlabel('Days of Adult Lifespan')
	development_adult.set_ylabel('Days of Development')
	label_string = 'r^2 = ' + str(computeStatistics.quick_pearson(my_developspans, my_adultspans))[:5]
	development_adult.annotate(label_string, (2, 3), textcoords = 'data', size = 10)	

	development_life.set_title('Development vs. Total Lifespan')
	development_life.set_aspect('auto')
	development_life.set_xlabel('Days of Total Lifespan')
	development_life.set_ylabel('Days of Development')
	label_string = 'r^2 = ' + str(computeStatistics.quick_pearson(my_developspans, my_lifespans))[:5]
	development_life.annotate(label_string, (4, 3), textcoords = 'data', size = 10)	

	adult_life.set_title('Adult vs. Total Lifespan')
	adult_life.set_aspect('auto')
	adult_life.set_xlabel('Days of Adult Lifespan')
	adult_life.set_ylabel('Days of Total Lifespan')
	label_string = 'r^2 = ' + str(computeStatistics.quick_pearson(my_adultspans, my_lifespans))[:5]
	adult_life.annotate(label_string, (4, 15), textcoords = 'data', size = 10)	

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return


#def spans

def geometry_table(adult_df, refresh_values = False):




	all_measures = ['autofluorescence', 'eggs', 'movement', 'size', 'texture', 'health']
	geometry_dicts = [computeStatistics.one_d_geometries(adult_df, a_measure) for a_measure in all_measures]
#	geometries = sorted(list(geometry_dicts[0].keys()))
#	geometries.remove('lifespan')
#	geometries = [geometry.replace('_', ' ').title() for geometry in geometries]
	geometries = ['Start', 'Rate', 'End', 'Absolute Inflection', 'Self Inflection']
	print(geometries)
	geometry_names = ['Start', 'Rate', 'End', 'Absolute Deviation', 'Self Deviation']

	individual_measures = all_measures[:-1]
	grammar_names = [adult_df.display_names(a_measure) for a_measure in all_measures]	
	grammar_names.extend(['Youthfulness Index', 'Predicted Survival', 'Linear Prognosis', 'Non-Mottled Prognosis'])	
	
#	table_data = [['0.031, 0.001', '0.684, 0.468', '-0.388]']]
	table_data = pd.DataFrame([], index = grammar_names, columns = geometry_names)

#	health_variance = computeStatistics.quick_pearson(*adult_df.flat_data(['health', 'ghost_age']))
#	table_data.loc[grammar_names[-1], 'Total Lifespan Variance Explained'] = str(health_variance)[:5]
#	table_data.loc[grammar_names[-1], 'Unique Lifespan Variance Explained'] = 'N/A'



	# AGE:
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	# Prepare my health variable.
	(variable_data, health_data, life_data) = computeStatistics.svr_data(adult_df, ['intensity_80', 'adjusted_size', 'adjusted_size_rate', 'life_texture', 'cumulative_eggs', 'cumulative_eggs_rate', 'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'], dependent_variable = 'age')
	health_data = np.expand_dims(health_data, 1)
	health_data = health_data*(-1/24)
	health_data = health_data - np.nanmin(health_data)	
	health_data = health_data/np.mean(health_data[:, :, 0])	
	geometry_dict = computeStatistics.one_d_geometries(adult_df, health_data)
	geometry_dicts.append(geometry_dict)



	# SURVIVAL:
	# Prepare my health variable.
	health_data = adult_df.mloc(measures = ['health'])
	(health_data, my_unit, fancy_name) = adult_df.display_variables(health_data, 'health')
	health_to_survival = define_survival(adult_df)
	survival_data = health_to_survival(health_data)
	health_data = survival_data
	geometry_dict = computeStatistics.one_d_geometries(adult_df, health_data)
	geometry_dicts.append(geometry_dict)




	# LINEAR:
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	my_cohorts = life_cohorts
	# Prepare my health variable.
	health_data = computeStatistics.multiple_regression_combine(adult_df, ['intensity_80', 'adjusted_size', 'adjusted_size_rate', 'life_texture', 'cumulative_eggs', 'cumulative_eggs_rate', 'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'], dependent_variable = 'ghost_age')[0]
	health_data = health_data*(-1/24)
	geometry_dict = computeStatistics.one_d_geometries(adult_df, health_data)
	geometry_dicts.append(geometry_dict)






	# MOTTLED:
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_cohorts = life_cohorts
	mottled_mask = pd.read_csv(r'C:\Google Drive\Aging Research\WormAgingMechanics\data\2016.04.24 Mottled Annotations\manual_mottled.tsv', sep = '\t').loc[:, 'Mottled?'].values
	non_mottled_worms = np.array(adult_df.worms)[np.invert(mottled_mask)]	
	non_mottled_indices = [adult_df.worm_indices[a_worm] for a_worm in non_mottled_worms]
	# Make bins of lifespans.
	(life_cohorts, bin_lifes, my_bins, my_colors) = selectData.adult_cohort_bins(adult_df, my_worms = adult_df.worms, bin_width_days = 2)
	my_adultspans = selectData.get_adultspans(adult_df)/24	
	# Prepare my "inflection" data.	
	geometry_dict = computeStatistics.one_d_geometries(adult_df, health_data)
	geometry_dicts.append(geometry_dict)




	if refresh_values:
		for i in range(0, len(grammar_names)):
			for j in range(0, len(geometries)):
				program_name = geometries[j].lower().replace(' ', '_')
				if j in [1, 2]:
					# Fit a polynomial for a trendline and for r^2.
					(ydata, xdata) = (geometry_dicts[i][program_name], geometry_dicts[i]['lifespan'])
					p_array = np.polyfit(xdata, ydata, 2)
					my_estimator = np.array([p_array[-i]*xdata**(i-1) for i in range(1, len(p_array)+1)])
					my_estimator = my_estimator.sum(axis = 0)
#					xrange = np.linspace(np.min(xdata), np.max(xdata), 200)
#					my_trendline = np.array([p_array[-i]*xrange**(i-1) for i in range(1, len(p_array)+1)])
#					my_trendline = my_trendline.sum(axis = 0)
#					my_subfigure.plot(xrange, my_trendline, color = 'black')
					my_r = '%.3f' % computeStatistics.quick_pearson(my_estimator, geometry_dicts[i][program_name], r_mode = True)
					my_rsquared = '%.3f' % computeStatistics.quick_pearson(my_estimator, geometry_dicts[i][program_name])
				

				else:
					my_r = '%.3f' % computeStatistics.quick_pearson(geometry_dicts[i][program_name], geometry_dicts[i]['lifespan'], r_mode = True)
					my_rsquared = '%.3f' % computeStatistics.quick_pearson(geometry_dicts[i][program_name], geometry_dicts[i]['lifespan'])
				table_data.iloc[i, j] = my_r + ', ' + my_rsquared			




#	for i in range(0, len(individual_measures)):
#		total_variance = computeStatistics.quick_pearson(*adult_df.flat_data([individual_measures[i], 'ghost_age']))
#		table_data.loc[grammar_names[i], 'Total Lifespan Variance Explained'] = str(total_variance)[:5]
#		
#		(variable_data, data_without, life_data) = computeStatistics.svr_data(adult_df, independent_variables = [a_measure for a_measure in individual_measures if a_measure != individual_measures[i]], dependent_variable = 'ghost_age')
#		nonunique_part = computeStatistics.quick_pearson(adult_df.flat_data(['ghost_age'])[0], np.ndarray.flatten(data_without))
#		unique_part = health_variance - nonunique_part
#		table_data.loc[grammar_names[i], 'Unique Lifespan Variance Explained'] = str(unique_part)[:5]
#		pass

	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 3)
	table_part = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)


	# Make the table nicely.
	table_part.axis('off')
	table_part.xaxis.set_visible(False)
	table_part.yaxis.set_visible(False)
	table_part.set_title('Correlations (Pearson $r$, $r^2$) with Adult Lifespan')
	my_table = table_part.table(colWidths = [0.10]*5, cellText = np.array(table_data), rowLabels = table_data.index, colLabels = table_data.columns, loc = 'center', cellLoc = 'center')
	my_table.set_fontsize(12)
	plt.tight_layout()

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return
	
def correlation_table_categories(adult_df):
	'''
	Make a table of correlations.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 4)
	table_part = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)

	all_measures = {'autofluorescence': ['intensity_80'],		
		'size': ['adjusted_size', 'adjusted_size_rate'],		
		'eggs': ['cumulative_eggs', 'cumulative_eggs_rate'],		
		'texture': ['life_texture'],		
		'movement': ['bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'],
		'health': []
	}
	all_raws = []
	for category in all_measures.keys():
		all_raws.extend([a_measure for a_measure in all_measures[category]])		
	print(all_raws)
	grammar_names = [adult_df.display_names(a_measure) for a_measure in all_measures.keys()]
	measure_categories = list(all_measures.keys())	
	measure_categories.remove('health')	
	
	table_data = pd.DataFrame(index = grammar_names, columns = ['Total Lifespan Variance Explained', 'Unique Lifespan Variance Explained'])

	health_variance = computeStatistics.quick_pearson(*adult_df.flat_data(['health', 'ghost_age']))
	table_data.loc[grammar_names[-1], 'Total Lifespan Variance Explained'] = str(health_variance)[:5]
	table_data.loc[grammar_names[-1], 'Unique Lifespan Variance Explained'] = 'N/A'

	for i in range(0, len(measure_categories)):
		total_variance = computeStatistics.quick_pearson(*adult_df.flat_data([measure_categories[i], 'ghost_age']))
		table_data.loc[grammar_names[i], 'Total Lifespan Variance Explained'] = str(total_variance)[:5]
		
		(variable_data, data_without, life_data) = computeStatistics.svr_data(adult_df, independent_variables = [a_measure for a_measure in all_raws if a_measure not in all_measures[measure_categories[i]]], dependent_variable = 'ghost_age')
		nonunique_part = computeStatistics.quick_pearson(adult_df.flat_data(['ghost_age'])[0], np.ndarray.flatten(data_without))
		unique_part = health_variance - nonunique_part
		table_data.loc[grammar_names[i], 'Unique Lifespan Variance Explained'] = str(unique_part)[:5]

	# Make the table nicely.
	table_part.axis('off')
	table_part.xaxis.set_visible(False)
	table_part.yaxis.set_visible(False)
	my_table = table_part.table(colWidths = [0.2]*3, cellText = np.array(table_data), rowLabels = table_data.index, colLabels = table_data.columns, loc = 'center', cellLoc = 'center')
	my_table.set_fontsize(14)
	plt.tight_layout()

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def correlation_table_raw(adult_df):
	'''
	Make a table of correlations.
	'''
	# Set up my figure.
	my_figure = plt.figure()
	my_figure.set_size_inches(12, 4)
	table_part = plotFigures.consistent_subgrid_coordinates((1, 1), (0, 0), my_width = 1, my_height = 1)

	all_measures = {'autofluorescence': ['intensity_80'],		
		'size': ['adjusted_size', 'adjusted_size_rate'],		
		'eggs': ['cumulative_eggs', 'cumulative_eggs_rate'],		
		'texture': ['life_texture'],		
		'movement': ['bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate'],
		'health': []
	}
	all_raws = []
	for category in all_measures.keys():
		all_raws.extend([a_measure for a_measure in all_measures[category]])		
	print(all_raws)
	grammar_names = [adult_df.display_names(a_measure) for a_measure in all_raws]
	
	table_data = pd.DataFrame(index = grammar_names, columns = ['Total Lifespan Variance Explained', 'Unique Lifespan Variance Explained'])

	health_variance = computeStatistics.quick_pearson(*adult_df.flat_data(['health', 'ghost_age']))
	table_data.loc[grammar_names[-1], 'Total Lifespan Variance Explained'] = str(health_variance)[:5]
	table_data.loc[grammar_names[-1], 'Unique Lifespan Variance Explained'] = 'N/A'

	for i in range(0, len(all_raws)):
		(variable_data, data_only, life_data) = computeStatistics.svr_data(adult_df, independent_variables = [all_raws[i]], dependent_variable = 'ghost_age')

		total_variance = computeStatistics.quick_pearson(adult_df.flat_data(['ghost_age'])[0], np.ndarray.flatten(data_only))
		table_data.loc[grammar_names[i], 'Total Lifespan Variance Explained'] = str(total_variance)[:5]
		
		(variable_data, data_without, life_data) = computeStatistics.svr_data(adult_df, independent_variables = [a_measure for a_measure in all_raws if a_measure != all_raws[i]], dependent_variable = 'ghost_age')
		nonunique_part = computeStatistics.quick_pearson(adult_df.flat_data(['ghost_age'])[0], np.ndarray.flatten(data_without))
		unique_part = health_variance - nonunique_part
		table_data.loc[grammar_names[i], 'Unique Lifespan Variance Explained'] = str(unique_part)[:5]

	# Make the table nicely.
	table_part.axis('off')
	table_part.xaxis.set_visible(False)
	table_part.yaxis.set_visible(False)
	my_table = table_part.table(colWidths = [0.2]*3, cellText = np.array(table_data), rowLabels = table_data.index, colLabels = table_data.columns, loc = 'center', cellLoc = 'center')
	my_table.set_fontsize(14)
	plt.tight_layout()

	# Save out my figure.
	graph_name = inspect.stack()[0][3]
	plt.savefig(adult_df.save_directory + os.path.sep + graph_name + figure_ending, dpi = 300, bbox_inches = 'tight')
	plotFigures.remove_whitespace(adult_df.save_directory + os.path.sep + graph_name + figure_ending)	
	return

def main():
	return

if __name__ == "__main__":
	main()
