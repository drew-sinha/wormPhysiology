# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 06:02:01 2015

@author: Willie
"""

import sys
import pickle
import os
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.ndimage.filters as filters
import sklearn.model_selection as model_selection
import json
import pathlib

class BasicWormDF():
    '''
        Provides a basic version of Willie's CompleteWormDF
    '''
        
    def __init__(self, data_directories, **kwargs):
        # Set some default values if parameters not given in provided kwargs.
        default_args = {
            'adult_only': False,    # Restrict storage of data in the df to adult_only data
            'do_smoothing': False,  # Toggle smoothing
            'measures_nosmoothing': ['age', 'egg_age','ghost_age','great_lawn_area','great_lawn_max_diameter'],
            'bad_worm_kws':[],      # List of strs of annotations in 'Notes' field to use to screen out animals
            'scale_data': True,     # Toggle automatic storage of data as z-scores
            'regressor_fp':None,    # (str/pathlib.Path) filepath for health regressor
        }
        [kwargs.setdefault(k,v) for k,v in default_args.items()]
        
        # Set arguments as attributes in the dataframe for book-keeping.
        [setattr(self,attr_name,attr_val) 
            for attr_name, attr_val in kwargs.items()]
        
        print('Working on the following directories:')
        [print(my_directory) for my_directory in data_directories]
        
        # Read in the raw data and set up some basic information.       
        self.raw = self.read_trajectories(data_directories)
        self.worms = sorted(list(self.raw.keys()))
        
        all_measures = [list(self.raw[worm].columns) for worm in self.worms]
        self.measures = list({item for items in all_measures for item in items}) # Flatten this list
        
        # Just in case a measurement wasn't taken for all individuals
        for worm in self.worms:
            for measure in self.measures:
                if measure not in self.raw[worm].columns:
                    self.raw[worm].loc[:,measure] = np.nan
        
        # Reorder raw data columns for consistent ordering when filling the df and setting up indexing
        for worm in self.worms:
            self.raw[worm] = self.raw[worm][self.measures]
        
        all_shapes = []     
        for a_worm in self.raw.keys():
            my_shape = self.raw[a_worm].shape
            all_shapes.append(my_shape)
        my_shape = np.max(np.array(all_shapes), axis = 0)
        max_times = my_shape[0]
        max_measurements = my_shape[1]

        # Set up convenience variables to reference variables and worms.
        self.worm_indices = {self.worms[i]:i for i in range(0, len(self.worms))}
        self.measure_indices = {self.measures[i]:i for i in range(0, len(self.measures))}
        self.times = [str(a_time//8) + '.' + str(a_time-8*(a_time//8)) for a_time in list(range(0, max_times))]
        self.time_indices = {self.times[i]:i for i in range(0, len(self.times))}
        self.ages = [int(a_time.split('.')[0]) + int(a_time.split('.')[1])/8 for a_time in self.times]

        # Fill in my frame with actual data!
        self.data = np.empty((len(self.worms), max_measurements, max_times))
        self.data.fill(np.nan)
        for i in range(0, len(self.worms)):
            a_worm = self.worms[i]
            worm_data = self.raw[a_worm].values
            (time_shape, measure_shape) = worm_data.shape
            self.data[i, 0:measure_shape, 0:time_shape] = worm_data.transpose()

        # Set up some information for re-scaling.
        self.means = np.empty(len(self.measures))
        self.means[:] = np.nan
        self.stds = np.empty(len(self.measures))
        self.stds[:] = np.nan
        
        self.process_eggs()
        self.adjust_size()
        
        print('Time-interpolating data')
        self.time_normalized_data()    

        self.add_rates()
        
        if self.scale_data:
            self.scale_normalized_data()

        if self.regressor_fp is not None:
            with pathlib.Path(self.regressor_fp).open('rb') as reg_file:
                regressor = pickle.load(reg_data)['fit_regressor']
            self.add_health(health_var_name='health',smooth=False,regressor=regressor)
            
        if self.do_smoothing:
            self.smooth_data_causalMA(measures_tosmooth = set(self.measures).difference(self.measures_nosmoothing)) # Default smoothing.
    
    def mloc(self, worms = None, measures = None, times = None):
        '''
        Selects information from my dictionary of normalized time dataframes as if it were a 3-dimensional array.
        '''
        mloc_data = self.data
        if worms != None:
            worm_indices = [self.worm_indices[a_worm] for a_worm in worms]
            mloc_data = mloc_data[worm_indices, :, :]
        if measures != None:
            measure_indices = [self.measure_indices[a_measure] for a_measure in measures]
            mloc_data = mloc_data[:, measure_indices, :]
        if times != None:
            times = [self.time_indices[a_time] for a_time in times]
            mloc_data = mloc_data[:, :, times]
        return mloc_data

    def flat_data(self, measures):
        '''
        Send back a list of flattened arrays.
        '''
        my_flats = [np.ndarray.flatten(self.mloc(measures = [a_measure])) for a_measure in measures]
        return my_flats

    def get_center_std(self, base_dictionary):
        '''
        Get the centers and standard deviations of all variables in base_dictionary.
        '''
        overall_frame = []
        for a_worm in base_dictionary.keys():
            my_frame = base_dictionary[a_worm].values
            my_frame = my_frame[~np.isnan(my_frame).any(axis = 1)]
            overall_frame.append(my_frame)
        overall_frame = np.vstack(overall_frame)        
        my_means = np.mean(overall_frame, axis = 0)
        my_stds = np.std(overall_frame, axis = 0)
        my_means[-4:] = 0
        my_stds[-4:] = 1
        return (my_means, my_stds)
    
    def identify_worms(self, a_measurement, a_time):
        '''
        For a_measurement at a_time, list the worms from lowest to highest values.
        '''
        my_ordering = self.mloc(measures = [a_measurement], times = [a_time])[:, 0, 0]
        my_ranks = my_ordering.argsort()
        my_ordering.sort()
        my_worms = []
        for i in range(0, my_ordering.shape[0]):
            if ~np.isnan(my_ordering[i]):
                my_worms.append(self.worms[my_ranks[i]])
        return my_worms
    
    def scale_normalized_data(self):
        '''
        Rescale normalized data so that it's in terms of standard deviations from the overall mean.
        '''
        for var_index in range(len(self.measures)):
            if np.isnan(self.means[var_index]) and self.measures[var_index] not in ['age','ghost_age','egg_age']:
                a_var = self.measures[var_index]            
                my_data = np.ndarray.flatten(self.mloc(measures = [a_var]))
                my_data = my_data[~np.isnan(my_data)]
                self.means[var_index] = np.mean(my_data)
                self.stds[var_index] = np.std(my_data)
                self.data[:, var_index, :] = (self.data[:, var_index, :] - self.means[var_index])/self.stds[var_index]     
            elif self.measures[var_index] in ['age','ghost_age','egg_age']:
                self.means[var_index] = 0
                self.stds[var_index] = 1
        return
    
    def time_normalized_data(self):
        '''
        Normalize my data so that times between time points are equal.
        '''
        normed_data = []
        max_times = 0
        for a_worm in self.worms:
            worm_index = self.worm_indices[a_worm]
            birth_time = self.mloc([a_worm], ['age'])
            if self.adult_only:
                birth_time = self.mloc([a_worm], ['egg_age'])
            death_time = self.mloc([a_worm], ['ghost_age'])
            start_index = np.abs(birth_time[~np.isnan(birth_time)]).argmin()
            end_index = np.abs(death_time[~np.isnan(death_time)]).argmin()

            age_array = self.mloc([a_worm], ['age'])[0, 0, start_index: end_index + 1]
            adult_start = age_array[0]
            
            normed_array = np.empty((len(self.measures), 1000)).astype('float')
            normed_array.fill(np.nan)
        
            adult_span = age_array[-1] - age_array[0]
            time_points = int(adult_span//3) + 1
            max_times = max(max_times, time_points)

            for i in range(0, time_points):
                i_age = adult_start + (i*3)
    
                age_sort = np.sort(np.append(age_array, i_age))
                my_indices = np.where(age_sort == i_age)[0]
                my_indices = np.where(abs(age_sort - i_age) < 0.001)[0]
                indices_shape = my_indices.shape[0]
    
                if indices_shape == 2:
                    normed_array[:, i] = self.data[worm_index, :, start_index + my_indices[0]]
                elif indices_shape == 1:
                    before_position = my_indices[0] - 1
                    after_position = my_indices[0]
                    before_age = age_array[before_position]
                    after_age = age_array[after_position]
                    age_range = after_age - before_age
                    before_portion = (after_age - i_age)/age_range
                    after_portion = (i_age - before_age)/age_range
                    normed_array[:, i] = before_portion*self.data[worm_index, :, start_index + before_position] + after_portion*self.data[worm_index, :, start_index + after_position]
                else:
                    raise BaseException('Too many instances of i_age found in age_sort.')
            normed_data.append(normed_array)
        normed_data = np.array(normed_data)[:, :, :max_times]
        self.data = normed_data
        self.times = [str(a_time//8) + '.' + str(a_time-8*(a_time//8)) for a_time in list(range(0, max_times))] # Again - assuming 3 hr time points to yield 1/8 day time intervals.
        self.time_indices = {self.times[i]:i for i in range(0, len(self.times))}
        self.ages = [int(a_time.split('.')[0]) + int(a_time.split('.')[1])/8 for a_time in self.times]
        return
                    
    def add_column(self, column_data, column_index, column_name):
        '''
        Add a new column of processed data to my array.
        '''
        self.data = np.concatenate([self.data[:, :column_index, :], column_data, self.data[:, column_index:, :]], axis = 1) 
        self.measures.insert(column_index, column_name)
        self.measure_indices = {self.measures[i]:i for i in range(0, len(self.measures))}
        
        mean_std_index = column_index
        if column_index < 0:
            mean_std_index = len(self.measures) + column_index - 1 # Willie had an off by one error here! (length inc. after adding add'l item)
        self.means = np.concatenate([self.means[:mean_std_index], [np.nan], self.means[mean_std_index:]], axis = 0) 
        self.stds = np.concatenate([self.stds[:mean_std_index], [np.nan], self.stds[mean_std_index:]], axis = 0)    
        return
        
    def process_eggs(self):
        '''
        Add a "cumulative eggs" variable to each worm dataframe in worm_frames.
        '''
        if 'visible_eggs' in self.measures and 'visible_area' in self.measures:
            new_data_eggs = []
            new_data_area = []
            for i in range(0, len(self.worms)):
                # Extract some needed information.
                a_worm = self.worms[i]
                visible_eggs = self.mloc([a_worm], ['visible_eggs'])
                visible_area = self.mloc([a_worm], ['visible_area'])
                egg_started = self.mloc([a_worm], ['egg_age'])
                start_index = np.abs(egg_started[~np.isnan(egg_started)]).argmin()
                
                # Eliminate false positive eggs from count.
                base_eggs = visible_eggs[0, 0, start_index - 1] 
                cumulative_eggs = np.array(visible_eggs)
                cumulative_eggs = cumulative_eggs - base_eggs
                cumulative_eggs[:, :, :start_index] = 0
                base_area = visible_area[0, 0, start_index - 1] 
                cumulative_area = np.array(visible_area)
                cumulative_area = cumulative_area - base_area
                cumulative_area[:, :, :start_index] = 0
        
                # Find running maximum of visible eggs to use as cumulative eggs laid.
                cumulative_eggs = np.maximum.accumulate(cumulative_eggs, axis = 2)          
                cumulative_area = np.maximum.accumulate(cumulative_area, axis = 2)          
                new_data_eggs.append(cumulative_eggs)           
                new_data_area.append(cumulative_area)           
    
            # Add my eggs into the columns.
            new_data_eggs = np.concatenate(new_data_eggs, axis = 0)
            column_index = self.measure_indices['visible_eggs']
            self.add_column(new_data_eggs, column_index, 'cumulative_eggs')
            new_data_area = np.concatenate(new_data_area, axis = 0)
            column_index = self.measure_indices['visible_area']
            self.add_column(new_data_area, column_index, 'cumulative_area')
        return
    
    def smooth_data_causalMA(self,measures_tosmooth=[],**smooth_kws):
        '''
            Smooth collated data using a trailing moving average filter
            
            Parameters
                measures_tosmooth - list of strs indicating which measures to smooth
                smooth_kws - kws to provide the smoothing filter, including
                
            Returns
                (none) self.data is appropriately smoothed....
        '''
        
        window_length = smooth_kws.get('window_length',7) # default to 7 point
        if len(measures_tosmooth) == 0: measures_tosmooth = self.measures
        
        for measure in measures_tosmooth:
            print('Smoothing {}'.format(measure))
            var_data = self.mloc(measures=[measure])[:,0,:]
            filtered_data = np.apply_along_axis(
                filters.uniform_filter,1,
                var_data,
                **{'size':window_length,
                    'mode':'nearest',
                    'origin':(window_length//2)})
            self.data[:,self.measure_indices[measure],:] = filtered_data

    def read_trajectories(self, data_directories):
        '''
        Reads in trajectories from .tsv files in working_directory's measured_health subdirectory and then groups them together nicely in a WormData class. Also adds meta-information from metadata in data_directory.
        '''
        if len(data_directories) == 1:
            data_directories = data_directories[0]
        
        # List of worms to exclude.
        not_yet_done = [
#           '2016.03.31 spe-9 16 005',
#           '2016.03.31 spe-9 16 132',
#           '2016.03.31 spe-9 16 196',
#           '2016.03.31 spe-9 16 200',
#           '2016.03.31 spe-9 16 201',
#           '2016.03.31 spe-9 16 202',
#           '2016.03.31 spe-9 16 204',
#           '2016.03.31 spe-9 16 208'
        ]
        never_eggs = [
            '2016.02.26 spe-9 11D 130',
            '2016.02.26 spe-9 11C 085',
            '2016.02.29 spe-9 12A 07',
            '2016.03.25 spe-9 15B 003',
            '2016.03.25 spe-9 15B 126',
            '2016.03.25 spe-9 15A 43',
            '2016.03.04 spe-9 13C 67',
            '2016.03.31 spe-9 16 154'   
        ]
        
        # Read in my measured_health data from the scope.
        if type(data_directories) == type([]):
            health_directories = [data_directory + os.path.sep + 'measured_health' for data_directory in data_directories]
            my_tsvs = []
            
            if len(self.bad_worm_kws) > 0:
                for data_directory in data_directories:
                    annotation_tsv = [data_directory + os.path.sep + my_file for my_file in os.listdir(data_directory) if '.tsv' in my_file][0]
                    annotation_data = pd.read_csv(annotation_tsv,sep='\t')
                    bad_worms = np.array([
                        worm[1:] for worm, entry in zip(annotation_data['Worm'],annotation_data['Notes'])
                        if (pd.isnull(entry)) 
                        or ('DEAD' not in entry)
                        or any([bad_kw in entry for bad_kw in self.bad_worm_kws])])
                    for a_file in os.listdir(data_directory + os.path.sep + 'measured_health'):
                        if ((a_file.split('.')[-1] == 'tsv') and
                            (a_file.split('.')[0] not in bad_worms)):
                            my_tsvs.extend([data_directory + os.path.sep + 'measured_health' + os.path.sep + a_file])
                        else:
                            print('Skipping '+ data_directory + ' ' + a_file + ': Bad worm')
            else:
                for health_directory in health_directories:
                    my_tsvs.extend([health_directory + os.path.sep + a_file for a_file in os.listdir(health_directory) if a_file.split('.')[-1] == 'tsv'])

            # Exclude worms.
            for a_worm in not_yet_done:
                if a_worm + '.tsv' in my_tsvs:
                    print('\tSkipping ' + a_worm + ', it is not yet done processing.')
                    my_tsvs.remove(a_worm + '.tsv')
            
            for a_worm in never_eggs:
                #worm_file = [a_dir for a_dir in health_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir][0] + os.path.sep + a_worm.split(' ')[-1] + '.tsv'
                worm_file = [a_dir for a_dir in health_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir]
                if len(worm_file)>0:
                    worm_file = worm_file[0] + os.path.sep + a_worm.split(' ')[-1] + '.tsv'
                    if worm_file in my_tsvs:
                        print('\tSkipping ' + a_worm + ', it never laid eggs.')
                        my_tsvs.remove(worm_file)       

            worm_frames = {a_file.split(os.path.sep)[-3].replace(' Run ', ' ') + ' ' + a_file.split(os.path.sep)[-1].split('.')[-2]: pd.read_csv(a_file, sep = '\t', index_col = 0) for a_file in my_tsvs}      

        # Read in my measured_health data from my special directory.
        elif type(data_directories) == type(''):    # TODO - think about removing this....
            my_tsvs = [a_file for a_file in os.listdir(data_directories) if a_file.split('.')[-1] == 'tsv']
            
            # Exclude worms.
            for a_worm in not_yet_done:
                if a_worm + '.tsv' in my_tsvs:
                    print('\tSkipping ' + a_worm + ', it is not yet done processing.')
                    my_tsvs.remove(a_worm + '.tsv')
            for a_worm in never_eggs:
                if a_worm + '.tsv' in my_tsvs:
                    print('\tSkipping ' + a_worm + ', it never laid eggs.')
                    my_tsvs.remove(a_worm + '.tsv')     
            
            # Actually read them in.
            worm_frames = {a_file[:-4]: pd.read_csv(data_directories + os.path.sep + a_file, sep = '\t', index_col = 0) for a_file in my_tsvs}
        return worm_frames
    
    def adjust_size(self, adjustment_parameters=(0.69133050735497115, 8501.9616379946274)):
        '''
        Adjust total_size for any bias in measurement by the automated method as compared to the manually drawn masks.
        Defaults to the the defined adjustment parameters previously found by linear regression for manually drawn masks.
        ''' 
        (m, b) = adjustment_parameters
        old_size = self.mloc(measures = ['total_size'])
        old_size_index = self.measure_indices['total_size']
        new_size = (m*old_size) + b
        self.add_column(new_size, old_size_index, 'adjusted_size')
        return

    def add_rates(self, rate_variables = None):
        '''
        Add rates of change for some variables.
        '''
        if rate_variables == None:
            rate_variables = ['cumulative_area', 'cumulative_eggs', 'adjusted_size']
        for a_variable in rate_variables:
            print('\tComputing rate of change for ' + a_variable + '.')             
            
            # Compute rates of change.
            my_data = self.mloc(measures = [a_variable])
            my_rate = my_data[:, :, 1:] - my_data[:, :, :-1]
            zfiller = np.zeros(my_data.shape[:2])
            my_rate = np.concatenate([my_rate, zfiller[:, :, np.newaxis]], axis = 2)

            # Actually add the rates data.
            old_index = self.measure_indices[a_variable]
            self.add_column(my_rate, old_index, a_variable + '_rate')
        return

    def add_healths(self, health_var_name='health',smooth=False, **hs_params):
        '''
        Add 'health' measure to df.
        '''
        assert hs_params.get('regressor') is not None
        health_scores = self.calc_healthscore(regressor) # Use default settings for this.
        column_data = np.expand_dims(health_scores, axis = 1)
        self.add_column(column_data, -3, health_var_name)
        if self.scale_data: self.scale_normalized_data()
        if smooth:
            self.smooth_data_causalMA(
                measures_tosmooth = [health_var_name]) # Default smoothing.

    
    def calc_healthscore(self,regressor,measurements=None,
        worms=None,times=None,predict_mode='simple',regressed_variable='ghost_age'):
        '''
            Calculate health scores for measurement data in this dataframe using the specified regressor
            
            Parameters
                regressor - (sklearn-compatible) estimator object implementing predict, fit, etc.
                measurements - list of strs corresponding to measurements to use as independent variables in regression
                regressed_variable - str corresponding to (outcome) measurement as dependent variable in regression; used if predict_mode is 'cv'
                savepath - (str/pathlib.Path) object giving the save location to save out regressor and associated metadata
                worms, times - lists of strs used to limit regression per mloc
                predict_mode - str detailing procedure by which to generate the score by prediction; currently 'simple' for a standard call to estimator.predict or 'cv' for 4-fold CV
                
            Returns
                health_score - array containing health scores with shape [animals,time]
        '''
        if measurements is None:
            measurements = ['intensity_80',       
                'adjusted_size', 'adjusted_size_rate',        
                'cumulative_eggs', 'cumulative_eggs_rate',        
                'life_texture',        
                'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate']
        
        measurement_data = self.mloc(measures=measurements,worms=worms,times=times) #[animals, measurements, time]
        measurement_data_flat = measurements_data.swapaxes(1,2).reshape(-1,len(measurements)) #[animals x time,measurements]
        
        nan_mask = np.isnan(measurement_data_flat).any(axis=1)
        
        if predict_mode == 'simple':
            health_score_flat = regressor.predict(measurement_data)
        elif predict_mode == 'cv':
            regressed_data = self.mloc(measures=[regressed_variable],worms=worms,times=times)[:,0,:] #[animals,time]
            regressed_data_flat = remaining_life.flatten() #[animals xtime]
            
            kfold_cv = model_selection.KFold(n_splits=4,shuffle=True)
            health_score_flat = model_selection.cross_val_predict(
                regressor,
                biomarker_data_flat[~nan_mask,:],
                remaining_life_flat[~nan_mask],
                cv = kfold_cv,
                n_jobs = 4) # Animals x time
        
        health_score = np.full_like(regressed_data_flat,np.nan)
        health_score[~nan_mask] = health_score_flat
        health_score = np.reshape(health_score,regressed_data.shape)
        
        return health_score
    
    def fit_regressor(self, regressor, measurements=None, regressed_variable='ghost_age', 
        savepath=None,
        worms=None,times=None):
        '''
            Convenience function for fitting a specified regressor to the data contained in this dataframe.
            
            Parameters
                regressor_to_use - regressor object (i.e. sklearn-like estimator implementing fit) to fit
                measurements - list of strs corresponding to measurements to use as independent variables in regression
                regressed_variable - str corresponding to (outcome) measurement as dependent variable in regression
                savepath - (str/pathlib.Path) object giving the save location to save out regressor and associated metadata
                worms, times - lists of strs used to limit regression per mloc
            
            Returns
                regressor - regressor object fit to data

        '''
        if measurements is None:
            measurements = ['intensity_80',       
                'adjusted_size', 'adjusted_size_rate',        
                'cumulative_eggs', 'cumulative_eggs_rate',        
                'life_texture',        
                'bulk_movement', 'stimulated_rate_a', 'stimulated_rate_b', 'unstimulated_rate']
        
        measurement_data = self.mloc(measures=measurements,worms=worms,times=times) #[animals, measurements, time]
        measurement_data_flat = measurements_data.swapaxes(1,2).reshape(-1,len(measurements)) #[animals x time,measurements]

        regressed_data = self.mloc(measures=[regressed_variable],worms=worms,times=times)[:,0,:] #[animals,time]
        regressed_data_flat = remaining_life.flatten() #[animals xtime]
        
        nan_mask = np.isnan(measurement_data_flat).any(axis=1)
        
        regressor.fit(measurement_data_flat[~nan_mask,:],
            regressed_data_flat[~nan_mask,:])
        
        if savepath is not None:
            print('Saving regressor at '+ str(savepath))
            with pathlib.Path(savepath).open('wb') as my_file:
                pickle.dump({'fit_regressor':regressor,
                    'measurements':measurements,
                    'regressed_variable':regressed_variable,
                    'sample_weights':sample_weights,
                    'worms':worms,
                    'times':times},
                my_file)
        
        return regressor
    
    def display_variables(self, an_array, my_var):
        '''
        Convert an_array of my_var data to display units.
        '''
        my_units = {
            'intensity_90': 'Standard Deviations', 
            'intensity_80': 'Standard Deviations', 
            'cumulative_eggs': 'Cumulative Oocytes Laid',
            'cumulative_eggs_rate': 'Oocytes Laid Per Hours',
            'cumulative_area': 'mm^2',
            'visible_eggs': 'Number of Oocytes',
            'total_size': 'mm^2', 
            'age_texture': 'Textural Age', 
            'bulk_movement': 'mm Displacement Per Hours',
            'stimulated_rate_a': 'mm/s',
            'stimulated_rate_b': 'mm/s',
            'unstimulated_rate': 'mm/s',
            'area': 'mm^2',
            'life_texture': 'Texture Prediction (Days Remaining)',
            'adjusted_size': 'mm^2',
            'adjusted_size_rate': 'mm^2 Per Hour',
            'great_lawn_area': 'mm^2', 
            'texture': 'Predicted Days of Life Remaining',
            'eggs': 'Predicted Days of Life Remaining',
            'autofluorescence': 'Predicted Days of Life Remaining',
            'movement': 'Predicted Days of Life Remaining',
            'size': 'Predicted Days of Life Remaining',
            'health': 'Predicted Days of Life Remaining'
        }
        unit_multipliers = {
            'intensity_90': None, 
            'intensity_80': None, 
            'cumulative_eggs': 1,
            'cumulative_eggs_rate': 1/3,
            'cumulative_area': (1.304/1000)**2,
            'visible_eggs': 1,
            'total_size': (1.304/1000)**2, 
            'age_texture': 1, 
            'bulk_movement': (1.304/1000)/3,
            'stimulated_rate_a': (1.304/1000),
            'stimulated_rate_b': (1.304/1000),
            'unstimulated_rate': (1.304/1000),
            'area': 1/100000,
            'life_texture': -1,
            'adjusted_size': (1.304/1000)**2,
            'adjusted_size_rate': ((1.304/1000)**2)/3,
            'great_lawn_area': (1.304/1000)**2, 
            'texture': (-1/24),
            'eggs': (-1/24),
            'autofluorescence': (-1/24),
            'movement': (-1/24),
            'size': (-1/24),
            'health': (-1/24),
        }

        if my_var in my_units.keys():
            my_unit = my_units[my_var]
            fancy_name = self.display_names(my_var)
            unit_multiplier = unit_multipliers[my_var]
        else:
            my_unit = 'Raw Numbers'
            fancy_name = 'No Fancy Name for ' + my_var
            unit_multiplier = 1
    
        if my_unit != 'Standard Deviations' and unit_multiplier != None:
            my_index = self.measures.index(my_var)
            my_mean = self.means[my_index]
            my_std = self.stds[my_index]
            new_array = unit_multiplier*((an_array*my_std) + my_mean)
        else:
            new_array = an_array
        return (new_array, my_unit, fancy_name)

def get_worm_names(data_directories):
    '''
        Returns full names for worms with measured health over all data_directories (outsourced from CompleteWormDF.read_trajectories
    '''
    never_eggs = [
        '2016.02.26 spe-9 11D 130',
        '2016.02.26 spe-9 11C 085',
        '2016.02.29 spe-9 12A 07',
        '2016.03.25 spe-9 15B 003',
        '2016.03.25 spe-9 15B 126',
        '2016.03.25 spe-9 15A 43',
        '2016.03.04 spe-9 13C 67',
        '2016.03.31 spe-9 16 154'   
    ]
    
    health_directories = [data_directory + os.path.sep + 'measured_health' for data_directory in data_directories]
    my_tsvs = []
    for health_directory in health_directories:
        my_tsvs.extend([health_directory + os.path.sep + a_file for a_file in os.listdir(health_directory) if a_file.split('.')[-1] == 'tsv'])
    
    #** Comment this out since note field was used to filter out worms previously... should be good, right?
    for a_worm in never_eggs:
        #worm_file = [a_dir for a_dir in health_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir][0] + os.path.sep + a_worm.split(' ')[-1] + '.tsv'
        worm_file = [a_dir for a_dir in health_directories if ' '.join(a_worm.split(' ')[:-2]) + ' Run ' + a_worm.split(' ')[-2] in a_dir]
        if len(worm_file)>0:
            worm_file = worm_file[0] + os.path.sep + a_worm.split(' ')[-1] + '.tsv'
            if worm_file in my_tsvs:
                print('\tSkipping ' + a_worm + ', it never laid eggs.')
                my_tsvs.remove(worm_file)
    
    return [a_file.split(os.path.sep)[-3].replace(' Run ', ' ') + ' ' + a_file.split(os.path.sep)[-1].split('.')[-2] for a_file in my_tsvs]

def resample(a_trajectory, new_points):
    '''
    Resample a_trajectory to have new_points points.
    '''
    a_trajectory = a_trajectory[~np.isnan(a_trajectory).any(axis = 1)]
    (my_points, my_dimensions) = a_trajectory.shape
    new_trajectory = np.zeros((new_points, my_dimensions))
    for i in range(0, my_dimensions):
        new_trajectory[:, i] = np.interp(np.arange(new_points)/new_points, np.arange(my_points)/my_points, a_trajectory[:, i])
    return new_trajectory


def main():
    return

if __name__ == "__main__":
    main()
