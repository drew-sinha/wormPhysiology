# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:21:55 2015

@author: Willie

THE FOLLOWING IS MY HIERARCHY OF IMPORTS:
    myRuns CAN IMPORT ANYTHING. ANYTHING!!!
    mainFigures can import anything below this.
    supplementFigures can import anything below this.
    plotFigures can import anything below this.
    characterizeTrajectories can import anything below this.
    computeStatistics can import anything below this.
    selectData can import anything below this.
    organizeData can import anything below this.
    extractFeatures can import anything below this.
    backgroundSubtraction, edgeMorphology, shapeModelFitting, and textureClassification can import anything below this.
    imageOperations can import anything below this.
    folderStuff can import anything below this.
    Python libraries.
"""

import sys
import os.path
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
os.chdir(file_dir)

import numpy as np
import json
import pathlib
import os
import pickle
import shutil
import pandas as pd 
import sklearn  
import random
import scipy.cluster
import pathlib
import time
import scipy.stats
import scipy.interpolate
import freeimage
import concurrent.futures
import multiprocessing
import time 
import matplotlib.pyplot as plt

import zplib.image.resample as zplib_image_resample
import zplib.image.mask as zplib_image_mask

import basicOperations.folderStuff as folderStuff
import basicOperations.imageOperations as imageOperations
import wormFinding.backgroundSubtraction as backgroundSubtraction   
#import wormFinding.shapeModelFitting as shapeModelFitting
import wormFinding.textureClassification as textureClassification
import wormFinding.edgeMorphology as edgeMorphology
import measurePhysiology.extractFeatures as extractFeatures
import measurePhysiology.organizeData as organizeData
import analyzeHealth.selectData as selectData
import analyzeHealth.computeStatistics as computeStatistics
import analyzeHealth.characterizeTrajectories as characterizeTrajectories
import graphingFigures.plotFigures as plotFigures
import graphingFigures.cannedFigures as cannedFigures
import graphingFigures.supplementFigures as supplementFigures
import graphingFigures.mainFigures as mainFigures

print('Getting started!', flush = True) 
# Default to my_mode 0.
my_mode = 0
if len(sys.argv) > 1:
    my_mode = int(sys.argv[1])

# Locations of all the files I need.
#save_directory = r'C:\Users\Willie\Desktop\save_dir'
save_directory = r'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/age-1percombinedweightedSVM_health/'
#working_directory = r'\\heavenly.wucon.wustl.edu\wzhang\work_dir'
working_directory = r'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/work_dir/'
#human_directory = r'\\heavenly.wucon.wustl.edu\wzhang\human_dir'
human_directory = r'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/'
#data_directories = [
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.16 spe-9 Run 9',      #0
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10A',    #1
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10B',    #2
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11A',    #3
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11B',    #4
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11C',    #5
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11D',    #6
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12A',    #7
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12B',    #8
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13A',    #9
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13B',    #10
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13C',    #11
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.14 spe-9 Run 14',     #12
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15A',    #13
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15B',    #14
    #r'\\zpl-iscope.wucon.wustl.edu\iscopearray\Zhang_William\2016.03.31 spe-9 Run 16',      #15
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17A', #16
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17B', #17
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17C', #18
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17D', #19
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17E', #20
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.12 spe-9 age-1 Run 18B',   #21
    #r'\\zpl-scope.wucon.wustl.edu\scopearray\ZhangWillie\2016.05.24 spe-9 age-1 Run 24A',  #22 7 worms alive as of 20160624
#]
data_directories = [
    r'\\mnt\iscopearray\Zhang_William\2016.02.16 spe-9 Run 9',      #0
    r'\\mnt\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10A',    #1
    r'\\mnt\iscopearray\Zhang_William\2016.02.20 spe-9 Run 10B',    #2
    r'\\mnt\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11A',    #3
    r'\\mnt\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11B',    #4
    r'\\mnt\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11C',    #5
    r'\\mnt\iscopearray\Zhang_William\2016.02.26 spe-9 Run 11D',    #6
    r'\\mnt\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12A',    #7
    r'\\mnt\iscopearray\Zhang_William\2016.02.29 spe-9 Run 12B',    #8
    r'\\mnt\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13A',    #9
    r'\\mnt\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13B',    #10
    r'\\mnt\iscopearray\Zhang_William\2016.03.04 spe-9 Run 13C',    #11
    r'\\mnt\iscopearray\Zhang_William\2016.03.14 spe-9 Run 14',     #12
    r'\\mnt\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15A',    #13
    r'\\mnt\iscopearray\Zhang_William\2016.03.25 spe-9 Run 15B',    #14
    r'\\mnt\iscopearray\Zhang_William\2016.03.31 spe-9 Run 16',      #15
    r'\\mnt\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17A', #16
    r'\\mnt\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17B', #17
    r'\\mnt\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17C', #18
    r'\\mnt\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17D', #19
    r'\\mnt\scopearray\ZhangWillie\2016.05.02 spe-9 age-1 osxi346 Run 17E', #20
    r'\\mnt\scopearray\ZhangWillie\2016.05.12 spe-9 age-1 Run 18B',   #21
    r'\\mnt\scopearray\ZhangWillie\2016.05.24 spe-9 age-1 Run 24A',  #22 7 worms alive as of 20160624
    r'\\mnt\scopearray\ZhangWillie\2016.07.01 spe-9 age-1 Run 21',  #23
]
extra_directories = [
    {'W': r'\\mnt\scopearray\ZhangWillie\2016.02.16 spe-9 Run 9'},      #0
    None,                                                                                     #1
    None,                                                                                     #2
    None,                                                                                     #3
    None,                                                                                     #4
    None,                                                                                     #5
    None,                                                                                     #6
    {'W': r'\\mnt\scopearray\ZhangWillie\2016.02.29 spe-9 Run 12A'},    #7
    {'W': r'\\mnt\scopearray\ZhangWillie\2016.02.29 spe-9 Run 12B'},    #8
    None,                                                                                     #9
    {'W': r'\\mnt\scopearray\ZhangWillie\2016.03.04 spe-9 Run 13B'},    #10
    {'W': r'\\mnt\scopearray\ZhangWillie\2016.03.04 spe-9 Run 13C'},    #11
    None,                                                                                     #12
    None,                                                                                     #13
    None,                                                                                     #14
    None,                                                                                     #15
    None,                                                                                     #16
    None,                                                                                     #17
    None,                                                                                     #18
    None,                                                                                     #19
    None,                                                                                     #20
    None,                                                                                     #21
    None,                                                                                     #22
    None,                                                                                     #23
]
experiment_directories = [
    'W',                                                                                      #0
    None,                                                                                     #1
    None,                                                                                     #2
    None,                                                                                     #3
    None,                                                                                     #4
    None,                                                                                     #5
    None,                                                                                     #6
    'W',                                                                                      #7
    'W',                                                                                      #8
    None,                                                                                     #9
    'W',                                                                                      #10
    'W',                                                                                      #11
    None,                                                                                     #12
    None,                                                                                     #13
    None,                                                                                     #14   
    None,                                                                                     #15       
    None,                                                                                     #16
    None,                                                                                     #17
    None,                                                                                     #18
    None,                                                                                     #19
    None,                                                                                     #20
    None,                                                                                     #21
    None,                                                                                     #22
    None,                                                                                     #23
]
annotation_directories = [
    r'\\mnt\scopearray\Sinha_Drew\20160216_spe9Acquisition',            #0
    None,                                                                                     #1
    None,                                                                                     #2
    None,                                                                                     #3
    None,                                                                                     #4
    None,                                                                                     #5
    None,                                                                                     #6
    r'\\mnt\scopearray\Sinha_Drew\20160229_spe9Acquisition_DevVarA',    #7
    r'\\mnt\scopearray\Sinha_Drew\20160229_spe9Acquisition_DevVarB',    #8
    None,                                                                                     #9
    r'\\mnt\scopearray\Sinha_Drew\20160304_spe9Acquisition_DevVarB',    #10
    r'\\mnt\scopearray\Sinha_Drew\20160304_spe9Acquisition_DevVarC',    #11
    None,                                                                                     #12
    None,                                                                                     #13
    None,                                                                                     #14
    None,                                                                                     #15
    None,                                                                                     #16
    None,                                                                                     #17
    None,                                                                                     #18
    None,                                                                                     #19
    None,                                                                                     #20
    None,                                                                                     #21
    None,                                                                                     #22
    None,                                                                                     #23
]
#directory_bolus = folderStuff.DirectoryBolus(working_directory, human_directory, data_directories, extra_directories, experiment_directories, annotation_directories, done = 16, ready = [16,23])    
directory_bolus = folderStuff.DirectoryBolus(working_directory, human_directory, data_directories, extra_directories, experiment_directories, annotation_directories, done = 23, ready = 24)    
if sys.platform == 'linux':
    human_directory = folderStuff.linux_path(human_directory)
    working_directory = folderStuff.linux_path(working_directory)

# Initialize a checker for manual image annotation.
# Use the following if you have issues with file permissions: sudo chmod -R g+w /mnt/scopearray
if my_mode == 1:
    checker = organizeData.HumanCheckpoints(data_directories[-1])

# Do the image analysis on heavenly/debug it.
if my_mode == 2:
    start_time = time.time()
    organizeData.measure_experiments(
        # Directory information.
        directory_bolus,

        # Parameters for broken down runs for debugging.
        parallel = False, #True
        only_worm = '/mnt/scopearray/ZhangWillie/2016.07.01 spe-9 age-1 Run 21/00', #None,#'/mnt/scopearray/ZhangWillie/2016.05.24 spe-9 age-1 Run 24A/04', 
        only_time = None,
        #parallel = False, 
#       only_worm = directory_bolus.data_directories[15] + os.path.sep + '132', 
#       only_time = '2016-02-23t0041',

        # Metadata information.
        refresh_metadata = True
    )
    print('Ran processing for {:.1f} s'.format(time.time()-start_time))

# Do the analysis.
if my_mode == 3: 
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True})   
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_dir_out':'/mnt/bulkdata/wzhang/temp/'})
    #with open('/mnt/bulkdata/wzhang/human_dir/debug_SVRload/df_rerun.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)
       
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/mnt/bulkdata/wzhang/human_dir/spe-9_health_SVR'})   
    #with open('/mnt/bulkdata/wzhang/human_dir/age-1_health/df_age-1.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)
        
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/mnt/bulkdata/wzhang/human_dir/spe-9_health_SVR'})   
    #with open('/mnt/bulkdata/wzhang/human_dir/spe-9_health/df_spe-9.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)
        
    # Make an SVR for health from age-1 data
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/mnt/bulkdata/wzhang/human_dir/age-1_health_SVR_oldhp/'})
    #with open('/mnt/bulkdata/wzhang/human_dir/spe-9perage-1SVM_health/df_spe-9perage-1SVM_health.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)
        
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_dir_out':'/mnt/bulkdata/wzhang/human_dir/spe-9age-1combined_health_SVR'})
    
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/spe-9age-1combined_health_SVR/'})
    #with open('/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/combined_speage_SVM_data/df_combined_speage_SVM_data.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)

    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/spe-9age-1combined_health_SVR/'})
    #with open('/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/spe-9percombinedSVM_health/df_spe-9percombinedSVM_health.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)
        
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, {'adult_only': True, 'svm_directory':'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/spe-9age-1combined_health_SVR/'})
    #with open('/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/age-1percombinedSVM_health/df_age-1percombinedSVM_health.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df},my_file)

    # For building combined SVR
    ## Build sample weights specified measured_healths
    #worm_names = characterizeTrajectories.get_worm_names(
        #directory_bolus.data_directories[directory_bolus.ready[0]:directory_bolus.ready[1]] if type(directory_bolus.ready) is list else directory_bolus.data_directories[:directory_bolus.ready])
    #worm_types = ['age-1']
    ##spe-9 labeled with a zero; mutants labeled by position in worm_types+1
    #worm_labels = np.array([mutant_num+1 if mutant_type in a_worm else 0 for a_worm in worm_names for mutant_num,mutant_type in enumerate(worm_types)])
    #worm_pop_sizes = np.array([np.count_nonzero(worm_labels == label_num) for label_num in np.arange(len(worm_types)+1)])
    #print(worm_pop_sizes)
    
    #sample_weights = np.array(
        #[worm_pop_sizes[0]/worm_pop_sizes[worm_label] for worm_label in worm_labels])
    
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, 
        #{'adult_only': True, 'svm_dir_out':'/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/spe-9age-1combined_weighted_health_SVR/', 'sample_weights':sample_weights})
    #adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, 
        #{'adult_only': True, 'sample_weights':sample_weights})
    #with open('/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/spe-9age-1percombinedweightedSVM_health/df_spe-9age-1percombinedweightedSVM_health.pickle','wb') as my_file:
        #pickle.dump({'adult_df':adult_df,'sample_weights':sample_weights},my_file)
    
    svm_directory='/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/utilities/spe-9age-1combined_weighted_health_SVR/'
    adult_df = characterizeTrajectories.CompleteWormDF(directory_bolus, save_directory, 
        {'adult_only': True, 'svm_directory':svm_directory})
    with open('/media/Data/Work/ZPLab/Analysis/MutantHealth/worm_health_data/age-1percombinedweightedSVM_health/df_age-1percombinedweightedSVM_health.pickle','wb') as my_file:
        pickle.dump({'adult_df':adult_df,'svm_directory':svm_directory},my_file)
