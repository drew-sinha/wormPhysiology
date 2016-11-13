import json
import os
import pathlib
import subprocess
import sys

import measurePhysiology.organizeHealth as organizeHealth

# Constants
work_dir = '/scratch/sinhad/work_dir/'  # Where mask data will be aved
human_dir = '/scratch/sinhad/utilities/'

def configure_pipeline(*dir_args):
    # dir_args - set of Path variables
    animals_toprocess = []
    
    for data_dir in dir_args:
        checker = organizeHealth.HumanCheckpoints(data_dir)
        checker.clean_experiment_directory(experiment_directory=data_dir)
        base = organizeData.BaseMeasurer(data_dir, work_dir, human_dir)
        for animal_subdir in base.worm_dirs:
            if animal_subdir.split(os.path.sep)[-1] in checker.good_worms and animal_subdir.split(os.path.sep)[-1] in checker.dead_worms:
                animals_toprocess.append(animal_subdir)
        
    # Save the json with the worms of interest to run
    with (pathlib.Path(os.getcwd()) / 'enumerated_animals.json').open() as subdir_file:
        json.dumps({'animals_toprocess':animals_toprocess}, subdir_file)
    
    # Write the job file corresponding to this particular job
    jobscript_template = string.Template(
        '''
        #PBS -N $job_name
        #PBS -t 1-$num_toprocess
        #PBS -l nodes=1:ppn=1
        #PBS -l walltime=06:00:00
        #PBS -M drew.sinha@wustl.edu
        #PBS -d /scratch/sinhad/
        
        python cluster_compute.py process $${PBS_ARRAYID}
        ''')
    jobscript_template.substitute(job_name='process_animals',
        num_toprocess=repr(len(animals_toprocess)))
    with (pathlib.Path(os.getcwd()) / 'job_script.sh').open() as job_script:
        job_script.write(jobscript_template)
    
    # Submit this job through subprocess and qsub
    subprocess.call('qsub ' + str(pathlib.Path(os.getcwd()) / 'job_script.sh'))

def process_animal(animal_id):
    with (pathlib.Path(os.getcwd()) / 'enumerated_animals.json').open() as subdir_file:
        processing_data = json.loads({'animals_toprocess':animals_toprocess}, subdir_file)
        
    worm_measurer = organizeData.WormMeasurer(processing_data[animal_id], working_dir, human_dir)
    worm_measurer.measure_a_worm_try()

if __name__ is "__main__":
    if sys.argv[1] is 'configure':
        configure_pipeline(*sys.argv[2:])
    elif sys.argv[1] is 'process':
        process_animal(sys.argv[2])
