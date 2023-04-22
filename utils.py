import numpy as np

SBATCH_PATH = '/home/itaitz/EC-KitY'

BIG_DATASETS = ['adult', 'magic', 'coil2000', 'agaricus_lepiota', 'mushroom', 'ring']

# 10 datasets of ~1000 samples each
MEDIUM_DATASETS = ['titanic', 'parity5+5', 'flare', 'credit_g', 'german', 'xd6', 'tokyo1', 'tic_tac_toe', 'pima', 'breast']

# 10 datasets of 200-350 samples each
SMALL_DATASETS = ['prnn_crabs', 'sonar', 'biomed', 'heart_statlog', 'spect', 'breast_cancer', 'heart_h', 'cleve', 'bupa']

thresholds = {
    'adult': 0.1,
    'magic': 0.07,
    'coil2000': 0.03,
    'agaricus_lepiota': 0.04,
    'mushroom': 0.06,
    'ring': 0.02,
    'CIFAR-10': 0.1
}

linear_gen_weight = lambda gen: gen + 1
square_gen_weight = lambda gen: (gen + 1) ** 2
exp_gen_weight = lambda gen: np.e ** (gen + 1)

CIFAR10_TRAIN_SAMPLES = 5000
CIFAR10_TEST_SAMPLES = 1000

def generate_sbatch_str(gen, idx, vector, use_gpu=True):
    if use_gpu:
        return f'''#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 6-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name gpu_job_{gen}_{idx}			### name of the job
#SBATCH --output=jobs/gpu_job_{gen}_{idx}.out			### output log for running job - %J for job number
#SBATCH --gpus=1				### number of GPUs, allocating more than 1 requires IT team's permission
#SBATCH --wait

#SBATCH --mem=24G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

echo "SLURM_JOBID"=$SLURM_JOBID

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate ec_env 				### activate a conda environment, replace my_env with your conda environment
python "/sise/home/itaitz/EC-KitY/nn_evaluator.py" {' '.join([str(x) for x in vector])}
'''

    else:
        return f'''#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH

#SBATCH --partition main ### specify partition name where to run a job. main - 7 days time limit
#SBATCH --time 0-01:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name cpu_job_{gen}_{idx} ### name of the job. replace my_job with your desired job name
#SBATCH --output jobs/cpu_job_{gen}_{idx}.out ### output log for running job - %J is the job number variable

#SBATCH --cpus-per-task=6 # 6 cpus per task – use for multithreading, usually with --tasks=1
#SBATCH --wait ### wait for task to finish before exiting

### Print some data to output file ###
echo "SLURM_JOBID”=$SLURM_JOBID
### Start you code below ####
module load anaconda ### load anaconda module
source activate ec_env ### activating Conda environment, environment must be configured before running the job
python "/sise/home/itaitz/EC-KitY/blackjack_evaluator.py" {' '.join([str(x) for x in vector])}
'''
