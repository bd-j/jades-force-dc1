#!/bin/bash

#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p gpu # Partition to submit to
#SBATCH -t 10:00:00 # Runtime
#SBATCH -J force_val_test
#SBATCH -o /n/holyscratch01/eisenstein_lab/bdjohnson/jades_force/validation/logs/valtest_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/eisenstein_lab/bdjohnson/jades_force/validation/logs/valtest_%A_%a.err # Standard err goes to this file

module purge
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/10.1.243-fasrc01
module load Anaconda3/5.0.1-fasrc01

# where code will be run
jdir=${SCRATCH}/eisenstein_lab/bdjohnson/jades_force/validation

patchid=$SLURM_ARRAY_TASK_ID


source activate jadespho
cd $jdir
python sample_validation.py --logging --patch_num=$patchid --outfile=validation_sample_patchid${patchid}.h5
