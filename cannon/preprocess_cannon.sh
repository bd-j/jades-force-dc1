#!/bin/bash

#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p shared,test,conroy # Partition to submit to
#SBATCH -t 06:00:00 # Runtime
#SBATCH -J preprocess_jades
#SBATCH -o /n/scratchlfs02/eisenstein_lab/bdjohnson/jades_force/logs/preprocess_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/scratchlfs02/eisenstein_lab/bdjohnson/jades_force/logs/preprocess_%A_%a.err # Standard err goes to this file

module purge
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/10.1.243-fasrc01
module load Anaconda3/5.0.1-fasrc01

# where code will be run
jdir=/n/scratchlfs02/eisenstein_lab/bdjohnson/jades_force/cannon/
# where the images are
fdir=/n/scratchlfs/eisenstein_lab/stacchella/mosaic/st

source activate jadespho
cd $jdir
python preprocess.py --frames_directory=$fdir \
                     --store_directory=${jdir}/stores \
                     --store_name=mini-challenge-19-st