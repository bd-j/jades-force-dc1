#!/bin/bash

#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem-per-cpu=2000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p shared # Partition to submit to
#SBATCH -t 01:00:00 # Runtime
#SBATCH -J preprocess_jades
#SBATCH -o /n/holyscratch01/eisenstein_lab/bdjohnson/jades_force/cannon/logs/preprocess_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/eisenstein_lab/bdjohnson/jades_force/cannon/logs/preprocess_%A_%a.err # Standard err goes to this file

module purge
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01
module load cuda/10.1.243-fasrc01
module load Anaconda3/5.0.1-fasrc01

# where code will be run
jadesdir=$SCRATCH/eisenstein_lab/bdjohnson/jades_force/cannon/
# where the images are
mosdir=$SCRATCH/eisenstein_lab/stacchella/mosaic/mosaic
framedir=$SCRATCH/eisenstein_lab/bdjohnson/jades_force/data/2019-mini-challenge/mosaics/st/trimmed


source activate jadespho
cd $jadesdir
mkdir -p $framedir
python preprocess_mosaic.py --frames_directory=$framedir \
                            --mosaics_directory=$mosdir \
                            --store_directory=${jadesdir}/stores \
                            --store_name=mini-challenge-19-mosaic-st