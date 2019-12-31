module purge

# modules
module load git/2.17.0-fasrc01

# --- compilers ---
#module load gcc/8.2.0-fasrc01
module load intel/19.0.5-fasrc01

# --- MPI ---
module load openmpi/4.0.1-fasrc01

# --- HDF5 ---
# Single threaded, core module
#module load hdf5/1.10.1-fasrc03
# MPI (openmpi 4.0.1)
module load hdf5/1.10.5-fasrc01

# --- compiler/mpi/h5py ---
# module load gcc/4.9.3-fasrc01 openmpi/2.1.0-fasrc01
# module load gcc/4.9.3-fasrc01 openmpi/2.1.0-fasrc01 hdf5/1.10.1-fasrc01
# module load intel/19.0.5-fasrc01 hdf5/1.10.5-fasrc03
# module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01

# --- Cuda ---
#module load cuda/9.2.88-fasrc01
module load cuda/10.1.243-fasrc01
