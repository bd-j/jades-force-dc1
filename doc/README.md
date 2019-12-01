
Useful modules on Odyssey
=====

These should be added to `.bash_profile`, or a little script can be run at the beginning of each job.

```bash
module purge
module load git/2.17.0-fasrc01

# --- compiler/mpi/hdf5 ---
module load intel/19.0.5-fasrc01 openmpi/4.0.1-fasrc01 hdf5/1.10.5-fasrc01

# --- Cuda ---
module load cuda/10.1.243-fasrc01
```

Create conda environment
====

```bash
# --- Load Python and create a basic environment ---
# -without hdf5, matplotlib, others
module load Anaconda3/5.0.1-fasrc01
# -with hdf5, matplotlib, others
# module load Anaconda3/5.0.1-fasrc02

conda env create -f jadespho_environment.yml #-p ~/jadespho
source activate jadespho
# --- install things with HPC specific binaries ---
pip install -v --no-binary=mpi4py mpi4py
# CC=gcc HDF5_MPI="ON" HDF5_VERSION=1.10.5 pip install -v --no-binary=h5py h5py
pip install -v --no-binary=h5py h5py
pip install pycuda
conda install --freeze-installed pymc3

# --- install forcepho (optional, can be run from source direcory) ---
git clone git@github.com:bd-j/forcepho.git
cd forcepho
python setup.py install
```

Then need to do
```
cp forcepho/*h ~/.conda/envs/jadespho/lib/python3.7/site-packages/forcepho-0.2-py3.7.egg/forcepho/
```

to delete an env:

```
conda remove --prefix ~/<env_name> --all
```

Compilation directories
====
Both pycuda and theano/pymc3 write compiled things to cache directories.  
The defaults are wherever you built the environment, which may be unwritable or slow during jobs.

```bash
mkdir /n/scratchlfs/<group>/<user>/pycudacache
mkdir /n/scratchlfs/<group>/<user>/theanocache
```

Then anytime you build a pycuda SourceModule add
```python
SourceModule("""C code here """, cache_dir="/n/scratchlfs/.../<user>/bdjohnson/pycudacache/")
```

and you also have to do something like
```bash
export THEANO_FLAGS="base_compiledir=/n/scratchlfs/.../<user>/theanocache/"
```

Submit Job (Odyssey)
=====

Single core job

```bash
#!/bin/bash

#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#BATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1000 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -p gpu # Partition to submit to
#SBATCH -t 06:00:00 # Runtime
#SBATCH -J force_smoke_test
#SBATCH -o /n/scratchlfs/eisenstein_lab/bdjohnson/jades_force/logs/smoketest_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/scratchlfs/eisenstein_lab/bdjohnson/jades_force/logs/smoketest%A_%a.err # Standard err goes to this file

MYSCRATCH=/n/scratchlfs/eisenstein_lab/$USER
./ody_modules.sh
export THEANO_FLAGS="base_compiledir=$MYSCRATCH/theanocache" #,floatX=float32"
source activate jadespho
date
python run_patch_gpu_test_simple.py
```

Multi-core job: same as above, but the final line is

```
srun -n $SLURM_NTASKS --mpi=pmi2 python run_patch_gpu_test_simple.py
```

Interactive Job (Odyssey)
=======
```bash
srun --pty -p gpu -t 0-06:00 --mem 8000 --gres=gpu:1 /bin/bash
```

From the odyssey docs: While on GPU node, you can run `nvidia-smi` to get information about the assigned GPU

Not sure if it's necessary or how to enable MPS server.  On ascent one does
```bash
-alloc_flags "gpumps"
```

Note that for the gpu_test queue the time limit is 1 hour



Profiling 
======
output.%h.%p

use `::KernelName:<int>` where `<int>` is the index of the kernel invocation that you want to profile

```bash
# detailed profiling of the kernel
jsrun -n1 -g1 -a1  nvprof --analysis-metrics -o /gpfs/wolf/gen126/scratch/bdjohnson/large_prof_metrics%h.%p.nvvp python run_patch_gpu_test_simple.py 

# FLOP count
jsrun -n1 -g1 -a1  nvprof --kernels ::EvaluateProposal:1 --metrics flop_count_sp python run_patch_gpu_test_simple.py 


```