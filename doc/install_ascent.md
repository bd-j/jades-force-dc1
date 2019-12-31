Demo
=====

```bash
git clone https://github.com/olcf/cpu_gpu_dgemm.git
cd cpu_gpu_dgemm/
module load cuda
module load essl
make
```

Interactive Job
=======
```bash
bsub -P GEN126 -nnodes 1 -W 30 -alloc_flags "gpumps" -Is /bin/bash
```

to enable MPS server add 
```bash
-alloc_flags "gpumps"
```

Submit job
===
```bash
jsrun -n1 -g1 -a1 ./cpu_gpu_dgemm
```

* `-n` number of resource sets
* `-g` number of GPUS per resource set
* `-a` number of processes per resource set

By default GPUs on summit are in exclusive mode (each GPU can anly talk to a single process) so `-n1 -g1 -a6` will fail to find enough free GPUs to complete more than one version of the code.

bsub works for MPI as well - each rank is a separate process, number of ranks = `a`

Useful modules
=====

These should be added to .bash_profile

```bash
module purge
module load git/2.13.0
module load gcc/4.8.5
#module load spectrum-mpi/10.2.0.11-20190201
module load spectrum-mpi/10.3.0.0-20190419
module load hdf5/1.10.3
module load cuda/9.2.148
module load python/3.7.0-anaconda3-5.3.0
module load py-h5py/2.8.0-py3
module load py-mpi4py/3.0.0-py3
```


Create conda environment
====

```bash
#Load Python and create a basic environment
module load python/3.7.0-anaconda3-5.3.0
conda env create -f test_environment.yml -p ~/test_env
source activate test_env
conda install numba
conda install pymc3
pip install emcee
pip install pycuda
pip install future_fstrings
# install things with HPC specific binaries
# pip install -v --no-binary=mpi4py mpi4py
# CC=gcc HDF5_MPI="ON" HDF5_VERSION=1.10.3 pip install -v --no-binary=h5py h5py
```

to delete an env:

```
conda remove --prefix ~/<env_name> --all
```

Compilation directories
====
Both pycuda and theano/pymc3 write compiled things to cache directories.  The defaults are wherever you built the environment, which is probably on /nfs and unwritable by actual jobs.  So

```bash
mkdir /gpfs/wolf/gen126/scratch/<user>/pycudacache
mkdir /gpfs/wolf/gen126/scratch/<user>/theanocache
```

Then anytime you build a pycuda SourceModule add
```python
SourceModule("""C code here """, cache_dir="/gpfs/wolf/gen126/<user>/bdjohnson/pycudacache/")
```

and you also have to do something like
```bash
export THEANO_FLAGS="base_compiledir=/gpfs/wolf/gen126/scratch/<user>/theanocache/"
```

Run
=====

In order to enable CUDA-Aware MPI on Summit/Ascent, you must pass the `--smpiargs="-gpu"` flag to `jsrun`. If you are not familiar with CUDA-Aware MPI...  but actually I don't think we need this since the MPI processes are not sharing device data.

then get on an interactive node and run the run_patches with
```bash
jsrun -n1 -g0 -a1 python run_patches_commtest.py  # one cpu process, no gpu
jsrun -n1 -g0 -c10 -a10 python run_patches_commtest.py # ten cpu process, no gpu
jsrun -n1 -g1 -a1 python run_patches_commtest.py   # one cpu process, one gpu
jsrun -n1 -g1 -c10 -a10 python run_patches_commtest.py  # ten cpu process, one gpu
```


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