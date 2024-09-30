#!/bin/bash

#SBATCH -J viper
#SBATCH -N 1

#SBATCH --account=p71704
#SBATCH --ntasks-per-core=2

#SBATCH --partition=skylake_0384
#SBATCH --qos=skylake_0384

source ~/.bashrc # for conda init!
conda deactivate
module purge

# load Intel compilers
module load intel-oneapi-compilers/2022.2.1-gcc-9.5.0-xg435ds 
module load intel-oneapi-tbb/2021.7.1-intel-2021.7.1-xrpassj
module load intel-oneapi-mpi/2021.7.1-intel-2021.7.1-fzg6q4x
module load intel-oneapi-mkl/2022.2.1-intel-2021.7.1-p7jisxw


# load conda
module load miniconda3/4.10.3-intel-2021.7.1-j6woa7k

# load conda env
conda activate on-the-fly

export I_MPI_DEBUG=100
export I_MPI_PIN_RESPECT_CPUSET=0
export MKL_NUM_THREADS=2

viperleed calc
