#!/bin/bash

export I_MPI_ADJUST_REDUCE=3

#module load intel/2017u5
#BIN=/data0/software/vasp/intel/vasp.5.4.4/bin/vasp_ncl
#export FI_PROVIDER=tcp
#export I_MPI_FABRICS=shm:tcp

export FI_PROVIDER=tcp
export I_MPI_OFI_PROVIDER=tcp
export FI_NO_ADDRCONFIG=1
export I_MPI_OFI_LIBRARY_INTERNAL=1

module load intel/2020u1

module load wannier90/3.1.0

BIN=/data0/software/vasp/intel/vasp.6.3.0-wannier90-neb-cell-sol/bin/vasp_ncl

#   sbatch --partition=hcpu64 --ntasks=16 job.sh
#   grep "running on" OUTCAR
#   squeue -w acn072
#   scontrol show node acn072 | grep CPU
#   grep E-fermi OUTCAR

cp KPOINTS.pw KPOINTS

cp INCAR.wannier INCAR

mpirun -np 16 $BIN
