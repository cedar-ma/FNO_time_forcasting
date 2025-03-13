#!/bin/bash
#SBATCH -J vof6_10o10_fractures           # Job name 
#SBATCH -o /scratch/08780/cedar996/outfile/outfile.o%j       # Name of stdout output file
#SBATCH -e /scratch/08780/cedar996/errfile/outfile.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH --ntasks-per-node 128             # Total # of mpi tasks
#SBATCH -t 10:20:00        # Run time (hh:mm:ss)
    # Send email at begin and end of job
#SBATCH -A OTH21076       # Project/Allocation name (req'd if you have more tha
CUDA_VISIBLE_DEVICES=0 python train_lightning.py
