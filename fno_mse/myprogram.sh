#!/bin/bash
#SBATCH -J msebinary_200_fractures           # Job name 
#SBATCH -o /scratch/08780/cedar996/outfile/outfile.o%j       # Name of stdout output file
#SBATCH -e /scratch/08780/cedar996/errfile/outfile.e%j       # Name of stderr error file
#SBATCH -p gpu-h100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH --ntasks-per-node 1             # Total # of mpi tasks
#SBATCH -t 04:20:00        # Run time (hh:mm:ss)
    # Send email at begin and end of job
#SBATCH -A OTH21076       # Project/Allocation name (req'd if you have more tha
CUDA_VISIBLE_DEVICES=0 python t2train_lightning.py #input_file=input.yml data_loc=$SCRATCH/ms-net/input
