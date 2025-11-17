#!/bin/bash
#SBATCH -n 5
#SBATCH --ntasks-per-node=5
#SBATCH -p batch-AMD
#SBATCH --time=900:00:00

source ~/.bashrc
conda activate jax_gpu

python -u rf-temp-optu-vali-2.py > rf-temp-optu-vali-2.out
