#!/bin/bash
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -p batch-AMD
#SBATCH --time=900:00:00

source ~/.bashrc
conda activate tcc1

python -u rf-temp-optu-vali-1.py > rf-temp-optu-vali-1.out
