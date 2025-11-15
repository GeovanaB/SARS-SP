#!/bin/bash
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -p batch-AMD
#SBATCH --time=900:00:00

source ~/.bashrc
conda activate tcc_darts

python -u catboost-temp-optu-vali-1.py > catboost-temp-optu-vali-1.out
