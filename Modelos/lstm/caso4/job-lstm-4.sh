#!/bin/bash
#SBATCH -n 10
#SBATCH --ntasks-per-node=10
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:1
#SBATCH --time=900:00:00

source ~/.bashrc
conda activate tcc1

# p/ rodar um modulo cuda
module load cuda/12.2

python -u lstm-temp-optu-vali-4.py > lstm-temp-optu-vali-4.out
