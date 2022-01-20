#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
source ../thesis_env/bin/activate
python train_hierarchical.py --gpus 1 --max_steps 400000 --gamma 1 --delta 5 --zeta 1.0 --dataset boxhead

