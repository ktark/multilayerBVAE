#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
source ../thesis_env/bin/activate
python train_hierarchical.py --gpus 1 --max_steps 400000 --gamma 1 --delta 12.5 --zeta 2.5 --latent_dim_level1 20 --latent_dim_level0 200 --hier_groups 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 --level0_training_start_iter 1e5 --loss_function bvae_l1_first --dataset boxhead

