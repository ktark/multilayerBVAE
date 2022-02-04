#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
source ../thesis_env_cuda11/bin/activate
python train_hierarchical_demo.py --gpus 1 --max_steps 600000 --gamma 1 --delta 8 --zeta 1 --latent_dim_level1 20 --latent_dim_level0 200 --hier_groups 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 --level0_training_start_iter 0 --C_stop_iter 1 --C_min 0 --C_max 0 --loss_function bvae --dataset boxheadsimple --lr 0.0003