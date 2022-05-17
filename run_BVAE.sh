#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
#source -- load your environmetn
python train_single_initial_BVAE.py --gpus 1 --max_steps 200000 --gamma 5 --C_min 0 --C_max 0 --C_stop_iter 100000 --latent_dim 10 --dataset dsprites --seed 2
