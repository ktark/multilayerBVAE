#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
source ../thesis_env_cuda11/bin/activate
export WANDB_API_KEY=536da542ee9b110c15555f219ff08d8d3fbc9ffb
python train_single_initial_BVAE.py --gpus 1 --max_steps 1500 --gamma 100 --C_min 0 --C_max 20 --C_stop_iter 100000 --latent_dim 10 --dataset dsprites --seed 2
