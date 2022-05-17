#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
#source -- load your environmetn
python train_five_layer.py --gpus 1 --max_steps 200000 --gamma 1.5 --latent_dims 500 400 300 200 100 --dataset boxheadsimple2 --lr 0.0003   --l1 0.0001 --l2 0.05
