#!/bin/bash
#SBATCH -J thesis_hierarchical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

# your code goes below
module load python/3.8.6
source ../thesis_env_cuda11/bin/activate
export WANDB_API_KEY=536da542ee9b110c15555f219ff08d8d3fbc9ffb
python train_five_layer.py --gpus 1 --max_steps 100000 --gamma 1.5 --latent_dims 500 400 300 200 100 --dataset boxheadsimple2 --lr 0.0003  --name "trial - no b on layer 1 - latent fc parent (2 layers) + leaky relu + corr visuals" --l1 0.0001 --l2 0.05