# Multilayer BVAE

## Setup:
 - Create virtual environment with Python 3.8.
 - Install requirements `pip install -r requirements.txt`.
 - set Weights and Biases API token `export WANDB_API_KEY=<YOUR KEY HERE>>`.
 - load your environment to slurm job shell scripts `*.sh`.
 - copy datasets to folder `dataset/`. Supported datasets `boxheadsimple/`,`boxheadsimple2/`, `boxhead_07/`. Dataset generation from [Boxhead data and experiment repository](https://github.com/yukunchen113/compvae)

## To run experiments:
- Experiments are defined as slurm jobs
- Five-layer beta-vae training. Edit parameters in `run_five_level.sh`. To run as slurm job `sbatch run_five_level.sh`. 
- Beta-vae training. Edit parameters in `train_single_initial_BVAE.sh`. To run as slurm job `sbatch train_single_initial_BVAE.sh`. 




