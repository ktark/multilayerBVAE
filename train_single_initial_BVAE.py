from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from src.utils import *
from src.modules import *
import wandb
from src.data_loaders import *
from src.architectures import *
from src.callbacks import *
from src.utils import seed_everything
from argparse import ArgumentParser

import os


def main(hparams):
    # os.environ['WANDB_API_KEY'] = '536da542ee9b110c15555f219ff08d8d3fbc9ffb0'
    seed_everything(int(hparams.seed))
    wandb.login()
    pl.seed_everything(int(hparams.seed), workers=True)
    print(hparams)
    if hparams.dataset == "dsprites":
        ds = Dsprites()
        vae = BVAE(nc=1, decoder_dist='bernoulli', latent_dim=int(hparams.latent_dim), input_height=64,
                   gamma=int(hparams.gamma),
                   max_iter=int(hparams.max_steps), lr=5e-4,
                   beta1=0.9, beta2=0.999, C_min=float(hparams.C_min),
                   C_max=float(hparams.C_max), C_stop_iter=int(hparams.C_stop_iter), reparameters_coef=float(hparams.reparameters_coef))

    else:
        ds = BoxHead(dataset=hparams.dataset)
        vae = VAEh(nc=3, decoder_dist='gaussian', latent_dim=int(hparams.latent_dim), input_height=64,
                   gamma=int(hparams.gamma),
                   max_iter=int(hparams.max_steps), lr=3e-4,
                   beta1=0.9, beta2=0.999, C_min=float(hparams.C_min),
                   C_max=float(hparams.C_max), C_stop_iter=int(hparams.C_stop_iter),  reparameters_coef=float(hparams.reparameters_coef))

    ds_dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)



    wandb_logger = WandbLogger(
        name=f'BVAE - test image every epoch: ds {hparams.dataset} | '
             f'gamma/betaVAE {str(hparams.gamma)} | '
             f'C: {str(hparams.C_min)}-{str(hparams.C_max)}/{str(hparams.C_stop_iter)} | '
             f'latents: {str(hparams.latent_dim)} |'
             f'steps: {str(hparams.max_steps)} |'
             f'seed: {str(hparams.seed)} |'
             f'repar: {str(hparams.reparameters_coef)}',
        project='thesis', job_type='train', log_model=False)

    # wandb_logger.watch(vae, log_freq=10000)  # log network topology and weights
    if hparams.dataset == "dsprites":
        sample_ids = [125140, 333500, 578560]
    else:
        sample_ids = [15, 4, 400]


    epoch_end_example_image_S = ImagePredictionLogger2(sample=sample_ids[0], ds=ds, wandb_logger=wandb_logger)  # Square
    epoch_end_example_image_S2 = ImagePredictionLogger2(sample=sample_ids[1], ds=ds, wandb_logger=wandb_logger)  # Ellipse
    epoch_end_example_image_S3 = ImagePredictionLogger2(sample=sample_ids[2], ds=ds, wandb_logger=wandb_logger)  # Ellipse

    #mu_val_logger = get_first_images_mu_logger(ds, wandb_logger)

    wandb_logger.watch(vae, log_freq=1000)  # log network topology and weights

    trainer = pl.Trainer(gpus=hparams.gpus,  max_steps=int(hparams.max_steps), log_every_n_steps=100,
                         enable_progress_bar = False,
                         logger=wandb_logger, callbacks=[ModelSummary(max_depth=-1),
                                                         epoch_end_example_image_S,
                                                         epoch_end_example_image_S2,
                                                         epoch_end_example_image_S3

                                                         #,
 #                                                        mu_val_logger
                                                         ])
    trainer.fit(vae, ds_dl)
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=-1)
    parser.add_argument("--max_steps", default=1500000)
    parser.add_argument("--C_min", default=0.0)
    parser.add_argument("--C_max", default=0.0)
    parser.add_argument("--gamma", default=5.0)  # beta from BVAE
    parser.add_argument("--C_stop_iter", default=1)  # C_stop iteration
    parser.add_argument("--latent_dim", default=12)  # latent_dim
    parser.add_argument("--dataset", default="boxhead2")  # dataset
    parser.add_argument("--seed", default=2)  # dataset
    parser.add_argument("--reparameters_coef", default=1.0)

    args = parser.parse_args()

    main(args)
