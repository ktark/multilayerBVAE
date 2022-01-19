from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from src.utils import *
from src.modules import *
import wandb
from src.data_loaders import *
from src.architectures import *
from src.callbacks import *
from argparse import ArgumentParser

import os


def main(hparams):
    # os.environ['WANDB_API_KEY'] = '536da542ee9b110c15555f219ff08d8d3fbc9ffb0'

    wandb.login()
    pl.seed_everything(2, workers=True)
    print(hparams)
    ds = BoxHead()
    ds_dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True, worker_init_fn=np.random.seed(0))

    vae = VAEhier(nc=3, decoder_dist='gaussian', latent_dim_level0=int(hparams.latent_dim_level0),
                  latent_dim_level1=int(hparams.latent_dim_level1), C_max=float(hparams.C_max),
                  C_min=float(hparams.C_min), gamma=float(hparams.gamma),
                  zeta=float(hparams.zeta), delta=float(hparams.delta),
                  C_stop_iter=int(hparams.C_stop_iter), hier_groups=hparams.hier_groups)

    wandb_logger = WandbLogger(
        name=f'Hierarchy: ds {hparams.dataset} | '
             f'gamma/betaVAE {str(hparams.gamma)} | '
             f'C: {str(hparams.C_min)}-{str(hparams.C_max)}/{str(hparams.C_stop_iter)} | '
             f'latents: {str(hparams.latent_dim_level0)}/{str(hparams.latent_dim_level1)} |'
             f'hier recon: {str(hparams.zeta)} |'
             f'hier KL: {str(hparams.delta)} |'
             f'steps: {str(hparams.max_steps)}',
        project='thesis', job_type='train', log_model=False)

    wandb_logger.watch(vae, log_freq=1000)  # log network topology and weights
    epoch_end_example_image_S = ImagePredictionLogger(sample=6, ds=ds, wandb_logger=wandb_logger)  # Square
    epoch_end_example_image_s2 = ImagePredictionLogger(sample=400, ds=ds, wandb_logger=wandb_logger)  # Square
    test_image400 = ImagePredictionLoggerHierarchy(sample=400, ds=ds, wandb_logger=wandb_logger)
    test_image6 = ImagePredictionLoggerHierarchy(sample=6, ds=ds, wandb_logger=wandb_logger)

    trainer = pl.Trainer(gpus=hparams.gpus,  max_steps=int(hparams.max_steps), log_every_n_steps=100,
                         enable_progress_bar=False,
                         logger=wandb_logger, callbacks=[ModelSummary(max_depth=-1),
                                                         epoch_end_example_image_S,
                                                         test_image400,
                                                         epoch_end_example_image_s2,
                                                         test_image6])
    trainer.fit(vae, ds_dl)
    wandb.finish()


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=-1)
    parser.add_argument("--max_steps", default=1500000)
    parser.add_argument("--C_min", default=0.0)
    parser.add_argument("--C_max", default=0.0)
    parser.add_argument("--gamma", default=5)  # beta from BVAE
    parser.add_argument("--delta", default=2)  # hierarchy KL coef
    parser.add_argument("--C_stop_iter", default=1)  # C_stop iteration
    parser.add_argument("--zeta", default=0)  # hierarchy reconstrunction
    parser.add_argument("--latent_dim_level1", default=12)  # latent_dim level 1
    parser.add_argument("--latent_dim_level0", default=48)  # latent_dim level 0
    parser.add_argument("--hier_groups", default=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])  # hierarchy reconstrunction
    parser.add_argument("--dataset", default="boxhead2")  # dataset

    args = parser.parse_args()

    main(args)
