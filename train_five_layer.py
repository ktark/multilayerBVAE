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
    seed_everything(int(hparams.seed))
    wandb.login()
    pl.seed_everything(int(hparams.seed), workers=True)
    print(hparams)

    ds = BoxHead(dataset=hparams.dataset)
    ds_t = BoxHeadWithLabels(dataset=hparams.dataset) #for testing only

    vae = VAEFiveLevel(nc=3, decoder_dist='gaussian', latent_dims=hparams.latent_dims,
               gamma=float(hparams.gamma), l1_regularization = float(hparams.l1), l2_regularization = float(hparams.l2),
               max_iter=int(hparams.max_steps), lr=float(hparams.lr),
               beta1=0.9, beta2=0.999)

    ds_dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    ds_test = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8) #for testing same dataset as training

    wandb_logger = WandbLogger(
        name=f'{hparams.name} : ds {hparams.dataset} | '
             f'gamma/betaVAE {str(hparams.gamma)} | '
             f'latents: {str(hparams.latent_dims)} |'
             f'steps: {str(hparams.max_steps)} |'
             f'L1:{str(hparams.l1)} |'
             f'L2:{str(hparams.l2)} |'
             f'lr:{str(hparams.lr)} |'
             
             f'seed: {str(hparams.seed)}',
        project='thesis_2', job_type='train', log_model=True)

    # wandb_logger.watch(vae, log_freq=10000)  # log network topology and weights

    sample_ids = [15, 4, 400]

    epoch_end_example_image_S = ImagePredictionLoggerFiveLevel(sample=sample_ids[0], ds=ds, wandb_logger=wandb_logger)
    epoch_end_example_image_S2 = ImagePredictionLoggerFiveLevel(sample=sample_ids[1], ds=ds, wandb_logger=wandb_logger)
    epoch_end_example_image_S3 = ImagePredictionLoggerFiveLevel(sample=sample_ids[2], ds=ds, wandb_logger=wandb_logger)

    test_logger = TestImagePredictionLoggerFiveLevel(sample=sample_ids[1], ds=ds, ds_t=ds_t, wandb_logger=wandb_logger)

    wandb_logger.watch(vae, log_freq=10000)  # log network topology and weights

    trainer = pl.Trainer(gpus=hparams.gpus,  max_steps=int(hparams.max_steps), log_every_n_steps=100,
                         enable_progress_bar = False,
                         logger=wandb_logger, callbacks=[ModelSummary(max_depth=-1),
                                                         epoch_end_example_image_S,
                                                         epoch_end_example_image_S2,
                                                         epoch_end_example_image_S3,
                                                         test_logger
                                                         ])
    trainer.fit(vae, ds_dl)
    trainer.test(vae, ds_test)

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=-1)
    parser.add_argument("--max_steps", default=1500000)
    parser.add_argument("--gamma", default=1.0) 
    parser.add_argument("--l1", default=1.0)  # L1 loss coef
    parser.add_argument("--l2", default=1.0)  # L2 loss coef
    parser.add_argument("--dataset", default="boxhead2") 
    parser.add_argument("--seed", default=2)  
    parser.add_argument("--lr", default=0.0003)  
    parser.add_argument("--latent_dims", nargs="*", type=int,
                        default=[500,400,300, 200, 100])
    parser.add_argument("--name", default="")  

    args = parser.parse_args()

    main(args)
