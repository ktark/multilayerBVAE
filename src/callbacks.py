from pytorch_lightning.callbacks import Callback, ModelSummary
from src.modules import *


class ImagePredictionLogger2(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        self.epoch_count += 1
        # print('test epoch count', self.epoch_count, 'div 10', self.epoch_count%10==0)

        if self.epoch_count % 50 == 1:

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            # k = vae.forward(val_imgs)
            # x_recon, mu, logvar = k
            # std = torch.exp(logvar / 2)
            # q = torch.distributions.Normal(mu, std)
            # z = q.rsample()
            pl_module.eval()
            k = pl_module.encoder(val_imgs)
            mu = k[:, :pl_module.latent_dim]

            # metrics ={}
            # for idx, i in enumerate(np.arange(0, k.shape[1]/2, 1)):
            #   metrics['mu_image_'+str(self.sample)+'_'+str(idx)] = k[0:1,int(idx):int(idx)+1].detach().cpu().numpy()[0][0]
            #   metrics['logvar_image_'+str(self.sample)+'_'+str(idx)] = k[0:1,idx+int(k.shape[1]/2):idx+int(k.shape[1]/2)+1].detach().cpu().numpy()[0][0]
            # wandb_logger.log_metrics(metrics)

            z = mu  # Take only first half of k (mu values) removed k[:, :10]
            with torch.no_grad():
                pred = pl_module.decoder(z.to(pl_module.device)).cpu()
            x2_recon = torch.sigmoid(pred).data
            recon_image = make_grid(x2_recon, normalize=True)
            print_orig_recon = []
            print_orig_recon.append(val_imgs.cpu())
            print_orig_recon.append(x2_recon)
            print_orig_recon = torch.cat(print_orig_recon, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('test_image_recon_' + str(self.sample), [(recon_image.numpy())])

            print_images = []
            z_size = z.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    print_images.append(sigm_pred)

            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=True, scale_each=True, nrow=12, pad_value=1)
            self.wandb_logger.log_image('test_image_z_' + str(self.sample), [(images_grid.permute(1, 2, 0).numpy())])
            pl_module.train()


class ImagePredictionLoggerLevel1(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        self.epoch_count += 1
        # print('test epoch count', self.epoch_count, 'div 10', self.epoch_count%10==0)

        if self.epoch_count % 50 == 1:
            # Bring the tensors to CPU

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            # k = vae.forward(val_imgs)
            # x_recon, mu, logvar = k
            # std = torch.exp(logvar / 2)
            # q = torch.distributions.Normal(mu, std)
            # z = q.rsample()
            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_hier = hier_dist_concat[:, :pl_module.latent_dim_level1]

            z = mu_hier  # Take only first half of k (mu values) removed k[:, :10]
            with torch.no_grad():
                pred = pl_module.decoder_level1(z.to(pl_module.device)).cpu()
            x2_recon = torch.sigmoid(pred).data
            recon_image = make_grid(x2_recon, normalize=True)
            print_orig_recon = []
            print_orig_recon.append(val_imgs.cpu())
            print_orig_recon.append(x2_recon)
            print_orig_recon = torch.cat(print_orig_recon, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('hier_test_image_recon_' + str(self.sample), [(recon_image.numpy())])

            pl_module.train()


class saveModelLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint('test20_CBVAE.pth')
        # torch.cuda.empty_cache()


from pytorch_lightning.callbacks import Callback, ModelSummary
from torchvision.utils import make_grid
import gc  # garbage collector


class ImagePredictionLogger(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        self.epoch_count += 1
        # print('test epoch count', self.epoch_count, 'div 10', self.epoch_count%10==0)

        if self.epoch_count % 5 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]

            z_level1 = mu_level1
            z_level0 = mu_level0

            with torch.no_grad():
                pred_level1 = pl_module.decoder_level1(z_level1.to(pl_module.device)).cpu()
                pred_level0 = pl_module.decoder(z_level0.to(pl_module.device)).cpu()

            level1_recon = torch.sigmoid(pred_level1).data
            level1_recon_pad = nn.functional.pad(level1_recon, pad=[4, 4, 4, 4], value=0.0)

            level0_recon = torch.sigmoid(pred_level0).data
            level0_recon_pad = nn.functional.pad(level0_recon, pad=[4, 4, 4, 4], value=1.0)

            orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

            recon_img = []
            recon_img.append(orig_pad)
            recon_img.append(level0_recon_pad)
            recon_img.append(level1_recon_pad)

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('test_image_recon_' + str(self.sample), [(recon_image.numpy())])
            pl_module.train()


class ImagePredictionLoggerHierarchy(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        self.epoch_count += 1
        # print('test epoch count', self.epoch_count, 'div 10', self.epoch_count%10==0)

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            # Higher level images
            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level1.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_level1(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level0.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))

            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 12]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 12]:
                    merged_image.append(img)
                group_counter += group_indices * 12
                level1_counter += 12

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=True, scale_each=True, nrow=12, pad_value=1)

            self.wandb_logger.log_image('hier_image_z_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()
