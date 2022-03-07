from pytorch_lightning.callbacks import Callback, ModelSummary
from src.modules import *
from pytorch_lightning.callbacks import Callback, ModelSummary
from torchvision.utils import make_grid
from PIL import Image, ImageFont, ImageDraw
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torchvision import transforms
import time



class ImagePredictionLogger2(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

#        if self.epoch_count % 50 == 1:
        if 1==1:

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()
            k = pl_module.encoder(val_imgs)
            mu = k[:, :pl_module.latent_dim]

            # Log test image means
            # metrics ={}
            # for idx, i in enumerate(np.arange(0, k.shape[1]/2, 1)):
            #   metrics['mu_image_'+str(self.sample)+'_'+str(idx)] = k[0:1,int(idx):int(idx)+1].detach().cpu().numpy()[0][0]
            #   metrics['logvar_image_'+str(self.sample)+'_'+str(idx)] = k[0:1,idx+int(k.shape[1]/2):idx+int(k.shape[1]/2)+1].detach().cpu().numpy()[0][0]
            # wandb_logger.log_metrics(metrics)

            z = mu  # Take only first half of k (mu values) removed k[:, :10]
            with torch.no_grad():
                pred = pl_module.decoder(z.to(pl_module.device)).cpu()
            x2_recon = torch.sigmoid(pred).data
            print_orig_recon = [val_imgs.cpu(), x2_recon]
            print_orig_recon = torch.cat(print_orig_recon, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])

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
            self.wandb_logger.log_image('train_images/test_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])
            pl_module.train()


class ImagePredictionLoggerLevel1(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 50 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_hier = hier_dist_concat[:, :pl_module.latent_dim_level1]

            z = mu_hier  # Take only first half of k (mu values) removed k[:, :10]
            with torch.no_grad():
                pred = pl_module.decoder_level1(z.to(pl_module.device)).cpu()
            x2_recon = torch.sigmoid(pred).data
            print_orig_recon = [val_imgs.cpu(), x2_recon]
            print_orig_recon = torch.cat(print_orig_recon, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/hier_test_image_recon_' + str(self.sample),
                                        [(recon_image.numpy())])

            pl_module.train()


class SaveModelLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = 'model_epoch_' + str(trainer.global_step) +'_'+timestr+ '.pth'
        trainer.save_checkpoint(file_name)
        # trainer.logger.save(file_name)

class SaveModelEvery50EpochLogger(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1
        if self.epoch_count % 50 == 1:
            file_name = 'model_epoch_' + str(trainer.global_step) + '.pth'
            trainer.save_checkpoint(file_name)
        # trainer.logger.save(file_name)

class SaveFinalModelLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, trainer, pl_module):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = 'final_model_' + str(trainer.global_step)+'_'+timestr + '.pth'
        trainer.save_checkpoint(file_name)
        # trainer.logger.save(file_name)

class ImagePredictionLogger(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

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
            level1_recon = (level1_recon - torch.min(level1_recon)).div(torch.max(level1_recon) - torch.min(level1_recon))
            level1_recon_pad = nn.functional.pad(level1_recon, pad=[4, 4, 4, 4], value=0.0)

            level0_recon = torch.sigmoid(pred_level0).data
            level0_recon = (level0_recon - torch.min(level0_recon)).div(torch.max(level0_recon) - torch.min(level0_recon))
            level0_recon_pad = nn.functional.pad(level0_recon, pad=[4, 4, 4, 4], value=1.0)

            orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

            recon_img = [orig_pad, level0_recon_pad, level1_recon_pad]

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])
            pl_module.train()


class ImagePredictionLoggerLayer(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()

            distributions = pl_module.encoder(val_imgs)

            hier0 = pl_module.encoder_level1_0(distributions[:, [0, 20]])
            hier1 = pl_module.encoder_level1_1(distributions[:, [1, 21]])
            hier2 = pl_module.encoder_level1_2(distributions[:, [2, 22]])
            hier3 = pl_module.encoder_level1_3(distributions[:, [3, 23]])
            hier4 = pl_module.encoder_level1_4(distributions[:, [4, 24]])
            hier5 = pl_module.encoder_level1_5(distributions[:, [5, 25]])
            hier6 = pl_module.encoder_level1_6(distributions[:, [6, 26]])
            hier7 = pl_module.encoder_level1_7(distributions[:, [7, 27]])
            hier8 = pl_module.encoder_level1_8(distributions[:, [8, 28]])
            hier9 = pl_module.encoder_level1_9(distributions[:, [9, 29]])
            hier10 = pl_module.encoder_level1_10(distributions[:, [10, 30]])
            hier11 = pl_module.encoder_level1_11(distributions[:, [11, 31]])
            hier12 = pl_module.encoder_level1_12(distributions[:, [12, 32]])
            hier13 = pl_module.encoder_level1_13(distributions[:, [13, 33]])
            hier14 = pl_module.encoder_level1_14(distributions[:, [14, 34]])
            hier15 = pl_module.encoder_level1_15(distributions[:, [15, 35]])
            hier16 = pl_module.encoder_level1_16(distributions[:, [16, 36]])
            hier17 = pl_module.encoder_level1_17(distributions[:, [17, 37]])
            hier18 = pl_module.encoder_level1_18(distributions[:, [18, 38]])
            hier19 = pl_module.encoder_level1_19(distributions[:, [19, 39]])

            cat_hier = torch.cat((hier0[:, :4], hier1[:, :4], hier2[:, :4], hier3[:, :4], hier4[:, :4], hier5[:, :4],
                                  hier6[:, :4], hier7[:, :4], hier8[:, :4], hier9[:, :4], hier10[:, :4], hier11[:, :4],
                                  hier12[:, :4], hier13[:, :4], hier14[:, :4], hier15[:, :4], hier16[:, :4],
                                  hier17[:, :4],
                                  hier18[:, :4], hier19[:, :4],
                                  hier0[:, 4:], hier1[:, 4:], hier2[:, 4:], hier3[:, 4:], hier4[:, 4:], hier5[:, 4:],
                                  hier6[:, 4:], hier7[:, 4:], hier8[:, 4:], hier9[:, 4:], hier10[:, 4:], hier11[:, 4:],
                                  hier12[:, 4:], hier13[:, 4:], hier14[:, 4:], hier15[:, 4:], hier16[:, 4:],
                                  hier17[:, 4:],
                                  hier18[:, 4:], hier19[:, 4:]), axis=1)
            # print('cat_hier', cat_hier.shape, hier1.shape, hier2.shape)
            mu_level1 = cat_hier[:, :pl_module.latent_dim_level1]
            #logvar_level1 = cat_hier[:, pl_module.latent_dim_level1:]

            #mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
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

            recon_img = [orig_pad, level0_recon_pad, level1_recon_pad]

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=True, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])
            pl_module.train()

class ImagePredictionLoggerSingleDecoder(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]

            z_level1 = mu_level1
            z_level0 = mu_level0

            with torch.no_grad():
                z0_connector = pl_module.l0_connector(z_level0)
                z1_connector = pl_module.l1_connector(z_level1)

                pred_level1 = pl_module.decoder(z0_connector.to(pl_module.device)).cpu()
                pred_level0 = pl_module.decoder(z1_connector.to(pl_module.device)).cpu()

            level1_recon = torch.sigmoid(pred_level1).data
            level1_recon_pad = nn.functional.pad(level1_recon, pad=[4, 4, 4, 4], value=0.0)

            level0_recon = torch.sigmoid(pred_level0).data
            level0_recon_pad = nn.functional.pad(level0_recon, pad=[4, 4, 4, 4], value=1.0)

            orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

            recon_img = [orig_pad, level0_recon_pad, level1_recon_pad]

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])
            pl_module.train()

class ImagePredictionLoggerHierarchy(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            # Higher level images

            hier_kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72,
                                                    background_color="black", text_color="white")
            kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72)
            l0_low, l0_high, l1_low, l1_high = get_visualization_latent_border(trainer, pl_module)
            l0_low.cpu()
            l0_high.cpu()
            l1_low.cpu()
            l1_high.cpu()
            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
#                for z_change in np.linspace(l1_low[i].item(), l1_high[i].item(), 12):  # (np.arange(-3, 3, 0.5)):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level1.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_level1(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level1.append(torch.from_numpy(hier_kl_images[i]))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
#                for z_change in np.linspace(l0_low[i].item(), l0_high[i].item(), 12): #(np.arange(-3, 3, 0.5)):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level0.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level0.append(torch.from_numpy(kl_images[i]))

            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 13]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 13]:
                    merged_image.append(img)
                group_counter += group_indices * 13
                level1_counter += 13

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=13, pad_value=1)

            self.wandb_logger.log_image('train_images/hier_image_z_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()

class ImagePredictionLoggerHierarchyForward(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            pl_module.eval()

            distributions = pl_module.encoder(val_imgs)
            mu_level1 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level1 = distributions[:, pl_module.latent_dim_level0:]

            hier0 = pl_module.encoder_level1_0(distributions[:, [0, 20]])
            hier1 = pl_module.encoder_level1_1(distributions[:, [1, 21]])
            hier2 = pl_module.encoder_level1_2(distributions[:, [2, 22]])
            hier3 = pl_module.encoder_level1_3(distributions[:, [3, 23]])
            hier4 = pl_module.encoder_level1_4(distributions[:, [4, 24]])
            hier5 = pl_module.encoder_level1_5(distributions[:, [5, 25]])
            hier6 = pl_module.encoder_level1_6(distributions[:, [6, 26]])
            hier7 = pl_module.encoder_level1_7(distributions[:, [7, 27]])
            hier8 = pl_module.encoder_level1_8(distributions[:, [8, 28]])
            hier9 = pl_module.encoder_level1_9(distributions[:, [9, 29]])
            hier10 = pl_module.encoder_level1_10(distributions[:, [10, 30]])
            hier11 = pl_module.encoder_level1_11(distributions[:, [11, 31]])
            hier12 = pl_module.encoder_level1_12(distributions[:, [12, 32]])
            hier13 = pl_module.encoder_level1_13(distributions[:, [13, 33]])
            hier14 = pl_module.encoder_level1_14(distributions[:, [14, 34]])
            hier15 = pl_module.encoder_level1_15(distributions[:, [15, 35]])
            hier16 = pl_module.encoder_level1_16(distributions[:, [16, 36]])
            hier17 = pl_module.encoder_level1_17(distributions[:, [17, 37]])
            hier18 = pl_module.encoder_level1_18(distributions[:, [18, 38]])
            hier19 = pl_module.encoder_level1_19(distributions[:, [19, 39]])

            cat_hier = torch.cat((hier0[:, :4], hier1[:, :4], hier2[:, :4], hier3[:, :4], hier4[:, :4], hier5[:, :4],
                                  hier6[:, :4], hier7[:, :4], hier8[:, :4], hier9[:, :4], hier10[:, :4], hier11[:, :4],
                                  hier12[:, :4], hier13[:, :4], hier14[:, :4], hier15[:, :4], hier16[:, :4],
                                  hier17[:, :4],
                                  hier18[:, :4], hier19[:, :4],
                                  hier0[:, 4:], hier1[:, 4:], hier2[:, 4:], hier3[:, 4:], hier4[:, 4:], hier5[:, 4:],
                                  hier6[:, 4:], hier7[:, 4:], hier8[:, 4:], hier9[:, 4:], hier10[:, 4:], hier11[:, 4:],
                                  hier12[:, 4:], hier13[:, 4:], hier14[:, 4:], hier15[:, 4:], hier16[:, 4:],
                                  hier17[:, 4:],
                                  hier18[:, 4:], hier19[:, 4:]), axis=1)

            mu_level0 = cat_hier[:, :pl_module.latent_dim_level1]
            logvar_level0 = cat_hier[:, pl_module.latent_dim_level1:]


            # distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            # mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            # logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            # mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            # logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            # Higher level images

            hier_kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72,
                                                    background_color="black", text_color="white")
            kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72)
            l0_low, l0_high, l1_low, l1_high = get_visualization_latent_border(trainer, pl_module)
            l0_low.cpu()
            l0_high.cpu()
            l1_low.cpu()
            l1_high.cpu()
            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
#                for z_change in np.linspace(l1_low[i].item(), l1_high[i].item(), 12):  # (np.arange(-3, 3, 0.5)):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level1.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level1.append(torch.from_numpy(hier_kl_images[i]))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
#                for z_change in np.linspace(l0_low[i].item(), l0_high[i].item(), 12): #(np.arange(-3, 3, 0.5)):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level0.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_level1(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level0.append(torch.from_numpy(kl_images[i]))

            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 13]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 13]:
                    merged_image.append(img)
                group_counter += group_indices * 13
                level1_counter += 13

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=13, pad_value=1)

            self.wandb_logger.log_image('train_images/hier_image_z_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()


class ImagePredictionLoggerHierarchySingleDecoder(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:

            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()
            pl_module.eval()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            # Higher level images

            hier_kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72,
                                                    background_color="black", text_color="white")
            kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72)

            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level1.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        z1_connector = pl_module.l1_connector(z_copy)
                        pred = pl_module.decoder(z1_connector.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level1.append(torch.from_numpy(hier_kl_images[i]))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = z_level0.clone()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        z0_connector = pl_module.l0_connector(z_copy)
                        pred = pl_module.decoder(z0_connector.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level0.append(torch.from_numpy(kl_images[i]))

            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 13]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 13]:
                    merged_image.append(img)
                group_counter += group_indices * 13
                level1_counter += 13

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=13, pad_value=1)

            self.wandb_logger.log_image('train_images/hier_image_z_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()

class ImagePredictionLoggerLatentActivation(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU
            pl_module.eval()
            pl_module.cuda()
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            with torch.no_grad():
                pred0_level0 = pl_module.decoder(torch.zeros_like(z_level0)).cpu()
                pred0_level1 = pl_module.decoder_level1(torch.zeros_like(z_level1)).cpu()

            zero_pred_level0 = torch.sigmoid(pred0_level0).data
            zero_pred_level1 = torch.sigmoid(pred0_level1).data

            # Higher level images

            hier_kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72,
                                                    background_color="black",
                                                    text_color="white", mode="KL")
            kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72, mode="KL")

            data = next(iter(trainer.train_dataloader)).cuda()
            _, mu_l0, _, _, mu_l1, _ = pl_module.forward(data)
            mu_l0.cpu()
            mu_l1.cpu()

            scatter_images = get_scatter_images(pl_module, mu_l0, mu_l1)
            l0_low, l0_high, l1_low, l1_high = torch.min(mu_l0, dim=0).values, torch.max(mu_l0, dim=0).values, torch.min(mu_l1, dim=0).values, torch.max(mu_l1, dim=0).values
            l0_low.cpu()
            l0_high.cpu()
            l1_low.cpu()
            l1_high.cpu()
            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
                for z_change in np.linspace(l1_low[i].item(), l1_high[i].item(), 12): #(np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level1)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_level1(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred-pred0_level1).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))

                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level1.append(torch.from_numpy(hier_kl_images[i]))
                print_images_level1.append(torch.zeros_like(torch.from_numpy(hier_kl_images[i])))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
                for z_change in np.linspace(l0_low[i].item(), l0_high[i].item(), 12):#(np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level0)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred-pred0_level0).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))

                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level0.append(torch.from_numpy(kl_images[i]))
                print_images_level0.append(scatter_images[i])


            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 14]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 14]:
                    merged_image.append(img)
                group_counter += group_indices * 14
                level1_counter += 14

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=14, pad_value=1)
            self.wandb_logger.log_image('train_images/latent_info_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()


# Layers
class ImagePredictionLoggerLatentActivationLayers(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU
            pl_module.eval()
            pl_module.cuda()
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            distributions = pl_module.encoder(val_imgs)
            #, hier_dist_concat
            hier0 = pl_module.encoder_level1_0(distributions[:, [0, 20]])
            hier1 = pl_module.encoder_level1_1(distributions[:, [1, 21]])
            hier2 = pl_module.encoder_level1_2(distributions[:, [2, 22]])
            hier3 = pl_module.encoder_level1_3(distributions[:, [3, 23]])
            hier4 = pl_module.encoder_level1_4(distributions[:, [4, 24]])
            hier5 = pl_module.encoder_level1_5(distributions[:, [5, 25]])
            hier6 = pl_module.encoder_level1_6(distributions[:, [6, 26]])
            hier7 = pl_module.encoder_level1_7(distributions[:, [7, 27]])
            hier8 = pl_module.encoder_level1_8(distributions[:, [8, 28]])
            hier9 = pl_module.encoder_level1_9(distributions[:, [9, 29]])
            hier10 = pl_module.encoder_level1_10(distributions[:, [10, 30]])
            hier11 = pl_module.encoder_level1_11(distributions[:, [11, 31]])
            hier12 = pl_module.encoder_level1_12(distributions[:, [12, 32]])
            hier13 = pl_module.encoder_level1_13(distributions[:, [13, 33]])
            hier14 = pl_module.encoder_level1_14(distributions[:, [14, 34]])
            hier15 = pl_module.encoder_level1_15(distributions[:, [15, 35]])
            hier16 = pl_module.encoder_level1_16(distributions[:, [16, 36]])
            hier17 = pl_module.encoder_level1_17(distributions[:, [17, 37]])
            hier18 = pl_module.encoder_level1_18(distributions[:, [18, 38]])
            hier19 = pl_module.encoder_level1_19(distributions[:, [19, 39]])

            cat_hier = torch.cat((hier0[:, :4], hier1[:, :4], hier2[:, :4], hier3[:, :4], hier4[:, :4], hier5[:, :4],
                                  hier6[:, :4], hier7[:, :4], hier8[:, :4], hier9[:, :4], hier10[:, :4], hier11[:, :4],
                                  hier12[:, :4], hier13[:, :4], hier14[:, :4], hier15[:, :4], hier16[:, :4],
                                  hier17[:, :4],
                                  hier18[:, :4], hier19[:, :4],
                                  hier0[:, 4:], hier1[:, 4:], hier2[:, 4:], hier3[:, 4:], hier4[:, 4:], hier5[:, 4:],
                                  hier6[:, 4:], hier7[:, 4:], hier8[:, 4:], hier9[:, 4:], hier10[:, 4:], hier11[:, 4:],
                                  hier12[:, 4:], hier13[:, 4:], hier14[:, 4:], hier15[:, 4:], hier16[:, 4:],
                                  hier17[:, 4:],
                                  hier18[:, 4:], hier19[:, 4:]), axis=1)
            # print('cat_hier', cat_hier.shape, hier1.shape, hier2.shape)
            mu_level1 = cat_hier[:, :pl_module.latent_dim_level1]
            logvar_level1 = cat_hier[:, pl_module.latent_dim_level1:]



            # mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            # logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            with torch.no_grad():
                pred0_level0 = pl_module.decoder(torch.zeros_like(z_level0)).cpu()
                pred0_level1 = pl_module.decoder_level1(torch.zeros_like(z_level1)).cpu()

            zero_pred_level0 = torch.sigmoid(pred0_level0).data
            zero_pred_level1 = torch.sigmoid(pred0_level1).data

            # Higher level images

            hier_kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72,
                                                    background_color="black",
                                                    text_color="white", mode="KL")
            kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72, mode="KL")

            data = next(iter(trainer.train_dataloader)).cuda()
            _, mu_l0, _, _, mu_l1, _ = pl_module.forward(data)
            mu_l0.cpu()
            mu_l1.cpu()

            scatter_images = get_scatter_images_layer(pl_module, mu_l0, mu_l1)
            l0_low, l0_high, l1_low, l1_high = torch.min(mu_l0, dim=0).values, torch.max(mu_l0, dim=0).values, torch.min(mu_l1, dim=0).values, torch.max(mu_l1, dim=0).values
            l0_low.cpu()
            l0_high.cpu()
            l1_low.cpu()
            l1_high.cpu()
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
                for z_change in np.linspace(l0_low[i].item(), l0_high[i].item(), 12): #(np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level0)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred-pred0_level0).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred)) #normalize
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level0.append(torch.from_numpy(hier_kl_images[i]))
                print_images_level0.append(torch.zeros_like(torch.from_numpy(hier_kl_images[i])))

            # Lower level images
            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
                for z_change in np.linspace(l1_low[i].item(), l1_high[i].item(), 12):#(np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level1)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_level1(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred-pred0_level1).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))  # normalize
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level1.append(torch.from_numpy(kl_images[i]))
                print_images_level1.append(scatter_images[i])


            merged_image = []
            group_counter = 0
            level0_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level0[level0_counter:level0_counter + 14]:
                    merged_image.append(img)
                for img in print_images_level1[group_counter:group_counter + group_indices * 14]:
                    merged_image.append(img)
                group_counter += group_indices * 14
                level0_counter += 14

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=14, pad_value=1)
            self.wandb_logger.log_image('train_images/latent_info_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()


class ImagePredictionLoggerLatentActivationSingleDecoder(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU
            pl_module.eval()
            pl_module.cuda()
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            distributions, hier_dist_concat = pl_module.encoder(val_imgs)
            mu_level1 = hier_dist_concat[:, :pl_module.latent_dim_level1]
            logvar_level1 = hier_dist_concat[:, pl_module.latent_dim_level1:]
            mu_level0 = distributions[:, :pl_module.latent_dim_level0]
            logvar_level0 = distributions[:, pl_module.latent_dim_level0:]

            _, dim_wise_kld, _ = kl_divergence(mu_level0, logvar_level0)

            _, hierarchical_kl, _ = kl_divergence(mu_level1, logvar_level1)

            z_level1 = mu_level1
            z_level0 = mu_level0

            print_images_level1 = []
            print_images_level0 = []

            with torch.no_grad():
                z0_connector = pl_module.l0_connector(torch.zeros_like(z_level0))
                z1_connector = pl_module.l1_connector(torch.zeros_like(z_level1))
                pred0_level0 = pl_module.decoder(z0_connector).cpu()
                pred0_level1 = pl_module.decoder(z1_connector).cpu()

            zero_pred_level0 = torch.sigmoid(pred0_level0).data
            zero_pred_level1 = torch.sigmoid(pred0_level1).data

            # Higher level images

            hier_kl_images = create_kl_value_images(hierarchical_kl, mu_level1, logvar_level1, 72,
                                                    background_color="black",
                                                    text_color="white", mode="KL")
            kl_images = create_kl_value_images(dim_wise_kld, mu_level0, logvar_level0, 72, mode="KL")

            z_level1_size = z_level1.size(1)
            for i in np.arange(0, z_level1_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level1)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        z1_connector = pl_module.l1_connector(z_copy.to(pl_module.device))
                        pred = pl_module.decoder(z1_connector).cpu()
                    sigm_pred = torch.sigmoid(pred).data - zero_pred_level1
                    print_images_level1.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=0.0))
                print_images_level1.append(torch.from_numpy(hier_kl_images[i]))

            # Lower level images
            z_level0_size = z_level0.size(1)
            for i in np.arange(0, z_level0_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = torch.zeros_like(z_level0)
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        z0_connector = pl_module.l0_connector(z_copy.to(pl_module.device))
                        pred = pl_module.decoder(z0_connector).cpu()
                    sigm_pred = torch.sigmoid(pred).data - zero_pred_level0
                    print_images_level0.append(nn.functional.pad(sigm_pred, pad=[4, 4, 4, 4], value=1.0))
                print_images_level0.append(torch.from_numpy(kl_images[i]))

            merged_image = []
            group_counter = 0
            level1_counter = 0
            for idx, group_indices in enumerate(pl_module.hier_groups):
                for img in print_images_level1[level1_counter:level1_counter + 13]:
                    merged_image.append(img)
                for img in print_images_level0[group_counter:group_counter + group_indices * 13]:
                    merged_image.append(img)
                group_counter += group_indices * 13
                level1_counter += 13

            merged_image_cat = torch.cat(merged_image, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=True, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/latent_info_' + str(self.sample),
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()
class ImagePredictionLoggerMergedLatentActivation(Callback):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU
            pl_module.eval()
            pl_module.cuda()
            zero_image = torch.zeros((1, 1, 64, 64)).float().cuda()
            level0 = pl_module.latent_dim_level0
            level1 = pl_module.latent_dim_level1
            print_images = []

            # 0 latent features
            level0_zero_img = torch.zeros((1, level0)).cuda()
            level1_zero_img = torch.zeros((1, level1)).cuda()

            with torch.no_grad():
                pred0_level0 = pl_module.decoder(level0_zero_img)
                pred0_level1 = pl_module.decoder_level1(level1_zero_img)

            # "no info" predictions
            zero_pred_level0 = torch.sigmoid(pred0_level0).data
            zero_pred_level1 = torch.sigmoid(pred0_level1).data
            # return zero_pred_level0, zero_pred_level1

            # which latent feature vales to check
            check_levels = [-1, 1]
            # hierarhy indices
            hier_indices = pl_module.encoder.mu_indices
            recon_loss_between_layers_list = []
            for i in np.arange(0, level1, 1):
                for check in check_levels:
                    z_img = level1_zero_img.clone()
                    z_img[0, i] = check

                    # hier reconstr
                    with torch.no_grad():
                        reconst_z1 = pl_module.decoder_level1(z_img).cpu()
                    # to cpu
                    reconst_z1_sigm = torch.sigmoid(reconst_z1).data - zero_pred_level1.cpu()
                    reconst_z1_sigm = (reconst_z1_sigm - torch.min(reconst_z1_sigm)).div(torch.max(reconst_z1_sigm) - torch.min(reconst_z1_sigm))  # normalize
                    print_images.append(nn.functional.pad(reconst_z1_sigm, pad=[4, 4, 4, 4], value=0.0))

                    # l0 reconstr
                    l0_indices = hier_indices[i]  # print(z_img)
                    z0_img = copy.deepcopy(level0_zero_img)
                    z0_img[0, l0_indices] = check

                    # to cpu
                    reconst_z0 = pl_module.decoder(z0_img).cpu()
                    reconst_z0_sigm = torch.sigmoid(reconst_z0).data - zero_pred_level0.cpu()
                    reconst_z0_sigm = (reconst_z0_sigm - torch.min(reconst_z0_sigm)).div(torch.max(reconst_z0_sigm) - torch.min(reconst_z0_sigm))  # normalize

                    recon_loss_between_layers = F.mse_loss(reconst_z1_sigm.cpu(), reconst_z0_sigm.cpu(), reduction='sum')

                    print_images.append(nn.functional.pad(reconst_z0_sigm, pad=[4, 4, 4, 4], value=1.0))
                    print_images.append(torch.from_numpy(create_text_image(str(i) + ": "+str(recon_loss_between_layers.item()))))
            merged_image_cat = torch.cat(print_images, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=False, scale_each=True, nrow=3, pad_value=1)
            self.wandb_logger.log_image('train_images/latent_info_merged',
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()



# TO DO: currently for 2 decoders
class ImagePredictionLoggerMergedLatentActivationSingleDecoder(Callback):
    def __init__(self, wandb_logger=None):
        super().__init__()
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            # Bring the tensors to CPU
            pl_module.eval()
            pl_module.cuda()
            zero_image = torch.zeros((1, 1, 64, 64)).float().cuda()
            level0 = pl_module.latent_dim_level0
            level1 = pl_module.latent_dim_level1
            print_images = []

            # 0 latent features
            level0_zero_img = torch.zeros((1, level0)).cuda()
            level1_zero_img = torch.zeros((1, level1)).cuda()

            with torch.no_grad():
                pred0_level0 = pl_module.decoder(level0_zero_img)
                pred0_level1 = pl_module.decoder_level1(level1_zero_img)

            # "no info" predictions
            zero_pred_level0 = torch.sigmoid(pred0_level0).data
            zero_pred_level1 = torch.sigmoid(pred0_level1).data
            # return zero_pred_level0, zero_pred_level1

            # which latent feature vales to check
            check_levels = [-1, 1]
            # hierarhy indices
            hier_indices = pl_module.encoder.mu_indices
            recon_loss_between_layers_list = []
            for i in np.arange(0, level1, 1):
                for check in check_levels:
                    z_img = level1_zero_img.clone()
                    z_img[0, i] = check

                    # hier reconstr
                    with torch.no_grad():
                        reconst_z1 = pl_module.decoder_level1(z_img).cpu()
                    # to cpu
                    reconst_z1_sigm = torch.sigmoid(reconst_z1).data - zero_pred_level1.cpu()
                    print_images.append(nn.functional.pad(reconst_z1_sigm, pad=[4, 4, 4, 4], value=0.0))

                    # l0 reconstr
                    l0_indices = hier_indices[i]  # print(z_img)
                    z0_img = copy.deepcopy(level0_zero_img)
                    z0_img[0, l0_indices] = check

                    # to cpu
                    reconst_z0 = pl_module.decoder(z0_img).cpu()
                    reconst_z0_sigm = torch.sigmoid(reconst_z0).data - zero_pred_level0.cpu()
                    recon_loss_between_layers = F.mse_loss(reconst_z1_sigm.cpu(), reconst_z0_sigm.cpu(), reduction='sum')

                    print_images.append(nn.functional.pad(reconst_z0_sigm, pad=[4, 4, 4, 4], value=1.0))
                    print_images.append(torch.from_numpy(create_text_image(str(i) + ": "+str(recon_loss_between_layers.item()))))
            merged_image_cat = torch.cat(print_images, dim=0).cpu()
            merged_image_grid = make_grid(merged_image_cat, normalize=True, scale_each=True, nrow=3, pad_value=1)
            self.wandb_logger.log_image('train_images/latent_info_merged',
                                        [(merged_image_grid.permute(1, 2, 0).numpy())])
            pl_module.train()

def create_kl_value_images(kl_list, mu_list, logvar_list, img_size=72, text_color="black",
                           background_color="white", mode="ALL"):  # input size 64+2*4
    images_list = []
    font = ImageFont.load_default()
    w, h = font.getsize(6)

    for idx, val in enumerate(kl_list):
        img = Image.new('RGB', (img_size, img_size), background_color)
        str_number0 = str(idx)
        str_number1 = "KL" + ":{:.5f}".format(val)
        str_number2 = "mu" + ":{:.5f}".format(mu_list[0][idx].item())
        str_number3 = "sd" + ":{:.5f}".format(logvar_list[0][idx].item())

        draw = ImageDraw.Draw(img)
        draw.text(((img_size - w) / 15, (img_size - h) * 0.05), str_number0, font=font, fill=text_color)
        if mode == "ALL":
            draw.text(((img_size - w) / 15, (img_size - h) * 0.25), str_number1, font=font, fill=text_color)
            draw.text(((img_size - w) / 15, (img_size - h) * 0.5), str_number2, font=font, fill=text_color)
            draw.text(((img_size - w) / 15, (img_size - h) * 0.75), str_number3, font=font, fill=text_color)

        numpy_img = np.asarray(img) / 255
        numpy_img = np.moveaxis(numpy_img, 2, 0)
        numpy_img = np.expand_dims(numpy_img, axis=0)
        images_list.append(numpy_img)

    return images_list


def create_text_image(text, img_size=72, text_color="black",
                           background_color="white"):  # input size 64+2*4

    font = ImageFont.load_default()
    w, h = font.getsize(6)

    img = Image.new('RGB', (img_size, img_size), background_color)
    draw = ImageDraw.Draw(img)
    draw.text(((img_size - w) / 15, (img_size - h) * 0.05), text, font=font, fill=text_color)


    numpy_img = np.asarray(img) / 255
    numpy_img = np.moveaxis(numpy_img, 2, 0)
    numpy_img = np.expand_dims(numpy_img, axis=0)

    return numpy_img

def get_visualization_latent_border(trainer, pl_module):
    data = next(iter(trainer.train_dataloader)).cuda()
    _, mu_l0, _, _, mu_l1, _ = pl_module.forward(data)

    return torch.min(mu_l0, dim=0).values, torch.max(mu_l0, dim=0).values, torch.min(mu_l1, dim=0).values, torch.max(mu_l1, dim=0).values



from matplotlib.ticker import FormatStrFormatter

def get_scatter_images(pl_module, mu, mu_hier):
    images_list = []
    #  l0_indices = pl_module.encoder.mu_indices[idx]
    transforms2 = transforms.Compose([transforms.Resize((72, 72)), transforms.ToTensor()])
    for lat1_idx, l0_indices in enumerate(pl_module.encoder.mu_indices):

        for l0_idx in l0_indices:
            corr_coefficient = torch.corrcoef(torch.stack((mu.T[l0_idx], mu_hier.T[lat1_idx]), axis=0))[0, 1].item()

            fig = Figure()
            canvas = FigureCanvas(fig)
            fig.set_size_inches(2, 2)
            ax = fig.gca()

            ax.scatter(mu_hier.T[lat1_idx].cpu().detach().numpy(),mu.T[l0_idx].cpu().detach().numpy(),c='green',
                       alpha=0.5)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_title(":{:.5f}".format(corr_coefficient), size=18, pad=-15)

            # fig.tight_layout(pad=5)

            # To remove the huge white borders

            fig.canvas.draw()

            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            im = Image.fromarray(image_from_plot)
            x = transforms2(im)
            x_exp = np.expand_dims(x.numpy(), axis=0)
            images_list.append(torch.from_numpy(x_exp))
    return images_list


def get_scatter_images_layer(pl_module, mu, mu_hier):
    images_list = []
    #  l0_indices = pl_module.encoder.mu_indices[idx]
    transforms2 = transforms.Compose([transforms.Resize((72, 72)), transforms.ToTensor()])
    for lat1_idx in torch.arange(0,20,1):

        for l0_idx in torch.arange(0,4,1):
            corr_coefficient = torch.corrcoef(torch.stack((mu.T[lat1_idx], mu_hier.T[lat1_idx*4+l0_idx]), axis=0))[0, 1].item()

            fig = Figure()
            canvas = FigureCanvas(fig)
            fig.set_size_inches(2, 2)
            ax = fig.gca()

            ax.scatter(mu_hier.T[lat1_idx*4+l0_idx].cpu().detach().numpy(),mu.T[lat1_idx].cpu().detach().numpy(), c='green',
                       alpha=0.5)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_title(":{:.5f}".format(corr_coefficient), size=18, pad=-15)

            # fig.tight_layout(pad=5)

            # To remove the huge white borders

            fig.canvas.draw()

            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            im = Image.fromarray(image_from_plot)
            x = transforms2(im)
            x_exp = np.expand_dims(x.numpy(), axis=0)
            images_list.append(torch.from_numpy(x_exp))
    return images_list

def get_first_images_mu(trainer, pl_module, ds, logger):
    images_indices = np.arange(0, 50, 1)

    l0_cols = ['l0_' + str(k) for k in np.arange(0, pl_module.latent_dim, 1)]

    columns = ["image_id"]
    columns.extend(l0_cols)
    full_data = []

    for idx in images_indices:
        data_row = []
        data = ds.__getitem__(idx).reshape((-1, 3, 64, 64)).float().cuda()
        _, mu_l0, _ = pl_module.forward(data)
        data_row.extend([idx])
        data_row.extend(mu_l0.flatten().detach().cpu().numpy())  # , mu_l1])
        full_data.append(data_row)
    logger.log_table(key="validation_samples", columns=columns, data=full_data)


class get_first_images_mu_logger(Callback):
    def __init__(self, ds=None, logger=None):
        super().__init__()
        self.ds = ds
        self.logger = logger
        self.epoch_count = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        self.epoch_count += 1
        # print('test epoch count', self.epoch_count, 'div 10', self.epoch_count%10==0)

        if self.epoch_count % 50 == 1:
            pl_module.eval()
            get_first_images_mu(trainer, pl_module, self.ds, self.logger)
            pl_module.train()

class ImagePredictionLoggerThreeLevel(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()

            first_latent, second_latent, third_latent = pl_module.encoder(val_imgs)

            mu_first = first_latent[:, :pl_module.latent_dims[0]]
            logvar_first = first_latent[:, pl_module.latent_dims[0]:]
            sd_first = logvar_first.div(2).exp()
            _, dim_wise_kld_first, _ = kl_divergence(mu_first, logvar_first)

            mu_second = second_latent[:, :pl_module.latent_dims[1]]
            logvar_second = second_latent[:, pl_module.latent_dims[1]:]
            sd_second = logvar_second.div(2).exp()
            _, dim_wise_kld_second, _ = kl_divergence(mu_second, logvar_second)

            mu_third = third_latent[:, :pl_module.latent_dims[2]]
            logvar_third = third_latent[:, pl_module.latent_dims[2]:]
            sd_third = logvar_third.div(2).exp()
            _, dim_wise_kld_third, _ = kl_divergence(mu_third, logvar_third)

            with torch.no_grad():
                pred_first = pl_module.decoder_first_latent(mu_first.to(pl_module.device)).cpu()
                pred_second = pl_module.decoder_second_latent(mu_second.to(pl_module.device)).cpu()
                pred_third = pl_module.decoder_third_latent(mu_third.to(pl_module.device)).cpu()

            first_recon = torch.sigmoid(pred_first).data
            first_recon = (first_recon - torch.min(first_recon)).div(torch.max(first_recon) - torch.min(first_recon))
            first_recon_pad = nn.functional.pad(first_recon, pad=[4, 4, 4, 4], value=0.1)

            second_recon = torch.sigmoid(pred_second).data
            second_recon = (second_recon - torch.min(second_recon)).div(torch.max(second_recon) - torch.min(second_recon))
            second_recon_pad = nn.functional.pad(second_recon, pad=[4, 4, 4, 4], value=0.2)

            third_recon = torch.sigmoid(pred_third).data
            third_recon = (third_recon - torch.min(third_recon)).div(torch.max(third_recon) - torch.min(third_recon))
            third_recon_pad = nn.functional.pad(third_recon, pad=[4, 4, 4, 4], value=0.3)

            orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

            recon_img = [orig_pad, first_recon_pad, second_recon_pad, third_recon_pad]

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])

            # log z changes first latents
            first_latent_kl_images = create_kl_value_images(dim_wise_kld_first, mu_first, sd_first,
                                                            72, background_color="black", text_color="white")

            print_images = []
            z_size = mu_first.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_first.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_first_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(first_latent_kl_images[i]))

            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/first_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])

            # log z changes second latents
            second_latent_kl_images = create_kl_value_images(dim_wise_kld_second, mu_second, sd_second,
                                                            72, background_color="black", text_color="white")

            print_images = []
            z_size = mu_second.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_second.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_second_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(second_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/second_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])
            # log z changes third latents
            third_latent_kl_images = create_kl_value_images(dim_wise_kld_third, mu_third, sd_third,
                                                             72, background_color="black", text_color="white")
            print_images = []
            z_size = mu_third.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_third.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_third_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(third_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/third_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])
            pl_module.train()



class ImagePredictionLoggerFiveLevel(Callback):
    def __init__(self, sample=0, ds=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.epoch_count = 0
        self.wandb_logger = wandb_logger

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1

        if self.epoch_count % 5 == 1:
            val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

            pl_module.eval()

            first_latent, second_latent, third_latent, forth_latent, fifth_latent = pl_module.encoder(val_imgs)

            mu_first = first_latent[:, :pl_module.latent_dims[0]]
            logvar_first = first_latent[:, pl_module.latent_dims[0]:]
            sd_first = logvar_first.div(2).exp()
            _, dim_wise_kld_first, _ = kl_divergence(mu_first, logvar_first)

            mu_second = second_latent[:, :pl_module.latent_dims[1]]
            logvar_second = second_latent[:, pl_module.latent_dims[1]:]
            sd_second = logvar_second.div(2).exp()
            _, dim_wise_kld_second, _ = kl_divergence(mu_second, logvar_second)

            mu_third = third_latent[:, :pl_module.latent_dims[2]]
            logvar_third = third_latent[:, pl_module.latent_dims[2]:]
            sd_third = logvar_third.div(2).exp()
            _, dim_wise_kld_third, _ = kl_divergence(mu_third, logvar_third)

            mu_forth = forth_latent[:, :pl_module.latent_dims[3]]
            logvar_forth = forth_latent[:, pl_module.latent_dims[3]:]
            sd_forth = logvar_forth.div(2).exp()
            _, dim_wise_kld_forth, _ = kl_divergence(mu_forth, logvar_forth)

            mu_fifth = fifth_latent[:, :pl_module.latent_dims[4]]
            logvar_fifth = fifth_latent[:, pl_module.latent_dims[4]:]
            sd_fifth = logvar_fifth.div(2).exp()
            _, dim_wise_kld_fifth, _ = kl_divergence(mu_fifth, logvar_fifth)

            with torch.no_grad():
                    pred_first = pl_module.decoder_first_latent(mu_first.to(pl_module.device)).cpu()
                    pred_second = pl_module.decoder_second_latent(mu_second.to(pl_module.device)).cpu()
                    pred_third = pl_module.decoder_third_latent(mu_third.to(pl_module.device)).cpu()
                    pred_forth = pl_module.decoder_forth_latent(mu_forth.to(pl_module.device)).cpu()
                    pred_fifth = pl_module.decoder_fifth_latent(mu_fifth.to(pl_module.device)).cpu()

            first_recon = torch.sigmoid(pred_first).data
            first_recon = (first_recon - torch.min(first_recon)).div(torch.max(first_recon) - torch.min(first_recon))
            first_recon_pad = nn.functional.pad(first_recon, pad=[4, 4, 4, 4], value=0.1)

            second_recon = torch.sigmoid(pred_second).data
            second_recon = (second_recon - torch.min(second_recon)).div(torch.max(second_recon) - torch.min(second_recon))
            second_recon_pad = nn.functional.pad(second_recon, pad=[4, 4, 4, 4], value=0.2)

            third_recon = torch.sigmoid(pred_third).data
            third_recon = (third_recon - torch.min(third_recon)).div(torch.max(third_recon) - torch.min(third_recon))
            third_recon_pad = nn.functional.pad(third_recon, pad=[4, 4, 4, 4], value=0.3)

            forth_recon = torch.sigmoid(pred_forth).data
            forth_recon = (forth_recon - torch.min(forth_recon)).div(torch.max(forth_recon) - torch.min(forth_recon))
            forth_recon_pad = nn.functional.pad(forth_recon, pad=[4, 4, 4, 4], value=0.3)


            fifth_recon = torch.sigmoid(pred_fifth).data
            fifth_recon = (fifth_recon - torch.min(fifth_recon)).div(torch.max(fifth_recon) - torch.min(fifth_recon))
            fifth_recon_pad = nn.functional.pad(fifth_recon, pad=[4, 4, 4, 4], value=0.3)

            orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

            recon_img = [orig_pad, first_recon_pad, second_recon_pad, third_recon_pad, forth_recon_pad, fifth_recon_pad]

            print_orig_recon = torch.cat(recon_img, dim=0).cpu()
            recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
            recon_image = recon_image.permute(1, 2, 0)
            # Log the images as wandb Image
            self.wandb_logger.log_image('train_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])

            # log z changes first latents
            first_latent_kl_images = create_kl_value_images(dim_wise_kld_first, mu_first, sd_first,
                                                            72, background_color="black", text_color="white")

            print_images = []
            z_size = mu_first.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_first.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_first_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(first_latent_kl_images[i]))

            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/first_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])

            # log z changes second latents
            second_latent_kl_images = create_kl_value_images(dim_wise_kld_second, mu_second, sd_second,
                                                            72, background_color="black", text_color="white")
            print_images = []
            z_size = mu_second.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_second.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_second_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(second_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/second_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])
            # log z changes third latents
            third_latent_kl_images = create_kl_value_images(dim_wise_kld_third, mu_third, sd_third,
                                                             72, background_color="black", text_color="white")
            print_images = []
            z_size = mu_third.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_third.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_third_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(third_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/third_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])


            # log z changes forth latents
            forth_latent_kl_images = create_kl_value_images(dim_wise_kld_forth, mu_forth, sd_forth,
                                                            72, background_color="black", text_color="white")
            print_images = []
            z_size = mu_forth.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_forth.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_forth_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(forth_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/forth_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])


            # log z changes fifth latents
            fifth_latent_kl_images = create_kl_value_images(dim_wise_kld_fifth, mu_fifth, sd_fifth,
                                                            72, background_color="black", text_color="white")
            print_images = []
            z_size = mu_fifth.size(1)
            for i in np.arange(0, z_size, 1):
                for z_change in (np.arange(-3, 3, 0.5)):
                    z_copy = mu_fifth.clone().detach()
                    z_copy[0, i] = z_change
                    with torch.no_grad():
                        pred = pl_module.decoder_fifth_latent(z_copy.to(pl_module.device)).cpu()
                    sigm_pred = torch.sigmoid(pred).data
                    sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                    sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                    print_images.append(sigm_pred)
                print_images.append(torch.from_numpy(fifth_latent_kl_images[i]))
            all_images = torch.cat(print_images, dim=0).cpu()
            images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=13, pad_value=1)
            self.wandb_logger.log_image('train_images/fifth_latent_image_z_' + str(self.sample),
                                        [(images_grid.permute(1, 2, 0).numpy())])

            pl_module.train()



def create_corr_images(input_latent_corr):
  #gen_params = ['azimuths', 'floor_colors','wall_colors','eye1_colors','eye2_colors','eye3_colors','eye4_colors','box_colors','box_sizes']
  latents_corr = 74-input_latent_corr*74
  images_list = []
  font = ImageFont.load_default()
  #print('latents_corr_shape_1', latents_corr.shape[1])
  for i in (np.arange(0, latents_corr.shape[1], 1)):
    current_latent_reg = latents_corr[:,i]
    img = Image.new('RGB', (72, 72), 'white')
    draw = ImageDraw.Draw(img)
    draw.line((4, current_latent_reg[0], 4, 74), fill=(51, 102, 204), width=8)
    draw.line((12, current_latent_reg[1], 12, 74), fill=(153, 204, 255), width=8)
    draw.line((20, current_latent_reg[2], 20, 74), fill=(153, 153, 51), width=8)
    draw.line((28, current_latent_reg[3], 28, 74), fill=(102, 102, 153), width=8)
    draw.line((36, current_latent_reg[4], 36, 74), fill=(204, 153, 51), width=8)
    draw.line((44, current_latent_reg[5], 44, 74), fill=(0, 102, 102), width=8)
    draw.line((52, current_latent_reg[6], 52, 74), fill=(51, 153, 255), width=8)
    draw.line((60, current_latent_reg[7], 60, 74), fill=(153, 51, 0), width=8)
    draw.line((68, current_latent_reg[8], 68, 74), fill=(204, 204, 153), width=8)

    numpy_img = np.asarray(img) / 255
    numpy_img = np.moveaxis(numpy_img, 2, 0)
    numpy_img = np.expand_dims(numpy_img, axis=0)
    images_list.append(numpy_img)

  return images_list


class TestImagePredictionLoggerThreeLevel(Callback):
    def __init__(self, sample=0, ds=None, ds_t=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.wandb_logger = wandb_logger
        self.ds_t = ds_t


    def on_test_epoch_end(self, trainer, pl_module):
        l1, l2, l3 = get_all_latent_correlation(pl_module.gen_params, self.ds_t, pl_module)
        self.wandb_logger.log_table(key="test/first_latents_corr", columns=pl_module.gen_params, data=l1.T)
        self.wandb_logger.log_table(key="test/second_latents_corr", columns=pl_module.gen_params, data=l2.T)
        self.wandb_logger.log_table(key="test/third_latents_corr", columns=pl_module.gen_params, data=l3.T)

        val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

        pl_module.eval()

        first_latent, second_latent, third_latent = pl_module.encoder(val_imgs)

        mu_first = first_latent[:, :pl_module.latent_dims[0]]
        logvar_first = first_latent[:, pl_module.latent_dims[0]:]
        sd_first = logvar_first.div(2).exp()
        _, dim_wise_kld_first, _ = kl_divergence(mu_first, logvar_first)

        mu_second = second_latent[:, :pl_module.latent_dims[1]]
        logvar_second = second_latent[:, pl_module.latent_dims[1]:]
        sd_second = logvar_second.div(2).exp()
        _, dim_wise_kld_second, _ = kl_divergence(mu_second, logvar_second)

        mu_third = third_latent[:, :pl_module.latent_dims[2]]
        logvar_third = third_latent[:, pl_module.latent_dims[2]:]
        sd_third = logvar_third.div(2).exp()
        _, dim_wise_kld_third, _ = kl_divergence(mu_third, logvar_third)

        with torch.no_grad():
            pred_first = pl_module.decoder_first_latent(mu_first.to(pl_module.device)).cpu()
            pred_second = pl_module.decoder_second_latent(mu_second.to(pl_module.device)).cpu()
            pred_third = pl_module.decoder_third_latent(mu_third.to(pl_module.device)).cpu()

        first_recon = torch.sigmoid(pred_first).data
        first_recon = (first_recon - torch.min(first_recon)).div(torch.max(first_recon) - torch.min(first_recon))
        first_recon_pad = nn.functional.pad(first_recon, pad=[4, 4, 4, 4], value=0.1)

        second_recon = torch.sigmoid(pred_second).data
        second_recon = (second_recon - torch.min(second_recon)).div(torch.max(second_recon) - torch.min(second_recon))
        second_recon_pad = nn.functional.pad(second_recon, pad=[4, 4, 4, 4], value=0.2)

        third_recon = torch.sigmoid(pred_third).data
        third_recon = (third_recon - torch.min(third_recon)).div(torch.max(third_recon) - torch.min(third_recon))
        third_recon_pad = nn.functional.pad(third_recon, pad=[4, 4, 4, 4], value=0.3)

        orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

        recon_img = [orig_pad, first_recon_pad, second_recon_pad, third_recon_pad]

        print_orig_recon = torch.cat(recon_img, dim=0).cpu()
        recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
        recon_image = recon_image.permute(1, 2, 0)
        # Log the images as wandb Image
        self.wandb_logger.log_image('test_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])

        # log z changes first latents
        first_latent_kl_images = create_kl_value_images(dim_wise_kld_first, mu_first, sd_first,
                                                        72, background_color="black", text_color="white")
        first_latent_corr_images = create_corr_images(l1)
        # print('L1 shape: ', l1.shape)
        # print('first_latent_corr_images: ', str(len(first_latent_corr_images)))

        print_images = []
        z_size = mu_first.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_first.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_first_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(first_latent_kl_images[i]))

            print_images.append(torch.from_numpy(first_latent_corr_images[i]))

        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/first_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])

        # log z changes second latents
        second_latent_kl_images = create_kl_value_images(dim_wise_kld_second, mu_second, sd_second,
                                                        72, background_color="black", text_color="white")
        second_latent_corr_images = create_corr_images(l2)

        print_images = []
        z_size = mu_second.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_second.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_second_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(second_latent_kl_images[i]))
            print_images.append(torch.from_numpy(second_latent_corr_images[i]))

        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/second_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])
        # log z changes third latents
        third_latent_kl_images = create_kl_value_images(dim_wise_kld_third, mu_third, sd_third,
                                                          72, background_color="black", text_color="white")
        third_latent_corr_images = create_corr_images(l3)
        print_images = []
        z_size = mu_third.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_third.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_third_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(third_latent_kl_images[i]))
            print_images.append(torch.from_numpy(third_latent_corr_images[i]))
        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/third_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])
        pl_module.train()




class TestImagePredictionLoggerFiveLevel(Callback):
    def __init__(self, sample=0, ds=None, ds_t=None, wandb_logger=None):
        super().__init__()
        self.sample = sample
        self.ds = ds
        self.wandb_logger = wandb_logger
        self.ds_t = ds_t


    def on_test_epoch_end(self, trainer, pl_module):
        l1, l2, l3, l4, l5 = get_all_latent_correlation5(pl_module.gen_params, self.ds_t, pl_module)
        self.wandb_logger.log_table(key="test/first_latents_corr", columns=pl_module.gen_params, data=l1.T)
        self.wandb_logger.log_table(key="test/second_latents_corr", columns=pl_module.gen_params, data=l2.T)
        self.wandb_logger.log_table(key="test/third_latents_corr", columns=pl_module.gen_params, data=l3.T)
        self.wandb_logger.log_table(key="test/forth_latents_corr", columns=pl_module.gen_params, data=l4.T)
        self.wandb_logger.log_table(key="test/fifth_latents_corr", columns=pl_module.gen_params, data=l5.T)

        val_imgs = self.ds.__getitem__(self.sample).reshape((1, -1, 64, 64)).float().cuda()

        pl_module.eval()

        first_latent, second_latent, third_latent, forth_latent, fifth_latent = pl_module.encoder(val_imgs)

        mu_first = first_latent[:, :pl_module.latent_dims[0]]
        logvar_first = first_latent[:, pl_module.latent_dims[0]:]
        sd_first = logvar_first.div(2).exp()
        _, dim_wise_kld_first, _ = kl_divergence(mu_first, logvar_first)

        mu_second = second_latent[:, :pl_module.latent_dims[1]]
        logvar_second = second_latent[:, pl_module.latent_dims[1]:]
        sd_second = logvar_second.div(2).exp()
        _, dim_wise_kld_second, _ = kl_divergence(mu_second, logvar_second)

        mu_third = third_latent[:, :pl_module.latent_dims[2]]
        logvar_third = third_latent[:, pl_module.latent_dims[2]:]
        sd_third = logvar_third.div(2).exp()
        _, dim_wise_kld_third, _ = kl_divergence(mu_third, logvar_third)

        mu_forth = forth_latent[:, :pl_module.latent_dims[3]]
        logvar_forth = forth_latent[:, pl_module.latent_dims[3]:]
        sd_forth = logvar_forth.div(2).exp()
        _, dim_wise_kld_forth, _ = kl_divergence(mu_forth, logvar_forth)

        mu_fifth = fifth_latent[:, :pl_module.latent_dims[4]]
        logvar_fifth = fifth_latent[:, pl_module.latent_dims[4]:]
        sd_fifth = logvar_fifth.div(2).exp()
        _, dim_wise_kld_fifth, _ = kl_divergence(mu_fifth, logvar_fifth)


        with torch.no_grad():
            pred_first = pl_module.decoder_first_latent(mu_first.to(pl_module.device)).cpu()
            pred_second = pl_module.decoder_second_latent(mu_second.to(pl_module.device)).cpu()
            pred_third = pl_module.decoder_third_latent(mu_third.to(pl_module.device)).cpu()
            pred_forth = pl_module.decoder_forth_latent(mu_forth.to(pl_module.device)).cpu()
            pred_fifth = pl_module.decoder_fifth_latent(mu_fifth.to(pl_module.device)).cpu()

        first_recon = torch.sigmoid(pred_first).data
        first_recon = (first_recon - torch.min(first_recon)).div(torch.max(first_recon) - torch.min(first_recon))
        first_recon_pad = nn.functional.pad(first_recon, pad=[4, 4, 4, 4], value=0.1)

        second_recon = torch.sigmoid(pred_second).data
        second_recon = (second_recon - torch.min(second_recon)).div(torch.max(second_recon) - torch.min(second_recon))
        second_recon_pad = nn.functional.pad(second_recon, pad=[4, 4, 4, 4], value=0.2)

        third_recon = torch.sigmoid(pred_third).data
        third_recon = (third_recon - torch.min(third_recon)).div(torch.max(third_recon) - torch.min(third_recon))
        third_recon_pad = nn.functional.pad(third_recon, pad=[4, 4, 4, 4], value=0.3)

        forth_recon = torch.sigmoid(pred_forth).data
        forth_recon = (forth_recon - torch.min(forth_recon)).div(torch.max(forth_recon) - torch.min(forth_recon))
        forth_recon_pad = nn.functional.pad(forth_recon, pad=[4, 4, 4, 4], value=0.3)

        fifth_recon = torch.sigmoid(pred_fifth).data
        fifth_recon = (fifth_recon - torch.min(fifth_recon)).div(torch.max(fifth_recon) - torch.min(fifth_recon))
        fifth_recon_pad = nn.functional.pad(fifth_recon, pad=[4, 4, 4, 4], value=0.3)

        orig_pad = nn.functional.pad(val_imgs.cpu(), pad=[4, 4, 4, 4], value=0.5)

        recon_img = [orig_pad, first_recon_pad, second_recon_pad, third_recon_pad, forth_recon_pad, fifth_recon_pad]

        print_orig_recon = torch.cat(recon_img, dim=0).cpu()
        recon_image = make_grid(print_orig_recon, normalize=False, scale_each=True, nrow=1, pad_value=1)
        recon_image = recon_image.permute(1, 2, 0)
        # Log the images as wandb Image
        self.wandb_logger.log_image('test_images/test_image_recon_' + str(self.sample), [(recon_image.numpy())])

        # log z changes first latents
        first_latent_kl_images = create_kl_value_images(dim_wise_kld_first, mu_first, sd_first,
                                                        72, background_color="black", text_color="white")
        first_latent_corr_images = create_corr_images(l1)
        # print('L1 shape: ', l1.shape)
        # print('first_latent_corr_images: ', str(len(first_latent_corr_images)))

        print_images = []
        z_size = mu_first.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_first.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_first_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(first_latent_kl_images[i]))

            print_images.append(torch.from_numpy(first_latent_corr_images[i]))

        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/first_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])

        # log z changes second latents
        second_latent_kl_images = create_kl_value_images(dim_wise_kld_second, mu_second, sd_second,
                                                        72, background_color="black", text_color="white")
        second_latent_corr_images = create_corr_images(l2)

        print_images = []
        z_size = mu_second.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_second.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_second_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(second_latent_kl_images[i]))
            print_images.append(torch.from_numpy(second_latent_corr_images[i]))

        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/second_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])
        # log z changes third latents
        third_latent_kl_images = create_kl_value_images(dim_wise_kld_third, mu_third, sd_third,
                                                          72, background_color="black", text_color="white")
        third_latent_corr_images = create_corr_images(l3)
        print_images = []
        z_size = mu_third.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_third.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_third_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(third_latent_kl_images[i]))
            print_images.append(torch.from_numpy(third_latent_corr_images[i]))
        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/third_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])

        # log z changes forth latents
        forth_latent_kl_images = create_kl_value_images(dim_wise_kld_forth, mu_forth, sd_forth,
                                                          72, background_color="black", text_color="white")
        forth_latent_corr_images = create_corr_images(l4)
        print_images = []
        z_size = mu_forth.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_forth.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_forth_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(forth_latent_kl_images[i]))
            print_images.append(torch.from_numpy(forth_latent_corr_images[i]))
        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/forth_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])

        # log z changes fifth latents
        fifth_latent_kl_images = create_kl_value_images(dim_wise_kld_fifth, mu_fifth, sd_fifth,
                                                          72, background_color="black", text_color="white")
        fifth_latent_corr_images = create_corr_images(l5)
        print_images = []
        z_size = mu_fifth.size(1)
        for i in np.arange(0, z_size, 1):
            for z_change in (np.arange(-3, 3, 0.5)):
                z_copy = mu_fifth.clone().detach()
                z_copy[0, i] = z_change
                with torch.no_grad():
                    pred = pl_module.decoder_fifth_latent(z_copy.to(pl_module.device)).cpu()
                sigm_pred = torch.sigmoid(pred).data
                sigm_pred = (sigm_pred - torch.min(sigm_pred)).div(torch.max(sigm_pred) - torch.min(sigm_pred))
                sigm_pred = nn.functional.pad(sigm_pred.cpu(), pad=[4, 4, 4, 4], value=0.5)
                print_images.append(sigm_pred)
            print_images.append(torch.from_numpy(fifth_latent_kl_images[i]))
            print_images.append(torch.from_numpy(fifth_latent_corr_images[i]))
        all_images = torch.cat(print_images, dim=0).cpu()
        images_grid = make_grid(all_images, normalize=False, scale_each=True, nrow=14, pad_value=1)
        self.wandb_logger.log_image('test_images/fifth_latent_image_z_' + str(self.sample),
                                    [(images_grid.permute(1, 2, 0).numpy())])


        pl_module.train()