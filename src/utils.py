import copy
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
import numpy as np


def get_subgroup_indices(hier_groups, latent_dim):
    # hierarchical groups sum and latent dimensions must be equal size
    assert (np.sum(hier_groups) == latent_dim)

    previous_idx = 0
    group_idx = []
    mu_idx = []
    for k in hier_groups:
        idx = torch.arange(previous_idx, previous_idx + k)
        list_indexes = torch.cat((idx, idx + latent_dim))
        group_idx.append(list_indexes)
        mu_idx.append(idx)
        previous_idx += k
    return group_idx, mu_idx


# Define reparametrize function
def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_().detach()
    return mu + std * eps


# Get log loss (1 channel - bernoulli, 3 channels gaussian)
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss


# Calculate KL divergence from unit gaussian
def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    # mu B, len(latent_vector)

    # if mu.data.ndimension() == 4:
    #     mu = mu.view(mu.size(0), mu.size(1))
    # if logvar.data.ndimension() == 4:
    #     logvar = logvar.view(logvar.size(0), logvar.size(1)) #Batch size,

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) #shape (B, latent vector)
    total_kld = klds.sum(1).mean(0, True) #latent vector, mean over B
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# https://github.com/simplespy/SCAN-from-betaVAE/blob/e1d04dcc787b791369f49227db0bffdc63b38ea2/SCAN/model.py#L155
# KL of 2 distributions
def kl(mu1, logvar1, mu2, logvar2):
    level0mu = torch.mean(mu1, 1, keepdim=True)
    level0logvar = torch.max(logvar1, 1, keepdim=True).values
    mu = level0mu - mu2
    return torch.sum(
        0.5 * (logvar2 - level0logvar + (level0logvar - logvar2).exp() + torch.mul(mu, mu) / logvar2.exp() - 1),
        1).sum()


# Weights initialization

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def latent_layer_reconstruction(pl_module, batch_size):
    # Bring the tensors to CPU
    #pl_module.eval()
    # pl_module.cuda()
    switch_batchnorm(pl_module, turn_on=False)
    #zero_image = torch.zeros((batch_size, 1, 64, 64)).float().cuda()
    level0 = pl_module.latent_dim_level0
    level1 = pl_module.latent_dim_level1

    # 0 latent features
    level0_zero_img = torch.zeros((batch_size, level0)).cuda()
    level1_zero_img = torch.zeros((batch_size, level1)).cuda()

    #with torch.no_grad():
    pred0_level0 = pl_module.decoder(level0_zero_img)
    pred0_level1 = pl_module.decoder_level1(level1_zero_img)

    # "no info" predictions
    zero_pred_level0 = torch.sigmoid(pred0_level0)
    zero_pred_level1 = torch.sigmoid(pred0_level1)
    # return zero_pred_level0, zero_pred_level1

    # which latent feature vales to check
    check_levels = [-1, 1]
    # hierarhy indices
    hier_indices = pl_module.encoder.mu_indices
    recon_loss_between_layers_list = []
    value = 0
    for i in np.arange(0, level1, 1):
        for check in check_levels:
            z_img = copy.deepcopy(level1_zero_img)
            #batch_indices = torch.arange(0, batch_size,1).long()
            z_img[:, i] = check

            # hier reconstr
            reconst_z1 = pl_module.decoder_level1(z_img)
            # to cpu
            reconst_z1_sigm = torch.sigmoid(reconst_z1).data - zero_pred_level1

            # l0 reconstr
            l0_indices = hier_indices[i]  # print(z_img)
            z0_img = copy.deepcopy(level0_zero_img)
            z0_img[:, l0_indices] = check

            # to cpu
            reconst_z0 = pl_module.decoder(z0_img)
            reconst_z0_sigm = torch.sigmoid(reconst_z0) - zero_pred_level0

            recon_loss_between_layers = F.mse_loss(reconst_z1_sigm, reconst_z0_sigm, reduction='sum')


            #recon_loss_between_layers_list.append(recon_loss_between_layers)
            value = value + recon_loss_between_layers.item()
    logs = {}
    for idx, val in enumerate(recon_loss_between_layers_list):
        logs['train_kl/latent_recon_' + str(idx)] = val.item()
    pl_module.log_dict(
        logs,
        on_step=True, on_epoch=False, prog_bar=False, logger=True
    )
    switch_batchnorm(pl_module, turn_on=True)

            # print(recon_loss_between_layers)

            # imshow(make_grid(reconst_z1_sigm.detach().cpu(), normalize=True).permute(1, 2, 0).numpy())
            # imshow(make_grid(reconst_z0_sigm.detach().cpu(), normalize=True).permute(1, 2, 0).numpy())
            # show()
    #pl_module.train()
    return value #torch.FloatTensor(recon_loss_between_layers_list).sum()



def latent_layer_reconstruction_images(pl_module, images, images_hier, mu, mu_hier):
    # Bring the tensors to CPU
    #pl_module.eval()
    # pl_module.cuda()
    #zero_image = torch.zeros((batch_size, 1, 64, 64)).float().cuda()
    #level0 = pl_module.latent_dim_level0

    #stop batchnorm
    #switch_batchnorm(pl_module, turn_on=False)

    level1 = pl_module.latent_dim_level1
    batch_size = images.size(dim=0)

    # 0 latent features
    # level0_zero_img = torch.zeros((batch_size, level0)).cuda()
    # level1_zero_img = torch.zeros((batch_size, level1)).cuda()

    #with torch.no_grad():
    # pred0_level0 = pl_module.decoder(level0_zero_img)
    # pred0_level1 = pl_module.decoder_level1(level1_zero_img)
    #use mu directly for creating images without noise
    x_recon_hier_mu = pl_module.decoder_level1(mu_hier).view(images.size())
    x_recon_mu = pl_module.decoder(mu).view(images.size())

    # "no info" predictions
    zero_pred_level0 = torch.sigmoid(x_recon_mu)
    zero_pred_level1 = torch.sigmoid(x_recon_hier_mu)
    # return zero_pred_level0, zero_pred_level1

    # which latent feature vales to check
    check_levels = [-1, 1]
    # hierarhy indices
    hier_indices = pl_module.encoder.mu_indices
    recon_loss_between_layers_list = []
    value = 0
    for i in np.arange(0, level1, 1):
        for check in check_levels:
            z_img = mu_hier.clone()
            z_img.retain_grad()
            #batch_indices = torch.arange(0, batch_size).long()
            z_img[:, i] = check

            # hier reconstr
            reconst_z1 = pl_module.decoder_level1(z_img)
            # to cpu
            reconst_z1_sigm = torch.sigmoid(reconst_z1) - zero_pred_level1

            # l0 reconstr
            l0_indices = hier_indices[i]  # print(z_img)
            z0_img = mu.clone()
            z0_img.retain_grad()
            z0_img[:, l0_indices] = check

            # to cpu
            reconst_z0 = pl_module.decoder(z0_img)
            reconst_z0_sigm = torch.sigmoid(reconst_z0) - zero_pred_level0

            recon_loss_between_layers = F.mse_loss(reconst_z1_sigm, reconst_z0_sigm, reduction='sum')


            recon_loss_between_layers_list.append(recon_loss_between_layers)
            value = value + recon_loss_between_layers.item()/batch_size
    logs = {}
    for idx, val in enumerate(recon_loss_between_layers_list):
        logs['train_kl/latent_recon_' + str(idx)] = val.item()
    pl_module.log_dict(
        logs,
        on_step=True, on_epoch=False, prog_bar=False, logger=True
    )
    # stop batchnorm
    #switch_batchnorm(pl_module, turn_on=True)

            # print(recon_loss_between_layers)

            # imshow(make_grid(reconst_z1_sigm.detach().cpu(), normalize=True).permute(1, 2, 0).numpy())
            # imshow(make_grid(reconst_z0_sigm.detach().cpu(), normalize=True).permute(1, 2, 0).numpy())
            # show()
    #pl_module.train()
    return value #torch.FloatTensor(recon_loss_between_layers_list).sum()


def switch_batchnorm(pl_module, turn_on = True):
    for module in pl_module.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(turn_on)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(turn_on)
            if turn_on:
                module.train()
            else:
                module.eval()


def get_latent_levels_correlation_sum(pl_module, mu, mu_hier):
    repeats = torch.tensor(pl_module.hier_groups).cuda()
    mu_hier_trans_repeated = torch.repeat_interleave(mu_hier.T, repeats, dim=0)
    mu_trans = mu.T

    list_of_coeff = [torch.abs(torch.corrcoef(torch.stack((mu_trans[idx], mu_hier_trans_repeated[idx]), axis=0))[0, 1]) for idx in
                     torch.arange(0, mu_trans.size(0), 1)]
    corr = torch.stack(list_of_coeff).sum().div(mu_trans.size(0))
    return corr
