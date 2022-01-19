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
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
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
