import copy
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
import numpy as np
from scipy import stats


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



#returns eps for parent, child level. Child level eps same as parent
def reparametrize_eps(parent_size, child_repeats):
    ztorch = torch.zeros(parent_size).detach()
    child_repeats = child_repeats.cuda()
    eps = ztorch.data.new(ztorch.size()).normal_().detach().cuda()
    eps_child = eps.repeat_interleave(child_repeats, dim=1).detach().cuda()
    return eps, eps_child

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
    list_of_coeff_func = [4*cor-4*torch.pow(cor, 2) for cor in list_of_coeff]

    corr = torch.stack(list_of_coeff_func).sum().div(mu_trans.size(0))
    return corr


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_all_latent_correlation(generative_properties_list, ds_t, vae):
    stat_l1_all = np.array([])
    stat_l2_all = np.array([])
    stat_l3_all = np.array([])

    for gen_prop in generative_properties_list:
        stat_l1 = []
        stat_l2 = []
        stat_l3 = []
        prop = ds_t.get_all_labels()[gen_prop]

        for i in np.arange(0, vae.test_mu_latent1.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent1[:, i]).correlation)
            stat_l1.append(corr)
        stat_l1_all = np.vstack([stat_l1_all, stat_l1]) if stat_l1_all.size else np.array(stat_l1)

        for i in np.arange(0, vae.test_mu_latent2.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent2[:, i]).correlation)
            stat_l2.append(corr)
        stat_l2_all = np.vstack([stat_l2_all, stat_l2]) if stat_l2_all.size else np.array(stat_l2)

        for i in np.arange(0, vae.test_mu_latent3.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent3[:, i]).correlation)
            stat_l3.append(corr)
        stat_l3_all = np.vstack([stat_l3_all, stat_l3]) if stat_l3_all.size else np.array(stat_l3)

    return stat_l1_all, stat_l2_all, stat_l3_all

def get_all_latent_correlation5(generative_properties_list, ds_t, vae):
    stat_l1_all = np.array([])
    stat_l2_all = np.array([])
    stat_l3_all = np.array([])
    stat_l4_all = np.array([])
    stat_l5_all = np.array([])

    for gen_prop in generative_properties_list:
        stat_l1 = []
        stat_l2 = []
        stat_l3 = []
        stat_l4 = []
        stat_l5 = []

        prop = ds_t.get_all_labels()[gen_prop]

        for i in np.arange(0, vae.test_mu_latent1.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent1[:, i]).correlation)
            stat_l1.append(corr)
        stat_l1_all = np.vstack([stat_l1_all, stat_l1]) if stat_l1_all.size else np.array(stat_l1)

        for i in np.arange(0, vae.test_mu_latent2.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent2[:, i]).correlation)
            stat_l2.append(corr)
        stat_l2_all = np.vstack([stat_l2_all, stat_l2]) if stat_l2_all.size else np.array(stat_l2)

        for i in np.arange(0, vae.test_mu_latent3.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent3[:, i]).correlation)
            stat_l3.append(corr)
        stat_l3_all = np.vstack([stat_l3_all, stat_l3]) if stat_l3_all.size else np.array(stat_l3)

        for i in np.arange(0, vae.test_mu_latent4.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent4[:, i]).correlation)
            stat_l4.append(corr)
        stat_l4_all = np.vstack([stat_l4_all, stat_l4]) if stat_l4_all.size else np.array(stat_l4)

        for i in np.arange(0, vae.test_mu_latent5.shape[1], 1):
            corr = np.abs(stats.spearmanr(prop, vae.test_mu_latent5[:, i]).correlation)
            stat_l5.append(corr)
        stat_l5_all = np.vstack([stat_l5_all, stat_l5]) if stat_l5_all.size else np.array(stat_l5)

    return stat_l1_all, stat_l2_all, stat_l3_all, stat_l4_all, stat_l5_all


# from google disentanglement lib
from sklearn import ensemble
import scipy

def _compute_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = ensemble.GradientBoostingRegressor(verbose=0)
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)


def get_all_latent_multinomial_regr5(generative_properties_list, ds_t, vae):
    stat_l1_all = np.array([])
    stat_l2_all = np.array([])
    stat_l3_all = np.array([])
    stat_l4_all = np.array([])
    stat_l5_all = np.array([])

    for gen_prop in generative_properties_list:
        stat_l1 = []
        stat_l2 = []
        stat_l3 = []
        stat_l4 = []
        stat_l5 = []

        prop = ds_t.get_all_labels()[gen_prop]

        for i in np.arange(0, vae.test_mu_latent1.shape[1], 1):
            y = prop
            x = vae.test_mu_latent1[:, i][:, None]
            #model pipeline

            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            #append statistics
            stat_l1.append(metric)
        stat_l1_all = np.vstack([stat_l1_all, stat_l1]) if stat_l1_all.size else np.array(stat_l1)

        for i in np.arange(0, vae.test_mu_latent2.shape[1], 1):
            y = prop
            x = vae.test_mu_latent2[:, i][:, None]
            #model pipeline
            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            stat_l2.append(metric)
        stat_l2_all = np.vstack([stat_l2_all, stat_l2]) if stat_l2_all.size else np.array(stat_l2)

        for i in np.arange(0, vae.test_mu_latent3.shape[1], 1):
            y = prop
            x = vae.test_mu_latent3[:, i][:, None]
            #model pipeline
            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            stat_l3.append(metric)
        stat_l3_all = np.vstack([stat_l3_all, stat_l3]) if stat_l3_all.size else np.array(stat_l3)

        for i in np.arange(0, vae.test_mu_latent4.shape[1], 1):
            y = prop
            x = vae.test_mu_latent4[:, i][:, None]
            #model pipeline
            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            stat_l4.append(metric)
        stat_l4_all = np.vstack([stat_l4_all, stat_l4]) if stat_l4_all.size else np.array(stat_l4)

        for i in np.arange(0, vae.test_mu_latent5.shape[1], 1):
            y = prop
            x = vae.test_mu_latent5[:, i][:, None]
            #model pipeline
            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            stat_l5.append(metric)
        stat_l5_all = np.vstack([stat_l5_all, stat_l5]) if stat_l5_all.size else np.array(stat_l5)

    return stat_l1_all, stat_l2_all, stat_l3_all, stat_l4_all, stat_l5_all

def get_all_latent_multinomial_regr1(generative_properties_list, ds_t, vae):
    stat_l1_all = np.array([])

    for gen_prop in generative_properties_list:
        stat_l1 = []

        prop = ds_t.get_all_labels()[gen_prop]

        for i in np.arange(0, vae.test_mu_latent1.shape[1], 1):
            y = prop
            x = vae.test_mu_latent1[:, i][:, None]
            #model pipeline

            model = make_pipeline(PolynomialFeatures(4), LinearRegression())
            pred = model.fit(x, y).predict(x)
            metric = r2_score(y, pred)
            #append statistics
            stat_l1.append(metric)
        stat_l1_all = np.vstack([stat_l1_all, stat_l1]) if stat_l1_all.size else np.array(stat_l1)

    return stat_l1_all




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

