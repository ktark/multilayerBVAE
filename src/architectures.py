from src.modules import *
from src.utils import *
import pytorch_lightning as pl


class VAEh(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=10, input_height=64, nc=1, decoder_dist='bernoulli',
                 gamma=100, max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0.0, C_max=20.0, C_stop_iter=1e5,
                 reparemeters_coef=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_dist = decoder_dist
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.nc = nc
        # needs to be converted to FloatTensor on cuda
        self.C_min = torch.FloatTensor([C_min]).cuda()
        self.C_max = torch.FloatTensor([C_max]).cuda()
        self.reparemeters_coef = reparemeters_coef

        self.automatic_optimization = False  # turn off pytorch optimizer

        # model architecture
        self.encoder = BoxHeadSmallEncoder(nc=self.nc,
                                           latent_dim=self.latent_dim).encoder  # InitialEncoder(nc=self.nc, latent_dim = self.latent_dim).encoder # B, z_dim*2
        self.decoder = BoxHeadSmallDecoder(nc=self.nc,
                                           latent_dim=self.latent_dim).decoder  # InitialDecoder(nc=self.nc, latent_dim = self.latent_dim).decoder #B,  nc, 64, 64

        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]

        if self.reparameters_coef < 1.0:
            mu_first = mu[:, :int(self.latent_dim / 2)]
            logvar_first = logvar[:, :int(self.latent_dim / 2)]
            mu_second = mu[:, int(self.latent_dim / 2):]
            logvar_second = logvar[:, int(self.latent_dim / 2):]

            eps_parent, eps_child = reparametrize_eps(logvar_first[:, 0:1].size(),
                                                      torch.tensor([int(self.latent_dim / 2)]))

            # std_first = logvar_first.div(2).exp()
            std_second = logvar_second.div(2).exp()

            z_first = reparametrize(mu_first, logvar_first)
            z_second = mu_second + std_second * eps_child * (
                        1 - self.reparameters_coef) + std_second * std_second.data.new(
                std_second.size()).normal_().detach() * self.reparameters_coef
            z = torch.cat((z_first, z_second), dim=1)
        else:
            z = reparametrize(mu, logvar)
        x_recon = self.decoder(z).view(x.size())
        return x_recon, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(dim=0)
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon, mu, logvar = self(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])
        beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()

        beta_vae_loss.backward()
        opt.step()

        logs = {
            'beta_vae_loss': beta_vae_loss,
            'kl': mean_kld,
            'recon_loss': recon_loss,
            'C': C,
            'iter': self.global_iter

        }
        for idx, val in enumerate(dim_wise_kld):
            logs['kl ' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return beta_vae_loss


class BVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=10, input_height=64, nc=1, decoder_dist='bernoulli',
                 gamma=100, max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0.0, C_max=20.0, C_stop_iter=1e5,
                 reparameters_coef=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_dist = decoder_dist
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.nc = nc
        # needs to be converted to FloatTensor on cuda
        self.C_min = torch.FloatTensor([C_min]).cuda()
        self.C_max = torch.FloatTensor([C_max]).cuda()

        self.automatic_optimization = False  # turn off pytorch optimizer
        self.reparameters_coef = reparameters_coef

        # model architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, self.latent_dim * 2),  # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32 * 4 * 4),  # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),  # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 64, 64
        )
        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim]
        logvar = distributions[:, self.latent_dim:]

        if self.reparameters_coef < 1.0:
            mu_first = mu[:, :int(self.latent_dim / 2)]
            logvar_first = logvar[:, :int(self.latent_dim / 2)]
            mu_second = mu[:, int(self.latent_dim / 2):]
            logvar_second = logvar[:, int(self.latent_dim / 2):]

            eps_parent, eps_child = reparametrize_eps(logvar_first[:, 0:1].size(),
                                                      torch.tensor([int(self.latent_dim / 2)]))

            # std_first = logvar_first.div(2).exp()
            std_second = logvar_second.div(2).exp()

            z_first = reparametrize(mu_first, logvar_first)
            z_second = mu_second + std_second * eps_child * (
                        1 - self.reparameters_coef) + std_second * std_second.data.new(
                std_second.size()).normal_().detach() * self.reparameters_coef
            z = torch.cat((z_first, z_second), dim=1)
        else:
            z = reparametrize(mu, logvar)
        x_recon = self.decoder(z).view(x.size())
        return x_recon, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(dim=0)
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon, mu, logvar = self(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])
        beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()

        beta_vae_loss.backward()
        opt.step()

        logs = {
            'beta_vae_loss': beta_vae_loss,
            'kl': mean_kld,
            'recon_loss': recon_loss,
            'C': C,
            'iter': self.global_iter

        }
        for idx, val in enumerate(dim_wise_kld):
            logs['kl ' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return beta_vae_loss


class VAEhier(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim_level0=12, latent_dim_level1=9, input_height=64, nc=1,
                 hier_groups=[4, 1, 1, 1, 1, 1, 1, 1, 1], decoder_dist='bernoulli', gamma=100, zeta0=1, zeta=0.8,
                 delta=0.001,
                 max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0, C_max=20, C_stop_iter=1e5,
                 loss_function='bvae', level0_training_start_iter=0, laten_recon_coef=0, reparemeters_coef=1):
        super().__init__()
        self.latent_dim_level0 = latent_dim_level0
        self.latent_dim_level1 = latent_dim_level1
        self.latent_subgroups = latent_dim_level0 / latent_dim_level1
        self.laten_recon_coef = laten_recon_coef
        self.decoder_dist = decoder_dist
        self.hier_groups = hier_groups
        print('hier_groups VAEhier', self.hier_groups)
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.zeta = zeta
        # needs to be converted to FloatTensor on cuda
        self.C_min = torch.FloatTensor([C_min]).cuda()
        self.C_max = torch.FloatTensor([C_max]).cuda()
        self.loss_function = loss_function
        self.level0_training_start_iter = torch.tensor(level0_training_start_iter)
        self.level0_beta_vae = 0
        self.l1_regularization = 1
        self.dim_wise_kld = []
        self.hierarchical_kl = []
        self.zeta0 = zeta0
        self.reparameters_coef = reparemeters_coef

        self.automatic_optimization = False
        # nr of channels in image
        self.nc = nc

        # encoder
        self.encoder = HierInitialEncoder(nc=self.nc, latent_dim=self.latent_dim_level0, hier_groups=self.hier_groups)

        self.decoder = BoxHeadSmallDecoder(nc=self.nc, latent_dim=self.latent_dim_level0).decoder

        self.decoder_level1 = BoxHeadSmallDecoder(nc=self.nc, latent_dim=self.latent_dim_level1).decoder

        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        # self.weight_init()
        self.init_weights()

    def weight_init(self):
        for block in self._modules:
            print(type(self._modules[block]), self._modules[block])
            for m in self._modules[block]:
                kaiming_init(m)

    def init_weights(m):
        print(m)
        kaiming_init(m)

    def forward(self, x):
        distributions, hier_dist_concat = self.encoder(x)
        mu = distributions[:, :self.latent_dim_level0]
        logvar = distributions[:, self.latent_dim_level0:]

        mu_hier = hier_dist_concat[:, :self.latent_dim_level1]
        logvar_hier = hier_dist_concat[:, self.latent_dim_level1:]
        # print(logvar_hier.size(), torch.tensor(self.hier_groups).size())
        if self.reparameters_coef >= 1.0:
            z = reparametrize(mu, logvar)
            z_hier = reparametrize(mu_hier, logvar_hier)
        else:
            eps_parent, eps_child = reparametrize_eps(logvar_hier.size(), torch.tensor(self.hier_groups))
            std = logvar.div(2).exp()
            z = mu + std * eps_child * (1 - self.reparameters_coef) + std * std.data.new(
                std.size()).normal_().detach() * (self.reparameters_coef)
            z_hier = mu_hier + logvar_hier.div(2).exp() * eps_parent

        x_recon = self.decoder(z).view(x.size())
        x_recon_hier = self.decoder_level1(z_hier).view(x.size())

        return x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return torch.optim.Adamax(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(dim=0)
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier = self(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)

        recon_loss_hier = reconstruction_loss(x, x_recon_hier, self.decoder_dist)

        recon_loss_levels = reconstruction_loss(torch.sigmoid(x_recon_hier), x_recon, self.decoder_dist)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        l2_loss_additional = 0
        l1_loss_additional = 0
        total_kld_hier, hierarchical_kl, mean_kld_hier = kl_divergence(mu_hier, logvar_hier)
        latent_recon = 0
        if self.loss_function == 'bvae_latent':
            # empty_image = torch.zeros_like(x)
            latent_recon = latent_layer_reconstruction_images(self, x_recon, x_recon_hier, mu, mu_hier)
            # distributions_empty, hier_dist_concat_empty = self.encoder(empty_image)
            # latent_recon = latent_layer_reconstruction(self, batch_size)
        corr = 0
        if self.loss_function == 'bvae_corr':
            corr = get_latent_levels_correlation_sum(self, mu, mu_hier)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])

        # level 0 annealing
        C_anneal_level0 = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter - (
                self.C_max * (self.level0_training_start_iter / self.C_stop_iter)), self.C_min, self.C_max.data[0])
        anneal_coef = (self.trainer.global_step - self.level0_training_start_iter) / (
                self.level0_training_start_iter + 1)
        level0_anneal = torch.clamp(anneal_coef, torch.tensor([0]).item(), torch.tensor([1]).item())

        kld_diff_loss = 0

        if self.loss_function == 'bvae':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_kld_diff':

            already_visited_idx = 0
            for idx, k in enumerate(hierarchical_kl):
                nr_of_child_kl = self.hier_groups[idx]
                child_kld_sum = torch.sum(dim_wise_kld[already_visited_idx:already_visited_idx + nr_of_child_kl])
                kld_diff_loss += (k - child_kld_sum) ** 2
                already_visited_idx += nr_of_child_kl
            kld_diff_loss = kld_diff_loss / len(self.hier_groups)
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + 5000 * kld_diff_loss

        if self.loss_function == 'bvae_l1l2':
            l1_loss_additional = (sum(torch.sum(p.abs()) for p in self.encoder.additional_encoders.parameters()))
            l2_loss_additional = (sum(torch.norm(p) for p in self.encoder.additional_encoders.parameters()))
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + self.l1_regularization * l1_loss_additional + 10 * l2_loss_additional

        if self.loss_function == 'bvae_corr':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + corr * self.laten_recon_coef + recon_loss_levels * 20

        if self.loss_function == 'bvae_latent':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + latent_recon * self.laten_recon_coef

        if self.loss_function == 'bvae_l1_first':

            if self.trainer.global_step > self.level0_training_start_iter.item():
                self.level0_beta_vae = (recon_loss + self.gamma * (total_kld - C_anneal_level0).abs()) * level0_anneal
            else:
                self.level0_beta_vae = 0
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = self.level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_l1_first_recon':
            if self.trainer.global_step > self.level0_training_start_iter:
                level0_beta_vae = recon_loss + self.gamma * (total_kld - C_anneal_level0).abs() * level0_anneal
            else:
                level0_beta_vae = recon_loss
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_anneal_level0':
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_KL_layers':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + self.delta * kl_layers

        if self.loss_function == 'bvae_KL_layers_only':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # L1 regularization for additional layers
            l1_loss_additional = (sum(torch.sum(p.abs()) for p in self.encoder.additional_encoders.parameters()))
            l2_loss_additional = (sum(torch.norm(p) for p in self.encoder.additional_encoders.parameters()))
            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * kl_layers.abs() + self.l1_regularization * l1_loss_additional

        beta_vae_loss.backward()

        logs = {
            'train/beta_vae_loss': beta_vae_loss,
            'train/kl': mean_kld,
            'train/recon_loss': recon_loss,
            'train/C': C,
            'train/iter': self.global_iter,
            'train/kl_hier_total': total_kld_hier,
            'train/mean_kld_hier': mean_kld_hier,
            'train/recon_loss_hier': recon_loss_hier,
            'train/C_anneal_level0': C_anneal_level0,
            'train/latent_recon': latent_recon,
            'train/mean_corr': corr,
            'train/l1': l1_loss_additional,
            'train/l2': l2_loss_additional,
            'train/kld_diff_mse': kld_diff_loss

        }
        for idx, val in enumerate(dim_wise_kld):
            logs['train_kl/kl_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        for idx, val in enumerate(hierarchical_kl):
            logs['train_kl/kl_hier_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.dim_wise_kld = dim_wise_kld
        self.hierarchical_kl = hierarchical_kl
        opt.step()
        return beta_vae_loss


class VAEhierSingleDecoder(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim_level0=12, latent_dim_level1=9, input_height=64, nc=1,
                 hier_groups=[4, 1, 1, 1, 1, 1, 1, 1, 1], decoder_dist='bernoulli', gamma=100, zeta0=1, zeta=0.8,
                 delta=0.001,
                 max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0, C_max=20, C_stop_iter=1e5,
                 loss_function='bvae', level0_training_start_iter=0, laten_recon_coef=0):
        super().__init__()
        self.latent_dim_level0 = latent_dim_level0
        self.latent_dim_level1 = latent_dim_level1
        self.latent_subgroups = latent_dim_level0 / latent_dim_level1
        self.laten_recon_coef = laten_recon_coef
        self.decoder_dist = decoder_dist
        self.hier_groups = hier_groups
        print('hier_groups VAEhier', self.hier_groups)
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.zeta = zeta
        # needs to be converted to FloatTensor on cuda
        self.C_min = torch.FloatTensor([C_min]).cuda()
        self.C_max = torch.FloatTensor([C_max]).cuda()
        self.loss_function = loss_function
        self.level0_training_start_iter = torch.tensor(level0_training_start_iter)
        self.level0_beta_vae = 0
        self.l1_regularization = 0.0001
        self.dim_wise_kld = []
        self.hierarchical_kl = []
        self.zeta0 = zeta0

        self.automatic_optimization = False
        # nr of channels in image
        self.nc = nc

        # encoder
        self.encoder = HierInitialEncoder(nc=self.nc, latent_dim=self.latent_dim_level0, hier_groups=self.hier_groups)

        self.l0_connector = nn.Sequential(nn.Linear(self.latent_dim_level0, 1024),  # B, 1024
                                          nn.Dropout(p=0.3),
                                          nn.BatchNorm1d(1024))
        self.l1_connector = nn.Sequential(nn.Linear(self.latent_dim_level1, 1024),  # B, 1024
                                          nn.Dropout(p=0.3),
                                          nn.BatchNorm1d(1024))

        self.decoder = BoxHeadSmallDecoderFixedInput(nc=self.nc).decoder

        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        # self.weight_init()
        self.init_weights()

    def weight_init(self):
        for block in self._modules:
            print(type(self._modules[block]), self._modules[block])
            for m in self._modules[block]:
                kaiming_init(m)

    def init_weights(m):
        print(m)
        kaiming_init(m)

    def forward(self, x):
        distributions, hier_dist_concat = self.encoder(x)
        mu = distributions[:, :self.latent_dim_level0]
        logvar = distributions[:, self.latent_dim_level0:]

        z = reparametrize(mu, logvar)
        z0_connector = self.l0_connector(z)
        x_recon = self.decoder(z0_connector).view(x.size())

        mu_hier = hier_dist_concat[:, :self.latent_dim_level1]
        logvar_hier = hier_dist_concat[:, self.latent_dim_level1:]

        z_hier = reparametrize(mu_hier, logvar_hier)
        z1_connector = self.l1_connector(z_hier)
        x_recon_hier = self.decoder(z1_connector).view(x.size())

        return x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return torch.optim.Adamax(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(dim=0)
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier = self(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)

        recon_loss_hier = reconstruction_loss(x, x_recon_hier, self.decoder_dist)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        total_kld_hier, hierarchical_kl, mean_kld_hier = kl_divergence(mu_hier, logvar_hier)
        latent_recon = 0
        if self.loss_function == 'bvae_latent':
            # empty_image = torch.zeros_like(x)
            latent_recon = latent_layer_reconstruction_images(self, x_recon, x_recon_hier, mu, mu_hier)
            # distributions_empty, hier_dist_concat_empty = self.encoder(empty_image)
            # latent_recon = latent_layer_reconstruction(self, batch_size)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])

        # level 0 annealing
        C_anneal_level0 = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter - (
                self.C_max * (self.level0_training_start_iter / self.C_stop_iter)), self.C_min, self.C_max.data[0])
        anneal_coef = (self.trainer.global_step - self.level0_training_start_iter) / (
                self.level0_training_start_iter + 1)
        level0_anneal = torch.clamp(anneal_coef, torch.tensor([0]).item(), torch.tensor([1]).item())

        if self.loss_function == 'bvae':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_latent':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + latent_recon * self.laten_recon_coef

        if self.loss_function == 'bvae_l1_first':

            if self.trainer.global_step > self.level0_training_start_iter.item():
                self.level0_beta_vae = (recon_loss + self.gamma * (total_kld - C_anneal_level0).abs()) * level0_anneal
            else:
                self.level0_beta_vae = 0
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = self.level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_l1_first_recon':
            if self.trainer.global_step > self.level0_training_start_iter:
                level0_beta_vae = recon_loss + self.gamma * (total_kld - C_anneal_level0).abs() * level0_anneal
            else:
                level0_beta_vae = recon_loss
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_anneal_level0':
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_KL_layers':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + self.delta * kl_layers

        if self.loss_function == 'bvae_KL_layers_only':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # L1 regularization for additional layers
            l1_loss_additional = (sum(torch.sum(p.abs()) for p in self.encoder.additional_encoders.parameters()))

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * kl_layers.abs() + self.l1_regularization * l1_loss_additional

        beta_vae_loss.backward()

        logs = {
            'train/beta_vae_loss': beta_vae_loss,
            'train/kl': mean_kld,
            'train/recon_loss': recon_loss,
            'train/C': C,
            'train/iter': self.global_iter,
            'train/kl_hier_total': total_kld_hier,
            'train/mean_kld_hier': mean_kld_hier,
            'train/recon_loss_hier': recon_loss_hier,
            'train/C_anneal_level0': C_anneal_level0,
            'train/latent_recon': latent_recon

        }
        for idx, val in enumerate(dim_wise_kld):
            logs['train_kl/kl_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        for idx, val in enumerate(hierarchical_kl):
            logs['train_kl/kl_hier_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.dim_wise_kld = dim_wise_kld
        self.hierarchical_kl = hierarchical_kl
        opt.step()
        return beta_vae_loss


class VAEmulti(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim_level0=12, latent_dim_level1=9, input_height=64, nc=1,
                 hier_groups=[4, 1, 1, 1, 1, 1, 1, 1, 1], decoder_dist='bernoulli', gamma=100, zeta0=1, zeta=0.8,
                 delta=0.001,
                 max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0, C_max=20, C_stop_iter=1e5,
                 loss_function='bvae', level0_training_start_iter=0, laten_recon_coef=0, reparameters_coef=1.0):
        super().__init__()
        self.latent_dim_level0 = latent_dim_level0
        self.latent_dim_level1 = latent_dim_level1
        self.latent_subgroups = latent_dim_level0 / latent_dim_level1
        self.laten_recon_coef = laten_recon_coef
        self.decoder_dist = decoder_dist
        self.hier_groups = hier_groups
        # print('hier_groups VAEhier', self.hier_groups)
        self.C_stop_iter = C_stop_iter
        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.delta = delta
        self.zeta = zeta
        # needs to be converted to FloatTensor on cuda
        self.C_min = torch.FloatTensor([C_min]).cuda()
        self.C_max = torch.FloatTensor([C_max]).cuda()
        self.loss_function = loss_function
        self.level0_training_start_iter = torch.tensor(level0_training_start_iter)
        self.level0_beta_vae = 0
        self.l1_regularization = 0.0001
        self.dim_wise_kld = []
        self.hierarchical_kl = []
        self.zeta0 = zeta0
        self.reparameters_coef = reparameters_coef

        self.automatic_optimization = False
        # nr of channels in image
        self.nc = nc

        # encoder
        self.encoder = BoxHeadSmallEncoder(nc=self.nc, latent_dim=self.latent_dim_level0)

        ### encoders for hierarchical
        self.encoder_level1_0 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_1 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_2 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_3 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_4 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_5 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_6 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_7 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_8 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_9 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_10 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_11 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_12 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_13 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_14 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_15 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_16 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_17 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )
        self.encoder_level1_18 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )

        self.encoder_level1_19 = nn.Sequential(
            nn.Linear(1 * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4 * 2)
        )

        self.decoder = BoxHeadSmallDecoder(nc=self.nc, latent_dim=self.latent_dim_level0).decoder

        self.decoder_level1 = BoxHeadSmallDecoder(nc=self.nc, latent_dim=self.latent_dim_level1).decoder

        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        # self.weight_init()
        self.init_weights()

    def weight_init(self):
        for block in self._modules:
            print(type(self._modules[block]), self._modules[block])
            for m in self._modules[block]:
                kaiming_init(m)

    def init_weights(m):
        print(m)
        kaiming_init(m)

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.latent_dim_level0]
        logvar = distributions[:, self.latent_dim_level0:]

        hier0 = self.encoder_level1_0(distributions[:, [0, 20]])
        hier1 = self.encoder_level1_1(distributions[:, [1, 21]])
        hier2 = self.encoder_level1_2(distributions[:, [2, 22]])
        hier3 = self.encoder_level1_3(distributions[:, [3, 23]])
        hier4 = self.encoder_level1_4(distributions[:, [4, 24]])
        hier5 = self.encoder_level1_5(distributions[:, [5, 25]])
        hier6 = self.encoder_level1_6(distributions[:, [6, 26]])
        hier7 = self.encoder_level1_7(distributions[:, [7, 27]])
        hier8 = self.encoder_level1_8(distributions[:, [8, 28]])
        hier9 = self.encoder_level1_9(distributions[:, [9, 29]])
        hier10 = self.encoder_level1_10(distributions[:, [10, 30]])
        hier11 = self.encoder_level1_11(distributions[:, [11, 31]])
        hier12 = self.encoder_level1_12(distributions[:, [12, 32]])
        hier13 = self.encoder_level1_13(distributions[:, [13, 33]])
        hier14 = self.encoder_level1_14(distributions[:, [14, 34]])
        hier15 = self.encoder_level1_15(distributions[:, [15, 35]])
        hier16 = self.encoder_level1_16(distributions[:, [16, 36]])
        hier17 = self.encoder_level1_17(distributions[:, [17, 37]])
        hier18 = self.encoder_level1_18(distributions[:, [18, 38]])
        hier19 = self.encoder_level1_19(distributions[:, [19, 39]])

        cat_hier = torch.cat((hier0[:, :4], hier1[:, :4], hier2[:, :4], hier3[:, :4], hier4[:, :4], hier5[:, :4],
                              hier6[:, :4], hier7[:, :4], hier8[:, :4], hier9[:, :4], hier10[:, :4], hier11[:, :4],
                              hier12[:, :4], hier13[:, :4], hier14[:, :4], hier15[:, :4], hier16[:, :4], hier17[:, :4],
                              hier18[:, :4], hier19[:, :4],
                              hier0[:, 4:], hier1[:, 4:], hier2[:, 4:], hier3[:, 4:], hier4[:, 4:], hier5[:, 4:],
                              hier6[:, 4:], hier7[:, 4:], hier8[:, 4:], hier9[:, 4:], hier10[:, 4:], hier11[:, 4:],
                              hier12[:, 4:], hier13[:, 4:], hier14[:, 4:], hier15[:, 4:], hier16[:, 4:], hier17[:, 4:],
                              hier18[:, 4:], hier19[:, 4:]), axis=1)

        mu_hier = cat_hier[:, :self.latent_dim_level1]
        logvar_hier = cat_hier[:, self.latent_dim_level1:]

        if self.reparameters_coef >= 1.0:
            z = reparametrize(mu, logvar)
            z_hier = reparametrize(mu_hier, logvar_hier)
        else:
            eps_parent, eps_child = reparametrize_eps(logvar.size(), torch.tensor(self.hier_groups))
            std = logvar.div(2).exp()
            std_hier = logvar_hier.div(2).exp()

            z_hier = mu_hier + std_hier * eps_child * (1 - self.reparameters_coef) + std_hier * std_hier.data.new(
                std_hier.size()).normal_().detach() * self.reparameters_coef
            z = mu + std * eps_parent

        # z = reparametrize(mu, logvar)
        # z_hier = reparametrize(mu_hier, logvar_hier)

        x_recon = self.decoder(z).view(x.size())

        # print('cat_hier', cat_hier.shape, hier1.shape, hier2.shape)

        x_recon_hier = self.decoder_level1(z_hier).view(x.size())

        return x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return torch.optim.Adamax(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(dim=0)
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier = self(x)
        recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)

        recon_loss_hier = reconstruction_loss(x, x_recon_hier, self.decoder_dist)

        recon_loss_levels = reconstruction_loss(torch.sigmoid(x_recon_hier), x_recon, self.decoder_dist)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        total_kld_hier, hierarchical_kl, mean_kld_hier = kl_divergence(mu_hier, logvar_hier)
        latent_recon = 0
        if self.loss_function == 'bvae_latent':
            # empty_image = torch.zeros_like(x)
            latent_recon = latent_layer_reconstruction_images(self, x_recon, x_recon_hier, mu, mu_hier)
            # distributions_empty, hier_dist_concat_empty = self.encoder(empty_image)
            # latent_recon = latent_layer_reconstruction(self, batch_size)
        corr = 0
        if self.loss_function == 'bvae_corr':
            corr = get_latent_levels_correlation_sum(self, mu, mu_hier)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])

        # level 0 annealing
        C_anneal_level0 = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter - (
                self.C_max * (self.level0_training_start_iter / self.C_stop_iter)), self.C_min, self.C_max.data[0])
        anneal_coef = (self.trainer.global_step - self.level0_training_start_iter) / (
                self.level0_training_start_iter + 1)
        level0_anneal = torch.clamp(anneal_coef, torch.tensor([0]).item(), torch.tensor([1]).item())

        if self.loss_function == 'bvae':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_corr':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + corr * self.laten_recon_coef + recon_loss_levels * 20

        if self.loss_function == 'bvae_latent':
            beta_vae_loss = self.zeta0 * recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + latent_recon * self.laten_recon_coef

        if self.loss_function == 'bvae_l1_first':

            if self.trainer.global_step > self.level0_training_start_iter.item():
                self.level0_beta_vae = (recon_loss + self.gamma * (total_kld - C_anneal_level0).abs()) * level0_anneal
            else:
                self.level0_beta_vae = 0
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = self.level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_l1_first_recon':
            if self.trainer.global_step > self.level0_training_start_iter:
                level0_beta_vae = recon_loss + self.gamma * (total_kld - C_anneal_level0).abs() * level0_anneal
            else:
                level0_beta_vae = recon_loss
            level1_beta_vae = self.zeta * recon_loss_hier + self.delta * total_kld_hier.abs()
            beta_vae_loss = level0_beta_vae + level1_beta_vae

        if self.loss_function == 'bvae_anneal_level0':
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs()

        if self.loss_function == 'bvae_KL_layers':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + self.delta * kl_layers

        if self.loss_function == 'bvae_KL_layers_only':
            # Calculate hier level 1 KL to level 0
            hierarchical_kl = []
            hierarchical_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hierarchical_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hierarchical_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hierarchical_kl = torch.stack(hierarchical_kl)
            kl_layers = torch.sum(stacked_hierarchical_kl)

            # L1 regularization for additional layers
            l1_loss_additional = (sum(torch.sum(p.abs()) for p in self.encoder.additional_encoders.parameters()))

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * kl_layers.abs() + self.l1_regularization * l1_loss_additional

        beta_vae_loss.backward()

        logs = {
            'train/beta_vae_loss': beta_vae_loss,
            'train/kl': mean_kld,
            'train/recon_loss': recon_loss,
            'train/C': C,
            'train/iter': self.global_iter,
            'train/kl_hier_total': total_kld_hier,
            'train/mean_kld_hier': mean_kld_hier,
            'train/recon_loss_hier': recon_loss_hier,
            'train/C_anneal_level0': C_anneal_level0,
            'train/latent_recon': latent_recon,
            'train/mean_corr': corr

        }
        for idx, val in enumerate(dim_wise_kld):
            logs['train_kl/kl_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        for idx, val in enumerate(hierarchical_kl):
            logs['train_kl/kl_hier_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.dim_wise_kld = dim_wise_kld
        self.hierarchical_kl = hierarchical_kl
        opt.step()
        return beta_vae_loss


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


class VAEThreeLevel(pl.LightningModule):
    def __init__(self, latent_dims=[100, 100, 100], nc=1,
                 decoder_dist='bernoulli', gamma=1.0,
                 max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, l1_regularization=0.1, l2_regularization=10,
                 loss_function='bvae'):
        super().__init__()

        self.decoder_dist = decoder_dist

        self.global_iter = 0
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.loss_function = loss_function
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.latent_dims = latent_dims

        self.dim_wise_kld = []
        self.hierarchical_kl = []

        self.automatic_optimization = False
        # nr of channels in image
        self.nc = nc

        # encoder
        self.encoder = ThreeLevelEncoder(nc=self.nc, latent_dims=self.latent_dims)

        self.decoder_first_latent = SmallDecoder(nc=self.nc, latent_dim=self.latent_dims[0]).decoder
        self.decoder_second_latent = SmallDecoder(nc=self.nc, latent_dim=self.latent_dims[1]).decoder
        self.decoder_third_latent = SmallDecoder(nc=self.nc, latent_dim=self.latent_dims[2]).decoder

        # log hyperparameters
        self.save_hyperparameters()

        # Initialize weights
        # self.weight_init()
        self.init_weights()

    def weight_init(self):
        for block in self._modules:
            print(type(self._modules[block]), self._modules[block])
            for m in self._modules[block]:
                kaiming_init(m)

    def init_weights(m):
        kaiming_init(m)

    def forward(self, x):
        first_latent, second_latent, third_latent = self.encoder(x)

        mu_first = first_latent[:, :self.latent_dims[0]]
        logvar_first = first_latent[:, self.latent_dims[0]:]

        mu_second = second_latent[:, :self.latent_dims[1]]
        logvar_second = second_latent[:, self.latent_dims[1]:]

        mu_third = third_latent[:, :self.latent_dims[2]]
        logvar_third = third_latent[:, self.latent_dims[2]:]

        z_first = reparametrize(mu_first, logvar_first)
        z_second = reparametrize(mu_second, logvar_second)
        z_third = reparametrize(mu_third, logvar_third)

        x_recon_first = self.decoder_first_latent(z_first).view(x.size())
        x_recon_second = self.decoder_second_latent(z_second).view(x.size())
        x_recon_third = self.decoder_third_latent(z_third).view(x.size())

        return x_recon_first, mu_first, logvar_first, \
               x_recon_second, mu_second, logvar_second, \
               x_recon_third, mu_third, logvar_third

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        return torch.optim.Adamax(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def training_step(self, batch, batch_idx):
        x = batch.float()
        x = x.detach()
        self.global_iter = self.trainer.global_step + 1

        opt = self.optimizers()
        opt.zero_grad()

        x_recon_first, mu_first, logvar_first, \
        x_recon_second, mu_second, logvar_second, \
        x_recon_third, mu_third, logvar_third = self(x)

        recon_loss_first = reconstruction_loss(x, x_recon_first, self.decoder_dist)
        recon_loss_second = reconstruction_loss(x, x_recon_second, self.decoder_dist)
        recon_loss_third = reconstruction_loss(x, x_recon_third, self.decoder_dist)

        total_kld_first, dim_wise_kld_first, mean_kld_first = kl_divergence(mu_first, logvar_first)
        total_kld_second, dim_wise_kld_second, mean_kld_second = kl_divergence(mu_second, logvar_second)
        total_kld_third, dim_wise_kld_third, mean_kld_third = kl_divergence(mu_third, logvar_third)

        # if self.loss_function == 'bvae':
        l1_loss_second_latents = (sum(torch.sum(p.abs()) for p in self.encoder.second_latents.parameters()))
        l1_loss_third_latents = (sum(torch.sum(p.abs()) for p in self.encoder.third_latents.parameters()))

        l2_loss_second_latents = (sum(torch.norm(p) for p in self.encoder.second_latents.parameters()))
        l2_loss_third_latents = (sum(torch.norm(p) for p in self.encoder.third_latents.parameters()))

        beta_vae_loss = recon_loss_first + recon_loss_second + recon_loss_third + \
                        self.gamma * (total_kld_first + total_kld_second + total_kld_third) + \
                        self.l1_regularization * (l1_loss_second_latents + l1_loss_third_latents) + \
                        self.l2_regularization * (l2_loss_second_latents + l2_loss_third_latents)

        beta_vae_loss.backward()

        logs = {
            'train/beta_vae_loss': beta_vae_loss,
            'train/kl_first': mean_kld_first,
            'train/kl_second': mean_kld_second,
            'train/kl_third': mean_kld_third,

            'train/recon_first': recon_loss_first,
            'train/recon_second': recon_loss_second,
            'train/recon_third': recon_loss_third,

            'train/iter': self.global_iter,

            'train/l1_second': l1_loss_second_latents,
            'train/l1_third': l1_loss_third_latents,

            'train/l2_second': l2_loss_second_latents,
            'train/l2_third': l2_loss_third_latents,

        }
        for idx, val in enumerate(dim_wise_kld_first):
            logs['train_kl/kl_first_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        for idx, val in enumerate(dim_wise_kld_second):
            logs['train_kl/kl_second_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        for idx, val in enumerate(dim_wise_kld_third):
            logs['train_kl/kl_third_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        opt.step()
        return beta_vae_loss
