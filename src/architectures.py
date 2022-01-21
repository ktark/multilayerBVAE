from src.modules import *
from src.utils import *
import pytorch_lightning as pl


class VAEh(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=10, input_height=64, nc=1, decoder_dist='bernoulli',
                 gamma=100, max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0.0, C_max=20.0, C_stop_iter=1e5):
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
                 hier_groups=[4, 1, 1, 1, 1, 1, 1, 1, 1], decoder_dist='bernoulli', gamma=100, zeta=0.8, delta=0.001,
                 max_iter=1.5e6, lr=5e-4, beta1=0.9, beta2=0.999, C_min=0, C_max=20, C_stop_iter=1e5,
                 loss_function='bvae'):
        super().__init__()
        self.latent_dim_level0 = latent_dim_level0
        self.latent_dim_level1 = latent_dim_level1
        self.latent_subgroups = latent_dim_level0 / latent_dim_level1
        self.decoder_dist = decoder_dist
        self.hier_groups = hier_groups
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

        z = reparametrize(mu, logvar)
        x_recon = self.decoder(z).view(x.size())

        mu_hier = hier_dist_concat[:, :self.latent_dim_level1]
        logvar_hier = hier_dist_concat[:, self.latent_dim_level1:]
        # print('hier ',mu_hier.shape, logvar_hier.shape, mu_hier)

        z_hier = reparametrize(mu_hier, logvar_hier)
        x_recon_hier = self.decoder_level1(z_hier).view(x.size())

        return x_recon, mu, logvar, x_recon_hier, mu_hier, logvar_hier

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

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

        total_kld_hier, hier_kl, _ = kl_divergence(mu_hier, logvar_hier)

        # calculate C value
        C = torch.clamp((self.C_max / self.C_stop_iter) * self.global_iter, self.C_min, self.C_max.data[0])

        if self.loss_function == 'bvae':
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * self.gamma * (total_kld_hier - 1 * C).abs()

        if self.loss_function == 'bvae_anneal_level0':
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * (total_kld_hier).abs()

        if self.loss_function == 'bvae_KL_layers':
            # Calculate hier level 1 KL to level 0
            hier_kl = []
            hier_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hier_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hier_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hier_kl = torch.stack(hier_kl)
            kl_layers = torch.sum(stacked_hier_kl)

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * total_kld_hier.abs() + self.delta * kl_layers

        if self.loss_function == 'bvae_KL_layers_only':
            # Calculate hier level 1 KL to level 0
            hier_kl = []
            hier_indices = self.encoder.mu_indices

            for idx, indices in enumerate(hier_indices):
                indices_torch = indices.clone().detach().cuda()
                idx_torch = torch.tensor(idx).cuda()
                hier_kl.append(
                    kl(torch.index_select(mu, 1, indices_torch), torch.index_select(logvar, 1, indices_torch),
                       torch.index_select(mu_hier, 1, idx_torch), torch.index_select(logvar_hier, 1, idx_torch)))

            stacked_hier_kl = torch.stack(hier_kl)
            kl_layers = torch.sum(stacked_hier_kl)

            # get loss
            beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs() + self.zeta * recon_loss_hier + \
                            self.delta * kl_layers.abs()

        beta_vae_loss.backward()
        opt.step()

        logs = {
            'beta_vae_loss': beta_vae_loss,
            'kl': mean_kld,
            'recon_loss': recon_loss,
            'C': C,
            'iter': self.global_iter,
            'kl_hier_total': total_kld_hier,
            'recon_loss_hier': recon_loss_hier
        }
        for idx, val in enumerate(dim_wise_kld):
            logs['kl_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        for idx, val in enumerate(hier_kl):
            logs['kl_hier_' + str(idx)] = val
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

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
