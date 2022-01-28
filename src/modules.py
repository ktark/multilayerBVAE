from src.utils import *


# Simple layer to resize
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class InitialEncoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super(InitialEncoder, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32,  4,  4
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),  # B, 512
            nn.Linear(32 * 4 * 4, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, self.latent_dim * 2),  # B, z_dim*2
        )

    def forward(self, x):
        level0 = self.encoder(x)
        return level0


class InitialDecoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super(InitialDecoder, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 32 * 4 * 4),  # B, 512
            nn.BatchNorm1d(512),
            nn.ReLU(True),  # NO RELU HERE?
            View((-1, 32, 4, 4)),  # B,  32,  4,  4

            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.nc, 4, 2, 1),  # B,  nc, 64, 64
        )

    def forward(self, x):
        output = self.decoder(x)
        return output


class BoxHeadSmallEncoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super(BoxHeadSmallEncoder, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 64, 4, 2, padding="valid"),  # 1          # B,  32, 32, 64
            nn.LeakyReLU(0.3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, 2, padding="valid"),  # 1          # B,  16, 16, 128
            nn.LeakyReLU(0.3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 2, 2, padding="valid"),  # B,   8,  8, 256
            nn.LeakyReLU(0.3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 2, 2, padding="valid"),  # B,  4,  4,  512
            nn.LeakyReLU(0.3),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(512),
            View((-1, 512 * 3 * 3)),  # B, 2048
            nn.Linear(4608, 1024),  # B, 1024
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.latent_dim * 2),  # B, z_dim*2
        )

    def forward(self, x):
        level0 = self.encoder(x)
        return level0


class BoxHeadSmallDecoder(nn.Module):
    def __init__(self, nc, latent_dim):
        super(BoxHeadSmallDecoder, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),  # B, 1024
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4 * 4 * 512),  # B, 256
            nn.BatchNorm1d(4 * 4 * 512),
            View((-1, 512, 4, 4)),  # B,  32,  4,  4
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 256, 4, 1, 1),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, 4, 2, 2),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 4, 1, 2),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 3, 4, 2),
            nn.LeakyReLU(0.3)

        )


class HierInitialEncoder(nn.Module):
    def __init__(self, nc, latent_dim, hier_groups):
        super(HierInitialEncoder, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.hier_groups = hier_groups
        self.subgroup_indices, self.mu_indices = get_subgroup_indices(self.hier_groups, self.latent_dim)

        ######Temp excluded
        # self.encoder = InitialEncoder(nc = self.nc, latent_dim = self.latent_dim) # B, z_dim*2

        self.encoder = BoxHeadSmallEncoder(nc=self.nc, latent_dim=self.latent_dim)
        self.additional_encoders = nn.ModuleList()

        for level0_nodes_grouped in hier_groups:
            self.additional_encoders.append(nn.Sequential(nn.Linear(level0_nodes_grouped * 2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1 * 2)))

    def forward(self, x):
        level0 = self.encoder(x)

        hier_dist_params = torch.Tensor([]).cuda()

        # pass through additional encoders part
        for idx, indices in enumerate(self.subgroup_indices):
            hier_dist_params = torch.cat((hier_dist_params, self.additional_encoders[idx](level0[:, indices])),
                                         axis=1)  # B, len(hier_groups), 2

        index_size = hier_dist_params.size(1)
        mu_indexes = torch.arange(0, index_size, 2).cuda()
        logvar_indexes = torch.arange(1, index_size, 2).cuda()

        # get hierarchical mu and logvar
        mu_tensor = torch.index_select(hier_dist_params, 1, mu_indexes)
        logvar_tensor = torch.index_select(hier_dist_params, 1, logvar_indexes)

        hier_dist_concat = torch.cat((mu_tensor, logvar_tensor), axis=1)  # B, len(hier_groups)*2

        return level0, hier_dist_concat
