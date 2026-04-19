import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )

    def forward(self,x):
        x = self.mlp(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, z_dim=256, N=1024):
        super().__init__()
        self.num_points = num_points

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N * 3)
        )

    def forward(self, z):
        # z: (B, z_dim)
        B = z.shape[0]

        x = self.mlp(z)                 # (B, N*3)
        x = x.view(B, self.num_points, 3)

        return x

class PointCloudAE(nn.Module):
    def __init__(self, z_dim=256, num_points=1024):
        super().__init__()
        self.encoder = PointNetEncoder(z_dim)
        self.decoder = MLPDecoder(z_dim, num_points)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z