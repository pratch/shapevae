import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, z_dim=1024):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.ReLU(),
            nn.Conv1d(64,128,1),
            nn.ReLU(),
            nn.Conv1d(128,z_dim,1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

    def forward(self,x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N) for Conv1d (recall conv2d is like MLP on the channel dim = 3 here)
        x = self.point_mlp(x)
        return x

class MLPDecoder(nn.Module):
    def __init__(self, z_dim=1024, N=1024):
        super().__init__()
        self.num_points = N

        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, N * 3)
        )

    def forward(self, z):
        # z: (B, z_dim)
        B = z.shape[0]

        x = self.mlp(z) # (B, N*3)
        x = x.view(B, self.num_points, 3)

        return x

class PointCloudAE(nn.Module):
    def __init__(self, z_dim=1024, num_points=1024):
        super().__init__()
        self.encoder = PointNetEncoder(z_dim)
        self.decoder = MLPDecoder(z_dim, num_points)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z