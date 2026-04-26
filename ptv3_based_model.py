import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, "/home/palakons/Pointcept")  # adjust this path based on your singularity home
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)
from dataclasses import dataclass


@dataclass
class VAEConfig:
    hidden_dim: int = 64
    latent_dim: int = 128
    num_points: int = 1024
    variational: bool = True  # False -> AE, True -> VAE
    grid_size: float = (
        0.01  # for voxelization in PTv3 encoder; adjust based on point cloud scale
    )


class PointVAE(nn.Module):
    def __init__(
        self,
        cfg: VAEConfig,
    ):
        super().__init__()
        self.ptv3_condition = "chair"  # dummy condition, not used in this context
        self.cfg = cfg

        C = cfg.hidden_dim
        D = cfg.latent_dim
        N = cfg.num_points

        self.backbone = PointTransformerV3(
            in_channels=C,
            pdnorm_conditions=("chair",),  # dummy condition, not used in this context
            enc_mode=True,  # we only use the encoder part of PTv3, the decoder is a simple MLP
            # stride=(2, 2, 2, 2),
            # enc_depths=(2, 2, 2, 6, 2),
            # enc_channels=(32, 64, 128, 256, 512),
            # enc_num_head=(2, 4, 8, 16, 32),
            # enc_patch_size=(48, 48, 48, 48, 48),
            # dec_depths=(2, 2, 2, 2),
            # dec_channels=(64, 64, 128, 256),
            # dec_num_head=(4, 4, 8, 16),
            # dec_patch_size=(48, 48, 48, 48),
            stride=(2,),
            enc_depths=(2, 2,),
            enc_channels=(32, 64,),
            enc_num_head=(2, 4,),
            enc_patch_size=(48, 48, ),
            dec_depths=(2,),
            dec_channels=(64,),
            dec_num_head=(4, ),
            dec_patch_size=(48, ),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(3, C),
            nn.LayerNorm(C),
            nn.SiLU(),
        )

        self.post_backbone_norm = nn.LayerNorm(C)

        self.to_latent = nn.Sequential(
            nn.Linear(C, C),
            nn.SiLU(),
        )

        if cfg.variational:  # select VAE or AE
            self.to_mu = nn.Linear(C, D)
            self.to_logvar = nn.Linear(C, D)
        else:
            self.to_z = nn.Linear(C, D)

        self.decoder = nn.Sequential(  # use PTv3 as encoder ONLY!
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D * 4),
            nn.SiLU(),
            nn.Linear(D * 4, N * 3),
        )

    @staticmethod
    def _build_batch_and_offset(B: int, N: int, device: torch.device):
        batch = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(N)
        offset = torch.arange(1, B + 1, device=device, dtype=torch.long) * N
        #append 0 in fornt of offset to make it [B+1], where offset[i] is the starting index of the i-th batch in the flattened point cloud.
        # offset = torch.cat([torch.tensor([0], device=device, dtype=torch.long), offset], dim=0)
        return batch, offset

    def encode(self, x: torch.Tensor):
        """
        x: [B, N, 3]
        returns:
            z, mu, logvar
        """
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must be [B, N, 3], got {tuple(x.shape)}")

        B, N, _ = x.shape
        device = x.device

        coord = x.reshape(B * N, 3)  # [B*N, 3]
        batch, offset = self._build_batch_and_offset(B, N, device)
        feat = self.input_proj(coord)  # [B*N, C]

        backbone_input = {
            "coord": coord,
            "feat": feat,
            "batch": batch,
            "offset": offset,
            "grid_size": self.cfg.grid_size,
            "condition": self.ptv3_condition,
        }
        backbone_out = self.backbone(backbone_input)

        feat_out = backbone_out["feat"]  # [B*N, C]
        batch = backbone_out["batch"]  # [B*N]
        offset = backbone_out["offset"]  # [B+1]
        feat_out = self.post_backbone_norm(feat_out)

        pooled = torch.stack([feat_out[(offset[i-1] if i>0 else 0) : offset[i ]].max(dim=0).values for i in range(B)], dim=0)  # [B, C]
        # Global pooling -> shape code
        # pooled = feat_out.max(dim=1).values  # [B, C]
        pooled = self.to_latent(pooled)  # [B, C]

        if self.cfg.variational:
            mu = self.to_mu(pooled)  # [B, D]
            logvar = self.to_logvar(pooled)  # [B, D]
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            z = self.to_z(pooled)  # [B, D]
            return z, None, None

    def decode(self, z: torch.Tensor):
        """
        z: [B, D]
        returns:
            x_hat: [B, N, 3]
        """
        B = z.shape[0]
        out = self.decoder(z)  # [B, N*3]
        x_hat = out.view(B, self.cfg.num_points, 3)
        return x_hat

    def forward(self, x: torch.Tensor):
        """
        returns dict for convenience in notebook usage
        """
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }
