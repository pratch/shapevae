import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm,trange
import os

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

class AECLIPProjectionHead(nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
        print(f"Initializing PointVAE with config: {cfg},D: {D}, C: {C}, N: {N}")

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
    def encoder(self, x: torch.Tensor):
        return self.encode(x)
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
        # print(f"coord_in {coord.shape}, feat_in shape: {feat.shape}, batch_in shape: {batch.shape}, offset_in shape: {offset.shape}") 

        
        backbone_out = self.backbone(backbone_input)

        feat_out = backbone_out["feat"]  # [M, C] after PTv3 downsampling
        batch_out = backbone_out["batch"]  # [M], sample index per point
        offset_out = backbone_out["offset"]  # cumulative counts (kept for debugging/compat)
        coord_out = backbone_out["coord"]  # [M, 3] coordinates after PTv3 processing (not used for pooling, but useful for debugging and potential future use)
        # print(f"keys in backbone_out: {backbone_out.keys()}")  # keys in backbone_out: dict_keys(['feat', 'coord', 'batch', 'offset'])
        # print(f"coord_out {coord_out.shape}, feat_out shape: {feat_out.shape}, batch_out shape: {batch_out.shape}, offset_out shape: {offset_out.shape}") #feat_out shape: torch.Size([1024, 64]), batch shape: torch.Size([863338]), offset shape: torch.Size([1024])

        feat_out = self.post_backbone_norm(feat_out)
        # print(f"feat_out after norm shape: {feat_out.shape}")  # feat_out after norm shape: torch.Size([1024, 64])
        pooled = self.to_latent(feat_out)  # [B, C]
        # print(f"pooled after to_latent shape: {pooled.shape}")  # pooled after to_latent shape: torch.Size([16, 64])

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
        z, mu, logvar = self.encoder(x)
        x_hat = self.decode(z)
        
        return x_hat, z
        return {
            "x_hat": x_hat,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

def check_voxel_collisions(pts, grid_size):
    grid_coord = torch.div(
        pts - pts.min(dim=0).values,
        grid_size,
        rounding_mode="trunc"
    ).int()

    unique_voxels, counts = torch.unique(grid_coord, dim=0, return_counts=True)

    num_points = pts.shape[0]
    num_unique = unique_voxels.shape[0]
    num_collided_points = counts[counts > 1].sum().item()
    num_collision_voxels = (counts > 1).sum().item()
    max_points_in_one_voxel = counts.max().item()

    # print(f"N points                : {num_points}")
    # print(f"Unique voxels           : {num_unique}")
    # print(f"Collision voxels        : {num_collision_voxels}")
    # print(f"Points in collided bins : {num_collided_points}")
    # print(f"Max points in one voxel : {max_points_in_one_voxel}")
    # print(f"Voxel occupancy ratio   : {num_unique / num_points:.4f}")

    return {
        "grid_coord": grid_coord,
        "counts": counts,
        "num_points": num_points,
        "num_unique": num_unique,
        "collision_voxels": num_collision_voxels,
        "collided_points": num_collided_points,
        "max_points_in_one_voxel": max_points_in_one_voxel,
        "occupancy_ratio": num_unique / num_points,
    }

def train_adaptor(  config,
    point_ae_model,
    adaptor_model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    device,
):
    '''
    gt: batch.clip_latent (B, 512)
    pred: model(batch.points) -> (B, 512)
    '''

    best_val_loss = float('inf')
    point_ae_model.eval()  # freeze point AE during adaptor training

    # outer bar: do not leave previous bars in notebook
    ppbar = tqdm(range(config.num_epochs), desc="Adaptor Training", leave=False, position=0)
    for epoch in ppbar:
        adaptor_model.train()

        train_loss = 0.0
        # inner training bar at position 1 so it replaces/clears correctly
        for batch in train_loader:
            points = batch['points'].to(device)  # (B, N, 3)
            clip_latent = batch['clip_latent'].to(device)  # (B, 512)

            optimizer.zero_grad()
            z = point_ae_model.encode(points)  # (B, 1024)
            # if model is PTV3-based, it may return (z, mu, logvar),  retuirn the first element as z
            if isinstance(z, tuple):
                z = z[0]    
                
            pred_clip_latent = adaptor_model(z)  # (B, 512)
            loss = loss_fn(pred_clip_latent, clip_latent)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * points.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        adaptor_model.eval()
        val_loss = 0.0
        # validation bar at same position as training to avoid stacking
        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(device)  # (B, N, 3)
                clip_latent = batch['clip_latent'].to(device)  # (B, 512)

                z = point_ae_model.encoder(points)  # (B, 1024)
                if isinstance(z, tuple):
                    z = z[0]    
                pred_clip_latent = adaptor_model(z)  # (B, 512)
                loss = loss_fn(pred_clip_latent, clip_latent)
                val_loss += loss.item() * points.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)



        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.run_root, f"best_adaptor_{config.name}.pt")
            torch.save(adaptor_model.state_dict(), best_path)
    

        ppbar.set_description(f"Train:{avg_train_loss:.4f}, Val:{avg_val_loss:.4f}")
    #save model at end of training as well, for checkpointing
    final_path = os.path.join(config.run_root, f"final_adaptor_{config.name}.pt")
    torch.save(adaptor_model.state_dict(), final_path)
    print(f"Best adaptor model saved at: {best_path} with val loss: {best_val_loss:.4f}")
    print(f"Final adaptor model saved at: {final_path}")

    return config.run_root, {'best_val_loss': best_val_loss}