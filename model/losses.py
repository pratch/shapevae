
from pytorch3d.loss import chamfer_distance
import torch

def repulsion_exp_loss(pts, h=0.04):
    # penalize very-close pairs smoothly
    B, N, _ = pts.shape
    D = torch.cdist(pts, pts)                      # (B, N, N)
    # remove diagonal by adding large value
    D = D + torch.eye(N, device=pts.device).unsqueeze(0) * 1e6
    loss = torch.exp(-(D**2) / (h**2)).sum() / (B * N * (N-1))
    return loss

def loss_fn_cd_plus_real_repulsion(recon, points):
    cd = chamfer_distance(
        recon.float(),
        points.float(),
        batch_reduction="mean",
        point_reduction="mean",
    )[0]
    rep = repulsion_exp_loss(recon)
    return cd + 0.1 * rep

loss_fn_cd = lambda recon, points: chamfer_distance(
    recon.float(),
    points.float(),
    batch_reduction="mean",
    point_reduction="mean",
)[0]

def loss_fn_cd_plus_repulsion(recon, points):
    cd = chamfer_distance(
        recon,
        points,
        batch_reduction="mean",
        point_reduction="mean",
    )[0]
    rep = repulsion_exp_loss(recon)
    return cd + 0.1 * rep

def loss_fn_cd(recon, points):
    return chamfer_distance(
        recon.float(),
        points.float(),
        batch_reduction="mean",
        point_reduction="mean",
    )[0]