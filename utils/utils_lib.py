import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os,clip
import time
from tqdm import tqdm, trange

from shapenet_dataset import ShapeNetDataset
import sys
sys.path.insert(0, "/home/palakons/shapevae")#files are relative to singularity home, coz the server is running singularity from there
from model.ptv3_based_model import PointVAE, VAEConfig

from visualize import visualize_reconstructions, plot_pointclouds,visualize_interpolations



from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from pytorch3d.loss import chamfer_distance
import numpy as np

import wandb



from model.base_model import PointCloudAE

from model.ptv3_based_model import PointVAE, VAEConfig,AECLIPProjectionHead,train_adaptor

# count categories
from collections import Counter


def loadmodel_from_wandb(run_id, device,num_points=1024):
    api = wandb.Api()
    run_allcat = api.run(f"alephnir-vistec/shapevae/{run_id}")

    print("  [loadmodel_from_wandb] run_dir:", run_allcat.summary["run_dir"].replace("/ist-nas/ist-share/vision/pratchp", ""))
    best_checkpoint_path = os.path.join(run_allcat.summary["run_dir"].replace("/ist-nas/ist-share/vision/pratchp", ""), "checkpoints/best.pt")

    # print("    Found checkpoint:", best_checkpoint_path)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    # print(f"keys in checkpoint: {checkpoint.keys()}") #['epoch', 'best_val', 'model_state_dict', 'optimizer_state_dict', 'config']
    model_state_dict = checkpoint["model_state_dict"]
    if "ptv3" in run_allcat.name:
        cfg = VAEConfig(hidden_dim=64, latent_dim=128, num_points=num_points, variational=("vae" in run_allcat.name)   , grid_size=.01)
        point_ae_model = PointVAE(cfg=cfg).to(device)
    elif "baseline" in run_allcat.name:
        point_ae_model = PointCloudAE(z_dim=z_dim, num_points=num_points)
    else:
        print("    Unrecognized model type in run name. Skipping visualization.")

    point_ae_model.load_state_dict(model_state_dict)
    point_ae_model.to(device)
    point_ae_model.eval()
    return point_ae_model,run_allcat

def load_dataset(data_dir, object_classes,verbose=False,batch_size=16, num_workers=0, seed=42,split_ratios=(0.8, 0.1, 0.1),device="cuda"):
    torch.manual_seed(seed)
    if len(object_classes) == 0:
        object_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"Found object classes: {object_classes}")
    datasets = [ShapeNetDataset(data_dir=data_dir, object_class=obj_cls) for obj_cls in object_classes]
    dataset = torch.utils.data.ConcatDataset(datasets)

    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    train_idx, val_idx, test_idx = np.split(all_indices, [int(split_ratios[0]*len(dataset)), int((split_ratios[0]+split_ratios[1])*len(dataset))])
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    if verbose:
        # plot 3d scatter of first batch
        for batch in train_loader:
            pcs = batch['points']  # shape (B, N, 3)
            ids = batch['object_id']
            pc_list = list(zip(ids, pcs))
            plot_pointclouds(pc_list, n_cols=8)
            break
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx


def count_categories(loader):

    all_classes = []
    tt = tqdm(loader, desc="Counting categories")
    for batch in tt:
        class_labels = batch['category']
        all_classes.extend(class_labels)
        tt.set_description(f"{Counter(all_classes)}")
    return Counter(all_classes)

def train_clip2z_adaptor(  config,

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
            z = point_ae_model.encoder(points)  # (B, 1024)
            if isinstance(z, tuple):
                z = z[0]
            z_pred = adaptor_model(clip_latent)  # (B, 1024)
            loss = loss_fn(z_pred, z)
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
                z_pred = adaptor_model(clip_latent)  # (B, 1024)
                loss = loss_fn(z_pred, z)
                val_loss += loss.item() * points.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)



        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.run_root, f"best_clip2z_adaptor_{config.name}.pt")
            torch.save(adaptor_model.state_dict(), best_path)
    

        ppbar.set_description(f"Train:{avg_train_loss:.4f}, Val:{avg_val_loss:.4f}")
    #save model at end of training as well, for checkpointing
    final_path = os.path.join(config.run_root, f"final_clip2z_adaptor_{config.name}.pt")
    torch.save(adaptor_model.state_dict(), final_path)
    print(f"Best adaptor model saved at: {best_path} with val loss: {best_val_loss:.4f}")
    print(f"Final adaptor model saved at: {final_path}")

    return config.run_root, {'best_val_loss': best_val_loss}
def best_worst_pointclouds_by_keyword(
    point_ae_model,
    model_adaptor,
    clip_model,
    keywords,
    train_loader,
    device,
    max_batches=None,
):
    clip_model.eval()
    clip_model.to(device)
    model_adaptor.eval()

    kw_tokens = clip.tokenize(keywords).to(device)                   # (K,)
    kw_clip_latents = clip_model.encode_text(kw_tokens).detach()    # (K,512)
    kw_norm = F.normalize(kw_clip_latents, dim=1).float()                    # (K,512)

    overall_min_sim = [float('inf')] * len(keywords)
    overall_max_sim = [float('-inf')] * len(keywords)
    pc_max = [None] * len(keywords)
    pc_min = [None] * len(keywords)

    # Avoid accumulating grads and reduce overhead
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if (max_batches is not None) and (i >= max_batches):
                break
            points = batch['points'].to(device)  # (B, N, 3)

            z = point_ae_model.encoder(points)  # (B, 1024)
            if isinstance(z, tuple):
                z = z[0]
            pred_clip_latent = model_adaptor(z)  # (B, 512)
            
            pred_norm = F.normalize(pred_clip_latent, dim=1).float()     # (B,512)
            sims = pred_norm @ kw_norm.t()                      # (B, K)

            # for keyword index k, column = sims[:, k]
            for k_idx, keyword in enumerate(keywords):
                col = sims[:, k_idx]                           # (B,)
                batch_min_sim, batch_min_idx = torch.min(col, dim=0)
                batch_max_sim, batch_max_idx = torch.max(col, dim=0)
                if batch_min_sim < overall_min_sim[k_idx]:
                    overall_min_sim[k_idx] = batch_min_sim.item()
                    pc_min[k_idx] = points[batch_min_idx].cpu().numpy()  # (N, 3)
                if batch_max_sim > overall_max_sim[k_idx]:
                    overall_max_sim[k_idx] = batch_max_sim.item()
                    pc_max[k_idx] = points[batch_max_idx].cpu().numpy()  # (N, 3)
    return zip(overall_min_sim, pc_min), zip(overall_max_sim, pc_max)
def eval_clip_similarity(
    point_ae_model,
    model_adaptor_z2clip,
    clip_model,
    keywords,
    dataloaders, #("train", train_loader_ptv3_allcat), ("val", val_loader_ptv3_allcat)
    device,
):
    clip_model.eval()
    clip_model.to(device)
    model_adaptor_z2clip.eval()

    kw_tokens = clip.tokenize(keywords).to(device)                   # (K,)
    kw_clip_latents = clip_model.encode_text(kw_tokens).detach()    # (K,512)
    kw_norm = F.normalize(kw_clip_latents, dim=1).float()                    # (K,512)
    results = {}
    for name, dataloader in dataloaders: 
        print(f"Evaluating {name}: finding best/worst pointclouds for keywords: {keywords}")
        min_results, max_results = best_worst_pointclouds_by_keyword(
            point_ae_model=point_ae_model,
            model_adaptor=model_adaptor_z2clip,
            clip_model=clip_model,
            keywords=keywords,
            train_loader=dataloader,
            device=device,
        )

        sim_points_pairs: Sequence[Tuple[str, torch.Tensor | np.ndarray]] = list(zip(keywords*2, [a[1] for a in list(min_results)+list(max_results)]))
        results[name] = sim_points_pairs

        print(f"Plotting for {name} set:")
        plot_pointclouds(sim_points_pairs, n_cols=len(keywords),truncate_length=20)
    return results