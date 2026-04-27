import math
import os
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return points


def _unnormalize_batch_points(
    points_np: np.ndarray,
    centroids_np: np.ndarray,
    scales_np: np.ndarray,
) -> np.ndarray:
    scales = scales_np.reshape(-1, 1, 1)
    centroids = centroids_np.reshape(-1, 1, 3)
    return points_np * scales + centroids


def _load_norm_params_for_paths(norm_paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(norm_paths)
    centroids = np.zeros((n, 3), dtype=np.float32)
    scales = np.ones((n,), dtype=np.float32)

    for i, norm_path in enumerate(norm_paths):
        if not norm_path or not os.path.isfile(norm_path):
            continue
        try:
            with np.load(norm_path) as norm_payload:
                centroids[i] = norm_payload["centroid"].astype(np.float32)
                scales[i] = np.float32(norm_payload["scale"])
        except Exception:
            # Fallback to identity unnormalization for missing/corrupt files.
            centroids[i] = np.zeros((3,), dtype=np.float32)
            scales[i] = np.float32(1.0)

    return centroids, scales


def _encode_points(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode") and callable(model.encode):
        encoded = model.encode(points)
        if isinstance(encoded, tuple):
            return encoded[0]
        return encoded

    if hasattr(model, "encoder") and callable(model.encoder):
        return model.encoder(points)

    out = model(points)
    if isinstance(out, tuple) and len(out) >= 2:
        return out[1]
    raise ValueError("Unable to extract latent codes from model")


def _decode_latents(model: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "decode") and callable(model.decode):
        return model.decode(latents)

    if hasattr(model, "decoder") and callable(model.decoder):
        return model.decoder(latents)

    raise ValueError("Model does not expose decode() or decoder for latent interpolation")


def _build_2d_latent_grid(
    z00: torch.Tensor,
    z10: torch.Tensor,
    z01: torch.Tensor,
    z11: torch.Tensor,
    rows: int,
    cols: int,
    device: torch.device,
) -> torch.Tensor:
    u = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    v = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    wu = torch.from_numpy(uu).to(device=device)
    wv = torch.from_numpy(vv).to(device=device)
    one = torch.ones_like(wu)

    # Bilinear blend of four corner latents.
    w00 = (one - wu) * (one - wv)
    w10 = wu * (one - wv)
    w01 = (one - wu) * wv
    w11 = wu * wv

    z_grid = (
        w00.unsqueeze(-1) * z00.unsqueeze(0).unsqueeze(0)
        + w10.unsqueeze(-1) * z10.unsqueeze(0).unsqueeze(0)
        + w01.unsqueeze(-1) * z01.unsqueeze(0).unsqueeze(0)
        + w11.unsqueeze(-1) * z11.unsqueeze(0).unsqueeze(0)
    )
    return z_grid.view(rows * cols, -1)


def plot_pointclouds(
    pc_list: Sequence[Tuple[str, torch.Tensor | np.ndarray]],
    n_cols: int = 4,
    point_size: float = 2,
    color: str = "dodgerblue",
    alpha: float = 0.8,
) -> None:
    n = len(pc_list)
    if n == 0:
        return

    n_rows = math.ceil(n / n_cols)
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    fig.patch.set_facecolor("white")

    for i, (model_id, points) in enumerate(pc_list):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        points_np = _to_numpy(points)

        ax.scatter(
            points_np[:, 0],
            points_np[:, 2],
            points_np[:, 1],
            s=point_size,
            c=color,
            alpha=alpha,
        )
        ax.set_title(f"ID: {str(model_id)[:8]}", fontsize=8)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

    total_plots = n_rows * n_cols
    for j in range(n, total_plots):
        ax = fig.add_subplot(n_rows, n_cols, j + 1, projection="3d")
        ax.set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


def visualize_reconstructions(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: str | torch.device,
    num_batches: int = 1,
    n_cols: int = 8,
    input_color: str = "dodgerblue",
    recon_color: str = "orangered",
    input_alpha: float = 0.35,
    recon_alpha: float = 0.75,
    unnormalize: bool = False,
) -> None:
    fig = make_reconstruction_figure(
        model=model,
        loader=loader,
        device=device,
        num_batches=num_batches,
        n_cols=n_cols,
        input_color=input_color,
        recon_color=recon_color,
        input_alpha=input_alpha,
        recon_alpha=recon_alpha,
        unnormalize=unnormalize,
    )
    plt.show()


def visualize_interpolations(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: str | torch.device,
    grid_size: int | Tuple[int, int] = 5,
    input_color: str = "dodgerblue",
    interp_color: str = "orangered",
    input_alpha: float = 0.85,
    interp_alpha: float = 0.85,
    title: str | None = None,
) -> None:
    fig = make_interpolation_figure(
        model=model,
        loader=loader,
        device=device,
        grid_size=grid_size,
        input_color=input_color,
        interp_color=interp_color,
        input_alpha=input_alpha,
        interp_alpha=interp_alpha,
        title=title,
        unnormalize=False,
    )
    plt.show()


def make_reconstruction_figure(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: str | torch.device,
    num_batches: int = 1,
    n_cols: int = 8,
    input_color: str = "dodgerblue",
    recon_color: str = "orangered",
    input_alpha: float = 0.35,
    recon_alpha: float = 0.75,
    max_samples: int | None = None,
    unnormalize: bool = False,
):
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")

    collected_points: List[torch.Tensor] = []
    collected_norm_paths: List[str] = []
    collected_ids: List[str] = []
    has_all_ids = True
    has_all_norm_paths = True

    loader_iter = iter(loader)
    for _ in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        collected_points.append(batch["points"])
        if "norm_path" in batch:
            collected_norm_paths.extend([str(x) for x in batch["norm_path"]])
        else:
            has_all_norm_paths = False

        if "object_id" in batch:
            collected_ids.extend([str(x) for x in batch["object_id"]])
        else:
            has_all_ids = False

    if not collected_points:
        raise ValueError("No batches available from loader")

    points = torch.cat(collected_points, dim=0).to(device)

    model.eval()
    with torch.no_grad():
        recon, _ = model(points)

    points_np = points.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()

    n_show = points_np.shape[0]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1 when provided")
        n_show = min(n_show, max_samples)

    points_np = points_np[:n_show]
    recon_np = recon_np[:n_show]
    if unnormalize and has_all_norm_paths and collected_norm_paths:
        show_norm_paths = collected_norm_paths[:n_show]
        centroids_np, scales_np = _load_norm_params_for_paths(show_norm_paths)
        points_np = _unnormalize_batch_points(points_np, centroids_np, scales_np)
        recon_np = _unnormalize_batch_points(recon_np, centroids_np, scales_np)

    if has_all_ids:
        collected_ids = collected_ids[:n_show]

    n_rows = math.ceil(n_show / n_cols)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))
    fig.patch.set_facecolor("white")

    for i in range(n_show):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        p_in = points_np[i]
        p_out = recon_np[i]

        # set axis limits to be based on gt points
        xmin, ymin, zmin = p_in.min(axis=0)
        xmax, ymax, zmax = p_in.max(axis=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)  # because y-axis uses p[:, 2]
        ax.set_zlim(ymin, ymax)  # because z-axis uses p[:, 1]

        ax.scatter(
            p_in[:, 0],
            p_in[:, 2],
            p_in[:, 1],
            s=2,
            c=input_color,
            alpha=input_alpha,
            label="input",
        )
        ax.scatter(
            p_out[:, 0],
            p_out[:, 2],
            p_out[:, 1],
            s=2,
            c=recon_color,
            alpha=recon_alpha,
            label="recon",
        )

        if has_all_ids and i < len(collected_ids):
            ax.set_title(f"ID: {collected_ids[i][:8]}", fontsize=8)

        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.subplots_adjust(wspace=0.02, hspace=0.08, top=0.90)
    return fig


def make_interpolation_figure(
    model: torch.nn.Module,
    loader: Iterable[dict],
    device: str | torch.device,
    grid_size: int | Tuple[int, int] = 5,
    input_color: str = "dodgerblue",
    interp_color: str = "orangered",
    input_alpha: float = 0.85,
    interp_alpha: float = 0.85,
    title: str | None = None,
    unnormalize: bool = False,
):
    if isinstance(grid_size, int):
        rows = cols = grid_size
    else:
        rows, cols = grid_size

    if rows < 2 or cols < 2:
        raise ValueError("grid_size must be >= 2 in each dimension")

    device_obj = torch.device(device)
    collected_points: List[torch.Tensor] = []
    collected_norm_paths: List[str] = []
    collected_ids: List[str] = []
    has_all_ids = True
    has_all_norm_paths = True

    loader_iter = iter(loader)
    while sum(x.shape[0] for x in collected_points) < 4:
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        pts = batch["points"]
        needed = 4 - sum(x.shape[0] for x in collected_points)
        take = min(needed, pts.shape[0])
        if take <= 0:
            break

        collected_points.append(pts[:take])
        if "norm_path" in batch:
            collected_norm_paths.extend([str(x) for x in batch["norm_path"][:take]])
        else:
            has_all_norm_paths = False

        if "object_id" in batch:
            collected_ids.extend([str(x) for x in batch["object_id"][:take]])
        else:
            has_all_ids = False

    if sum(x.shape[0] for x in collected_points) < 4:
        raise ValueError("Need at least 4 samples from loader for interpolation figure")

    anchor_points = torch.cat(collected_points, dim=0)[:4].to(device_obj)

    model.eval()
    with torch.no_grad():
        anchor_z = _encode_points(model, anchor_points)
        z_grid = _build_2d_latent_grid(
            z00=anchor_z[0],
            z10=anchor_z[1],
            z01=anchor_z[2],
            z11=anchor_z[3],
            rows=rows,
            cols=cols,
            device=device_obj,
        )
        decoded_grid = _decode_latents(model, z_grid)

    decoded_np = decoded_grid.detach().cpu().numpy()
    anchor_np = anchor_points.detach().cpu().numpy()

    if unnormalize and has_all_norm_paths and collected_norm_paths:
        anchor_norm_paths = collected_norm_paths[:4]
        anchor_centroids_np, anchor_scales_np = _load_norm_params_for_paths(anchor_norm_paths)
        anchor_centroids = torch.from_numpy(anchor_centroids_np).to(device_obj)
        anchor_scales = torch.from_numpy(anchor_scales_np).to(device_obj)

        centroid_grid = _build_2d_latent_grid(
            z00=anchor_centroids[0],
            z10=anchor_centroids[1],
            z01=anchor_centroids[2],
            z11=anchor_centroids[3],
            rows=rows,
            cols=cols,
            device=device_obj,
        )
        scale_grid = _build_2d_latent_grid(
            z00=anchor_scales[0].view(1),
            z10=anchor_scales[1].view(1),
            z01=anchor_scales[2].view(1),
            z11=anchor_scales[3].view(1),
            rows=rows,
            cols=cols,
            device=device_obj,
        ).squeeze(-1)

        centroid_grid_np = centroid_grid.detach().cpu().numpy().astype(np.float32)
        scale_grid_np = scale_grid.detach().cpu().numpy().astype(np.float32)
        decoded_np = _unnormalize_batch_points(decoded_np, centroid_grid_np, scale_grid_np)

        anchor_np = _unnormalize_batch_points(anchor_np, anchor_centroids_np, anchor_scales_np)

    corner_indices = {
        0: (0, 0),
        1: (0, cols - 1),
        2: (rows - 1, 0),
        3: (rows - 1, cols - 1),
    }
    corner_linear = {r * cols + c: idx for idx, (r, c) in corner_indices.items()}

    all_points = [decoded_np[i] for i in range(rows * cols)]
    xyz_min = np.min(np.concatenate(all_points, axis=0), axis=0)
    xyz_max = np.max(np.concatenate(all_points, axis=0), axis=0)

    fig = plt.figure(figsize=(2.5 * cols, 2.5 * rows))
    fig.patch.set_facecolor("white")

    for linear_idx in range(rows * cols):
        ax = fig.add_subplot(rows, cols, linear_idx + 1, projection="3d")
        p = decoded_np[linear_idx]

        ax.scatter(
            p[:, 0],
            p[:, 2],
            p[:, 1],
            s=2,
            c=interp_color,
            alpha=interp_alpha,
        )

        if linear_idx in corner_linear:
            corner_id = corner_linear[linear_idx]
            p_anchor = anchor_np[corner_id]
            ax.scatter(
                p_anchor[:, 0],
                p_anchor[:, 2],
                p_anchor[:, 1],
                s=2,
                c=input_color,
                alpha=input_alpha,
            )
            if has_all_ids and corner_id < len(collected_ids):
                ax.set_title(f"ID: {collected_ids[corner_id][:8]}", fontsize=8)

        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[2], xyz_max[2])
        ax.set_zlim(xyz_min[1], xyz_max[1])
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

    fig.suptitle(f"Latent Interpolation (grid={rows}x{cols})"+(f" - {title}" if title else ""), fontsize=12)
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.92)
    return fig
