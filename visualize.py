import math
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(points: torch.Tensor | np.ndarray) -> np.ndarray:
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    return points


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
):
    if num_batches < 1:
        raise ValueError("num_batches must be >= 1")
    if n_cols < 1:
        raise ValueError("n_cols must be >= 1")

    collected_points: List[torch.Tensor] = []
    collected_ids: List[str] = []
    has_all_ids = True

    loader_iter = iter(loader)
    for _ in range(num_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        collected_points.append(batch["points"])
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
    if has_all_ids:
        collected_ids = collected_ids[:n_show]

    n_rows = math.ceil(n_show / n_cols)

    fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * n_rows))
    fig.patch.set_facecolor("white")

    for i in range(n_show):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        p_in = points_np[i]
        p_out = recon_np[i]

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
