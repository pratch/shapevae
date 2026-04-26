from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from pytorch3d.loss import chamfer_distance


def build_diverse_anchor_loader(
    source_loader: DataLoader,
    n_select: int = 4,
    seed: int = 42,
    compute_device: Optional[str | torch.device] = None,
    show_progress: bool = False,
) -> Tuple[DataLoader, List[int], List[str]]:
    """Greedy farthest-point sampling over a loader's dataset using Chamfer distance.

    Returns a loader containing only selected anchors, selected dataset indices, and IDs.
    """
    dataset = source_loader.dataset
    n_items = len(dataset)

    if n_items < n_select:
        raise ValueError(f"Need at least {n_select} samples, but found {n_items}")

    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(compute_device)

    points_cpu: List[torch.Tensor] = []
    object_ids: List[str] = []
    for i in range(n_items):
        sample = dataset[i]
        points_cpu.append(sample["points"].float().cpu())
        object_ids.append(str(sample.get("object_id", i)))

    rng = np.random.default_rng(seed)
    first_idx = int(rng.integers(0, n_items))
    selected = [first_idx]
    selected_set = {first_idx}

    min_dist_to_selected = torch.full((n_items,), float("inf"), dtype=torch.float32)

    iterator_factory = range
    if show_progress:
        from tqdm.auto import tqdm

        iterator_factory = lambda x: tqdm(x, leave=False, desc="Selecting diverse interpolation anchors")

    for _ in range(1, n_select):
        last_idx = selected[-1]
        last_pc = points_cpu[last_idx].to(compute_device).unsqueeze(0)

        with torch.no_grad():
            for i in iterator_factory(range(n_items)):
                if i in selected_set:
                    continue
                cand_pc = points_cpu[i].to(compute_device).unsqueeze(0)
                d = chamfer_distance(
                    last_pc,
                    cand_pc,
                    batch_reduction="mean",
                    point_reduction="mean",
                )[0].item()
                if d < float(min_dist_to_selected[i]):
                    min_dist_to_selected[i] = d

        min_dist_to_selected[list(selected_set)] = -float("inf")
        next_idx = int(torch.argmax(min_dist_to_selected).item())
        selected.append(next_idx)
        selected_set.add(next_idx)

    selected_ids = [object_ids[i] for i in selected]

    anchor_set = Subset(dataset, selected)
    anchor_loader = DataLoader(
        anchor_set,
        batch_size=n_select,
        shuffle=False,
        num_workers=0,
        pin_memory=getattr(source_loader, "pin_memory", False),
    )

    return anchor_loader, selected, selected_ids
