from __future__ import annotations

import csv
import getpass
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import wandb
from utils.interpolation_anchors import build_diverse_anchor_loader
from visualize import make_interpolation_figure, make_reconstruction_figure


@dataclass
class ExperimentConfig:
    name: str
    num_epochs: int = 100
    seed: int = 42
    use_amp: bool = True
    save_every: int = 10
    run_root: str = "runs"
    epoch_log_every: int = 10
    show_epoch_progress: bool = True


def _safe_path_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    token = token.strip("._")
    return token or "unknown"


def build_run_dir(config: ExperimentConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    username = _safe_path_token(getpass.getuser())
    exp_name = _safe_path_token(config.name)
    run_name = f"{ts}_{username}_{exp_name}"
    run_root = Path(os.path.expandvars(config.run_root)).expanduser()
    run_dir = run_root / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    progress_bar: Optional[tqdm] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> float:
    model.train()
    total = 0.0

    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            recon, _ = model(points)
            loss = loss_fn(recon, points)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
        if progress_bar is not None:
            progress_bar.update(1)
            postfix = {
                "phase": "train",
                "loss": f"{loss.item():.6f}",
            }
            if epoch is not None and num_epochs is not None:
                postfix["epoch"] = f"{epoch}/{num_epochs}"
            progress_bar.set_postfix(postfix)

    return total / max(1, len(loader))


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool,
    progress_bar: Optional[tqdm] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> float:
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                recon, _ = model(points)
                loss = loss_fn(recon, points)

            total += loss.item()
            if progress_bar is not None:
                progress_bar.update(1)
                postfix = {
                    "phase": "val",
                    "loss": f"{loss.item():.6f}",
                }
                if epoch is not None and num_epochs is not None:
                    postfix["epoch"] = f"{epoch}/{num_epochs}"
                progress_bar.set_postfix(postfix)

    return total / max(1, len(loader))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    config: ExperimentConfig,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def append_metrics(path: Path, epoch: int, train_loss: float, val_loss: float) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerow([epoch, train_loss, val_loss])


def init_wandb(config: ExperimentConfig):
    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        return None

    try:
        run = wandb.init(
            project="shapevae",
            name=config.name,
            config=asdict(config),
            # reinit=True,
        )
        return run
    except Exception:
        return None


def run_training(
    config: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: Optional[str] = None,
) -> Tuple[Path, Dict[str, float]]:
    set_seed(config.seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True

    run_dir = build_run_dir(config)
    save_json(run_dir / "config.json", asdict(config))

    model = model.to(device_obj)

    use_amp = config.use_amp and device_obj.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wandb_run = init_wandb(config)

    best_val = float("inf")
    best_epoch = -1
    log_every = max(1, config.epoch_log_every)

    # W&B qualitative logging defaults (kept outside ExperimentConfig by design)
    log_val_reconstructions = True
    recon_every = 10
    recon_num_batches = 1
    recon_n_cols = 8
    recon_max_samples = 16
    log_val_interpolations = True
    interp_every = 10
    interp_grid_size = 5
    interp_vis_loader = val_loader

    if log_val_interpolations:
        try:
            interp_vis_loader, selected_anchor_indices, selected_anchor_ids = build_diverse_anchor_loader(
                source_loader=val_loader,
                n_select=4,
                seed=config.seed,
                compute_device=device_obj,
                show_progress=False,
            )
            tqdm.write(
                f"[{config.name}] interpolation anchors selected: "
                f"idx={selected_anchor_indices}, ids={selected_anchor_ids}"
            )
        except Exception as exc:
            tqdm.write(
                f"[{config.name}] failed to build diverse interpolation anchors; "
                f"falling back to val_loader: {exc}"
            )

    total_steps = config.num_epochs * (len(train_loader) + len(val_loader))
    global_pbar = None
    if config.show_epoch_progress:
        global_pbar = tqdm(
            total=total_steps,
            desc=f"train:{config.name}",
            leave=True,
        )

    try:
        for epoch in range(1, config.num_epochs + 1):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device_obj,
                use_amp=use_amp,
                scaler=scaler,
                progress_bar=global_pbar,
                epoch=epoch,
                num_epochs=config.num_epochs,
            )
            val_loss = validate_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device_obj,
                use_amp=use_amp,
                progress_bar=global_pbar,
                epoch=epoch,
                num_epochs=config.num_epochs,
            )

            append_metrics(run_dir / "metrics.csv", epoch, train_loss, val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    config,
                )

            if config.save_every > 0 and epoch % config.save_every == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val,
                    config,
                )

            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/best": best_val,
            }

            if (
                wandb_run is not None
                and log_val_reconstructions
                and (epoch % recon_every == 0 or epoch == 1)  
            ):
                try:

                    fig = make_reconstruction_figure(
                        model=model,
                        loader=val_loader,
                        device=device_obj,
                        num_batches=max(1, recon_num_batches),
                        n_cols=max(1, recon_n_cols),
                        max_samples=recon_max_samples,
                    )
                    metrics["val/reconstruction"] = wandb.Image(fig, caption=f"epoch={epoch}")
                    plt.close(fig)
                except Exception as exc:
                    tqdm.write(f"[{config.name}] skipped val reconstruction logging at epoch {epoch}: {exc}")

            if (
                wandb_run is not None
                and log_val_interpolations
                and (epoch % interp_every == 0 or epoch == 1)
            ):
                try:
                    interp_fig = make_interpolation_figure(
                        model=model,
                        loader=interp_vis_loader,
                        device=device_obj,
                        grid_size=interp_grid_size,
                    )
                    metrics["val/interpolation"] = wandb.Image(
                        interp_fig,
                        caption=f"epoch={epoch}, grid={interp_grid_size}x{interp_grid_size}",
                    )
                    plt.close(interp_fig)
                except Exception as exc:
                    tqdm.write(f"[{config.name}] skipped val interpolation logging at epoch {epoch}: {exc}")

            if wandb_run is not None:
                wandb_run.log(metrics, step=epoch)

            should_log = (epoch % log_every == 0) or (epoch == config.num_epochs)
            if should_log:
                tqdm.write(
                    f"[{config.name}] epoch {epoch}/{config.num_epochs} "
                    f"train {train_loss:.6f} val {val_loss:.6f} best {best_val:.6f}"
                )

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            config.num_epochs,
            best_val,
            config,
        )

        summary = {
            "best_val": best_val,
            "best_epoch": best_epoch,
        }
        save_json(run_dir / "summary.json", summary)

        if wandb_run is not None:
            wandb_run.summary["best_val"] = best_val
            wandb_run.summary["best_epoch"] = best_epoch

        return run_dir, summary
    finally:
        if global_pbar is not None:
            global_pbar.close()

        if wandb_run is not None:
            wandb_run.finish()
