from __future__ import annotations

import csv
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


@dataclass
class ExperimentConfig:
    name: str
    num_epochs: int = 100
    seed: int = 42
    use_amp: bool = True
    save_every: int = 10
    run_root: str = "runs"


def build_run_dir(config: ExperimentConfig) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{ts}_{config.name}"
    run_dir = Path(config.run_root) / run_name
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
) -> float:
    model.train()
    total = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
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
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total / max(1, len(loader))


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for batch in pbar:
            points = batch["points"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                recon, _ = model(points)
                loss = loss_fn(recon, points)

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

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
        import wandb  # type: ignore

        run = wandb.init(
            project="shapevae",
            name=config.name,
            config=asdict(config),
            reinit=True,
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

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device_obj,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device_obj,
            use_amp=use_amp,
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

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/best": best_val,
                }
            )

        print(
            f"[{config.name}] epoch {epoch:03d}/{config.num_epochs} "
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
        wandb_run.finish()

    return run_dir, summary
