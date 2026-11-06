"""Training utilities for neural network models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    gradient_clip: float = 1.0
    device: str = "auto"  # auto, cpu, cuda
    save_best: bool = True
    checkpoint_dir: str = "./checkpoints"


@dataclass
class TrainingHistory:
    """Training history tracker."""
    train_loss: list[float]
    val_loss: list[float]
    train_metrics: list[dict]
    val_metrics: list[dict]
    best_epoch: int
    best_val_loss: float


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for training."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    optimizer_type: str = "adamw",
) -> optim.Optimizer:
    """Create optimizer for model."""
    if optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "reduce_on_plateau",
    **kwargs,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    if scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            min_lr=kwargs.get("min_lr", 1e-6),
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 1e-6),
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
    compute_metrics: Optional[Callable] = None,
) -> tuple[float, dict]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_metrics = []

    for batch in dataloader:
        # Handle different batch formats
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch.get("inputs")
            targets = batch.get("targets")
        else:
            raise ValueError("Batch must be (inputs, targets) tuple or dict with 'inputs' and 'targets' keys")

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Handle multi-output models
        if isinstance(outputs, tuple):
            loss = criterion(outputs[0], targets)
        else:
            loss = criterion(outputs, targets)

        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()

        if compute_metrics:
            if isinstance(outputs, tuple):
                metrics = compute_metrics(outputs[0], targets)
            else:
                metrics = compute_metrics(outputs, targets)
            all_metrics.append(metrics)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_metrics: Optional[Callable] = None,
) -> tuple[float, dict]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_metrics = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            elif isinstance(batch, dict):
                inputs = batch.get("inputs")
                targets = batch.get("targets")
            else:
                raise ValueError("Batch must be (inputs, targets) tuple or dict with 'inputs' and 'targets' keys")

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], targets)
            else:
                loss = criterion(outputs, targets)

            total_loss += loss.item()

            if compute_metrics:
                if isinstance(outputs, tuple):
                    metrics = compute_metrics(outputs[0], targets)
                else:
                    metrics = compute_metrics(outputs, targets)
                all_metrics.append(metrics)

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    criterion: Optional[nn.Module] = None,
    compute_metrics: Optional[Callable] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
) -> TrainingHistory:
    """Train a model with validation."""
    device = get_device(config.device)
    model = model.to(device)

    if criterion is None:
        criterion = nn.MSELoss()

    if optimizer is None:
        optimizer = create_optimizer(model, config.learning_rate, config.weight_decay)

    if scheduler is None:
        scheduler = create_scheduler(
            optimizer,
            "reduce_on_plateau",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        train_metrics=[],
        val_metrics=[],
        best_epoch=0,
        best_val_loss=float("inf"),
    )

    for epoch in range(config.epochs):
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config.gradient_clip, compute_metrics
        )

        # Validation
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, compute_metrics
        )

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Track history
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_metrics.append(train_metrics)
        history.val_metrics.append(val_metrics)

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            if config.save_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                }, checkpoint_dir / "best_model.pt")

        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    if config.save_best and (checkpoint_dir / "best_model.pt").exists():
        checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {history.best_epoch+1}")

    return history


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Make predictions with a trained model."""
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)

    all_predictions = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _ = batch
            elif isinstance(batch, dict):
                inputs = batch.get("inputs")
            else:
                inputs = batch

            inputs = inputs.to(device)
            outputs = model(inputs)

            # Handle multi-output models (predictions, uncertainties)
            if isinstance(outputs, tuple):
                preds, uncerts = outputs
                all_uncertainties.append(uncerts.cpu().numpy())
            else:
                preds = outputs

            all_predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    uncertainties = np.concatenate(all_uncertainties, axis=0) if all_uncertainties else None

    return predictions, uncertainties


def save_model(
    model: nn.Module,
    path: str,
    config: Optional[TrainingConfig] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Save model checkpoint."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    if config:
        save_dict['config'] = config
    if metadata:
        save_dict['metadata'] = metadata

    torch.save(save_dict, path)


def load_model(
    model_class: type,
    path: str,
    device: Optional[torch.device] = None,
    **model_kwargs,
) -> nn.Module:
    """Load model from checkpoint."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device)

    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model