import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    **kwargs
) -> Tuple[float, float]:
    """
    Run one full training pass over the data loader. Use knowledge distillation if keyword arguments "teacher"
    and "kd_cfg" are provided.

    Args:
        model:     Neural network in training mode.
        loader:    Training DataLoader.
        optimizer: Optimiser for weight updates.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        Tuple of (average_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    teacher = kwargs.get("teacher", False)
    kd_cfg = kwargs.get("kd_cfg", False)
    if teacher and kd_cfg:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            with torch.no_grad():
                t_logits = teacher(imgs)
            loss = criterion(out, t_logits, labels, kd_cfg.temperature, kd_cfg.alpha,)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    else:

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n
