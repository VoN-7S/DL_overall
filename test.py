import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    **kwargs
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set without gradient computation.

    Args:
        model:     Neural network in eval mode.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        Tuple of (average_loss, accuracy) over the validation set.
    """
    model.eval()
    teacher = kwargs.get("teacher", False)
    kd_cfg = kwargs.get("kd_cfg", False)
    if teacher and kd_cfg:
        for imgs, labels in loader:
            print("batch")
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            with torch.no_grad():
                t_logits = teacher(imgs)
            loss = criterion(out, t_logits, labels, kd_cfg.temperature, kd_cfg.alpha,)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    else:

        for imgs, labels in loader:
            print("batch")
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n
