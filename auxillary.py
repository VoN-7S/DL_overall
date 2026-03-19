"""
auxillary.py
------------
Shared helper functions used across all experiments.

Functions
---------
set_seed     -- Fix all random seeds for reproducibility.
get_device   -- Return the best available compute device.
save_results -- Save loss curve plot and parameters JSON to disk.
"""

import os
import json
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    """
    Fix all random seeds to ensure reproducible results.

    Sets seeds for Python, NumPy, PyTorch CPU, PyTorch GPU,
    and disables cuDNN non-deterministic algorithms.

    Args:
        seed: Integer seed value to use across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Checks in order: CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU.

    Returns
    -------
    torch.device
        The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(
    params: dict,
    tloss_list: list,
    vloss_list: list,
    save_dir: str,
    name: str,
) -> None:
    """
    Save training and validation loss curve as PNG and parameters as JSON.

    Creates the save directory if it does not exist. Saves two files:
        {save_dir}/{name}_train_val.png -- loss curve plot
        {save_dir}/{name}_params.json   -- all experiment parameters

    Args:
        params:     Dictionary of all experiment parameters to save.
        tloss_list: Training loss value per epoch.
        vloss_list: Validation loss value per epoch.
        save_dir:   Directory path where files will be saved.
        name:       Name prefix used for both output filenames.
    """
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, name + "_params.json")
    with open(json_path, "w") as f:
        json.dump(params, f, indent=2)


    epochs = range(1, len(tloss_list) + 1)
    plt.figure()
    plt.plot(epochs, tloss_list, color="r", label="Training Loss", marker= "x")
    plt.plot(epochs, vloss_list, color="b", label="Validation Loss", marker= "o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss - " + name)
    plt.legend()
    plt.grid()

    png_path = os.path.join(save_dir, name + "_train_val.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
