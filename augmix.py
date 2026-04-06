"""
robustness.py
-------------
HW2 Tasks 1 & 2 — Robustness evaluation on CIFAR-10-C and AugMix training.

Task 1
------
Load the fine-tuned ResNet-18 from HW1b and evaluate it on:
  (a) clean CIFAR-10 test set
  (b) CIFAR-10-C corrupted test set (19 corruption types × 5 severity levels)

Task 2
------
Re-train ResNet-18 using AugMix (Jensen-Shannon consistency loss) and repeat
the same evaluation, reporting clean and corrupted accuracies side-by-side.

CIFAR-10-C
----------
Download from: https://zenodo.org/record/2535967
Expected structure after extraction:
    data/CIFAR-10-C/
        labels.npy          (50 000,) — same for all corruptions
        brightness.npy      (50 000, 32, 32, 3)
        contrast.npy
        ... (19 files total)

If the directory is absent, the script will print a download reminder and exit.

Reference
---------
Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations," ICLR 2019.
"""

from __future__ import annotations

import os
import json
from copy import deepcopy
from dataclasses import dataclass, asdict
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.ResNet import ResNet, BasicBlock
from auxillary import set_seed, get_device, save_results, AugMixTransform, augmix_loss
from test import validate
from parameters import RobustnessConfig, TrainingConfig, get_training_configs

# ==============================================================================
#  Constants
# ==============================================================================

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)

_CIFAR10C_DIR = "./data/CIFAR-10-C"

_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
]

_NUM_CLASSES = 10


# ==============================================================================
#  Dataset helpers
# ==============================================================================

def _get_clean_loader(batch_size: int, num_workers: int) -> DataLoader:
    """Return CIFAR-10 clean test DataLoader with standard normalization."""
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def _get_augmix_train_loader(batch_size: int, num_workers: int) -> DataLoader:
    """
    Return CIFAR-10 training DataLoader with AugMix transform.

    Each sample is (stacked_views, label) where stacked_views is (3, C, H, W):
    index 0 = clean, 1 = aug1, 2 = aug2.
    """
    base_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    augmix_tf = AugMixTransform(base_transform=base_tf, severity=0.5, width=3, alpha=1.0)
    ds = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=augmix_tf
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def _get_cifar10c_loader(
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    """
    Build a DataLoader for one CIFAR-10-C (corruption, severity) pair.

    CIFAR-10-C stores all 5 severity levels concatenated in a single .npy file
    of shape (50000, 32, 32, 3). Severity 1 = indices 0-9999, ..., 5 = 40000-49999.

    Args:
        corruption: Corruption name (must match filename without .npy).
        severity:   1–5.
        batch_size: Mini-batch size.
        num_workers: DataLoader workers.

    Returns:
        DataLoader, or None if the file is missing.
    """
    data_path  = os.path.join(_CIFAR10C_DIR, corruption + ".npy")
    label_path = os.path.join(_CIFAR10C_DIR, "labels.npy")

    if not os.path.exists(data_path) or not os.path.exists(label_path):
        return None

    data   = np.load(data_path)                     # (50000, 32, 32, 3) uint8
    labels = np.load(label_path).astype(np.int64)   # (50000,)

    start = (severity - 1) * 10_000
    end   = severity * 10_000
    data   = data[start:end]
    labels = labels[start:end]

    # Convert uint8 HWC numpy → float CHW tensor, then normalize
    mean = torch.tensor(_CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(_CIFAR_STD).view(3, 1, 1)
    imgs = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
    imgs = (imgs - mean) / std
    lbls = torch.from_numpy(labels)

    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ==============================================================================
#  CIFAR-10-C evaluation
# ==============================================================================

@torch.no_grad()
def evaluate_cifar10c(
    model: nn.Module,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate model accuracy on all 19 CIFAR-10-C corruption types × 5 severities.

    Returns a nested dict: results[corruption][severity] = accuracy (float).
    Missing .npy files are skipped with a warning.
    """
    model.eval()
    results: Dict[str, Dict[int, float]] = {}

    for corruption in _CORRUPTIONS:
        results[corruption] = {}
        for severity in range(1, 6):
            loader = _get_cifar10c_loader(corruption, severity, batch_size, num_workers)
            if loader is None:
                print(f"  [WARN] Missing {corruption}.npy — skipping")
                break
            _, acc = validate(model, loader, nn.CrossEntropyLoss(), device)
            results[corruption][severity] = round(acc, 4)
            print(f"  {corruption:<22} sev={severity}  acc={acc:.4f}")

    return results


def mean_corruption_error(
    results: Dict[str, Dict[int, float]],
    clean_acc: float,
) -> float:
    """
    Compute mean Corruption Error (mCE) averaged over all corruptions and severities.

    CE_c = (1 - mean_severity_acc_c)
    mCE = mean over all corruptions of CE_c

    Args:
        results:   Output of evaluate_cifar10c().
        clean_acc: Clean test accuracy (used only for reporting; not in mCE formula).

    Returns:
        mCE as a float in [0, 1].
    """
    ces = []
    for corruption, sev_dict in results.items():
        if sev_dict:
            avg_acc = sum(sev_dict.values()) / len(sev_dict)
            ces.append(1.0 - avg_acc)
    return float(np.mean(ces)) if ces else float("nan")


def plot_corruption_results(
    results_vanilla: Dict[str, Dict[int, float]],
    results_augmix:  Dict[str, Dict[int, float]],
    save_dir: str,
) -> None:
    """
    Bar chart: mean accuracy per corruption type, vanilla vs AugMix.

    Args:
        results_vanilla: evaluate_cifar10c() output for vanilla model.
        results_augmix:  evaluate_cifar10c() output for AugMix model.
        save_dir:        Directory to save the figure.
    """
    corruptions = [c for c in _CORRUPTIONS if results_vanilla.get(c)]
    van_means   = [sum(results_vanilla[c].values()) / len(results_vanilla[c]) for c in corruptions]
    aug_means   = [sum(results_augmix[c].values()) / len(results_augmix[c]) for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x - width/2, van_means, width, label="Vanilla fine-tune", color="steelblue")
    ax.bar(x + width/2, aug_means, width, label="AugMix fine-tune",  color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean accuracy (severities 1–5)")
    ax.set_title("CIFAR-10-C robustness: Vanilla vs AugMix")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "cifar10c_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved corruption comparison plot → {save_dir}/cifar10c_comparison.png")


# ==============================================================================
#  AugMix training loop
# ==============================================================================

def _train_one_epoch_augmix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lam: float = 12.0,
) -> Tuple[float, float]:
    """
    One training epoch with the AugMix loss (CE + lambda * JS).

    The DataLoader must use AugMixTransform so each batch yields
    imgs of shape (N, 3, C, H, W).

    Args:
        model:     Network in train mode.
        loader:    DataLoader with AugMixTransform.
        optimizer: Optimizer.
        device:    Compute device.
        lam:       JS loss weight lambda.

    Returns:
        Tuple of (mean_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for imgs_stacked, labels in loader:
        imgs_stacked = imgs_stacked.to(device)   # (N, 3, C, H, W)
        labels       = labels.to(device)

        optimizer.zero_grad()
        loss, logits_clean = augmix_loss(model, imgs_stacked, labels, lam=lam)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += logits_clean.argmax(1).eq(labels).sum().item()
        n          += labels.size(0)

    return total_loss / n, correct / n


# ==============================================================================
#  Task runners
# ==============================================================================

def run_task1(train_cfg: TrainingConfig, augmix_cfg: RobustnessConfig, device: torch.device) -> Dict[str, Dict[int, float]]:
    """
    Task 1: Evaluate vanilla HW1b model on clean CIFAR-10 and CIFAR-10-C.

    Loads the best checkpoint from HW1b transfer learning (option 2).
    Saves per-corruption accuracy table as JSON.

    Args:
        cfg:    RobustnessConfig with paths and batch settings.
        device: Compute device.

    Returns:
        Corruption results dict for downstream comparison.
    """
    save_dir = "./results/hw2/robustness_vanilla"
    os.makedirs(save_dir, exist_ok=True)

    # Load HW1b fine-tuned model (transfer option 2 = full fine-tune)
    import torchvision.models as tvm
    model = tvm.resnet18(weights=None)
    model.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, _NUM_CLASSES)
    ckpt = train_cfg.vanilla_ckpt
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Vanilla checkpoint not found: {ckpt}\n"
            "Run HW1b transfer learning first:  python main.py --task transfer --tl_option 2"
        )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"  Loaded vanilla checkpoint: {ckpt}")

    clean_loader = _get_clean_loader(train_cfg.batch_size, train_cfg.num_workers)
    _, clean_acc = validate(model, clean_loader, nn.CrossEntropyLoss(), device)
    print(f"\n  Clean test accuracy (vanilla): {clean_acc:.4f}")

    if not os.path.isdir(_CIFAR10C_DIR):
        print(
            f"\n  [ERROR] CIFAR-10-C not found at {_CIFAR10C_DIR}.\n"
            "  Download from: https://zenodo.org/record/2535967\n"
            "  Extract so that data/CIFAR-10-C/brightness.npy etc. exist."
        )
        return {}

    print("\n  Evaluating on CIFAR-10-C...")
    results = evaluate_cifar10c(model, train_cfg.batch_size, train_cfg.num_workers, device)
    mce = mean_corruption_error(results, clean_acc)
    print(f"\n  mCE (vanilla): {mce:.4f}   clean_acc: {clean_acc:.4f}")

    # Save results
    summary = {
        "clean_acc": clean_acc,
        "mCE":       mce,
        "per_corruption": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {save_dir}/results.json")

    return results


def run_task2(
    training_cfg: TrainingConfig,
    robustness_cfg: RobustnessConfig,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Dict[int, float]]]:
    """
    Task 2: Train ResNet-18 with AugMix and evaluate on clean + CIFAR-10-C.

    Saves:
        results/hw2/robustness_augmix/model.pth
        results/hw2/robustness_augmix/results.json
        results/hw2/robustness_augmix/loss_curve.png

    Args:
        cfg:    RobustnessConfig.
        device: Compute device.

    Returns:
        Tuple of (trained_model, corruption_results).
    """
    save_dir = "./results/hw2/robustness_augmix"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(training_cfg.seed)

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=_NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training_cfg.learning_rate,
        momentum=0.9, weight_decay=training_cfg.weight_decay,
    )

    train_loader = _get_augmix_train_loader(training_cfg.batch_size, training_cfg.num_workers)
    clean_loader = _get_clean_loader(training_cfg.batch_size, training_cfg.num_workers)

    best_acc = 0.0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print("\n" + "=" * 55)
    print(f"  Task 2: AugMix training | {training_cfg.epochs} epochs")
    print("=" * 55)

    for epoch in range(1, training_cfg.epochs + 1):
        tr_loss, tr_acc = _train_one_epoch_augmix(
            model, train_loader, optimizer, device, lam=robustness_cfg.augmix_lambda
        )
        val_loss, val_acc = validate(model, clean_loader, nn.CrossEntropyLoss(), device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            f"  [{epoch:02d}/{training_cfg.epochs}]"
            f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(deepcopy(model.state_dict()), os.path.join(save_dir, "model.pth"))
            print(f"    Checkpoint saved (val_acc={best_acc:.4f})")

    # Plot loss curves
    save_results(
        params={},
        tloss_list=train_losses,
        vloss_list=val_losses,
        save_dir=save_dir,
        name="augmix_training",
    )

    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location=device))

    _, clean_acc = validate(model, clean_loader, nn.CrossEntropyLoss(), device)
    print(f"\n  Clean test accuracy (AugMix): {clean_acc:.4f}")

    if not os.path.isdir(_CIFAR10C_DIR):
        print(f"\n  [ERROR] CIFAR-10-C not found at {_CIFAR10C_DIR}. Skipping corruption eval.")
        return model, {}

    print("\n  Evaluating on CIFAR-10-C...")
    results = evaluate_cifar10c(model, training_cfg.batch_size, training_cfg.num_workers, device)
    mce = mean_corruption_error(results, clean_acc)
    print(f"\n  mCE (AugMix): {mce:.4f}   clean_acc: {clean_acc:.4f}")

    summary = {
        "clean_acc": clean_acc,
        "mCE":       mce,
        "per_corruption": results,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {save_dir}/results.json")

    return model, results


# ==============================================================================
#  Entry point
# ==============================================================================


def run_robustness(params: Namespace) -> None:
    """
    Dispatcher for HW2 robustness tasks.

    Args:
        params: Parsed argparse Namespace (from get_params in parameters.py).
    """
    from parameters import get_robustness_config
    augmix_cfg    = get_robustness_config(params)
    train_cfg = get_training_configs(params)
    device = get_device()

    print("Task   : HW2 Robustness (CIFAR-10 / CIFAR-10-C)")
    print(f"Device : {device}")

    results_vanilla: Dict[str, Dict[int, float]] = {}
    results_augmix:  Dict[str, Dict[int, float]] = {}

    if params.hw2_task in ("task1", "both"):
        results_vanilla = run_task1(train_cfg, augmix_cfg, device)

    augmix_model: Optional[nn.Module] = None
    if params.hw2_task in ("task2", "both"):
        augmix_model, results_augmix = run_task2(train_cfg, augmix_cfg, device)
        # Save model reference for downstream use (Tasks 4/5)
        if augmix_model is not None:
            print("  AugMix model available at ./results/hw2/robustness_augmix/model.pth")

    if results_vanilla and results_augmix:
        plot_corruption_results(
            results_vanilla, results_augmix,
            save_dir="./results/hw2"
        )

    print("\nRobustness experiments complete.")