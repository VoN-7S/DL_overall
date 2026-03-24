import os
from copy import deepcopy
from dataclasses import asdict
from typing import List, Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from parameters import TransferConfig, TrainingConfig, get_transfer_configs, get_training_configs
from auxillary import set_seed, get_device, save_results
from test import validate
from train import train_one_epoch


# ==============================================================================
#  Dataset Stats
# ==============================================================================

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


# ==============================================================================
#  Data Loading
# ==============================================================================

def get_loaders(
    transfer_cfg: TransferConfig,
    training_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train and validation DataLoaders for the given option.

    Option 1: Resize to 224x224 with ImageNet normalization.
    Option 2: RandomCrop 32x32 with padding and CIFAR-10 normalization.

    Args:
        transfer_cfg: TransferConfig with option.
        training_cfg: TrainingConfig with batch_size and num_workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if transfer_cfg.option == 1:
        train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    train_ds = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=val_tf
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ==============================================================================
#  Model builders
# ==============================================================================

def build_model_option1(num_classes: int = 10) -> nn.Module:
    """
    Load pretrained ResNet-18, freeze all layers, replace the FC head.

    Args:
        num_classes: Number of output classes. Default 10 for CIFAR-10.

    Returns:
        ResNet-18 with frozen backbone and fresh classification head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("  Option 1 | Trainable: " + str(trainable) + " / " + str(total) + " params")
    return model


def build_model_option2(num_classes: int = 10) -> nn.Module:
    """
    Load pretrained ResNet-18, adapt for 32x32 input, unfreeze all layers.

    Args:
        num_classes: Number of output classes. Default 10 for CIFAR-10.

    Returns:
        Modified ResNet-18 with all parameters trainable.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("  Option 2 | Trainable: " + str(trainable) + " / " + str(total) + " params")
    return model


# ==============================================================================
#  Experiment Runner
# ==============================================================================

def run_one_option(
    transfer_cfg: TransferConfig,
    training_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """
    Run a single transfer learning experiment for one option.

    Builds the model and loaders, trains for training_cfg.epoch epochs,
    saves the best checkpoint, loss curve, and parameters to disk.

    Results saved to:
        results/transfer/transfer_resize/      (option 1)
        results/transfer/transfer_layerchange/ (option 2)

    Args:
        transfer_cfg: TransferConfig with option.
        training_cfg: TrainingConfig with epoch, learning_rate, etc.
        device:       Compute device.
    """
    set_seed(training_cfg.seed)

    folder_name = "transfer_resize" if transfer_cfg.option == 1 else "transfer_layerchange"
    save_dir    = os.path.join("./results/transfer", folder_name)
    os.makedirs(save_dir, exist_ok=True)

    if transfer_cfg.option == 1:
        model = build_model_option1().to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
    else:
        model = build_model_option2().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )

    train_loader, val_loader = get_loaders(transfer_cfg, training_cfg)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_weights = None
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print("\n" + "="*55)
    print(
        "  Transfer Learning Option " + str(transfer_cfg.option) +
        " | " + str(training_cfg.epoch) + " epochs" +
        " | device=" + str(device)
    )
    print("="*55)

    for epoch in range(1, training_cfg.epoch + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch).zfill(2) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc=" + str(round(tr_acc, 4)) +
            "  val_loss=" + str(round(val_loss, 4)) +
            "  val_acc=" + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(save_dir, "model.pth"))
            print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

    print("\n  Best val accuracy: " + str(round(best_acc, 4)))

    save_results(
        params = {**asdict(transfer_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir = save_dir,
        name = folder_name,
    )


def run_transfer(params: Namespace) -> None:
    """
    Run all requested transfer learning experiments.

    Reads --tl_option from params and runs one or both options.
    Each option saves its results to its own subfolder.

    Args:
        params: Parsed Namespace object from get_params().
    """
    transfer_configs = get_transfer_configs(params)
    training_cfg = get_training_configs(params)
    device = get_device()

    print("Task   : Transfer Learning (CIFAR-10)")
    print("Device : " + str(device))

    for transfer_cfg in transfer_configs:
        run_one_option(transfer_cfg, training_cfg, device)

    print("\nTransfer learning complete.")
    print("Results saved to: ./results/transfer/")
