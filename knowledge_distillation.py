import os
from copy import deepcopy
from dataclasses import asdict
from typing import List, Optional, Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from test import validate
from train import train_one_epoch
from models.SimpleCNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from parameters import KDConfig, TrainingConfig, get_kd_config, get_training_configs
from auxillary import set_seed, get_device, save_results
from loss_functions import *

try:
    from ptflops import get_model_complexity_info
    _PTFLOPS = True
except ImportError:
    _PTFLOPS = False


# ==============================================================================
#  Dataset Stats
# ==============================================================================

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


# ==============================================================================
#  Data Loading
# ==============================================================================

def get_loaders(
    training_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train and validation DataLoaders.

    Args:
        kd_cfg:       KDConfig with data_dir.
        training_cfg: TrainingConfig with batch_size and num_workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
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
#  FLOPs reporting
# ==============================================================================

def report_flops(model: nn.Module, input_shape: Tuple, name: str) -> None:
    """
    Print MACs and parameter count for a model using ptflops.

    Args:
        model:       Network to profile.
        input_shape: Input shape excluding batch dim, e.g. (3, 32, 32).
        name:        Display name for the printout.
    """
    if not _PTFLOPS:
        print("  [FLOPs] Skipped for " + name + " -- pip install ptflops")
        return
    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print("\n  -- " + name + " --")
    print("     MACs   : " + str(macs))
    print("     Params : " + str(params))


# ==============================================================================
#  Individual experiment runners
# ==============================================================================

def run_exp1(
    kd_cfg: KDConfig,
    training_cfg: TrainingConfig,
    device: torch.device,
) -> None:
    """
    Experiment 1: Train SimpleCNN from scratch with standard cross-entropy.

    Results saved to: results/kd/kd_simplecnn_scratch/

    Args:
        kd_cfg:       KDConfig.
        training_cfg: TrainingConfig with epoch, learning_rate, etc.
        device:       Compute device.
    """
    save_dir = "./results/kd/kd_simplecnn_scratch"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(training_cfg.seed)

    model     = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epoch
    )
    train_loader, val_loader = get_loaders(training_cfg)

    best_acc = 0.0
    best_weights = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    print("\n" + "="*55)
    print("  Exp 1: SimpleCNN from scratch | " + str(training_cfg.epoch) + " epochs")
    print("="*55)

    for epoch in range(1, training_cfg.epoch + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
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
        params = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir = save_dir,
        name = "kd_simplecnn_scratch",
    )


def run_exp2(
    kd_cfg: KDConfig,
    training_cfg: TrainingConfig,
    device: torch.device,
) -> Tuple[str, str]:
    """
    Experiment 2: Train ResNet-18 with and without label smoothing.

    Both models are saved in the same folder with distinct names.
    Returns paths to both checkpoints so Exp 3 and 4 can select the best.

    Results saved to: results/kd/kd_resnet/

    Args:
        kd_cfg:       KDConfig with smoothing value.
        training_cfg: TrainingConfig with epoch, learning_rate, etc.
        device:       Compute device.

    Returns:
        Tuple of (path_no_smoothing, path_smoothing) checkpoint file paths.
    """
    save_dir = "./results/kd/kd_resnet"
    os.makedirs(save_dir, exist_ok=True)

    path_no_ls = os.path.join(save_dir, "model_no_ls.pth")
    path_ls    = os.path.join(save_dir, "model_ls.pth")

    for use_ls in [False, True]:
        set_seed(training_cfg.seed)
        label     = "with label smoothing" if use_ls else "no label smoothing"
        suffix    = "ls" if use_ls else "no_ls"
        model     = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
        criterion = (
            LabelSmoothingLoss(10, kd_cfg.smoothing)
            if use_ls else nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_cfg.learning_rate,
            weight_decay=training_cfg.weight_decay,
        )
        train_loader, val_loader = get_loaders(training_cfg)

        best_acc = 0.0
        best_weights = None
        train_losses: List[float] = []
        val_losses: List[float] = []

        print("\n" + "="*55)
        print("  Exp 2: ResNet-18 " + label + " | " + str(training_cfg.epoch) + " epochs")
        print("="*55)

        for epoch in range(1, training_cfg.epoch + 1):
            tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            print(
                "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
                "  train_loss=" + str(round(tr_loss, 4)) +
                "  train_acc=" + str(round(tr_acc, 4)) +
                "  val_loss=" + str(round(val_loss, 4)) +
                "  val_acc=" + str(round(val_acc, 4))
            )

            if val_acc > best_acc:
                best_acc     = val_acc
                best_weights = deepcopy(model.state_dict())
                ckpt_path    = os.path.join(save_dir, "model_" + suffix + ".pth")
                torch.save(best_weights, ckpt_path)
                print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

        print("\n  Best val accuracy (" + label + "): " + str(round(best_acc, 4)))
        save_results(
            params = {**asdict(kd_cfg), **asdict(training_cfg)},
            tloss_list = train_losses,
            vloss_list = val_losses,
            save_dir = save_dir,
            name = "kd_resnet_" + suffix,
        )

    return path_no_ls, path_ls


def run_exp3(
    kd_cfg: KDConfig,
    training_cfg: TrainingConfig,
    device: torch.device,
    teacher_path: str,
) -> None:
    """
    Experiment 3: Train SimpleCNN using ResNet-18 teacher (Hinton KD).

    Results saved to: results/kd/kd_simplecnn_kd/

    Args:
        kd_cfg:       KDConfig with temperature and alpha.
        training_cfg: TrainingConfig with epoch, learning_rate, etc.
        device:       Compute device.
        teacher_path: Path to the best ResNet-18 checkpoint from Exp 2.
    """
    save_dir = "./results/kd/kd_simplecnn_kd"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(training_cfg.seed)

    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("  Teacher loaded: " + teacher_path)

    student   = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    train_loader, val_loader = get_loaders(training_cfg)

    best_acc = 0.0
    best_weights = None
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print("\n" + "="*55)
    print(
        "  Exp 3: Hinton KD ResNet -> SimpleCNN | " +
        str(training_cfg.epoch) + " epochs"
    )
    print(
        "  T=" + str(kd_cfg.temperature) +
        "  alpha=" + str(kd_cfg.alpha)
    )
    print("="*55)

    for epoch in range(1, training_cfg.epoch + 1):
        tr_loss, tr_acc   = train_one_epoch(
            student, train_loader, optimizer, hinton_kd_loss, device, teacher=teacher, kd_cfg= kd_cfg
        )
        val_loss, val_acc = validate(student, val_loader, hinton_kd_loss, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc=" + str(round(tr_acc, 4)) +
            "  val_loss=" + str(round(val_loss, 4)) +
            "  val_acc=" + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = deepcopy(student.state_dict())
            torch.save(best_weights, os.path.join(save_dir, "model.pth"))
            print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

    print("\n  Best val accuracy: " + str(round(best_acc, 4)))
    report_flops(teacher, (3, 32, 32), "ResNet-18 (teacher)")
    report_flops(student, (3, 32, 32), "SimpleCNN (student)")

    save_results(
        params = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir = save_dir,
        name = "kd_simplecnn_kd",
    )


def run_exp4(
    kd_cfg: KDConfig,
    training_cfg: TrainingConfig,
    device: torch.device,
    teacher_path: str,
) -> None:
    """
    Experiment 4: Train MobileNetV2 using ResNet-18 teacher (modified KD).

    Uses modified KD loss where the teacher provides confidence only for
    the true class and remaining probability is spread uniformly.

    Results saved to: results/kd/kd_mobilenet/

    Args:
        kd_cfg:       KDConfig with temperature and alpha.
        training_cfg: TrainingConfig with epoch, learning_rate, etc.
        device:       Compute device.
        teacher_path: Path to the best ResNet-18 checkpoint from Exp 2.
    """
    save_dir = "./results/kd/kd_mobilenet"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(training_cfg.seed)

    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    print("  Teacher loaded: " + teacher_path)

    student = MobileNetV2(num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    train_loader, val_loader = get_loaders(training_cfg)

    best_acc = 0.0
    best_weights = None
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print("\n" + "="*55)
    print(
        "  Exp 4: Modified KD ResNet -> MobileNetV2 | " +
        str(training_cfg.epoch) + " epochs"
    )
    print(
        "  T=" + str(kd_cfg.temperature) +
        "  alpha=" + str(kd_cfg.alpha)
    )
    print("="*55)

    for epoch in range(1, training_cfg.epoch + 1):
        tr_loss, tr_acc = train_one_epoch(
            student, train_loader, optimizer, modified_kd_loss, device, teacher=teacher, kd_cfg=kd_cfg
        )
        val_loss, val_acc = validate(student, val_loader, modified_kd_loss, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc=" + str(round(tr_acc, 4)) +
            "  val_loss=" + str(round(val_loss, 4)) +
            "  val_acc=" + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = deepcopy(student.state_dict())
            torch.save(best_weights, os.path.join(save_dir, "model.pth"))
            print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

    print("\n  Best val accuracy: " + str(round(best_acc, 4)))
    report_flops(teacher, (3, 32, 32), "ResNet-18   (teacher)")
    report_flops(student, (3, 32, 32), "MobileNetV2 (student)")

    save_results(
        params = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir = save_dir,
        name = "kd_mobilenet",
    )


# ==============================================================================
#  Main dispatcher
# ==============================================================================

def run_distillation(params: Namespace) -> None:
    """
    Run all requested knowledge distillation experiments.

    Args:
        params: Parsed Namespace object from get_params().
    """
    kd_cfg = get_kd_config(params)
    training_cfg = get_training_configs(params)
    device = get_device()

    print("Task   : Knowledge Distillation (CIFAR-10)")
    print("Device : " + str(device))

    experiments  = [kd_cfg.experiment] if kd_cfg.experiment else [1, 2, 3, 4]
    teacher_path: Optional[str] = None

    if 1 in experiments:
        run_exp1(kd_cfg, training_cfg, device)

    if 2 in experiments:
        path_no_ls, path_ls = run_exp2(kd_cfg, training_cfg, device)

        # Auto-select best teacher checkpoint for Exp 3 and 4
        m1 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
        m1.load_state_dict(torch.load(path_no_ls, map_location=device))
        m2 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
        m2.load_state_dict(torch.load(path_ls, map_location=device))

        _, val_loader = get_loaders(training_cfg)
        _, acc_no_ls = validate(m1, val_loader, nn.CrossEntropyLoss(), device)
        _, acc_ls = validate(m2, val_loader, LabelSmoothingLoss(10, kd_cfg.smoothing),  device)

        if acc_ls > acc_no_ls:
            teacher_path = path_ls
            print("\n[Teacher] Using model_ls.pth (val_acc=" + str(round(acc_ls, 4)) + ")")
        else:
            teacher_path = path_no_ls
            print("\n[Teacher] Using model_no_ls.pth (val_acc=" + str(round(acc_no_ls, 4)) + ")")

    if 3 in experiments:
        if teacher_path is None:
            default = "./results/kd/kd_resnet/model_no_ls.pth"
            if not os.path.exists(default):
                raise RuntimeError(
                    "Teacher checkpoint not found. Run Exp 2 first."
                )
            teacher_path = default
            print("\n[Teacher] Using default path: " + teacher_path)
        run_exp3(kd_cfg, training_cfg, device, teacher_path)

    if 4 in experiments:
        if teacher_path is None:
            default = "./results/kd/kd_resnet/model_no_ls.pth"
            if not os.path.exists(default):
                raise RuntimeError(
                    "Teacher checkpoint not found. Run Exp 2 first."
                )
            teacher_path = default
            print("\n[Teacher] Using default path: " + teacher_path)
        run_exp4(kd_cfg, training_cfg, device, teacher_path)

    print("\nKnowledge distillation complete.")
    print("Results saved to: ./results/kd/")
