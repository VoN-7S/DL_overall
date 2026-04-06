import os
from copy import deepcopy
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from argparse import Namespace
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from test import validate
from train import train_one_epoch
from models.SimpleCNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from parameters import KDConfig, TrainingConfig, get_kd_config, get_training_configs
from auxillary import set_seed, get_device, save_results
from loss_functions import *
from adversarial import pgd_attack

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

    model = SimpleCNN(num_classes=10).to(device)
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
            best_acc = val_acc
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
        label = "with label smoothing" if use_ls else "no label smoothing"
        suffix = "ls" if use_ls else "no_ls"
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
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
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
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
                best_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                ckpt_path = os.path.join(save_dir, "model_" + suffix + ".pth")
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

    student = SimpleCNN(num_classes=10).to(device)
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
        tr_loss, tr_acc = train_one_epoch(
            student, train_loader, optimizer, hinton_kd_loss, device, teacher=teacher, kd_cfg= kd_cfg
        )
        val_loss, val_acc = validate(student, val_loader, hinton_kd_loss, device, teacher=teacher, kd_cfg = kd_cfg)

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
        val_loss, val_acc = validate(student, val_loader, modified_kd_loss, device, teacher=teacher, kd_cfg=kd_cfg)

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

def run_exp4_augmix(kd_cfg: KDConfig, training_cfg: TrainingConfig, device: torch.device) -> None:
    """
    Task 4: Repeat KD experiments from HW1b using the AugMix teacher.
 
    Calls run_exp3 and run_exp4 from knowledge_distillation.py directly,
    passing the AugMix checkpoint as teacher_path.  Results land in:
        results/kd/kd_simplecnn_kd/   (same folder as HW1b Exp 3)
        results/kd/kd_mobilenet/       (same folder as HW1b Exp 4)
    so the two runs overwrite each other — run HW1b Exp 3/4 before Task 4
    if you need to preserve those results.
 
    Args:
        kd_cfg:       KDConfig (temperature, alpha, smoothing).
        training_cfg: TrainingConfig (epochs, lr, batch_size, etc.).
        device:       Compute device.
    """
    augmix_ckpt = "./results/hw2/robustness_augmix/model.pth"
    if not os.path.exists(augmix_ckpt):
        raise FileNotFoundError(
            f"AugMix teacher checkpoint not found: {augmix_ckpt}\n"
            "Run Task 2 first:  python main.py --task robustness --hw2_task task2"
        )
 
    print("\n" + "=" * 55)
    print("  Task 4: Hinton KD — AugMix teacher → SimpleCNN")
    print("=" * 55)
    run_exp3(kd_cfg, training_cfg, device, teacher_path=augmix_ckpt)
 
    print("\n" + "=" * 55)
    print("  Task 4: Modified KD — AugMix teacher → MobileNetV2")
    print("=" * 55)
    run_exp4(kd_cfg, training_cfg, device, teacher_path=augmix_ckpt)
 
 
# ==============================================================================
#  Task 5: Adversarial transferability
# ==============================================================================
 
 
@torch.no_grad()
def _eval_on_dataset(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Evaluate model accuracy on a pre-built tensor dataset.
 
    Args:
        model:      Model in eval mode.
        imgs:       Image tensor (N, C, H, W), already normalized.
        labels:     Label tensor (N,).
        batch_size: Mini-batch size for inference.
        device:     Compute device.
 
    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=False)
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
        n       += y.size(0)
    return correct / n
 
 
def run_task5(kd_cfg: KDConfig, training_cfg: TrainingConfig, device: torch.device) -> None:
    """
    Task 5: Adversarial transferability — teacher-crafted samples tested on student.
 
    Generates PGD20 L∞ (ε = 4/255) adversarial examples using the teacher's
    gradients, then evaluates both teacher and student on those examples.
    A large accuracy drop on the student (despite using teacher gradients) means
    the student learned similar decision boundaries.
 
    Covers two pairs matching HW1b experiments:
        Vanilla teacher (HW1b)  → SimpleCNN student   (Exp 3)
        AugMix teacher (Task 2) → MobileNetV2 student (Exp 4)
 
    Results saved to: results/hw2/distillation/task5_transferability.json
 
    Args:
        kd_cfg:       KDConfig (kept for signature consistency).
        training_cfg: TrainingConfig (batch_size, num_workers).
        device:       Compute device.
    """
    save_dir = "./results/hw2/distillation"
    os.makedirs(save_dir, exist_ok=True)
 
    linf_eps = 4.0 / 255.0
    alpha    = linf_eps / 4.0
    steps    = 20
 
    # Standard CIFAR-10 test loader (same transforms as knowledge_distillation.py)
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=val_tf
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=True,
    )
 
    # Checkpoint paths
    vanilla_teacher_ckpt = "./results/transfer/transfer_layerchange/model.pth"
    augmix_teacher_ckpt  = "./results/hw2/robustness_augmix/model.pth"
    simplecnn_ckpt       = "./results/kd/kd_simplecnn_kd/model.pth"
    mobilenet_ckpt       = "./results/kd/kd_mobilenet/model.pth"
 
    results = {}
 
    pairs = [
        (
            "vanilla_teacher_to_SimpleCNN",
            vanilla_teacher_ckpt, simplecnn_ckpt,
            "torchvision_resnet18", "simplecnn",
        ),
        (
            "augmix_teacher_to_MobileNetV2",
            augmix_teacher_ckpt, mobilenet_ckpt,
            "scratch_resnet18", "mobilenetv2",
        ),
    ]
 
    for tag, t_ckpt, s_ckpt, t_arch, s_arch in pairs:
        print(f"\n  Task 5: {tag}")
 
        # Skip gracefully if checkpoints are missing
        missing = [(p, lbl) for p, lbl in [(t_ckpt, "teacher"), (s_ckpt, "student")]
                   if not os.path.exists(p)]
        if missing:
            for p, lbl in missing:
                print(f"  [SKIP] {lbl} checkpoint not found: {p}")
            continue
 
        # Load teacher
        if t_arch == "torchvision_resnet18":
            import torchvision.models as tvm
            teacher = tvm.resnet18(weights=None)
            teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            teacher.maxpool = nn.Identity()
            teacher.fc      = nn.Linear(teacher.fc.in_features, 10)
        else:
            teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        teacher.load_state_dict(torch.load(t_ckpt, map_location=device))
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
 
        # Load student
        student = SimpleCNN(num_classes=10) if s_arch == "simplecnn" else MobileNetV2(num_classes=10)
        student.load_state_dict(torch.load(s_ckpt, map_location=device))
        student.to(device).eval()
 
        # Collect full test set into tensors so we can reuse the same data
        all_imgs, all_labels = [], []
        for imgs_b, labels_b in test_loader:
            all_imgs.append(imgs_b)
            all_labels.append(labels_b)
        all_imgs   = torch.cat(all_imgs)
        all_labels = torch.cat(all_labels)
 
        # Generate adversarial samples batch by batch using teacher gradients
        print(f"  Generating PGD20 L∞ (ε={linf_eps:.5f}) adversarial samples on teacher...")
        adv_imgs_list = []
        adv_loader = DataLoader(
            TensorDataset(all_imgs, all_labels),
            batch_size=training_cfg.batch_size,
            shuffle=False,
        )
        for imgs_b, labels_b in adv_loader:
            adv_b = pgd_attack(
                teacher, imgs_b, labels_b,
                linf_eps, alpha, steps, "linf", device,
            )
            adv_imgs_list.append(adv_b.cpu())
        adv_imgs = torch.cat(adv_imgs_list)
 
        # Evaluate both models on clean and adversarial samples
        clean_acc_t = _eval_on_dataset(teacher, all_imgs, all_labels, training_cfg.batch_size, device)
        clean_acc_s = _eval_on_dataset(student, all_imgs, all_labels, training_cfg.batch_size, device)
        adv_acc_t   = _eval_on_dataset(teacher, adv_imgs, all_labels, training_cfg.batch_size, device)
        adv_acc_s   = _eval_on_dataset(student, adv_imgs, all_labels, training_cfg.batch_size, device)
 
        print(f"  Teacher — clean: {clean_acc_t:.4f}  adv: {adv_acc_t:.4f}  drop: {clean_acc_t - adv_acc_t:.4f}")
        print(f"  Student — clean: {clean_acc_s:.4f}  adv: {adv_acc_s:.4f}  drop: {clean_acc_s - adv_acc_s:.4f}")
 
        results[tag] = {
            "teacher_clean_acc": round(clean_acc_t, 4),
            "student_clean_acc": round(clean_acc_s, 4),
            "teacher_adv_acc":   round(adv_acc_t,   4),
            "student_adv_acc":   round(adv_acc_s,   4),
            "teacher_acc_drop":  round(clean_acc_t - adv_acc_t, 4),
            "student_acc_drop":  round(clean_acc_s - adv_acc_s, 4),
        }
 
    out_path = os.path.join(save_dir, "task5_transferability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Task 5 results saved → {out_path}")


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


def run_hw2_distillation(params: Namespace) -> None:
    """
    Dispatcher for HW2 distillation tasks (Tasks 4 and 5).
 
    Args:
        params: Parsed argparse Namespace from get_params().
    """
    kd_cfg       = get_kd_config(params)
    training_cfg = get_training_configs(params)
    device       = get_device()
 
    print("Task   : HW2 Distillation (Tasks 4 & 5)")
    print(f"Device : {device}")
 
    if params.hw2_task in ("task4", "both"):
        run_exp4_augmix(kd_cfg, training_cfg, device)
 
    if params.hw2_task in ("task5", "both"):
        run_task5(kd_cfg, training_cfg, device)
 
    print("\nHW2 distillation complete.")
    print("Results saved under: ./results/kd/  and  ./results/hw2/distillation/")