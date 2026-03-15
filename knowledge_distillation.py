"""
knowledge_distillation.py
--------------------------
Everything related to knowledge distillation on CIFAR-10 (HW1b Part B).

Four experiments run in sequence or individually via --kd_experiment:

    Exp 1 (kd_simplecnn_scratch):
        Train SimpleCNN from scratch with standard cross-entropy.

    Exp 2 (kd_resnet):
        Train ResNet-18 with and without label smoothing.
        Both models saved in the same folder with distinct names.

    Exp 3 (kd_simplecnn_kd):
        Train SimpleCNN using the best ResNet-18 from Exp 2 as teacher.
        Uses Hinton knowledge distillation loss.

    Exp 4 (kd_mobilenet):
        Train MobileNetV2 using the best ResNet-18 from Exp 2 as teacher.
        Uses modified KD loss where only the true class gets the teacher
        probability and remaining mass is spread uniformly.

Results saved to:
    results/kd/kd_simplecnn_scratch/
    results/kd/kd_resnet/
    results/kd/kd_simplecnn_kd/
    results/kd/kd_mobilenet/

Usage
-----
    python main.py --task distillation --epoch 30
    python main.py --task distillation --kd_experiment 3
"""

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

from models.SimpleCNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from parameters import KDConfig, TrainingConfig, get_kd_config, get_training_configs
from auxillary import set_seed, get_device, save_results

try:
    from ptflops import get_model_complexity_info
    _PTFLOPS = True
except ImportError:
    _PTFLOPS = False


# ==============================================================================
#  Dataset statistics
# ==============================================================================

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


# ==============================================================================
#  Data loading
# ==============================================================================

def get_loaders(
    training_cfg: TrainingConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train and validation DataLoaders.

    Applies RandomCrop and RandomHorizontalFlip for training augmentation.
    Validation uses only ToTensor and normalisation.

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
#  Loss functions
# ==============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    The true class gets probability (1 - smoothing) and each other class
    gets smoothing / (C - 1). Prevents the model from becoming overconfident.

    Reference: Szegedy et al. (2016), Rethinking the Inception Architecture.

    Args:
        num_classes: Total number of output classes C.
        smoothing:   Smoothing factor in range [0, 1). 0 = standard CE.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothed cross-entropy loss.

        Args:
            logits:  Raw model outputs of shape (N, C).
            targets: True class indices of shape (N,).

        Returns:
            Scalar smoothed cross-entropy loss.
        """
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)
        soft = torch.full_like(logits, smooth_val)
        soft.scatter_(1, targets.unsqueeze(1), confidence)
        return -(soft * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def hinton_kd_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Hinton knowledge distillation loss.

    Combines soft loss (KL divergence between temperature-scaled teacher
    and student) with hard loss (standard cross-entropy against true labels).

    Total = alpha * T^2 * KL(teacher || student) + (1 - alpha) * CE

    Reference: Hinton et al. (2015), Distilling the Knowledge in a Neural Network.

    Args:
        s_logits:    Student logits of shape (N, C).
        t_logits:    Teacher logits of shape (N, C).
        labels:      True class indices of shape (N,).
        temperature: Softening temperature T.
        alpha:       Weight on the soft-target loss.

    Returns:
        Scalar combined KD loss.
    """
    soft_s = F.log_softmax(s_logits / temperature, dim=1)
    soft_t = F.softmax(t_logits    / temperature, dim=1)
    kd     = F.kl_div(soft_s, soft_t, reduction="batchmean") * (temperature ** 2)
    ce     = F.cross_entropy(s_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce


def modified_kd_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Modified KD loss using teacher confidence on the true class only.

    For each sample with true class y:
        target[y]    = softmax(teacher / T)[y]
        target[j!=y] = (1 - target[y]) / (C - 1)

    Encodes per-example difficulty while keeping wrong-class targets uniform.

    Args:
        s_logits:    Student logits of shape (N, C).
        t_logits:    Teacher logits of shape (N, C).
        labels:      True class indices of shape (N,).
        temperature: Softening temperature T applied to teacher.
        alpha:       Weight on the soft-target loss.

    Returns:
        Scalar combined modified KD loss.
    """
    N, C    = s_logits.shape
    t_probs = F.softmax(t_logits / temperature, dim=1)
    p_true  = t_probs.gather(1, labels.unsqueeze(1))

    uniform_other = (1.0 - p_true) / (C - 1)
    soft          = uniform_other.expand(N, C).clone()
    soft.scatter_(1, labels.unsqueeze(1), p_true)

    log_s = F.log_softmax(s_logits / temperature, dim=1)
    kd    = F.kl_div(log_s, soft, reduction="batchmean") * (temperature ** 2)
    ce    = F.cross_entropy(s_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce


# ==============================================================================
#  Training and validation
# ==============================================================================

def train_standard(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    One epoch of standard or label-smoothed training.

    Args:
        model:     Neural network in training mode.
        loader:    Training DataLoader.
        optimizer: Optimiser for weight updates.
        criterion: Loss function (CE or LabelSmoothingLoss).
        device:    Compute device.

    Returns:
        Tuple of (average_loss, accuracy) for this epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
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


def train_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kd_cfg: KDConfig,
    device: torch.device,
    use_modified: bool,
) -> Tuple[float, float]:
    """
    One epoch of knowledge distillation training.

    The teacher is always frozen in eval mode. Only the student receives
    gradient updates.

    Args:
        student:      Student model in training mode.
        teacher:      Frozen teacher model in eval mode.
        loader:       Training DataLoader.
        optimizer:    Student optimiser.
        kd_cfg:       KDConfig with temperature and alpha.
        device:       Compute device.
        use_modified: Use modified KD loss if True, Hinton loss if False.

    Returns:
        Tuple of (average_loss, accuracy) for the student this epoch.
    """
    student.train()
    teacher.eval()
    loss_fn = modified_kd_loss if use_modified else hinton_kd_loss
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        s_logits = student(imgs)
        with torch.no_grad():
            t_logits = teacher(imgs)
        loss = loss_fn(
            s_logits, t_logits, labels,
            kd_cfg.temperature, kd_cfg.alpha,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += s_logits.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set using standard CE loss.

    Args:
        model:  Neural network in eval mode.
        loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy) over the validation set.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = F.cross_entropy(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


# ==============================================================================
#  FLOPs reporting
# ==============================================================================

def report_flops(model: nn.Module, input_shape: Tuple, name: str) -> None:
    """
    Print MACs and parameter count for a model using ptflops.

    Skips silently if ptflops is not installed.

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

    best_acc               = 0.0
    best_weights           = None
    train_losses: List[float] = []
    val_losses:   List[float] = []

    print("\n" + "="*55)
    print("  Exp 1: SimpleCNN from scratch | " + str(training_cfg.epoch) + " epochs")
    print("="*55)

    for epoch in range(1, training_cfg.epoch + 1):
        tr_loss, tr_acc   = train_standard(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc="  + str(round(tr_acc, 4)) +
            "  val_loss="   + str(round(val_loss, 4)) +
            "  val_acc="    + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(save_dir, "model.pth"))
            print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

    print("\n  Best val accuracy: " + str(round(best_acc, 4)))
    save_results(
        params     = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir   = save_dir,
        name       = "kd_simplecnn_scratch",
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg.epoch
        )
        train_loader, val_loader = get_loaders(training_cfg)

        best_acc               = 0.0
        best_weights           = None
        train_losses: List[float] = []
        val_losses:   List[float] = []

        print("\n" + "="*55)
        print("  Exp 2: ResNet-18 " + label + " | " + str(training_cfg.epoch) + " epochs")
        print("="*55)

        for epoch in range(1, training_cfg.epoch + 1):
            tr_loss, tr_acc   = train_standard(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, device)
            scheduler.step()

            train_losses.append(tr_loss)
            val_losses.append(val_loss)

            print(
                "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
                "  train_loss=" + str(round(tr_loss, 4)) +
                "  train_acc="  + str(round(tr_acc, 4)) +
                "  val_loss="   + str(round(val_loss, 4)) +
                "  val_acc="    + str(round(val_acc, 4))
            )

            if val_acc > best_acc:
                best_acc     = val_acc
                best_weights = deepcopy(model.state_dict())
                ckpt_path    = os.path.join(save_dir, "model_" + suffix + ".pth")
                torch.save(best_weights, ckpt_path)
                print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

        print("\n  Best val accuracy (" + label + "): " + str(round(best_acc, 4)))
        save_results(
            params     = {**asdict(kd_cfg), **asdict(training_cfg)},
            tloss_list = train_losses,
            vloss_list = val_losses,
            save_dir   = save_dir,
            name       = "kd_resnet_" + suffix,
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

    best_acc               = 0.0
    best_weights           = None
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
        tr_loss, tr_acc   = train_kd(
            student, teacher, train_loader, optimizer,
            kd_cfg, device, use_modified=False,
        )
        val_loss, val_acc = validate(student, val_loader, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc="  + str(round(tr_acc, 4)) +
            "  val_loss="   + str(round(val_loss, 4)) +
            "  val_acc="    + str(round(val_acc, 4))
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
        params     = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir   = save_dir,
        name       = "kd_simplecnn_kd",
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

    student   = MobileNetV2(num_classes=10).to(device)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.epoch
    )
    train_loader, val_loader = get_loaders(training_cfg)

    best_acc               = 0.0
    best_weights           = None
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
        tr_loss, tr_acc   = train_kd(
            student, teacher, train_loader, optimizer,
            kd_cfg, device, use_modified=True,
        )
        val_loss, val_acc = validate(student, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  [" + str(epoch).zfill(2) + "/" + str(training_cfg.epoch) + "]" +
            "  train_loss=" + str(round(tr_loss, 4)) +
            "  train_acc="  + str(round(tr_acc, 4)) +
            "  val_loss="   + str(round(val_loss, 4)) +
            "  val_acc="    + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = deepcopy(student.state_dict())
            torch.save(best_weights, os.path.join(save_dir, "model.pth"))
            print("    Checkpoint saved (val_acc=" + str(round(best_acc, 4)) + ")")

    print("\n  Best val accuracy: " + str(round(best_acc, 4)))
    report_flops(teacher, (3, 32, 32), "ResNet-18   (teacher)")
    report_flops(student, (3, 32, 32), "MobileNetV2 (student)")

    save_results(
        params     = {**asdict(kd_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir   = save_dir,
        name       = "kd_mobilenet",
    )


# ==============================================================================
#  Main dispatcher
# ==============================================================================

def run_distillation(params: Namespace) -> None:
    """
    Run all requested knowledge distillation experiments.

    Reads --kd_experiment from params. If not set, runs all four in order.
    Experiments 3 and 4 require a teacher checkpoint. If running all four,
    the best model from Exp 2 is selected automatically. If running 3 or 4
    standalone, the teacher path must be provided via --kd_teacher_path
    or the checkpoint from Exp 2 must already exist at the default path.

    Args:
        params: Parsed Namespace object from get_params().
    """
    kd_cfg       = get_kd_config(params)
    training_cfg = get_training_configs(params)
    device       = get_device()

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
        _, acc_no_ls  = validate(m1, val_loader, device)
        _, acc_ls     = validate(m2, val_loader, device)

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
