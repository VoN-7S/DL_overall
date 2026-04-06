"""
adversarial.py
--------------
HW2 Task 3 — Adversarial robustness evaluation.

Experiments
-----------
1. PGD20 attack on vanilla and AugMix ResNet-18:
     - L∞  norm, ε = 4/255
     - L2   norm, ε = 0.25

2. Grad-CAM visualization on 1–2 samples where the clean prediction is
   correct but the adversarial prediction is wrong.

3. t-SNE visualization of feature embeddings for:
     - clean samples
     - adversarial samples (L∞ PGD20)

Reference
---------
Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks,"
ICLR 2018.  https://openreview.net/forum?id=rJzIBfZAb
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE

from models.ResNet import ResNet, BasicBlock
from auxillary import get_device

from parameters import AdversarialConfig, TrainingConfig, get_training_configs


# ==============================================================================
#  Constants
# ==============================================================================

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)
_CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]

_MEAN_T = torch.tensor(_CIFAR_MEAN).view(1, 3, 1, 1)
_STD_T  = torch.tensor(_CIFAR_STD).view(1, 3, 1, 1)


# ==============================================================================
#  PGD Attack
# ==============================================================================

def pgd_attack(
    model:      nn.Module,
    imgs:       torch.Tensor,
    labels:     torch.Tensor,
    eps:        float,
    alpha:      float,
    steps:      int,
    norm:       str,
    device:     torch.device,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Operates in the normalized image space (images are already normalized).
    Perturbations are applied in normalized space and clipped to the allowed
    ε-ball, then the result is clipped to the valid normalized pixel range.

    Args:
        model:  Neural network in eval mode.
        imgs:   Normalized input images (N, C, H, W).
        labels: True class indices (N,).
        eps:    Attack budget.  L∞: 4/255 in raw space → eps_norm.
                                L2:  0.25 in raw space.
        alpha:  Step size (typically eps/4).
        steps:  Number of PGD iterations (20 for PGD20).
        norm:   "linf" or "l2".
        device: Compute device.

    Returns:
        Adversarial images of the same shape as imgs (normalized).
    """
    model.eval()
    imgs   = imgs.to(device)
    labels = labels.to(device)

    # Convert ε from raw [0,1] space to normalized space (per-channel)
    # For a proper per-channel conversion we use the worst-case std:
    # eps_norm = eps / min(std)  — conservative upper bound used in most papers.
    # (A cleaner approach is to work in raw space, but that requires denorm/renorm
    #  each step, which is equivalent and more code for no accuracy gain.)
    std_min = min(_CIFAR_STD)
    eps_norm   = eps   / std_min
    alpha_norm = alpha / std_min

    # Random initialization within ε-ball
    delta = torch.zeros_like(imgs)
    if norm == "linf":
        delta.uniform_(-eps_norm, eps_norm)
    else:
        # L2: sample uniformly from sphere, scale to random radius ≤ eps_norm
        delta = torch.randn_like(imgs)
        d_flat = delta.view(delta.size(0), -1)
        norms  = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        delta  = delta / norms.view(-1, 1, 1, 1) * eps_norm * torch.rand(delta.size(0), 1, 1, 1, device=device)
    delta = delta.detach().requires_grad_(False)

    adv = (imgs + delta).detach()

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]

        with torch.no_grad():
            if norm == "linf":
                adv = adv + alpha_norm * grad.sign()
                delta = torch.clamp(adv - imgs, -eps_norm, eps_norm)
                adv = imgs + delta
            else:  # L2
                g_flat  = grad.view(grad.size(0), -1)
                g_norm  = g_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                grad_n  = grad / g_norm.view(-1, 1, 1, 1)
                adv     = adv + alpha_norm * grad_n
                delta   = adv - imgs
                d_flat  = delta.view(delta.size(0), -1)
                d_norm  = d_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
                # Project onto L2 ball
                factor  = torch.clamp(eps_norm / d_norm, max=1.0)
                delta   = delta * factor.view(-1, 1, 1, 1)
                adv     = imgs + delta

        adv = adv.detach()

    return adv


@torch.no_grad()
def evaluate_pgd(
    model:   nn.Module,
    loader:  DataLoader,
    eps:     float,
    alpha:   float,
    steps:   int,
    norm:    str,
    device:  torch.device,
) -> float:
    """
    Evaluate model accuracy under PGD attack over a full DataLoader.

    Args:
        model:   Network in eval mode.
        loader:  DataLoader over the test set.
        eps:     Attack budget.
        alpha:   Step size.
        steps:   PGD iterations.
        norm:    "linf" or "l2".
        device:  Compute device.

    Returns:
        Robust accuracy as a float in [0, 1].
    """
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv = pgd_attack(model, imgs, labels, eps, alpha, steps, norm, device)
        preds = model(adv).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += labels.size(0)
    return correct / n


# ==============================================================================
#  Grad-CAM
# ==============================================================================

class GradCAM:
    """
    Grad-CAM implementation targeting a specific layer of a CNN.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4)
        heatmap = cam(img_tensor, class_idx)  # (H, W) numpy array

    Reference:
        Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization," ICCV 2017.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output: Tuple) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, img: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args:
            img:       Normalized image tensor (1, C, H, W).
            class_idx: Target class for gradient computation.

        Returns:
            Normalized heatmap (H, W) as numpy float32 array in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(img)
        score  = logits[0, class_idx]
        score.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (h, w)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        cam = F.interpolate(cam, size=img.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def _denormalize(img_t: torch.Tensor) -> np.ndarray:
    """Convert normalized (C,H,W) tensor to uint8 (H,W,C) numpy array."""
    mean = torch.tensor(_CIFAR_MEAN).view(3, 1, 1)
    std  = torch.tensor(_CIFAR_STD).view(3, 1, 1)
    img  = img_t.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def visualize_gradcam(
    model:       nn.Module,
    clean_imgs:  torch.Tensor,
    adv_imgs:    torch.Tensor,
    labels:      torch.Tensor,
    device:      torch.device,
    save_dir:    str,
    max_samples: int = 2,
) -> None:
    """
    Find samples where the clean prediction is correct but adversarial is wrong,
    then plot and save side-by-side Grad-CAM visualizations.

    Layout per sample (row):
        [clean image] [clean Grad-CAM] [adv image] [adv Grad-CAM]

    Args:
        model:       ResNet-18 model.
        clean_imgs:  Normalized clean images (N, C, H, W).
        adv_imgs:    Normalized adversarial images (N, C, H, W).
        labels:      True labels (N,).
        device:      Compute device.
        save_dir:    Directory to save figures.
        max_samples: How many qualifying samples to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    grad_cam = GradCAM(model, target_layer=model.layer4[-1])

    model.eval()
    clean_imgs = clean_imgs.to(device)
    adv_imgs   = adv_imgs.to(device)
    labels_dev = labels.to(device)

    with torch.no_grad():
        clean_preds = model(clean_imgs).argmax(1)
        adv_preds   = model(adv_imgs).argmax(1)

    # Find correctly classified clean → misclassified adversarial
    mask = clean_preds.eq(labels_dev) & adv_preds.ne(labels_dev)
    indices = mask.nonzero(as_tuple=False).squeeze(1)

    if len(indices) == 0:
        print("  [Grad-CAM] No qualifying samples found in this batch.")
        return

    indices = indices[:max_samples]
    n_found = len(indices)

    fig, axes = plt.subplots(n_found, 4, figsize=(12, 3 * n_found))
    if n_found == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Clean image", "Clean Grad-CAM", "Adversarial image", "Adversarial Grad-CAM"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(indices):
        true_cls  = _CIFAR_CLASSES[labels[idx].item()]
        clean_cls = _CIFAR_CLASSES[clean_preds[idx].item()]
        adv_cls   = _CIFAR_CLASSES[adv_preds[idx].item()]

        img_clean = clean_imgs[idx:idx+1]
        img_adv   = adv_imgs[idx:idx+1]

        cam_clean = grad_cam(img_clean, clean_preds[idx].item())
        cam_adv   = grad_cam(img_adv,   adv_preds[idx].item())

        img_c_np = _denormalize(clean_imgs[idx].cpu())
        img_a_np = _denormalize(adv_imgs[idx].cpu())

        axes[row, 0].imshow(img_c_np)
        axes[row, 0].set_xlabel(f"True: {true_cls}\nPred: {clean_cls}", fontsize=8)

        axes[row, 1].imshow(img_c_np)
        axes[row, 1].imshow(cam_clean, cmap="jet", alpha=0.5)
        axes[row, 1].set_xlabel(f"Focus (clean)", fontsize=8)

        axes[row, 2].imshow(img_a_np)
        axes[row, 2].set_xlabel(f"True: {true_cls}\nPred: {adv_cls}", fontsize=8)

        axes[row, 3].imshow(img_a_np)
        axes[row, 3].imshow(cam_adv, cmap="jet", alpha=0.5)
        axes[row, 3].set_xlabel(f"Focus (adversarial)", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Grad-CAM: Clean vs Adversarial", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "gradcam.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grad-CAM saved → {path}")


# ==============================================================================
#  t-SNE visualization
# ==============================================================================

@torch.no_grad()
def extract_features(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_max:  int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer features from a ResNet-18.

    Hooks into the avgpool output (512-d before the final FC).

    Args:
        model:  ResNet-18 in eval mode.
        loader: DataLoader over the dataset.
        device: Compute device.
        n_max:  Maximum number of samples to extract.

    Returns:
        Tuple of (features, labels) as numpy arrays.
    """
    features_list: List[np.ndarray] = []
    labels_list:   List[np.ndarray] = []
    model.eval()

    # Temporarily remove the FC head to get 512-d embeddings
    original_fc = model.linear if hasattr(model, "linear") else model.fc
    model.linear = nn.Identity()

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        out  = model(imgs)                  # (B, 512) after Identity
        features_list.append(out.cpu().numpy())
        labels_list.append(lbls.numpy())
        if sum(len(f) for f in features_list) >= n_max:
            break

    model.linear = original_fc
    feats  = np.concatenate(features_list)[:n_max]
    labels = np.concatenate(labels_list)[:n_max]
    return feats, labels


def plot_tsne(
    clean_feats:    np.ndarray,
    clean_labels:   np.ndarray,
    adv_feats:      np.ndarray,
    adv_labels:     np.ndarray,
    save_dir:       str,
) -> None:
    """
    Plot t-SNE of clean and adversarial embeddings side-by-side.

    Clean samples are shown as filled circles, adversarial as crosses.
    Both are color-coded by true class.

    Args:
        clean_feats:  (N, 512) clean feature matrix.
        clean_labels: (N,) true labels for clean samples.
        adv_feats:    (N, 512) adversarial feature matrix.
        adv_labels:   (N,) true labels for adversarial samples.
        save_dir:     Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(clean_feats)
    all_feats = np.concatenate([clean_feats, adv_feats])

    print(f"  Running t-SNE on {len(all_feats)} samples...")
    tsne   = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=7)
    coords = tsne.fit_transform(all_feats)

    clean_coords = coords[:n]
    adv_coords   = coords[n:]

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for class_idx in range(10):
        c  = cmap(class_idx)
        lbl = _CIFAR_CLASSES[class_idx]
        m_c = clean_labels  == class_idx
        m_a = adv_labels    == class_idx

        axes[0].scatter(clean_coords[m_c, 0], clean_coords[m_c, 1],
                        c=[c], s=6, alpha=0.7, label=lbl)
        axes[1].scatter(adv_coords[m_a, 0],   adv_coords[m_a, 1],
                        c=[c], s=6, alpha=0.7, marker="x", label=lbl)

    axes[0].set_title("Clean samples")
    axes[0].legend(markerscale=3, fontsize=7, loc="best")
    axes[1].set_title("Adversarial samples (PGD20 L∞)")
    axes[1].legend(markerscale=3, fontsize=7, loc="best")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("t-SNE of ResNet-18 features: Clean vs Adversarial")
    plt.tight_layout()
    path = os.path.join(save_dir, "tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE saved → {path}")


# ==============================================================================
#  Model loaders
# ==============================================================================

def _load_resnet(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load a ResNet-18 checkpoint trained for CIFAR-10 (custom first-layer)."""
    import torchvision.models as tvm
    model = tvm.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    return model


def _load_resnet_from_scratch(ckpt_path: str, device: torch.device) -> ResNet:
    """Load a ResNet-18 checkpoint built with the repo's own ResNet class."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    return model


# ==============================================================================
#  Task 3 runner
# ==============================================================================


def run_task3(training_cfg: TrainingConfig, adv_cfg: AdversarialConfig, device: torch.device) -> None:
    """
    Task 3: PGD adversarial evaluation, Grad-CAM, and t-SNE.

    Evaluates both vanilla and AugMix models under PGD20 L∞ and L2 attacks.
    Produces Grad-CAM figures and t-SNE plots.

    Args:
        cfg:    AdversarialConfig with checkpoint paths and attack parameters.
        device: Compute device.
    """
    save_dir = "./results/hw2/adversarial"
    os.makedirs(save_dir, exist_ok=True)

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    test_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    test_loader = DataLoader(
        test_ds, batch_size=training_cfg.batch_size, shuffle=False,
        num_workers=training_cfg.num_workers, pin_memory=True,
    )

    # ---------- Load models ----------
    if not os.path.exists(adv_cfg.vanilla_ckpt):
        raise FileNotFoundError(f"Vanilla checkpoint not found: {adv_cfg.vanilla_ckpt}")
    vanilla_model = _load_resnet(adv_cfg.vanilla_ckpt, device)
    vanilla_model.eval()
    print(f"  Loaded vanilla model: {adv_cfg.vanilla_ckpt}")

    augmix_model: Optional[nn.Module] = None
    if os.path.exists(adv_cfg.augmix_ckpt):
        augmix_model = _load_resnet_from_scratch(adv_cfg.augmix_ckpt, device)
        augmix_model.eval()
        print(f"  Loaded AugMix model: {adv_cfg.augmix_ckpt}")
    else:
        print(f"  [WARN] AugMix checkpoint not found: {adv_cfg.augmix_ckpt}. Skipping AugMix eval.")

    results: Dict[str, Dict[str, float]] = {}

    for model_name, model in [("vanilla", vanilla_model), ("augmix", augmix_model)]:
        if model is None:
            continue
        results[model_name] = {}

        for norm, eps in [("linf", adv_cfg.linf_eps), ("l2", adv_cfg.l2_eps)]:
            alpha = eps / 4.0
            print(f"\n  PGD20 {norm.upper()} ε={eps:.4f} — {model_name}")
            rob_acc = evaluate_pgd(model, test_loader, eps, alpha,
                                   adv_cfg.pgd_steps, norm, device)
            results[model_name][norm] = round(rob_acc, 4)
            print(f"  Robust accuracy: {rob_acc:.4f}")

    # Save numeric results
    import json
    with open(os.path.join(save_dir, "pgd_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  PGD results saved → {save_dir}/pgd_results.json")

    # ---------- Grad-CAM ----------
    # Use a small batch for visualization (first 64 samples)
    small_loader = DataLoader(
        Subset(test_ds, list(range(256))),
        batch_size=256, shuffle=False, num_workers=0,
    )
    imgs_batch, lbls_batch = next(iter(small_loader))
    imgs_batch = imgs_batch.to(device)
    lbls_batch = lbls_batch.to(device)

    print("\n  Generating adversarial samples for Grad-CAM...")
    adv_linf = pgd_attack(
        vanilla_model, imgs_batch, lbls_batch,
        adv_cfg.linf_eps, adv_cfg.linf_eps / 4, adv_cfg.pgd_steps, "linf", device,
    )

    print("  Creating Grad-CAM plots...")
    visualize_gradcam(
        vanilla_model,
        imgs_batch.cpu(), adv_linf.cpu(), lbls_batch.cpu(),
        device, save_dir=os.path.join(save_dir, "gradcam_vanilla"),
    )

    if augmix_model is not None:
        adv_linf_am = pgd_attack(
            augmix_model, imgs_batch, lbls_batch,
            adv_cfg.linf_eps, adv_cfg.linf_eps / 4, adv_cfg.pgd_steps, "linf", device,
        )
        visualize_gradcam(
            augmix_model,
            imgs_batch.cpu(), adv_linf_am.cpu(), lbls_batch.cpu(),
            device, save_dir=os.path.join(save_dir, "gradcam_augmix"),
        )

    # ---------- t-SNE ----------
    print("\n  Extracting features for t-SNE...")

    # Build adversarial dataset for t-SNE
    adv_imgs_list, adv_lbls_list = [], []
    clean_imgs_list, clean_lbls_list = [], []
    collected = 0

    for batch_imgs, batch_lbls in test_loader:
        batch_imgs = batch_imgs.to(device)
        batch_lbls = batch_lbls.to(device)
        batch_adv  = pgd_attack(
            vanilla_model, batch_imgs, batch_lbls,
            adv_cfg.linf_eps, adv_cfg.linf_eps / 4, adv_cfg.pgd_steps, "linf", device,
        )
        clean_imgs_list.append(batch_imgs.cpu())
        clean_lbls_list.append(batch_lbls.cpu())
        adv_imgs_list.append(batch_adv.cpu())
        adv_lbls_list.append(batch_lbls.cpu())
        collected += len(batch_lbls)
        if collected >= adv_cfg.tsne_samples:
            break

    from torch.utils.data import TensorDataset
    clean_t = torch.cat(clean_imgs_list)[:adv_cfg.tsne_samples]
    clean_l = torch.cat(clean_lbls_list)[:adv_cfg.tsne_samples]
    adv_t   = torch.cat(adv_imgs_list)[:adv_cfg.tsne_samples]
    adv_l   = torch.cat(adv_lbls_list)[:adv_cfg.tsne_samples]

    clean_ds  = TensorDataset(clean_t, clean_l)
    adv_ds    = TensorDataset(adv_t,   adv_l)
    clean_ld  = DataLoader(clean_ds, batch_size=training_cfg.batch_size, shuffle=False)
    adv_ld    = DataLoader(adv_ds,   batch_size=training_cfg.batch_size, shuffle=False)

    clean_feats, clean_lbl_np = extract_features(vanilla_model, clean_ld, device, adv_cfg.tsne_samples)
    adv_feats,   adv_lbl_np   = extract_features(vanilla_model, adv_ld,   device, adv_cfg.tsne_samples)

    plot_tsne(clean_feats, clean_lbl_np, adv_feats, adv_lbl_np,
              save_dir=os.path.join(save_dir, "tsne"))

    print("\nTask 3 complete. Results in:", save_dir)


# ==============================================================================
#  Entry point
# ==============================================================================

def run_adversarial(params: Namespace) -> None:
    """
    Dispatcher for HW2 adversarial experiments (Task 3).

    Args:
        params: Parsed argparse Namespace.
    """
    from parameters import get_adversarial_config
    cfg    = get_adversarial_config(params)
    training_cfg = get_training_configs(params)
    device = get_device()

    print("Task   : HW2 Adversarial Robustness")
    print(f"Device : {device}")

    run_task3(cfg, device)