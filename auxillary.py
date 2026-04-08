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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance
from typing import List, Tuple


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


    if (tloss_list is not None) and (vloss_list is not None):
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

def _autocontrast(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _: float) -> Image.Image:
    return ImageOps.equalize(img)


def _rotate(img: Image.Image, severity: float) -> Image.Image:
    degrees = severity * 30.0
    return TF.rotate(img, degrees if np.random.rand() > 0.5 else -degrees)


def _solarize(img: Image.Image, severity: float) -> Image.Image:
    threshold = int(256 * (1.0 - severity))
    return ImageOps.solarize(img, threshold)


def _shear_x(img: Image.Image, severity: float) -> Image.Image:
    shear = severity * 0.3
    return img.transform(
        img.size, Image.AFFINE,
        (1, shear if np.random.rand() > 0.5 else -shear, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def _shear_y(img: Image.Image, severity: float) -> Image.Image:
    shear = severity * 0.3
    return img.transform(
        img.size, Image.AFFINE,
        (1, 0, 0, shear if np.random.rand() > 0.5 else -shear, 1, 0),
        resample=Image.BILINEAR,
    )


def _translate_x(img: Image.Image, severity: float) -> Image.Image:
    pixels = int(severity * img.size[0] * 0.33)
    return img.transform(
        img.size, Image.AFFINE,
        (1, 0, pixels if np.random.rand() > 0.5 else -pixels, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def _translate_y(img: Image.Image, severity: float) -> Image.Image:
    pixels = int(severity * img.size[1] * 0.33)
    return img.transform(
        img.size, Image.AFFINE,
        (1, 0, 0, 0, 1, pixels if np.random.rand() > 0.5 else -pixels),
        resample=Image.BILINEAR,
    )


def _posterize(img: Image.Image, severity: float) -> Image.Image:
    bits = max(1, int(8 - severity * 4))
    return ImageOps.posterize(img, bits)


def _color(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + (np.random.rand() * 2 - 1) * severity
    return ImageEnhance.Color(img).enhance(max(0.0, factor))


def _contrast(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + (np.random.rand() * 2 - 1) * severity
    return ImageEnhance.Contrast(img).enhance(max(0.01, factor))


def _brightness(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + (np.random.rand() * 2 - 1) * severity
    return ImageEnhance.Brightness(img).enhance(max(0.01, factor))


def _sharpness(img: Image.Image, severity: float) -> Image.Image:
    factor = 1.0 + (np.random.rand() * 2 - 1) * severity
    return ImageEnhance.Sharpness(img).enhance(max(0.01, factor))


_AUGMENTATIONS = [
    _autocontrast, _equalize, _rotate, _solarize,
    _shear_x, _shear_y, _translate_x, _translate_y,
    _posterize, _color, _contrast, _brightness, _sharpness,
]


# ==============================================================================
#  Single augmented copy
# ==============================================================================

def _augment_and_mix(
    img_pil: Image.Image,
    severity: float = 0.5,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> Image.Image:
    """
    Produce one AugMix-augmented copy of a PIL image.

    Args:
        img_pil:  Input PIL image.
        severity: Magnitude of each augmentation in [0, 1].
        width:    Number of parallel augmentation chains k.
        depth:    Chain length; -1 means sample uniformly from {1,2,3}.
        alpha:    Dirichlet concentration parameter.

    Returns:
        Augmented PIL image (float32 array mixed back to PIL).
    """
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    mixing_weights = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mixed = np.zeros_like(img_np)
    for i in range(width):
        chain_depth = np.random.randint(1, 4) if depth == -1 else depth
        chain_img = img_pil.copy()
        for _ in range(chain_depth):
            op = np.random.choice(_AUGMENTATIONS)
            chain_img = op(chain_img, severity)
        mixed += mixing_weights[i] * np.array(chain_img, dtype=np.float32) / 255.0

    result = (1 - m) * img_np + m * mixed
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# ==============================================================================
#  AugMix transform (drop-in for torchvision transforms)
# ==============================================================================

class AugMixTransform:
    """
    Drop-in torchvision-compatible transform that returns (x_clean, x_aug1, x_aug2).

    A shared spatial transform is applied once, then clean/augmented branches
    are individually postprocessed and stacked into shape (3, C, H, W).

    For the Jensen-Shannon loss, unpack with:
        x_clean, x_aug1, x_aug2 = imgs[:, 0], imgs[:, 1], imgs[:, 2]

    Args:
        shared_transform: Spatial transforms applied once to the base PIL image.
        post_transform:  Tensor conversion / normalization applied per branch.
        severity:       AugMix operation magnitude in [0, 1].
        width:          Number of chains per augmented view.
        alpha:          Dirichlet / Beta parameter.
    """

    def __init__(
        self,
        shared_transform: T.Compose,
        post_transform: T.Compose,
        severity: float = 0.5,
        width: int = 3,
        alpha: float = 1.0,
    ) -> None:
        self.shared_transform = shared_transform
        self.post_transform = post_transform
        self.severity = severity
        self.width = width
        self.alpha = alpha

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.shared_transform(img)
        x_clean = self.post_transform(img)
        x_aug1  = self.post_transform(_augment_and_mix(img, self.severity, self.width, alpha=self.alpha))
        x_aug2  = self.post_transform(_augment_and_mix(img, self.severity, self.width, alpha=self.alpha))
        return torch.stack([x_clean, x_aug1, x_aug2], dim=0)  # (3, C, H, W)


# ==============================================================================
#  Jensen-Shannon consistency loss
# ==============================================================================

def js_loss(
    logits_clean: torch.Tensor,
    logits_aug1:  torch.Tensor,
    logits_aug2:  torch.Tensor,
) -> torch.Tensor:
    """
    Compute three-way Jensen-Shannon divergence consistency loss.

    JS(p_clean || p_aug1 || p_aug2)
        = (1/3) * [ KL(p_clean || M) + KL(p_aug1 || M) + KL(p_aug2 || M) ]
    where M = (p_clean + p_aug1 + p_aug2) / 3.

    Args:
        logits_clean: Clean sample logits   (N, C).
        logits_aug1:  Augmented copy 1 logits (N, C).
        logits_aug2:  Augmented copy 2 logits (N, C).

    Returns:
        Scalar JS divergence (mean over batch).
    """
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1  = F.softmax(logits_aug1,  dim=1)
    p_aug2  = F.softmax(logits_aug2,  dim=1)

    M = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, min=1e-7, max=1.0)
    log_M = torch.log(M)

    js = (
        F.kl_div(log_M, p_clean, reduction="batchmean") +
        F.kl_div(log_M, p_aug1,  reduction="batchmean") +
        F.kl_div(log_M, p_aug2,  reduction="batchmean")
    ) / 3.0
    return js


def augmix_loss(
    model: nn.Module,
    imgs_stacked: torch.Tensor,
    labels: torch.Tensor,
    lam: float = 12.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full AugMix loss: CE on clean view + lambda * JS consistency.

    Args:
        model:         Network being trained.
        imgs_stacked:  Tensor of shape (N, 3, C, H, W) from AugMixTransform.
        labels:        Ground-truth class indices (N,).
        lam:           JS loss weight lambda (default 12 per the paper).

    Returns:
        Tuple of (total_loss, clean_logits) for accuracy tracking.
    """
    x_clean = imgs_stacked[:, 0]  # (N, C, H, W)
    x_aug1  = imgs_stacked[:, 1]
    x_aug2  = imgs_stacked[:, 2]

    logits_clean = model(x_clean)
    logits_aug1  = model(x_aug1)
    logits_aug2  = model(x_aug2)

    ce   = F.cross_entropy(logits_clean, labels)
    js   = js_loss(logits_clean, logits_aug1, logits_aug2)
    loss = ce + lam * js
    return loss, logits_clean
