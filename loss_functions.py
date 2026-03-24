import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
#  Loss Funcs
# ==============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    The true class gets probability (1 - smoothing) and each other class
    gets smoothing / (C - 1). Prevents the model from becoming overconfident.

    Args:
        num_classes: Total number of output classes C.
        smoothing:   Smoothing factor in range [0, 1). 0 = standard CE.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

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
    kd = F.kl_div(soft_s, soft_t, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(s_logits, labels)
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

    Encodes per-example difficulty while keeping wrong-class targets uniform.

    Args:
        s_logits:    Student logits of shape (N, C).
        t_logits:    Teacher logits of shape (N, C).
        labels:      True class indices of shape (N,).
        temperature: Softening temperature T applied to teacher.
        alpha:       Weight on the soft-target loss.

    Returns:
        Modified KD loss.
    """
    N, C = s_logits.shape
    t_probs = F.softmax(t_logits / temperature, dim=1)
    p_true  = t_probs.gather(1, labels.unsqueeze(1))

    uniform_other = (1.0 - p_true) / (C - 1)
    soft = uniform_other.expand(N, C).clone()
    soft.scatter_(1, labels.unsqueeze(1), p_true)

    log_s = F.log_softmax(s_logits / temperature, dim=1)
    kd = F.kl_div(log_s, soft, reduction="batchmean") * (temperature ** 2)
    ce = F.cross_entropy(s_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce