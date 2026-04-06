from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional
# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Training loop settings."""
    learning_rate: float  # optimiser step size
    batch_size: int       # samples per mini-batch
    epoch: int            # total training epochs
    regularizer: int      # 1 = L1, 2 = L2
    weight_decay: float   # regularisation strength (0 = disabled)
    seed: int             # global random seed
    log_interval: int     # print progress every N batches
    num_workers: int      # DataLoader worker processes

# ══════════════════════════════════════════════════════════════════════════════
#  Transfer Learning 
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferConfig:
    """
    Hyper-parameters and paths for one transfer-learning run on CIFAR-10.

    Attributes
    ----------
    option :
        1 -> resize images to 224x224, freeze backbone, train FC only.
        2 -> keep 32x32, replace conv1, fine-tune entire network.
    """
    option: int



# ══════════════════════════════════════════════════════════════════════════════
#  Knowledge Distillation  
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KDConfig:
    """
    Hyper-parameters for all four knowledge-distillation experiments.

    Attributes
    ----------
    label_smoothing :
        Smoothing factor ε for LabelSmoothingLoss (0 = standard CE).
    temperature :
        Softening temperature T for KD soft targets.
    alpha :
        Weight on the soft-target KD loss; (1 - alpha) weights the hard CE loss.
    """
    smoothing: float = 0.0
    temperature:float= 4.0
    alpha: float= 0.7
    experiment:int | None = None

@dataclass
class RobustnessConfig:
    """
    Configuration for HW2 Tasks 1 & 2 — CIFAR-10-C evaluation and AugMix training.
 
    Attributes
    ----------
    vanilla_ckpt:
        Path to the HW1b fine-tuned checkpoint (transfer option 2).
    batch_size:
        Mini-batch size for data loading and evaluation.
    num_workers:
        DataLoader worker processes.
    seed:
        Global random seed.
    epochs:
        Training epochs for the AugMix-trained model (Task 2).
    learning_rate:
        SGD learning rate for AugMix training.
    weight_decay:
        L2 regularization coefficient.
    augmix_lambda:
        Weight λ on the Jensen-Shannon consistency loss (default 12 per paper).
    """
    vanilla_ckpt:  str   = "./results/transfer/transfer_layerchange/model.pth"
    batch_size:    int   = 128
    num_workers:   int   = 2
    seed:          int   = 7
    epochs:        int   = 30
    learning_rate: float = 0.05
    weight_decay:  float = 5e-4
    augmix_lambda: float = 12.0
 
 
@dataclass
class AdversarialConfig:
    """
    Configuration for HW2 Task 3 — PGD attacks, Grad-CAM, t-SNE.
 
    Attributes
    ----------
    vanilla_ckpt:
        Path to the HW1b fine-tuned checkpoint.
    augmix_ckpt:
        Path to the AugMix-trained checkpoint from Task 2.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker processes.
    linf_eps:
        L∞ attack budget (4/255 in raw pixel space).
    l2_eps:
        L2 attack budget (0.25 in raw pixel space).
    pgd_steps:
        Number of PGD iterations (20 for PGD20).
    tsne_samples:
        Number of samples to extract for t-SNE visualization.
    """
    vanilla_ckpt:  str   = "./results/transfer/transfer_layerchange/model.pth"
    augmix_ckpt:   str   = "./results/hw2/robustness_augmix/model.pth"
    batch_size:    int   = 128
    num_workers:   int   = 2
    linf_eps:      float = 4.0 / 255.0
    l2_eps:        float = 0.25
    pgd_steps:     int   = 20
    tsne_samples:  int   = 1000
 
 
@dataclass
class HW2KDConfig:
    """
    Configuration for HW2 Tasks 4 & 5 — AugMix teacher KD and transferability.
 
    Attributes
    ----------
    vanilla_ckpt:
        Path to the HW1b fine-tuned teacher checkpoint.
    augmix_ckpt:
        Path to the AugMix-trained teacher checkpoint from Task 2.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker processes.
    seed:
        Global random seed.
    epochs:
        Number of student training epochs.
    learning_rate:
        Adam learning rate for student training.
    weight_decay:
        L2 regularization coefficient.
    temperature:
        KD softening temperature T.
    alpha:
        Weight on the soft-target KD loss.
    """
    vanilla_ckpt:  str   = "./results/transfer/transfer_layerchange/model.pth"
    augmix_ckpt:   str   = "./results/hw2/robustness_augmix/model.pth"
    batch_size:    int   = 128
    num_workers:   int   = 2
    seed:          int   = 7
    epochs:        int   = 30
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    temperature:   float = 4.0
    alpha:         float = 0.7

# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def get_params() -> Namespace:
    """
    Define and parse all command-line arguments for every task in the repo.

    The ``--task`` flag controls which experiment runs. Only the arguments
    relevant to that task need to be supplied.

    Returns
    -------
    Namespace
        Parsed argument values.
    """
    parser = ArgumentParser(description="Unified deep-learning experiment runner")

    # ── Task selector ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        choices=["transfer", "distillation", "robustness", "adversarial", "cutmix_distillation"],
        default="transfer",
        help="Which experiment to run.",
    )

    # ── Shared ────────────────────────────────────────────────────────
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-3,  type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--regularizer", default=2, type=int, choices=[1, 2])
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--train_eval", default = 2, type=int, choices=[0, 1, 2])

    # ── Transfer learning ─────────────────────────────────────────────
    parser.add_argument(
        "--tl_option", default="both", choices=["1", "2", "both"],
        help="Transfer learning option: 1=freeze backbone, 2=fine-tune all, both=run both.",
    )

    # ── Knowledge distillation ───────────────────────────────────────
    parser.add_argument(
        "--kd_experiment", default=None, type=int, choices=[1, 2, 3, 4],
        help="Run a single KD experiment (1-4). Omit to run all four.",
    )
    parser.add_argument("--kd_temperature",  default=4.0,  type=float)
    parser.add_argument("--kd_alpha",        default=0.7,  type=float)
    parser.add_argument("--kd_smoothing", default= 0.0, type=float)

    parser.add_argument(
        "--hw2_task", 
        choices=["task1", "task2", "task3", "task4", "task5", "both"], 
        default="both", 
        help=("Which HW2 sub-task to run. "))
    parser.add_argument(
        "--augmix_epochs",   
        default=30,   
        type=int,
        help="Training epochs for AugMix model (Task 2).")
    parser.add_argument(
        "--augmix_lr",       
        default=0.05, 
        type=float,
        help="Learning rate for AugMix training (Task 2).")
    parser.add_argument(
        "--augmix_lambda",   
        default=12.0, 
        type=float,
        help="Jensen-Shannon loss weight lambda for AugMix (Task 2).")
    parser.add_argument(
        "--linf_eps", 
        default=4.0/255.0, 
        type=float,
        help="PGD L∞ epsilon (default 4/255).")
    parser.add_argument(
        "--l2_eps", 
        default=0.25, 
        type=float,
        help="PGD L2 epsilon (default 0.25).")
    parser.add_argument(
        "--pgd_steps", 
        default=20, 
        type=int, 
        help="Number of PGD iterations.")
    parser.add_argument(
        "--tsne_samples", 
        default=1000, 
        type=int,
        help="Number of samples for t-SNE visualization.")
    parser.add_argument(
        "--hw2_kd_epochs",   
        default=30,   
        type=int,
        help="Student training epochs for HW2 KD experiments.")
    parser.add_argument(
        "--hw2_kd_lr",       
        default=1e-3, 
        type=float,
        help="Adam learning rate for HW2 student training.")
    parser.add_argument(
        "--vanilla_ckpt", 
        default="./results/transfer/transfer_layerchange/model.pth", 
        type=str,
        help="Path to HW1b fine-tuned ResNet-18 checkpoint.")
    parser.add_argument(
        "--augmix_ckpt",
        default="./results/hw2/robustness_augmix/model.pth",
        type=str,
        help="Path to AugMix-trained ResNet-18 checkpoint (from Task 2).",
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Configs
# ══════════════════════════════════════════════════════════════════════════════

def get_training_configs(p: Namespace) -> TrainingConfig:
    
    return TrainingConfig(
        learning_rate = p.learning_rate,
        batch_size    = p.batch_size,
        epoch         = p.epoch,
        regularizer   = p.regularizer,
        weight_decay  = p.weight_decay,
        seed          = p.seed,
        log_interval  = p.log_interval,
        num_workers   = p.num_workers,
    )


def get_transfer_configs(p: Namespace) -> List[TransferConfig]:
    """
    Build a list of TransferConfig instances from parsed args.

    Returns
    -------
    List[TransferConfig]
        One entry per option to run (1, 2, or both).
    """
    options = [1, 2] if p.tl_option == "both" else [int(p.tl_option)]
    return [
        TransferConfig(
            option = opt
        )
        for opt in options
    ]


def get_kd_config(p: Namespace) -> KDConfig:
    """
    Build a KDConfig from parsed args.

    Returns
    -------
    KDConfig
    """
    return KDConfig(
        smoothing       = p.kd_smoothing, 
        temperature     = p.kd_temperature,
        alpha           = p.kd_alpha,
        experiment      = p.kd_experiment
    )

def get_robustness_config(p: Namespace) -> RobustnessConfig:
    """
    Build a RobustnessConfig from parsed arguments.
 
    Returns
    -------
    RobustnessConfig
    """
    return RobustnessConfig(
        vanilla_ckpt  = getattr(p, "vanilla_ckpt",  "./results/transfer/transfer_layerchange/model.pth"),
        batch_size    = p.batch_size,
        num_workers   = p.num_workers,
        seed          = p.seed,
        epochs        = getattr(p, "augmix_epochs", 30),
        learning_rate = getattr(p, "augmix_lr",     0.05),
        weight_decay  = p.weight_decay,
        augmix_lambda = getattr(p, "augmix_lambda", 12.0),
    )
 
 
def get_adversarial_config(p: Namespace) -> AdversarialConfig:
    """
    Build an AdversarialConfig from parsed arguments.
 
    Returns
    -------
    AdversarialConfig
    """
    return AdversarialConfig(
        vanilla_ckpt  = getattr(p, "vanilla_ckpt",  "./results/transfer/transfer_layerchange/model.pth"),
        augmix_ckpt   = getattr(p, "augmix_ckpt",   "./results/hw2/robustness_augmix/model.pth"),
        batch_size    = p.batch_size,
        num_workers   = p.num_workers,
        linf_eps      = getattr(p, "linf_eps",       4.0 / 255.0),
        l2_eps        = getattr(p, "l2_eps",         0.25),
        pgd_steps     = getattr(p, "pgd_steps",      20),
        tsne_samples  = getattr(p, "tsne_samples",   1000),
    )
 
 
def get_hw2_kd_config(p: Namespace) -> HW2KDConfig:
    """
    Build a HW2KDConfig from parsed arguments.
 
    Returns
    -------
    HW2KDConfig
    """
    return HW2KDConfig(
        vanilla_ckpt  = getattr(p, "vanilla_ckpt",  "./results/transfer/transfer_layerchange/model.pth"),
        augmix_ckpt   = getattr(p, "augmix_ckpt",   "./results/hw2/robustness_augmix/model.pth"),
        batch_size    = p.batch_size,
        num_workers   = p.num_workers,
        seed          = p.seed,
        epochs        = getattr(p, "hw2_kd_epochs", 30),
        learning_rate = getattr(p, "hw2_kd_lr",     1e-3),
        weight_decay  = p.weight_decay,
        temperature   = p.kd_temperature,
        alpha         = p.kd_alpha,
    )