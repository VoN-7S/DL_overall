"""
parameters.py
-------------
Central configuration module for all experiments in this repository.

Each task (MNIST classification, transfer learning, knowledge distillation)
has its own dataclass.  All dataclasses are populated from the single
command-line interface defined in ``get_params()``.

"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List
# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Training loop settings for MNIST classification."""
    learning_rate: float  # optimiser step size
    batch_size: int       # samples per mini-batch
    epoch: int            # total training epochs
    regularizer: int      # 1 = L1, 2 = L2
    weight_decay: float   # regularisation strength (0 = disabled)
    seed: int             # global random seed
    log_interval: int     # print progress every N batches
    num_workers: int      # DataLoader worker processes

# ══════════════════════════════════════════════════════════════════════════════
#  MNIST Classification
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLPConfig:
    """MLP architecture for MNIST classification."""
    input_size: int           # 784 for MNIST (1×28×28 flattened)
    hidden_layers: List[int]  # neuron count per hidden layer, e.g. [196, 49]
    activation: str           # r=ReLU | lr=LeakyReLU | s=Sigmoid | t=Tanh
    bn: bool # apply BatchNorm1d after each linear layer
    dropout: float      # dropout probability (0 = disabled)
    num_classes: int          # 10 for MNIST


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
        1 → resize images to 224x224, freeze backbone, train FC only.
        2 → keep 32x32, replace conv1, fine-tune entire network.
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


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def get_params() -> Namespace:
    """
    Define and parse all command-line arguments for every task in the repo.

    The ``--task`` flag controls which experiment runs; only the arguments
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
        choices=["mnist", "transfer", "distillation"],
        default="mnist",
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



    # ── MNIST classification ───────────────────────────────────────────────────
    parser.add_argument("--mlp_input_size", default=784, type=int)
    parser.add_argument("--mlp_hidden_layers", nargs="+", default=[196, 49], type=int)
    parser.add_argument("--mlp_activation", default="r", choices=["r", "lr", "s", "t"])
    parser.add_argument("--mlp_bn",action="store_true")
    parser.add_argument("--mlp_dropout", default=0.0, type=float)
    parser.add_argument("--mlp_num_classes", default=10,    type=int)

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

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Config factory
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


def get_mlp_configs(p: Namespace) -> MLPConfig:
    """
    Parse arguments and return the three MNIST config dataclasses.

    Returns
    -------
    tuple[ModelConfig, TrainingConfig, SystemConfig]
    """

    model_cfg = MLPConfig(
        input_size         = p.mlp_input_size,
        hidden_layers      = p.mlp_hidden_layers,
        activation         = p.mlp_activation,
        bn                 = p.mlp_bn,
        dropout            = p.mlp_dropout,
        num_classes        = p.mlp_num_classes,
    )

    return model_cfg


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