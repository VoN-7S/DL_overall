"""
mnist.py
--------
Everything related to MNIST MLP classification (HW1a).

Contains the model definition, data loading, training loop,
validation, testing, and the main experiment runner.

Usage
-----
    python main.py --task mnist --epoch 25 --mlp_hidden_layers 256 128
"""

import os
from copy import deepcopy
from dataclasses import asdict
from typing import List, Tuple
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from parameters import MLPConfig, TrainingConfig, get_mlp_configs, get_training_configs
from auxillary import set_seed, get_device, save_results


# ==============================================================================
#  Model
# ==============================================================================

class MLPBase(nn.Module):
    """
    Flexible multi-layer perceptron for MNIST classification.

    Builds a sequential stack of Linear -> BatchNorm (optional) ->
    Activation -> Dropout layers based on the provided MLPConfig,
    followed by a final linear classification layer.

    Args:
        cfg: MLPConfig instance defining the architecture.
    """

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        layers = []
        in_size = cfg.input_size

        for size in cfg.hidden_layers:
            layers.append(nn.Linear(in_size, size))
            if cfg.bn:
                layers.append(nn.BatchNorm1d(size))
            layers.append(_get_activation(cfg.activation))
            layers.append(nn.Dropout(cfg.dropout))
            in_size = size

        layers.append(nn.Linear(in_size, cfg.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = x.view(x.size(0), -1)
        return self.network(x)


def _get_activation(activation: str) -> nn.Module:
    """
    Return the activation module for the given shorthand string.

    Args:
        activation: One of r (ReLU), lr (LeakyReLU), s (Sigmoid), t (Tanh).

    Returns:
        Corresponding PyTorch activation module.

    Raises:
        ValueError: If the activation string is not recognised.
    """
    if activation == "r":
        return nn.ReLU()
    elif activation == "lr":
        return nn.LeakyReLU()
    elif activation == "s":
        return nn.Sigmoid()
    elif activation == "t":
        return nn.Tanh()
    else:
        raise ValueError("Unknown activation: " + activation)


# ==============================================================================
#  Data loading
# ==============================================================================

def get_loaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Build MNIST training and validation DataLoaders.

    Applies ToTensor and normalisation using MNIST dataset statistics
    (mean=0.1307, std=0.3081). Uses SubsetRandomSampler over all 60000
    training samples.

    Args:
        cfg: TrainingConfig with batch_size and num_workers.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    val_ds = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    sampler = SubsetRandomSampler(list(range(len(train_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


# ==============================================================================
#  Training and validation
# ==============================================================================

def train_one_epoch(
    cfg: TrainingConfig,
    model: MLPBase,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Run one full pass over the training set.

    Computes loss plus optional L1 or L2 regularisation,
    runs backpropagation, and updates model weights.

    Args:
        cfg:          TrainingConfig with log_interval, batch_size,
                      regularizer, weight_decay.
        model:        The MLP model in training mode.
        criterion:    Loss function.
        optimizer:    Optimiser for weight updates.
        device:       Compute device.
        train_loader: DataLoader for the training set.

    Returns:
        Tuple of (average_loss, accuracy) over the full epoch.
    """
    model.train()
    total_loss = 0.0
    correct    = 0

    for batch_idx, (batch, labels) in enumerate(train_loader):
        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, labels)
        loss = loss + _get_regularization(cfg, model)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        correct    += torch.sum(
            torch.eq(torch.argmax(predictions.detach(), dim=1), labels)
        ).item()

        if (batch_idx + 1) % cfg.log_interval == 0:
            n_seen = (batch_idx + 1) * cfg.batch_size
            print(
                "  Batch " + str(batch_idx + 1) +
                " | Loss: " + str(round(total_loss / n_seen, 4)) +
                " | Acc: "  + str(round(correct    / n_seen, 4))
            )

    n = len(train_loader.dataset)
    return total_loss / n, correct / n


def validate(
    model: MLPBase,
    criterion: nn.Module,
    device: torch.device,
    val_loader: DataLoader,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Runs without gradient computation for efficiency. No weight updates.

    Args:
        model:      The MLP model in eval mode.
        criterion:  Loss function.
        device:     Compute device.
        val_loader: DataLoader for the validation set.

    Returns:
        Tuple of (average_loss, accuracy) over the full validation set.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0

    with torch.no_grad():
        for batch, labels in val_loader:
            batch, labels = batch.to(device), labels.to(device)
            predictions = model(batch)
            loss = criterion(predictions, labels)
            total_loss += loss.detach().item()
            correct    += torch.sum(
                torch.eq(torch.argmax(predictions.detach(), dim=1), labels)
            ).item()

    n = len(val_loader.dataset)
    return total_loss / n, correct / n


def _get_regularization(cfg: TrainingConfig, model: nn.Module) -> torch.Tensor:
    """
    Compute L1 or L2 regularisation penalty for all model parameters.

    Args:
        cfg:   TrainingConfig with weight_decay and regularizer.
        model: The model whose parameters are penalised.

    Returns:
        Scalar tensor representing the regularisation term.
    """
    return cfg.weight_decay * sum(
        torch.sum(torch.abs(p) ** cfg.regularizer)
        for p in model.parameters()
    )


# ==============================================================================
#  Testing
# ==============================================================================

def test(
    model: MLPBase,
    cfg: TrainingConfig,
    mlp_cfg: MLPConfig,
    model_path: str,
) -> None:
    """
    Evaluate the best saved model on the MNIST test set.

    Loads the checkpoint from model_path, runs inference on the full
    test set, and prints overall and per-class accuracy.

    Args:
        model:      The MLP model instance (weights will be loaded).
        cfg:        TrainingConfig with batch_size and num_workers.
        mlp_cfg:    MLPConfig with num_classes for per-class reporting.
        model_path: Path to the saved model checkpoint.
    """
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    correct       = 0
    class_correct = [0] * mlp_cfg.num_classes
    class_total   = [0] * mlp_cfg.num_classes

    with torch.no_grad():
        for batch, labels in test_loader:
            batch, labels = batch.to(device), labels.to(device)
            preds = torch.argmax(model(batch), dim=1)
            correct += torch.sum(torch.eq(preds, labels)).item()
            for pred, label in zip(preds, labels):
                class_correct[label] += (pred == label).item()
                class_total[label]   += 1

    print("\n===== TEST RESULTS =====")
    print(
        "Overall Accuracy: " + str(correct) +
        " / " + str(len(test_ds)) +
        " = " + str(round(correct / len(test_ds), 4))
    )
    for i in range(mlp_cfg.num_classes):
        print(
            "  Digit " + str(i) + ": " +
            str(round(class_correct[i] / class_total[i], 4))
        )


# ==============================================================================
#  Experiment runner
# ==============================================================================

def run_mnist(params: Namespace) -> None:
    """
    Run the full MNIST MLP classification experiment.

    Builds the model and data loaders from parsed params, trains for
    the configured number of epochs, saves the best checkpoint, runs
    the test set evaluation, and saves the loss curve and parameters.

    Results saved to:
        results/mnist/model.pth
        results/mnist/mnist_train_val.png
        results/mnist/mnist_params.json

    Args:
        params: Parsed Namespace object from get_params().
    """
    mlp_cfg      = get_mlp_configs(params)
    training_cfg = get_training_configs(params)

    set_seed(training_cfg.seed)
    device = get_device()

    print("Task   : MNIST classification")
    print("Device : " + str(device))

    model        = MLPBase(mlp_cfg).to(device)
    train_loader, val_loader = get_loaders(training_cfg)

    best_weights              = None
    best_acc                  = 0.0
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(training_cfg.epoch):
        print("\nEpoch " + str(epoch + 1) + "/" + str(training_cfg.epoch))

        criterion = nn.CrossEntropyLoss(reduction="sum")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=training_cfg.learning_rate
        )

        tr_loss, tr_acc   = train_one_epoch(
            training_cfg, model, criterion, optimizer, device, train_loader
        )
        val_loss, val_acc = validate(model, criterion, device, val_loader)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        print(
            "  Train loss: " + str(round(tr_loss, 4)) +
            "  acc: "        + str(round(tr_acc, 4)) +
            " | Val loss: "  + str(round(val_loss, 4)) +
            "  acc: "        + str(round(val_acc, 4))
        )

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = deepcopy(model.state_dict())
            print("  Best model updated.")

    model_path = "./results/mnist/model.pth"
    os.makedirs("./results/mnist", exist_ok=True)
    torch.save(best_weights, model_path)
    print("Model saved -> " + model_path)

    model.load_state_dict(best_weights)
    test(model, training_cfg, mlp_cfg, model_path)

    save_results(
        params     = {**asdict(mlp_cfg), **asdict(training_cfg)},
        tloss_list = train_losses,
        vloss_list = val_losses,
        save_dir   = "./results/mnist",
        name       = "mnist",
    )
