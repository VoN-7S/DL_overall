import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10 classification with Kaiming initialisation.

    Two convolutional layers (3-channel input, 32x32) followed by two
    fully connected layers.  Serves as the student model in the knowledge
    distillation experiments.

    Args:
        num_classes: Number of output classes. Default: 10.

    Architecture:
        Conv(3→32, k=3, pad=1) → ReLU → MaxPool(2) → 16x16
        Conv(32→64, k=3, pad=1) → ReLU → MaxPool(2) → 8x8
        Flatten → FC(64*8*8→128) → ReLU → FC(128→num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming (He) initialisation to conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))  # 32×32 → 32×32 (padding=1)
        x = F.max_pool2d(x, 2)    # → 16×16
        x = F.relu(self.conv2(x)) # → 16×16
        x = F.max_pool2d(x, 2)    # → 8×8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
