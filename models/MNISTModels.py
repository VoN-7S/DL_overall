import torch
import torch.nn as nn

from parameters import ModelConfig

class MLPBase(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        layers = []
        in_size = model_config.input_size
        for size in model_config.hidden_layers:
            layers += [
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size) if model_config.batch_normalization else nn.Identity(),
                get_activation(model_config),
                nn.Dropout(model_config.dropout_ratio),
                
            ]
            in_size = size
        
        self.layers = nn.Sequential(
            *layers,
            nn.Linear(in_size, model_config.num_classes),
        )
    
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    

def get_activation(model_cfg: ModelConfig) -> nn.modules.activation:
    if model_cfg.activation == "r":
        return nn.ReLU()
    elif model_cfg.activation == "lr":
        return nn.LeakyReLU()
    elif model_cfg.activation == "s":
        return nn.Sigmoid()
    elif model_cfg.activation == "t":
        return nn.Tanh()
    



            