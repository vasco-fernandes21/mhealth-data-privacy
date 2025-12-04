import torch.nn as nn
from opacus.validators import ModuleValidator


class UnifiedMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def get_model(input_dim: int, num_classes: int, device="cpu"):
    model = UnifiedMLP(input_dim, num_classes)
    # Ensure Opacus compatibility
    model = ModuleValidator.fix(model)
    return model.to(device)

