from src.models.unet import SinusoidalPositionEmbeddings
import torch.nn as nn
import torch


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(28), nn.Linear(28, 1), nn.GELU())

        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding="same"),
            nn.GroupNorm(1, 4),
            nn.SELU(),
            nn.Conv2d(4, 8, 3, padding="same"),
            nn.GroupNorm(1, 8),
            nn.SELU(),
            nn.Conv2d(8, 1, 3, padding="same"),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_mlp(t)

        return self.net(x_t + time_embed[:, None, None])
