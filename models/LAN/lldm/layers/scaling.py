import torch
import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)

        return x


