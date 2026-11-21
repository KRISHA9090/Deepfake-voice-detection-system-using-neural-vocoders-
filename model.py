"""Small RawNet2-inspired educational classifier.

This is NOT an exact reproduction of the published RawNet2 architecture.
It is a compact waveform model for demonstrating the training/inference
pipeline. The included demo WAVs are synthetic signal proxies, not speech.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SincLikeFrontEnd(nn.Module):
    """Learnable 1-D convolutional waveform front end (SincConv-inspired)."""
    def __init__(self, channels=32, kernel_size=101):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2,
                              bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        self.pool = nn.AvgPool1d(4) if pool else nn.Identity()

    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        y = self.bn2(self.conv2(y))
        y = F.leaky_relu(y + self.skip(x), 0.2)
        return self.pool(y)

class RawNet2Inspired(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.front = SincLikeFrontEnd(32, 101)
        self.blocks = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, waveform):
        # waveform: [batch, 1, samples]
        x = self.front(waveform)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.classifier(x)
