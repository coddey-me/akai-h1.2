# models/conv_embed_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbedV3(nn.Module):
    """
    Convolutional encoder for maze grids up to 100x100.
    Produces a flattened embedding vector.
    """
    def __init__(self, in_ch=1, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),  # (B,32,H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # (B,64,H/4,W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # (B,128,H/8,W/8)
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 13 * 13, embed_dim)  # 100x100 → 13x13 after 3 strided convs

    def forward(self, x):
        # x: (B,1,H,W)
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out  # (B,embed_dim)
