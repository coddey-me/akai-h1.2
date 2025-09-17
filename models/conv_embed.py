import torch
import torch.nn as nn

class ConvEmbed(nn.Module):
    """
    Small CNN embedder for binary maze grids.
    Input: (B, 1, H, W)
    Output: (B, embed_dim)
    """

    def __init__(self, in_ch=1, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # small conv stack
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)  # halves H and W
        )

        # we don't know flattened size yet → lazy linear
        self.flatten = nn.Flatten()
        self.linear = None  # will be built on first forward
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B,1,H,W)
        out = self.conv(x)
        out = self.flatten(out)

        if self.linear is None:
            # build linear layer dynamically once we know flattened dim
            in_features = out.shape[1]
            self.linear = nn.Linear(in_features, self.embed_dim).to(out.device)

        out = self.linear(out)
        out = self.relu(out)
        return out
