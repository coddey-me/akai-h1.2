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
        # small conv stack
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, padding=1),  # (B,8,H,W)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),     # (B,16,H,W)
            nn.ReLU(),
            nn.AvgPool2d(2)                                  # downsample H/2, W/2
        )
        # after pool the spatial dims are (H/2)*(W/2). We'll flatten and reduce.
        self.post = nn.Sequential(
            nn.Flatten(),                                   # (B, 16 * (H/2) * (W/2))
            nn.Linear(16 *  (8) * (8) // 4 if False else -1,  # placeholder, replaced in init
                      embed_dim),
            nn.ReLU()
        )

    def _build_linear(self, H, W, embed_dim):
        """Helper to replace placeholder linear with correct input size."""
        flattened = 16 * (H // 2) * (W // 2)
        self.post = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B,1,H,W)
        B, C, H, W = x.shape
        # ensure post linear is built with correct spatial dims
        if isinstance(self.post[1], nn.Linear) and self.post[1].in_features == -1:
            # replace placeholder with correct dimension
            self._build_linear(H, W, self.post[-2].out_features if len(self.post) >= 2 else 256)
        out = self.conv(x)
        out = self.post(out)
        return out
