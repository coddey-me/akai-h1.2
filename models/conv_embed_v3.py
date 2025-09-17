# models/conv_embed_v3.py
import torch
import torch.nn as nn

class ConvEmbedV3(nn.Module):
    def __init__(self, in_ch=1, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # ↓↓↓ critical line ↓↓↓
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  

        # After pooling we always have 128 features
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))

        # ↓↓↓ critical line ↓↓↓
        x = self.global_pool(x)           # (B,128,1,1)

        # flatten to (B,128)
        x = x.view(x.size(0), -1)

        x = self.fc(x)                    # (B,embed_dim)
        return x
