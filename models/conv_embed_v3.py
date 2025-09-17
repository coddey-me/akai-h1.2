# models/conv_embed_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbedV3(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # compute flattened size dynamically
        dummy_input = torch.zeros(1, 1, 50, 50)  # match your maze size here
        n_flatten = self._get_flatten_size(dummy_input)

        self.fc = nn.Linear(n_flatten, embed_dim)

    def _get_flatten_size(self, x):
        with torch.no_grad():
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

