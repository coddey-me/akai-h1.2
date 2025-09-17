# models/conv_embed_v3.py
import torch
import torch.nn as nn

class ConvEmbedV3(nn.Module):
    def __init__(self, in_ch=1, embed_dim=256, input_size=50):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, input_size, input_size)
            x = self.pool1(torch.relu(self.conv1(dummy)))
            x = self.pool2(torch.relu(self.conv2(x)))
            x = self.pool3(torch.relu(self.conv3(x)))
            n_flatten = x.view(1, -1).size(1)

        self.fc = nn.Linear(n_flatten, embed_dim)

    def forward(self, x):
        # x: (B,1,H,W)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
