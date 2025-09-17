import torch
import torch.nn as nn

class HighLevelPlanner(nn.Module):
    def __init__(self, input_size=64, hidden_size=32):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        return out, h
