import torch
import torch.nn as nn

class LowLevelExecutor(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, n_actions=4):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_actions)
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        logits = self.out(out)
        return logits, h
