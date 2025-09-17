import torch
import torch.nn as nn
from .high_level import HighLevelPlanner
from .low_level import LowLevelExecutor

class HRM(nn.Module):
    def __init__(self, maze_size=8, embed_dim=16):
        super().__init__()
        self.embed = nn.Linear(maze_size * maze_size, 64)
        self.high = HighLevelPlanner(input_size=64, hidden_size=32)
        self.low = LowLevelExecutor(input_size=32, hidden_size=64, n_actions=4)

    def forward(self, maze_batch, seq_len):
        """
        maze_batch: [B,1,H,W]
        seq_len: length of target sequence
        """
        B = maze_batch.size(0)
        x = maze_batch.view(B, -1)  # flatten grid
        high_in = x.unsqueeze(1).repeat(1, seq_len, 1)  # [B,seq,64]
        high_out, _ = self.high(high_in)
        logits, _ = self.low(high_out)
        return logits  # [B,seq,4]
