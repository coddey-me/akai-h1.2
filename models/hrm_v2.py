import torch
import torch.nn as nn
from .conv_embed import ConvEmbed

class HRM_V2(nn.Module):
    """
    HRM v2: ConvEmbed -> High-level GRU -> Low-level GRU -> action logits
    - maze_size: used only for clarity; ConvEmbed handles actual H,W at runtime
    - embed_dim: size of embedding produced by ConvEmbed
    """

    def __init__(self, embed_dim=256, high_hidden=128, low_hidden=256, n_actions=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.high_hidden = high_hidden
        self.low_hidden = low_hidden
        self.n_actions = n_actions

        # encoder
        self.encoder = ConvEmbed(in_ch=1, embed_dim=embed_dim)

        # high-level planner GRU (operates at seq length)
        self.high = nn.GRU(input_size=embed_dim, hidden_size=high_hidden, batch_first=True)

        # low-level executor GRU
        self.low = nn.GRU(input_size=high_hidden, hidden_size=low_hidden, batch_first=True)

        # final head -> action logits
        self.head = nn.Linear(low_hidden, n_actions)

    def forward(self, maze_batch, seq_len):
        """
        maze_batch: [B,1,H,W]
        seq_len: target sequence length (int)
        returns: logits [B, seq_len, n_actions]
        """
        B = maze_batch.size(0)

        # encode the whole maze into a single embedding per maze
        emb = self.encoder(maze_batch)          # [B, embed_dim]

        # repeat embedding across the sequence dimension for the planner
        high_in = emb.unsqueeze(1).repeat(1, seq_len, 1)  # [B, seq_len, embed_dim]

        # high-level planner (returns all outputs across seq)
        high_out, _ = self.high(high_in)       # [B, seq_len, high_hidden]

        # low-level executor consumes high-level outputs
        low_out, _ = self.low(high_out)        # [B, seq_len, low_hidden]

        logits = self.head(low_out)            # [B, seq_len, n_actions]
        return logits
