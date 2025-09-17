# models/hrm_v3.py
import torch
import torch.nn as nn
from .conv_embed_v3 import ConvEmbedV3

class HRM_V3(nn.Module):
    """
    Hierarchical Reasoning Model v3 (~1M parameters).
    Handles maze solving + text/code generation.
    """
    def __init__(self, embed_dim=256, high_hidden=512, low_hidden=512,
                 n_actions=4, vocab_size=500):
        super().__init__()
        self.encoder = ConvEmbedV3(in_ch=1, embed_dim=embed_dim, input_size=maze_size)

        # High-level reasoning layer (global context)
        self.high_rnn = nn.GRU(embed_dim, high_hidden, batch_first=True)

        # Low-level reasoning layer (per-step planning)
        self.low_rnn = nn.GRU(high_hidden, low_hidden, batch_first=True)

        # Maze action head
        self.action_head = nn.Linear(low_hidden, n_actions)

        # Text/code generation head
        self.token_head = nn.Linear(low_hidden, vocab_size)

    def forward(self, maze_batch, seq_len=10, task='maze', hidden=None):
        """
        task: 'maze' or 'text'
        maze_batch: (B,1,H,W)
        """
        B = maze_batch.size(0)
        emb = self.encoder(maze_batch)  # (B,embed_dim)

        # Expand embedding over sequence length
        emb_seq = emb.unsqueeze(1).repeat(1, seq_len, 1)  # (B,seq_len,embed_dim)

        # Pass through hierarchical layers
        high_out, _ = self.high_rnn(emb_seq)     # (B,seq_len,high_hidden)
        low_out, _ = self.low_rnn(high_out)      # (B,seq_len,low_hidden)

        if task == 'maze':
            logits = self.action_head(low_out)   # (B,seq_len,n_actions)
        else:
            logits = self.token_head(low_out)    # (B,seq_len,vocab_size)

        return logits
