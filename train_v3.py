# train_v3.py
import os
import math
import time
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# model & datasets (assumes the files you already have)
from models.hrm_v3 import HRM_V3
from models.tokenizer import CharTokenizer

from data.maze_dataset_v3 import MazeDatasetV3
from data.code_dataset import CodeDataset
from data.convo_dataset import ConvoDataset

# -----------------------------
# Hyperparameters & config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# model sizes (tweak to hit ~1M params)
embed_dim = 256
high_hidden = 512
low_hidden = 512

# training
maze_size = 100              # must match ConvEmbedV3 fc assumptions (100x100 default)
batch_size = 16
epochs = 10
lr = 1e-4
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# dataset sizes (you can scale up once pipeline is stable)
n_maze = 20000       # generate on-the-fly
n_code = 20000
n_convo = 20000

# tokenization
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size

# -----------------------------
# Datasets & DataLoaders
# -----------------------------
maze_ds = MazeDatasetV3(n_samples=n_maze, size=maze_size)
maze_loader = DataLoader(maze_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

code_ds = CodeDataset(tokenizer, n_samples=n_code, max_len=128)
code_loader = DataLoader(code_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

convo_ds = ConvoDataset(tokenizer, n_samples=n_convo, max_len=128)
convo_loader = DataLoader(convo_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# Create iterators that we can cycle through
maze_iter = iter(maze_loader)
code_iter = iter(code_loader)
convo_iter = iter(convo_loader)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = HRM_V3(embed_dim=embed_dim,
               high_hidden=high_hidden,
               low_hidden=low_hidden,
               n_actions=4,
               vocab_size=vocab_size).to(device)

# Losses
# we use ignore_index=-100 for padded positions in action sequences
action_criterion = nn.CrossEntropyLoss(ignore_index=-100)
token_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad_token in tokenizer

optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Helper functions
# -----------------------------

def pad_action_sequences(paths):
    """
    paths: list of 1D torch.long tensors of variable lengths
    returns: padded tensor (B, max_len) with -100 as padding for loss ignore
    """
    max_len = max([p.size(0) for p in paths]) if paths else 0
    padded = torch.full((len(paths), max_len), -100, dtype=torch.long)
    for i, p in enumerate(paths):
        if p.numel() > 0:
            padded[i, : p.size(0)] = p
    return padded, max_len

def pad_token_batch(inputs, targets):
    # inputs/targets are already fixed-length in our simple datasets (max_len)
    # Ensure tensors are on device
    return inputs.to(device), targets.to(device)

def eval_on_small_batches(model, n_eval=64):
    model.eval()
    maze_accs = []
    token_accs = []
    # Maze eval: sample a few
    with torch.no_grad():
        for _ in range(8):
            try:
                mazes, paths = next(maze_iter)
            except StopIteration:
                break
            mazes = mazes.to(device)
            padded, max_len = pad_action_sequences(paths)
            targets = padded.to(device)
            # ensure channel dimension
            if mazes.ndim == 3:
                mazes = mazes.unsqueeze(1)
            logits = model(mazes, seq_len=max_len, task='maze')  # [B,seq,4]
            preds = logits.argmax(dim=-1)  # [B,seq]
            mask = targets != -100
            correct = (preds == targets) & mask
            acc = correct.sum().item() / max(1, mask.sum().item())
            maze_accs.append(acc)
        # Token eval (code)
        for _ in range(4):
            try:
                inp, tgt = next(code_iter)
            except StopIteration
