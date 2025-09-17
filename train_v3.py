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
epochs = 5
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
from data.maze_dataset_v3 import MazeDatasetV3
from torch.utils.data import DataLoader

maze_dataset = MazeDatasetV3(precomputed_file="precomputed_mazes.pt")

from utils.collate_v3 import maze_collate_fn, collate_fn

maze_loader = DataLoader(
    maze_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn
)

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
               maze_size=maze_size,
               vocab_size=vocab_size).to(device)

# Losses
# we use ignore_index=-100 for padded positions in action sequences
action_criterion = nn.CrossEntropyLoss(ignore_index=-100)
token_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad_token in tokenizer

optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# Helper functions
# -----------------------------

def pad_action_sequences(paths, pad_value=0):
    """
    Pads a list of 1D tensors (paths) to the same length.
    Returns (padded_tensor, max_len)
    """
    if len(paths) == 0:
        return torch.empty(0), 1

    # get each path’s length
    lengths = [p.size(0) if isinstance(p, torch.Tensor) else len(p) for p in paths]
    max_len = max(lengths)

    # create padded tensor
    padded = torch.full((len(paths), max_len), pad_value, dtype=torch.long)

    for i, p in enumerate(paths):
        # make sure p is tensor
        p_tensor = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.long)
        padded[i, :len(p_tensor)] = p_tensor

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
            except StopIteration:
                break
            b_inp, b_tgt = pad_token_batch(inp, tgt)
            # create dummy mazes as context
            dummy = torch.zeros((b_inp.size(0), 1, maze_size, maze_size), device=device)
            logits = model(dummy, seq_len=b_inp.size(1), task='text')
            preds = logits.argmax(dim=-1)
            acc = (preds == b_tgt.to(device)).float().mean().item()
            token_accs.append(acc)
    # reset model to train mode
    model.train()
    maze_acc = sum(maze_accs) / max(1, len(maze_accs)) if maze_accs else 0.0
    token_acc = sum(token_accs) / max(1, len(token_accs)) if token_accs else 0.0
    return maze_acc, token_acc

# -----------------------------
# Training loop
# -----------------------------
print("Starting training")
global_step = 0
start_time = time.time()
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    epoch_steps = 0

    # We'll run for as many batches as the largest loader for this epoch
    max_batches = max(len(maze_loader), len(code_loader), len(convo_loader))
    # Round-robin: maze -> code -> convo -> repeat
    task_cycle = ["maze", "code", "convo"]

    for batch_idx in range(max_batches):
        for task in task_cycle:
            try:
                if task == "maze":
                    mazes, paths = next(maze_iter)
                    # mazes: (B,1,H,W) already, paths: list of tensors
                    # ensure channel dim
                    mazes = mazes.to(device)
                    if mazes.ndim == 3:
                        mazes = mazes.unsqueeze(1)
                    padded_targets, seq_len = pad_action_sequences(paths)
                    targets = padded_targets.to(device)  # (B,seq_len)
                    # Forward
                    logits = model(mazes, seq_len=seq_len, task='maze')  # (B,seq_len,4)
                    logits_flat = logits.view(-1, logits.size(-1))      # (B*seq_len,4)
                    targets_flat = targets.view(-1)                    # (B*seq_len,)
                    loss = action_criterion(logits_flat, targets_flat)
                elif task == "code":
                    inp, tgt = next(code_iter)
                    # inp/tgt: (B,L)
                    inp = inp.to(device)
                    tgt = tgt.to(device)
                    seq_len = inp.size(1)
                    # dummy maze context
                    dummy = torch.zeros((inp.size(0), 1, maze_size, maze_size), device=device)
                    logits = model(dummy, seq_len=seq_len, task='text')  # (B,seq_len,vocab)
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = tgt.view(-1)
                    loss = token_criterion(logits_flat, targets_flat)
                else:  # convo
                    inp, tgt = next(convo_iter)
                    inp = inp.to(device)
                    tgt = tgt.to(device)
                    seq_len = inp.size(1)
                    dummy = torch.zeros((inp.size(0), 1, maze_size, maze_size), device=device)
                    logits = model(dummy, seq_len=seq_len, task='text')
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = tgt.view(-1)
                    loss = token_criterion(logits_flat, targets_flat)

            except StopIteration:
                # Recreate iterator if exhausted
                if task == "maze":
                    maze_iter = iter(maze_loader)
                elif task == "code":
                    code_iter = iter(code_loader)
                else:
                    convo_iter = iter(convo_loader)
                continue  # skip this iteration (iterators reset)

            # backward & step
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

    avg_loss = epoch_loss / max(1, epoch_steps)
    elapsed = time.time() - start_time
    print(f"Epoch {epoch}/{epochs}  AvgLoss: {avg_loss:.4f}  Steps: {epoch_steps}  TimeElapsed: {elapsed/60:.2f}m")

    # Simple eval on small batches and save checkpoint
    maze_acc, token_acc = eval_on_small_batches(model)
    print(f"  [Eval] Maze acc: {maze_acc:.4f}  Token acc (code/convo): {token_acc:.4f}")

    ckpt_path = os.path.join(save_dir, f"hrm_v3_epoch{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("  Saved checkpoint:", ckpt_path)

print("Training complete.")
