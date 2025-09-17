# eval_v3.py
import torch
from models.hrm_v3 import HRM_V3
from data.maze_dataset_v3 import MazeDatasetV3
from torch.utils.data import DataLoader

# --- CONFIG ---
checkpoint_path = "checkpoints/hrm_v3_epoch10.pt"
precomputed_file = "precomputed_mazes.pt"   # or your test file
maze_size = 50  # your maze size during training
batch_size = 16
seq_len = 50    # how many steps to predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
model = HRM_V3(embed_dim=256, high_hidden=512, low_hidden=512,
               n_actions=4, vocab_size=500, maze_size=maze_size).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- LOAD DATA ---
dataset = MazeDatasetV3(precomputed_file)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- EVAL LOOP ---
correct = 0
total = 0

with torch.no_grad():
    for mazes, paths in loader:
        mazes, paths = mazes.to(device), paths.to(device)
        logits = model(mazes, seq_len=seq_len, task='maze')  # (B,seq_len,4)
        preds = logits.argmax(-1)  # predicted actions

        # Compare only up to the true path length
        min_len = min(paths.size(1), preds.size(1))
        match = (preds[:, :min_len] == paths[:, :min_len]).float().mean(dim=1)
        correct += match.sum().item()
        total += match.size(0)

print(f"Average path agreement: {correct/total:.4f}")
