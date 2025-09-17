import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data.dataset import MazeDataset
from models.hrm import HRM

#for v2
from models.hrm_v2 import HRM_V2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
maze_size = 50
batch_size = 32
epochs = 300
lr = 1e-3

# === Data ===
dataset = MazeDataset(n_samples=100000, size=maze_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

# === Model ===
#add iff to recognze v2
# choose model variant: 'v1' or 'v2'
model_variant = 'v2'   # set to 'v2' to use the new model

if model_variant == 'v2':
    # smaller configuration to keep training fast; tweak as needed
    model = HRM_V2(embed_dim=128, high_hidden=64, low_hidden=128, n_actions=4).to(device)
else:
 
    model = HRM(maze_size=8).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        mazes = [b[0] for b in batch]
        paths = [b[1] for b in batch]
        max_len = max(len(p) for p in paths)
        # pad paths
        padded_paths = torch.full((len(paths), max_len), 0, dtype=torch.long)
        for i,p in enumerate(paths):
            padded_paths[i,:len(p)] = p
        
        mazes = torch.cat(mazes, dim=0).to(device)  # (B,H,W) probably
        targets = padded_paths.to(device)
        
        # === FIX: ensure mazes has shape (B,1,H,W) ===
        if mazes.ndim == 3:
            mazes = mazes.unsqueeze(1)  # (B,1,H,W)
        
        if mazes.ndim == 4 and mazes.shape[1] != 1:
            # average across channels to get 1 channel
            mazes = mazes.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # === END FIX ===
        
        optimizer.zero_grad()
        logits = model(mazes, seq_len=max_len)

        logits = logits.view(-1,4)
        targets = targets.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
