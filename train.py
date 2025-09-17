import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from data.dataset import MazeDataset
from models.hrm import HRM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
maze_size = 8
batch_size = 32
epochs = 5
lr = 1e-3

# === Data ===
dataset = MazeDataset(n_samples=1000, size=maze_size)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

# === Model ===
model = HRM(maze_size=maze_size).to(device)
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
        mazes = torch.cat(mazes, dim=0).to(device)
        targets = padded_paths.to(device)

        optimizer.zero_grad()
        logits = model(mazes, seq_len=max_len)
        logits = logits.view(-1,4)
        targets = targets.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
