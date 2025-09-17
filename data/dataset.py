import torch
from torch.utils.data import Dataset
from .generate_maze import generate_maze

class MazeDataset(Dataset):
    def __init__(self, n_samples=1000, size=8):
        self.data = []
        for _ in range(n_samples):
            maze, path = generate_maze(size=size)
            if len(path) == 0:
                continue
            self.data.append((maze, path))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        maze, path = self.data[idx]
        maze_tensor = torch.tensor(maze, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        path_tensor = torch.tensor(path, dtype=torch.long)
        return maze_tensor, path_tensor
