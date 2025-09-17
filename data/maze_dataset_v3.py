# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset
from .generate_maze_v3 import generate_maze_and_path

class MazeDatasetV3(Dataset):
    """
    Generates mazes on-the-fly up to 100x100 with shortest paths.
    """
    def __init__(self, n_samples=10000, size=20):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        maze, path = generate_maze_and_path(size=self.size)
        maze_tensor = torch.tensor(maze, dtype=torch.float).unsqueeze(0)  # (1,H,W)
        path_tensor = torch.tensor(path, dtype=torch.long)
        return maze_tensor, path_tensor
