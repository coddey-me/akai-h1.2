# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset
from .generate_maze_v3 import generate_maze_and_path

# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset

class MazeDatasetV3(Dataset):
    """
    Loads precomputed mazes from a .pt file
    """
    def __init__(self, precomputed_file=None):
        data = torch.load(precomputed_file)
        self.mazes = data["mazes"]
        self.paths = data["paths"]

    def __len__(self):
        return len(self.mazes)

    def __getitem__(self, idx):
        return self.mazes[idx], self.paths[idx]
