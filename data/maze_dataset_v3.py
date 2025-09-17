# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset

class MazeDatasetV3(Dataset):
    """
    Loads precomputed mazes from a .pt file, filters out invalid paths.
    """
    def __init__(self, precomputed_file):
        data = torch.load(precomputed_file)
        filtered = [(m, p) for m, p in zip(data["mazes"], data["paths"]) if len(p) > 0]
        if len(filtered) == 0:
            raise ValueError("No valid mazes with non-empty paths found!")
        self.mazes, self.paths = zip(*filtered)

    def __len__(self):
        return len(self.mazes)

    def __getitem__(self, idx):
        return self.mazes[idx], self.paths[idx]
