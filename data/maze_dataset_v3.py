# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset
from .generate_maze_v3 import generate_maze_and_path

# data/maze_dataset_v3.py
# data/maze_dataset_v3.py
import torch
from torch.utils.data import Dataset

class MazeDatasetV3(Dataset):
    """
    Loads precomputed mazes from a .pt file.
    Supports dict with 'mazes'/'paths' or list of tuples.
    """
    def __init__(self, precomputed_file=None):
        data = torch.load(precomputed_file)

        # handle both formats
        if isinstance(data, dict) and "mazes" in data and "paths" in data:
            self.mazes = data["mazes"]
            self.paths = data["paths"]
        else:
            # assume list of (maze, path)
            mazes, paths = zip(*data)
            self.mazes = mazes
            self.paths = paths

        # convert to tensors if needed
        self.mazes = [torch.as_tensor(m).float() for m in self.mazes]
        self.paths = [torch.as_tensor(p).long() for p in self.paths]

    def __len__(self):
        return len(self.mazes)

    def __getitem__(self, idx):
        return self.mazes[idx], self.paths[idx]

