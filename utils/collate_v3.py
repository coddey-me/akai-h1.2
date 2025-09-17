# utils/collate_v3.py  (new file)
import torch
from torch.utils.data import DataLoader
import torch
def safe_maze_collate(batch):
    mazes, paths = zip(*batch)
    
    # Stack mazes (images must be same size HxW)
    mazes = torch.stack([m if m.ndim == 3 else m.unsqueeze(0) for m in mazes], dim=0)
    
    # Pad paths to max length
    lengths = [len(p) if isinstance(p, torch.Tensor) else len(torch.as_tensor(p)) for p in paths]
    max_len = max(lengths) if lengths else 1  # avoid zero-length
    padded_paths = torch.full((len(paths), max_len), -100, dtype=torch.long)
    for i, p in enumerate(paths):
        p_tensor = p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=torch.long)
        if p_tensor.numel() > 0:  # skip empty paths
            padded_paths[i, :p_tensor.size(0)] = p_tensor
    
    return mazes, padded_paths

def pad_collate(batch):
    mazes, paths = zip(*batch)
    mazes = torch.stack(mazes)  # mazes are always same shape
    
    max_len = max(len(p) for p in paths)  # longest path in batch
    padded_paths = torch.zeros(len(paths), max_len, dtype=torch.long)
    
    for i, p in enumerate(paths):
        padded_paths[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    
    return mazes, padded_paths



def collate_fn(batch):
    mazes, paths = zip(*batch)
    mazes = torch.stack(mazes)
    max_len = max(len(p) for p in paths)
    padded_paths = torch.zeros(len(paths), max_len, dtype=torch.long)
    for i, p in enumerate(paths):
        padded_paths[i, :len(p)] = torch.tensor(p, dtype=torch.long)
    return mazes, padded_paths

loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

def maze_collate_fn(batch):
    """
    batch: list of (maze_tensor, path_tensor)
    """
    mazes = [b[0] for b in batch]
    paths = [b[1] for b in batch]

    # stack mazes
    mazes = torch.stack(mazes, dim=0)  # [B,1,H,W]

    # pad paths to max length
    max_len = max(len(p) for p in paths)
    padded_paths = torch.full((len(paths), max_len), 0, dtype=torch.long)
    for i,p in enumerate(paths):
        padded_paths[i,:len(p)] = p

    return mazes, padded_paths
