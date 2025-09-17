# utils/collate_v3.py  (new file)
import torch
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
