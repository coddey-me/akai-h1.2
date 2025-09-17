# scripts/precompute_mazes.py
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from .data.generate_maze_v3 import generate_maze_and_path

def make_one(size):
    maze, path = generate_maze_and_path(size=size)
    return torch.tensor(maze, dtype=torch.float).unsqueeze(0), torch.tensor(path, dtype=torch.long)

def main(n_samples=10000, size=50, out_file="precomputed_mazes.pt"):
    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} CPU cores to generate mazes...")

    with Pool(n_workers) as pool:
        results = list(tqdm(pool.imap(partial(make_one, size), range(n_samples)), total=n_samples))

    mazes = [r[0] for r in results]
    paths = [r[1] for r in results]

    torch.save({"mazes": mazes, "paths": paths}, out_file)
    print(f"Saved {len(mazes)} mazes to {out_file}")

if __name__ == "__main__":
    # example usage:
    main(n_samples=20000, size=50, out_file="precomputed_mazes.pt")
