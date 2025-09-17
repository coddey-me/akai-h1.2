import numpy as np
from collections import deque

def generate_maze(size=8, wall_prob=0.2):
    """Generate random maze with start and goal, plus shortest path."""
    maze = (np.random.rand(size, size) < wall_prob).astype(np.int32)
    maze[0, 0] = 0  # start
    maze[-1, -1] = 0  # goal
    path = shortest_path(maze, (0, 0), (size - 1, size - 1))
    return maze, path

def shortest_path(grid, start, goal):
    """BFS shortest path; returns list of moves (0=U,1=D,2=L,3=R)."""
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move_ids = [0, 1, 2, 3]
    size = grid.shape[0]
    q = deque([(start, [])])
    visited = set([start])
    while q:
        (x, y), path = q.popleft()
        if (x, y) == goal:
            return path
        for (dx, dy), mid in zip(moves, move_ids):
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append(((nx, ny), path + [mid]))
    return []  # no path
