# data/generate_maze_v3.py
import numpy as np
from collections import deque

def generate_maze(size=20, p_block=0.3):
    """
    Generates a random maze with 0=open,1=wall and ensures start & end open.
    """
    maze = (np.random.rand(size, size) < p_block).astype(np.int32)
    maze[0,0] = 0
    maze[-1,-1] = 0
    return maze

def shortest_path(maze, start, goal):
    """
    BFS shortest path from start to goal.
    Returns a list of actions: 0=up,1=down,2=left,3=right
    """
    h,w = maze.shape
    q = deque([(start, [])])
    visited = set([start])
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        (x,y), path = q.popleft()
        if (x,y)==goal:
            return path
        for i,(dx,dy) in enumerate(dirs):
            nx,ny=x+dx,y+dy
            if 0<=nx<h and 0<=ny<w and maze[nx,ny]==0 and (nx,ny) not in visited:
                visited.add((nx,ny))
                q.append(((nx,ny), path+[i]))
    return []  # no path found

def generate_maze_and_path(size=20):
    """
    Generates maze and its shortest path.
    """
    maze = generate_maze(size)
    path = shortest_path(maze, (0,0), (size-1,size-1))
    return maze, path
