import matplotlib.pyplot as plt

def show_maze(maze, path=None):
    plt.imshow(maze, cmap='gray_r')
    if path:
        xs, ys = zip(*path)
        plt.plot(ys, xs, 'r-')
    plt.show()
