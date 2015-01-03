import numpy as np

N = 200
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))


def sample_scatterplot(ax, cmap):
    return ax.scatter(array_dg[:, 0], array_dg[:, 1], s=60, c=colors, cmap=cmap)
