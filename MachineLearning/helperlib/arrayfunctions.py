import numpy as np


def get_grid_points(npx=200, npy=200, limits=(-1, 1)):
    x_axis = np.linspace(*limits, npx)
    y_axis = np.linspace(*limits, npx)
    grid = np.array(np.meshgrid(x_axis, y_axis))
    grid_points = grid.reshape([2, npx * npy]).T
    return grid_points, x_axis, y_axis
