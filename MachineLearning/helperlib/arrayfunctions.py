import numpy as np


def get_grid_points(npx=200, npy=200, limits=(-1, 1)):
    x_axis = np.linspace(*limits, npx)
    y_axis = np.linspace(*limits, npx)
    grid = np.array(np.meshgrid(x_axis, y_axis))
    grid_points = grid.reshape([2, npx * npy]).T
    return grid_points, x_axis, y_axis


def accuracy(data_predicted, data_test):
    data_predicted_wrong = np.logical_xor(data_predicted, data_test)
    return 1 - sum(data_predicted_wrong) / len(data_predicted_wrong)