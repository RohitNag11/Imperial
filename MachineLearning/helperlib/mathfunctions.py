import numpy as np


def mat_mul(mat_array):
    res = mat_array[0]
    for mat in mat_array[1:]:
        res = np.matmul(res, mat)
    return res


def get_rot_mat(theta_deg=0, degree=2, axis='z'):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    if degree == 2:
        match axis:
            case 'z':
                return np.array([[c, s], [-s, c]])
            case _:
                raise Exception(
                    f"'{axis}' is an invalid axis. Tip: Choose between x, y or z.")
    if degree == 3:
        match axis:
            case 'x':
                return np.array([[1, 0, 0], [0, c, s], [0, s, c]])
            case 'y':
                return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            case 'z':
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            case _:
                raise Exception(
                    f"'{axis}' is an invalid axis. Tip: Choose between x, y or z.")
    raise Exception("Invalid degree. Tip: Choose between 2 or 3.")


def p_coordinate(x, covar):
    '''
    Calulcates the probability for a given coordinate from an assumed normal probability distributions.
    Note: mean = [0, 0]^T
    Returns: a float
    '''
    pow = -0.5 * mat_mul([x.T, np.linalg.inv(covar), x])
    coeff = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covar)))
    return coeff * np.exp(pow)


def p_grid(grid_points, covar):
    '''
    Calulcates the probability distribution function for all coordinates in a grid.
    Note: mean = [0, 0]^T
    Returns: a (1 x n) array for grid_points size of (n x 2)
    '''
    mul = np.matmul(grid_points, np.linalg.inv(covar)) * grid_points
    pow = -0.5 * np.sum(mul, axis=1)
    coeff = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covar)))
    return coeff * np.exp(pow)


def gaussian(x=0, mean=0, std=1):
    '''
    Calculates the gaussian function for a given x, mean and standard deviation.
    '''
    pow = -0.5 * ((x - mean) / std)**2
    coeff = 1 / (np.sqrt(2 * np.pi) * std)
    return coeff * np.exp(pow)
