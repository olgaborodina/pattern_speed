import numpy as np
import scipy.stats as scp

def rotate_bar(beta, X, Y, VX, VY):
    X_ = X * np.cos(beta) - Y * np.sin(beta)
    Y_ = X * np.sin(beta) + Y * np.cos(beta)

    VX_ = VX * np.cos(beta) - VY * np.sin(beta)
    VY_ = VX * np.sin(beta) + VY * np.cos(beta)
    return X_, Y_, VX_, VY_

def incline_galaxy(i, X, Y, VX, VY):
    VX_ = VX
    VZ_ = VY * np.sin(i)
    VY_ = VY * np.cos(i)

    X_ = X
    Z_ = Y * np.sin(i)
    Y_ = Y * np.cos(i)
    return X_, Y_, VZ_

def add_solid_body_rotation(X, Y, VX, VY, Omegap=0.4):
    PHI = np.arctan2(Y, X)
    VX -= np.sqrt(X ** 2 + Y ** 2) * Omegap * np.sin(PHI)
    VY += np.sqrt(X ** 2 + Y ** 2) * Omegap * np.cos(PHI)
    return VX, VY

def mean_in_pixel(X, Y, step, values):
    bounds_x = np.arange(X.min().round(1) - step, X.max().round(1) + step, step=step)
    bounds_y = np.arange(Y.min().round(1) - step, Y.max().round(1) + step, step=step)
    statistics, x_edge, y_edge, bins = scp.binned_statistic_2d(X.flatten(), Y.flatten(), values.flatten(),
                                                               'mean', bins=[bounds_x, bounds_y])
    return statistics

def centers_of_pixel(X, Y, step):
    bounds_x = np.arange(X.min().round(1) - step, X.max().round(1) + step, step=step)
    bounds_y = np.arange(Y.min().round(1) - step, Y.max().round(1) + step, step=step)
    return np.meshgrid(bounds_x[:-1] + step, bounds_y[:-1] + step, indexing='ij')