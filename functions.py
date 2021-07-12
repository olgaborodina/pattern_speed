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

def symmetrize_tw_integral(flux, x, y):
    flux_flat = flux.flatten()
    x_lon = x.flatten()
    y_lon = y.flatten()
    flux_len = np.arange(len(flux_flat))

    for y_lon_unique in np.unique(y_lon):

        # Take into account any NaNs in the image

        nan_idx = np.where(np.isnan(flux_flat) == False)

        y_lon_idx = np.where(y_lon[nan_idx] == y_lon_unique)

        if len(y_lon_idx[0]) == 0:
            continue

        # Find the most negative and positive x_lon at these positions.

        x_lon_slice = x_lon[nan_idx][y_lon_idx]

        x_lon_min = np.min(x_lon_slice)
        x_lon_max = np.max(x_lon_slice)

        # For things where we only have positive or negative data, remove everything

        if x_lon_min * x_lon_max > 0:

            x_lon_thresh = 0

        else:

            x_lon_thresh = np.min([np.abs(x_lon_max),
                                   np.abs(x_lon_min)])

        x_lon_idx = np.where(np.abs(x_lon[nan_idx][y_lon_idx]) > x_lon_thresh)

        if len(x_lon_idx[0]) == 0:
            continue

        final_idxs = flux_len[nan_idx][y_lon_idx][x_lon_idx]

        for final_idx in final_idxs:
            flux_flat[final_idx] = np.nan

    # Reshape the array and return

    flux_final = flux_flat.reshape(flux.shape)

    return flux_final

def make_symmetric(x_center_array, array):
    left_side  = x_center_array[:, 0] <= 0
    right_side = x_center_array[:, 0] >= 0
    array_ = array.copy()
    
    for ind in range(x_center_array.shape[1]):
        left_number  = np.isnan(array_[left_side,  ind]).sum()
        right_number = np.isnan(array_[right_side, ind]).sum()
        
        if left_number > right_number:
            array_[-left_number:, ind] = np.nan
        elif left_number < right_number:
            array_[:right_number, ind] = np.nan

    return array_

def nan_equal(a,b):
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def add_uncertanties(vr, phi, x, y, vr_scale, phi_scale, pa_scale):
    np.random.seed(100)
    vr_  = np.random.normal(loc=vr,  scale=vr_scale)
    phi_ = np.random.normal(loc=phi, scale=phi_scale)
    pa   = np.random.uniform(-pa_scale, pa_scale)
    x_, y_, vx_, vy_ = rotate_bar(np.deg2rad(pa), x, y, 0, 0)
    
    return vr_, phi_, x_, y_
    