import numpy as np
import scipy.stats as scp
from scipy.odr import Model, ODR, RealData

def rotate_bar(beta, X, Y, VX, VY):
    """
    Adds the rotation by an angle beta.
    Inputs: Angle beta (in radians), Cartesian coordinates X and Y, and velocity projections VX and VY.
    Outputs: X, Y, VX, VY after the rotation
    """
    X_ = X * np.cos(beta) - Y * np.sin(beta)
    Y_ = X * np.sin(beta) + Y * np.cos(beta)

    VX_ = VX * np.cos(beta) - VY * np.sin(beta)
    VY_ = VX * np.sin(beta) + VY * np.cos(beta)
    return X_, Y_, VX_, VY_

def incline_galaxy(i, X, Y, VX, VY):
    """
    Incines galaxy by an angle i.
    Inputs: Angle i (in radians), Cartesian coordinates X and Y, and velocity projections VX and VY.
    Outputs: X (which is the same as input...), Y and line of sight velocity.
    """
    VX_ = VX
    VZ_ = VY * np.sin(i)
    VY_ = VY * np.cos(i)

    X_ = X
    Z_ = Y * np.sin(i)
    Y_ = Y * np.cos(i)
    return X_, Y_, VZ_

def add_solid_body_rotation(X, Y, VX, VY, Omegap=0.4):
    """
    Adds solid body rotation.
    Inputs: Cartesian coordinates X and Y, velocity projections VX and VY, and angular velocity.
    Outputs: new velocity projections VX and VY.
    """
    PHI = np.arctan2(Y, X)
    VX -= np.sqrt(X ** 2 + Y ** 2) * Omegap * np.sin(PHI)
    VY += np.sqrt(X ** 2 + Y ** 2) * Omegap * np.cos(PHI)
    return VX, VY

def mean_in_pixel(X, Y, step, values):
    """
    Calculates the mean of values in each pixel made from X and Y data with sides equal to step value.
    Inputs: Cartesian coordinates X and Y (2D arrays with the same shape), step (float, the size of the pixel), values (2D array with the shape of X and Y).
    Outputs: mean values per pixel (2D array).
    """
    bounds_x = np.arange(X.min().round(1) - step, X.max().round(1) + step, step=step)
    bounds_y = np.arange(Y.min().round(1) - step, Y.max().round(1) + step, step=step)
    statistics, x_edge, y_edge, bins = scp.binned_statistic_2d(X.flatten(), Y.flatten(), values.flatten(),
                                                               'mean', bins=[bounds_x, bounds_y])
    return statistics

def centers_of_pixel(X, Y, step):
    """
    Calculates the center of each pixel.
    Inputs: Cartesian coordinates X and Y (2D arrays with the same shape), step (float, the size of the pixel).
    Outputs: centers of the pixels (two 2D arrays corresponding to X_centers and Y_centers).
    """    
    bounds_x = np.arange(X.min().round(1) - step, X.max().round(1) + step, step=step)
    bounds_y = np.arange(Y.min().round(1) - step, Y.max().round(1) + step, step=step)
    return np.meshgrid(bounds_x[:-1] + step, bounds_y[:-1] + step, indexing='ij')

def symmetrize_tw_integral(flux, x, y):
    """
    Makes slits symmetric. Tom's variant.
    Inputs: Flux (2D array), Cartesian coordinates X and Y (2D arrays with the same shape).
    Outputs: Symmetric flux with the shape of input flux.
    """    
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
    """
    Makes slits symmetric. My variant.
    Inputs: Cartesian coordinate X (2D array) and Flux (2D array, the same shape).
    Outputs: Symmetric flux with the shape of input flux.
    """    
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
    """
    Checks if array are the same, even if they have NaNs in them.
    Inputs: Any two arrays.
    Outputs: True/False.
    """ 
    try:
        np.testing.assert_equal(a,b)
    except AssertionError:
        return False
    return True

def add_uncertanties(vr, rho, x, y, vr_scale, rho_scale, pa_scale):
    """
    Adds uncertainties to the velocity, flux and positional angle with corresponding scales.
    Inputs: Values of velocity, flux and Cartesian coordinates and scales of randomization.
    Outputs: New values of velocity, flux and Cartesian coordinates.
    """ 
    np.random.seed(100)
    vr_  = np.random.normal(loc=vr,  scale=vr_scale)
    rho_ = np.random.normal(loc=rho, scale=rho_scale)
    pa   = np.random.uniform(-pa_scale, pa_scale)
    x_, y_, vx_, vy_ = rotate_bar(np.deg2rad(pa), x, y, 0, 0)
    
    return vr_, rho_, x_, y_

def linear_fit(theta, x):
    """
    Linear fitting function
    Inputs: Parameters (1D array with the length=2) and x
    Outputs: y
    """
    return theta[0] * x + theta[1]


def odr_fit(x, x_err, y, y_err):
    """
    Fits data with linear function
    Inputs: x and its errors x_err, y and its errors y_err (1D arrays all of the same shape)
    Outputs: best fitting parameters of linear function and their errors (all floats)
    """
    linear = Model(linear_fit)

    # Filter out any NaNs else ODR breaks.

    nan_idx = np.where(np.isnan(x) == False)

    # Fit using SciPy's ODR

    odr_data = RealData(x[nan_idx], y[nan_idx],
                        sx=x_err[nan_idx], sy=y_err[nan_idx])

    odr_run = ODR(odr_data, linear, beta0=[2, 0])

    odr_output = odr_run.run()

    m, c = odr_output.beta
    m_err, c_err = odr_output.sd_beta

    return m, m_err, c, c_err
    