import numpy as np
import scipy.stats as scp
from scipy.odr import Model, ODR, RealData

import os
import pickle

import emcee
import numpy as np
import pybar.pybar_tom as pybar
from tqdm import tqdm


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

def add_uncertanties(vr, rho, vr_scale, rho_scale):
    """
    Adds uncertainties to the velocity, flux with corresponding scales.
    Inputs: Values of velocity, flux and Cartesian coordinates and scales of randomization.
    Outputs: New values of velocity, flux and Cartesian coordinates.
    """ 
    np.random.seed(100)
    vr_  = np.random.normal(loc=vr,  scale=vr_scale)
    rho_ = np.random.normal(loc=rho, scale=rho_scale)
    return vr_, rho_


##################################################### Tom's code:#################################################

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



def log_prob(theta, flux, flux_err, vel, vel_err,
             x_cen, y_cen, centering_err,
             pa, pa_err, inclination,
             grid_step, slit_width):
    r0, c, m1, m2 = theta

    pa_bootstrap = pa + np.random.normal(loc=0, scale=pa_err)

    # Swing this around if it passes through 0/360 degrees.

    if pa_bootstrap < 0:
        pa_bootstrap += 360
    if pa_bootstrap > 360:
        pa_bootstrap -= 360

    x_cen_bootstrap = x_cen + np.random.normal(loc=0, scale=centering_err)
    y_cen_bootstrap = y_cen + np.random.normal(loc=0, scale=centering_err)

    grid_shape = flux.shape
    grid_x = (np.arange(grid_shape[1]) - x_cen_bootstrap) * grid_step
    grid_y = (np.arange(grid_shape[0]) - y_cen_bootstrap) * grid_step

    x_coords, y_coords = np.meshgrid(grid_x, grid_y)

    bar = pybar.mybar(Flux=flux, Flux_err=flux_err,
                      Velocity=vel, Velocity_err=vel_err,
                      Xin=x_coords, Yin=y_coords, PAnodes=pa_bootstrap,
                      inclin=inclination)

    bar.tremaine_weinberg(slit_width=slit_width)

    x_tw = bar.dfx_tw
    v_tw = bar.dfV_tw

    x_tw_err = bar.dfx_tw_err
    v_tw_err = bar.dfV_tw_err

    if m1 > 0 and m2 > 0 and 0 <= r0 <= np.nanmax(np.abs(x_tw)):
        return log_likelihood(theta, x_tw, v_tw, x_tw_err, v_tw_err)
    return -np.inf


def log_likelihood(theta, x_tw, v_tw, x_tw_err, v_tw_err):
    r0, c, m1, m2 = theta

    # Split up the data by r0 into the two slopes we want to fit

    idx_inner = np.where(np.abs(x_tw) <= r0)
    model_inner = x_tw[idx_inner] * m1 + c
    chisq_inner = np.nansum((v_tw[idx_inner] - model_inner) ** 2 /
                            (v_tw_err[idx_inner] ** 2 + (m1 * x_tw_err[idx_inner]) ** 2))

    idx_outer = np.where(np.abs(x_tw) > r0)
    model_outer = x_tw[idx_outer] * m2 + c
    chisq_outer = np.nansum((v_tw[idx_outer] - model_outer) ** 2 /
                            (v_tw_err[idx_outer] ** 2 + (m2 * x_tw_err[idx_outer]) ** 2))

    chisq = chisq_inner + chisq_outer

    return -0.5 * chisq


def bootstrap_tw(flux, flux_err,
                 vel, vel_err,
                 x_array, y_array,
                 grid_step=0.2, slit_width=1,
                 centering_err=1,
                 pa=45, pa_err=1,
                 inclination=30,
                 n_bootstraps=1000,
                 save_in_file=False,
                 bootstrap_filename='bootstraps.txt',
                 overwrite_bootstraps=False,
                 pattern_speed_filename='pattern_speeds.txt',
                 correction_type=None
                 ):
    """Bootstrapped errors for the Tremaine-Weinberg pattern speed method.
    Bootstrap wrapper (bootstwrapper?) around pybar. For n_bootstraps, will perturb the position angle and centre by
    pa_err and centering_err, respectively. pybar will then do the TW integral using this setup, and fitting is done
    accounting for x- and y-errors using SciPy's ODR routines. Will finally produce a pattern speed and associated
    errors (16th and 84th percentiles), and save those to pattern_speed_filename.
    Args:
        flux (numpy.ndarray): 2D array representing the flux (e.g. stellar mass, H alpha flux).
        flux_err (numpy.ndarray): 2D array representing per-pixel flux error.
        vel (numpy.ndarray): 2D array representing velocity associated with the flux above (e.g. stellar velocity, H
            alpha velocity). Should be in units of km/s
        vel_err (numpy.ndarray): 2D array representing per-pixel velocity error. N.B. these four arrays should all be
            the same size!
        grid_step (float, optional): The pixel size of the image, in arcsec. Defaults to 0.2, which is for MUSE data.
        slit_width (float, optional): The width of each slit, in arcsec. Defaults to 1, which is the pybar default.
        x_cen (float, optional): The x-centre of the galaxy, in pixels. Defaults to None, which will use half way along
            the image.
        y_cen (float, optional): The y-centre of the galaxy, in pixels. Defaults to None, which will use half way along
            the image.
        centering_err (float, optional): The error in this centering, in pixels. Defaults to 1 pixel.
        pa (float, optional): The position angle of the galaxy (measured left of north), in degrees. Defaults to 45
            degrees.
        pa_err (float, optional): The error on this position angle, in degrees. Defaults to 1 degree
        inclination (float, optional): The galaxy inclination, in degrees. Defaults to 30 degrees.
        dist (float, optional): The distance to the galaxy, in Mpc. Defaults to 1Mpc.
        n_bootstraps (int, optional): The number of bootstraps to run. Defaults to 1000.
        bootstrap_filename (str, optional): Will save the calculated m and c values from the straight line fit to a text
            file. This can speed up later runs. Defaults to 'bootstraps.txt'
        overwrite_bootstraps (bool, optional): If False, will attempt to read in existing bootstraps and fit from there.
            Otherwise will fit for every bootstrap iteration regardless of whether fitting has been done before.
            Defaults to False.
        pattern_speed_filename (str, optional): Where to save the final pattern speed (and error) output. Defaults to
            'pattern_speeds.txt'.
    """

    # If centres not specified, use centre of image.


    # Set up arrays for m and c. Load in any if applicable

    m_bootstrap = np.zeros(n_bootstraps)
    c_bootstrap = np.zeros_like(m_bootstrap)

    if not overwrite_bootstraps:

        try:
            m_loaded, c_loaded = np.loadtxt(bootstrap_filename,
                                            unpack=True)

            m_bootstrap[:len(m_loaded)] = m_loaded
            c_bootstrap[:len(c_loaded)] = c_loaded

        except OSError:
            pass

    for bootstrap_i in tqdm(range(n_bootstraps)):

        if m_bootstrap[bootstrap_i] != 0:
            # If we've already fitted here, just skip.

            continue

        pa_bootstrap = pa + np.random.normal(loc=0, scale=pa_err)

        # Swing this around if it passes through 0/360 degrees.

        if pa_bootstrap < 0:
            pa_bootstrap += 360
        if pa_bootstrap > 360:
            pa_bootstrap -= 360

        x_array += np.random.normal(loc=0, scale=centering_err)
        y_array += np.random.normal(loc=0, scale=centering_err)

        x_coords, y_coords = np.meshgrid(x_array, y_array, indexing='ij')

        bar = pybar.mybar(Flux=flux, Flux_err=flux_err,
                          Velocity=vel, Velocity_err=vel_err,
                          Xin=x_coords, Yin=y_coords, PAnodes=pa_bootstrap,
                          inclin=inclination)

        bar.tremaine_weinberg(slit_width=slit_width)

        x_tw = bar.dfx_tw
        v_tw = bar.dfV_tw

        x_tw_err = bar.dfx_tw_err
        v_tw_err = bar.dfV_tw_err

        m, m_err, c, c_err = odr_fit(x_tw, x_tw_err, v_tw, v_tw_err)

        m_bootstrap[bootstrap_i] = m * 100 / np.sin(np.deg2rad(inclination))
        c_bootstrap[bootstrap_i] = c * 100

    # Now we've bootstrapped, pull out the pattern speed and associated errors.

    omega_bar = np.nanmedian(m_bootstrap)
    omega_bar_err_up = np.nanpercentile(m_bootstrap, 84) - omega_bar
    omega_bar_err_down = omega_bar - np.nanpercentile(m_bootstrap, 16)
    
    # Write this pattern speed out
    
    if save_in_file:
        
        np.savetxt(pattern_speed_filename,
                   np.c_[omega_bar, omega_bar_err_up, omega_bar_err_down],
                   header='omega_bar, omega_bar_err_up, omega_bar_err_down (all km/s/kpc)')

        # Also save out each individual m and c

        np.savetxt(bootstrap_filename,
                   np.c_[m_bootstrap, c_bootstrap],
                   header='m, c')
    
    return omega_bar, omega_bar_err_up, omega_bar_err_down

