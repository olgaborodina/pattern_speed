import numpy as np
import scipy.stats as scp
from scipy.odr import Model, ODR, RealData

import os
import pickle
from multiprocessing import Pool
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

def mean_in_pixel(X, Y, step, values, if_xy=False):
    """
    Calculates the mean of values in each pixel made from X and Y data with sides equal to step value.
    Inputs: Cartesian coordinates X and Y (2D arrays with the same shape), step (float, the size of the pixel), values (2D array with the shape of X and Y).
    Outputs: mean values per pixel (2D array).
    """
    bound_max_x = np.nanmax([abs(X.min().round(1)), X.max().round(1)])
    bound_max_y = np.nanmax([abs(Y.min().round(1)), Y.max().round(1)])

    right_side_x = np.arange(0, bound_max_x + step, step=step)
    right_side_y = np.arange(0, bound_max_y + step, step=step)

    bounds_x = np.concatenate([np.sort(-right_side_x[1:]), right_side_x])
    bounds_y = np.concatenate([np.sort(-right_side_y[1:]), right_side_y])

    statistics, x_edge, y_edge, bins = scp.binned_statistic_2d(X.flatten(), Y.flatten(), values.flatten(),
                                                               'mean', bins=[bounds_x, bounds_y])

    if if_xy:
        x_center_array = x_edge[:-1] + step / 2
        y_center_array = y_edge[:-1] + step / 2
        return statistics, np.meshgrid(x_center_array, y_center_array, indexing='ij')
    else: return statistics

    

def centers_of_pixel(X, Y, step):
    """
    Calculates the center of each pixel.
    Inputs: Cartesian coordinates X and Y (2D arrays with the same shape), step (float, the size of the pixel).
    Outputs: centers of the pixels (two 2D arrays corresponding to X_centers and Y_centers).
    """
    bound_max_x = np.nanmax([abs(X.min().round(1)), X.max().round(1)])
    bound_max_y = np.nanmax([abs(Y.min().round(1)), Y.max().round(1)])
    
    bounds_x = np.arange(-bound_max_x, bound_max_x+step, step=step)
    bounds_y = np.arange(-bound_max_y, bound_max_y+step, step=step)
    return np.meshgrid(bounds_x, bounds_y, indexing='ij')

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
    np.random.seed(1)
    vr_  = np.random.normal(loc=vr,  scale=vr_scale)
    rho_ = np.random.normal(loc=rho, scale=rho_scale)
    
    return vr_, rho_


def linear_fit(theta, x):
    """
    Linear fitting function
    Inputs: Parameters (1D array with the length=2) and x
    Outputs: y
    """
    return theta[0] * x + theta[1]

def remove_nan(a):
    return np.array([np.nan if x=='NaN' else x for x in a])

def fit(bar, small_values_cut=True):
    x_tw = remove_nan(bar.dfx_tw)
    v_tw = remove_nan(bar.dfV_tw)

    x_tw_err = remove_nan(bar.dfx_tw_err)
    v_tw_err = remove_nan(bar.dfV_tw_err)
    
    if small_values_cut == True:
        not_bar_mask = ((bar.y_slits <
            abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) + 5 * bar.slit_width) &
                    (bar.y_slits >
            - abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) - 5 * bar.slit_width) &
            (np.abs(x_tw) >  bar.slit_width))
    else:
        not_bar_mask = ((bar.y_slits <
            abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) + 5 * bar.slit_width) &
                    (bar.y_slits >
            - abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) - 5 * bar.slit_width))

    if len(x_tw[not_bar_mask]) == 0:
        m = m_err = c = c_err = np.nan 
    else:
        m, m_err, c, c_err = odr_fit(x_tw[not_bar_mask], x_tw_err[not_bar_mask], v_tw[not_bar_mask], v_tw_err[not_bar_mask])
    return m, m_err, c, c_err, x_tw, v_tw, x_tw_err, v_tw_err

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

    odr_run = ODR(odr_data, linear, beta0=[0.4, 0])

    odr_output = odr_run.run()

    m, c = odr_output.beta
    m_err, c_err = odr_output.sd_beta

    return m, m_err, c, c_err
    
def bootstrap_iteration(flux, vel, flux_err, vel_err, 
         X, Y,
         step,
         centering_err,
         pa, pa_err, 
         inclination, beta,
         bar_length):
    
    pa_bootstrap = pa_err
    X += centering_err[0]
    Y += centering_err[1]
    
    bar = pybar.mybar(Flux=flux, Flux_err=flux_err,
              Velocity=vel, Velocity_err=vel_err,
              Yin=Y, Xin=X,
              inclin=np.rad2deg(inclination), PAnodes=np.rad2deg(pa_bootstrap), beta=np.rad2deg(beta))

    bar.tremaine_weinberg()

    x_tw = bar.dfx_tw
    v_tw = bar.dfV_tw
    
    x_tw_err = bar.dfx_tw_err
    v_tw_err = bar.dfV_tw_err

    bar_mask = ((bar.y_slits <
            abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) + 5 * bar.slit_width) |
                (bar.y_slits >
            - abs(bar.bar_length * np.cos(np.deg2rad(bar.inclin)) * np.sin(np.deg2rad(bar.beta))) - 5 * bar.slit_width) |
                (np.abs(x_tw) > step))
    if len(x_tw[bar_mask]) == 0:
        m = 0
        c = 0
        print('I am hereeee')
    else:
        m, m_err, c, c_err = odr_fit(x_tw[bar_mask], x_tw_err[bar_mask], v_tw[bar_mask], v_tw_err[ar_mask])
    return [m, c]

def bootstrap_tw(flux, vel, flux_err, vel_err, 
                 X, Y,
                 step,
                 centering_err,
                 pa, pa_err, 
                 inclination, beta,
                 bar_length,
                 n_bootstraps=1000,
                 save_in_file=False,
                 bootstrap_filename='test_bootstraps.dat',
                 pattern_speed_filename='test_patternspeeds.dat'
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

    m_bootstrap = []
    c_bootstrap = []

    random_pos = []
    for i in range(n_bootstraps):
        random_pos.append([np.random.normal(loc=pa, scale=pa_err), 
                           np.random.normal(loc=0, scale=centering_err), 
                           np.random.normal(loc=0, scale=centering_err)])


    def get_result(theta):
        m, c = theta
        m_bootstrap.append(m * 100 / np.sin(inclination))
        c_bootstrap.append(c * 100)

    pool = Pool(4)


    for bootstrap_i in tqdm(range(n_bootstraps)):
        pool.apply_async(bootstrap_iteration, args=(flux, vel, flux_err, vel_err, 
                 X, Y,
                 step,
                 random_pos[bootstrap_i][1:],
                 pa, random_pos[bootstrap_i][0], 
                 inclination, beta,
                 bar_length 
                 ), callback=get_result)
    pool.close()
    pool.join()

    m_bootstrap = np.array(m_bootstrap)
    c_bootstrap = np.array(c_bootstrap)

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

def pixelate(X, Y, RHO, VR, step):
    if step == 0:
        return RHO, VR, X, Y
    
    else:
        RHO_array         = mean_in_pixel(X, Y, step, RHO)
        VR_array          = mean_in_pixel(X, Y, step, VR * RHO) / RHO_array
        return RHO_array, VR_array, centers_of_pixel(X, Y, step)
