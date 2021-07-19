import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import cm
import pandas as pd
from tqdm import tqdm
import functions as f
from pathlib import Path
import pybar.pybar as pybar

import os
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2021/bin/universal-darwin'

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

plt.rc('text', usetex=True)

result = pd.DataFrame(columns=['i', 'beta','omega', 'omega_err_up', 'omega_err_down'])

SNratio = 10
t = 20
to_Myr = 9.78462
bar_length = 2.6
save_plot = False

FIG_DIR = Path('./../figures/')

beta_array = np.deg2rad(np.linspace(0, 89, 51))
i_array    = np.deg2rad(np.linspace(2, 89, 51))

for beta in beta_array:
    for i in i_array:
        X, Y, VX, VY, RHO = np.load(f'./../simulation/simulation/output_npy/data_{t}.npy')
        VX, VY = f.add_solid_body_rotation(X, Y, VX, VY, 0.4)
        X, Y, VX, VY = f.rotate_bar(beta, X, Y, VX, VY)
        X, Y, VR = f.incline_galaxy(i, X, Y, VX, VY)
        
        step = 0.1

        VR_array          = f.mean_in_pixel(X, Y, step, VR)
        RHO_array          = f.mean_in_pixel(X, Y, step, RHO)

        x_center_array, y_center_array = f.centers_of_pixel(X, Y, step)
        
        VR, RHO = f.add_uncertanties(VR, RHO, 0.1, 0.1 * RHO)
        
        VR_array_sym = f.make_symmetric(x_center_array, VR_array)
        RHO_array_sym = f.make_symmetric(x_center_array, RHO_array)
        
        not_bar_mask = (y_center_array > bar_length * np.sin(i)) | (y_center_array < - bar_length * np.sin(i))
        VR_array_sym[not_bar_mask]  = np.nan
        RHO_array_sym[not_bar_mask] = np.nan
        

        VR_err_array = 0.1 * np.ones_like(VR_array_sym)
        RHO_err_array = RHO_array_sym / SNratio
        
        omega, omega_err_up, omega_err_down = f.bootstrap_tw(flux=RHO_array_sym, flux_err=RHO_err_array,
                                               vel=VR_array_sym, vel_err=VR_err_array, 
                                               x_array=x_center_array[:,0], y_array=y_center_array[0],
                                               grid_step=step, slit_width=step,
                                               centering_err=step,
                                               pa=-90, pa_err=5, inclination=np.rad2deg(i),
                                               save_in_file=False,
                                               overwrite_bootstraps=True,
                                               n_bootstraps=1000
                                               )
        
        result_i = pd.DataFrame(data={'i':[i], 'beta': [beta], 
                                      'omega': [omega], 
                                      'omega_err_up' : [omega_err_up], 'omega_err_down' : [omega_err_down]})
        result = result.append(result_i)
        
        if save_plot:

            bar = pybar.mybar(Flux=RHO_array,
                      Velocity=VR_array,
                      Yin=y_center_array, Xin=x_center_array,
                      inclin=np.rad2deg(i), PAnodes=90)

            bar.tremaine_weinberg(slit_width=0.1)
            x_tw = bar.dfx_tw
            v_tw = bar.dfV_tw

            plasma = cm.get_cmap('plasma', len(y_center_array[0,1:-1]))
            colors = plasma.colors

            fig, ax = plt.subplots(1, 2, figsize=(15, 7))

            levels = np.linspace(np.nanmin(VR_array) * 100, np.nanmax(VR_array) * 100, 101)
            cbarticks = np.linspace(np.nanmin(VR_array).round(1) * 100, np.nanmax(VR_array).round(1) * 100, 5)
            norm = mc.BoundaryNorm(levels, 256)

            ax1 = ax[0]
            ax2 = ax[1]

            sc = ax1.scatter(x_center_array, y_center_array, c=VR_array * 100, 
                             cmap='Greys', norm=norm, marker='s', s=3)
            for index, y in enumerate(y_center_array[0,1:-1]):
                if index % 4 == 0:
                    ax1.plot([-8, 8], [y, y], c=colors[index])
            ax1.set_xlabel(r'$x$ [kpc]', fontsize=15)
            ax1.set_ylabel(r'$y$ [kpc]', fontsize=15)
            ax1.tick_params(labelsize=12, direction='in')
            ax1.annotate(r'$t=%.1f\, \rm Myr$' % (t * to_Myr), xy=(0.03, 0.94),
                         xycoords='axes fraction', fontsize=16, color='k')
            ax1.annotate(r'$i=%.1f ^{\circ}$' % (np.rad2deg(i)), xy=(0.03, 0.89),
                         xycoords='axes fraction', fontsize=16, color='k')
            ax1.annotate(r'$\beta=%.1f ^{\circ}$' % (np.rad2deg(beta)), xy=(0.03, 0.84),
                         xycoords='axes fraction', fontsize=16, color='k')

            ax1.set_ylim(-7, 7)
            cbar = plt.colorbar(sc, ax=ax1, orientation="horizontal", fraction=0.07, ticks=cbarticks)
            cbar.set_label(label=r'$V_{LOS}$ [km/s]', size=15)

            ax2.plot([np.nanmin(x_tw), np.nanmax(x_tw)], 
                     [np.nanmin(x_tw) * 40 * np.sin(i), np.nanmax(x_tw) * 40 * np.sin(i)], 
                    c='black', lw=3, label=r'GT: $\Omega_p = 40$ km/s/kpc')
            ax2.scatter(x_tw, v_tw * 100, s=20, c=colors, label=r'data', zorder=100)

            # ax2.set_ylim(-7, 7)
            # # ax2.set_xlim(-0.5, 0.5)
            ax2.tick_params(labelsize=12,direction='in')
            ax2.set_xlabel(r' $\langle x \rangle$ [kpc]', fontsize=15)
            ax2.set_ylabel(r' $\langle v \rangle$ [km/s]', fontsize=15)
            ax2.legend(fontsize=15, loc='upper left')
            ax2.grid(ls='dashed')
            plt.savefig(FIG_DIR / f'TW_method__{np.round(np.rad2deg(i), 0)}_{np.round(np.rad2deg(beta), 0)}_pybar.png', 
                        bbox_inches='tight', dpi=300)
            plt.close(fig)
            
result.to_csv(f'fit_pattern_speed.csv', index=False, sep=' ')
        