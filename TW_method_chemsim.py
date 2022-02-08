from read_snap import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib
from matplotlib import cm
import functions as f
import pybar.pybar_tom as pybar
from pathlib import Path


matplotlib.use('Agg')

path = '/Data/oborodina/internship/astro-node/chemistry_simulations/'
FIG_DIR = Path('/Data/oborodina/internship/astro-node/chemistry_simulations/figures/')

def fix_inf(a):
    a[a == np.inf] = np.nan
    a = a[np.logical_not(np.isnan(a))]
    return a

psi_bar = np.deg2rad(50)
i    = np.deg2rad(50)
to_Myr = 9.78462
bar_length = 3.0

err_vr = 0.1
err_rho = 0.05

halfbox = 120            # x size of computational box
halfboy = 120            # y size of computational box
halfboz = 10             # z size of computational box
omega   = 4.0            # bar pattern speed in units of km/s/100pc
phi     = np.radians(20) # bar viewing angle (i.e., angle between bar major axis and Sun-GC line)
isnap   = 185            # snapshot number

xsun    = 0.0   # x position of sun in 100pc
ysun    = -85.0 # y position of sun in 100pc
zsun    = 0.0   # z position of sun in 100pc

vxsun   = -220.0 # vx of sun in km/s
vysun   = 0.0    # vy of sun in km/s
vzsun   = 0.0    # vz of sun in km/s

arepoLength = 3.0856e20 # Length units (100pc) in cgs
arepoMass   = 1.988e33  # Mass units (1 Msun) in cgs 
arepoVel    = 1.0e5     # Velocity units (1 km/s) in cgs

chem        = True      # whether the snapshot contains chemistry

snapshot = Snapshot(isnap,path,halfbox=halfbox,halfboy=halfboy,halfboz=halfboz,arepoLength=arepoLength,arepoMass=arepoMass,arepoVel=arepoVel,vxyzsun=[xsun,ysun,zsun,vxsun,vysun,vzsun],chem=chem) 
snapshot.read_full()                                   # read snapshot
snapshot.rotate_full(omega*snapshot.t)    # rotate snapshot to align bar with y axis, minus viewing angle phi

## psi bar rotation
u_bar = [0, 0, 1]
snapshot.rotate_3D(u_bar, psi_bar)

## inclination
u_i = [1, 0, 0]
snapshot.rotate_3D(u_i, i)

doublets       = np.vstack((snapshot.x,snapshot.y)).T
xmin, xmax, dx = -110, 110, 0.1
ymin, ymax, dy = -110, 110, 0.1
xs = np.arange(xmin,xmax+dx,dx)
ys = np.arange(ymin,ymax+dy,dy)
xbins = np.arange(xmin-dx/2,xmax+3*dx/2,dx)
ybins = np.arange(ymin-dy/2,ymax+3*dy/2,dy)

density_CO = snapshot.nCO 
density_H2 = snapshot.nH2 
density_HI = snapshot.nHI 
# density = snapshot.nHI 

w_CO    = (density_CO*snapshot.volumes) /(dx*dy)*(arepoLength) # weights for binning
w_H2    = (density_H2*snapshot.volumes) /(dx*dy)*(arepoLength) # weights for binning
w_HI    = (density_HI*snapshot.volumes) /(dx*dy)*(arepoLength) # weights for binning
w    = w_CO.astype(float64) * np.nanmedian(w_H2) / np.nanmedian(w_CO) + w_HI.astype(float64)

# print(w_CO, w_CO.max(), '\n', w_H2, w_H2.max(), '\n', w, w.max())
#w    = (density*snapshot.volumes) /(dx*dy)*(arepoLength) # weights for binning
snapshot.vx, snapshot.vy = f.add_solid_body_rotation(snapshot.x, snapshot.y, snapshot.vx, snapshot.vy, 400)

gas_binned, bins = np.histogramdd(doublets, bins = (xbins,ybins),weights=w)     # surface density in cm^-2
binned_vx, bins = np.histogramdd(doublets, bins = (xbins,ybins), weights=snapshot.vx * snapshot.masses)  
binned_vy, bins = np.histogramdd(doublets, bins = (xbins,ybins), weights=snapshot.vy * snapshot.masses)
binned_vz, bins = np.histogramdd(doublets, bins = (xbins,ybins), weights=snapshot.vz * snapshot.masses)
binned_mass, bins = np.histogramdd(doublets, bins = (xbins,ybins), weights=snapshot.masses)

eps = binned_mass.mean() / 1e8
binned_vx /= (binned_mass + eps)
binned_vy /= (binned_mass + eps)
binned_vz /= (binned_mass + eps)

binned_x, binned_y = np.meshgrid(xbins[:-1] + dx/2, ybins[:-1] + dy/2, indexing='ij')

gas_binned = gas_binned.T
binned_vx = binned_vx.T / 100
binned_vy = binned_vy.T / 100
binned_vz = binned_vz.T / 100

binned_x = binned_x.T / 10
binned_y = binned_y.T / 10

VR = binned_vz# vy * np.sin(i)

print('Begin building the plot')


####################
# plot Total
####################

fig, ax = plt.subplots(figsize=(10,6),sharex=True,sharey=True)
u       = (arepoLength/snapshot.kpc_to_cm)
extent  = np.array([xmin,xmax,ymin,ymax])*u
cbarticks = np.logspace(-100,100,201)

# total
title   = 'Total'
cmap    = 'viridis'
cbarlbl = r'$[\rm cm^{-2}]$'
levels  = np.logspace(16,np.log10(gas_binned.max()),256)
norm    = mc.BoundaryNorm(levels,256)
im      = ax.imshow(gas_binned,norm=norm,extent=extent,cmap=cmap,origin='lower',interpolation='nearest')
#im = ax.scatter(binned_x, binned_y, c=gas_binned, norm=norm, cmap=cmap)
cbar    = fig.colorbar(im,ticks=cbarticks, format=ticker.FuncFormatter(fmt),shrink=0.8)
cbar.set_label(cbarlbl,fontsize=14)
ax.set_title(title,fontsize=20)
fig.savefig(FIG_DIR / 'rotated.pdf',bbox_inches='tight',dpi=200)


####################
# plot Vx
####################

fig, ax = plt.subplots(figsize=(10,6),sharex=True,sharey=True)
u       = (arepoLength/snapshot.kpc_to_cm)
extent  = np.array([xmin,xmax,ymin,ymax])*u
cbarticks = np.logspace(-100,100,201)

# total
title   = r'V_{LOS}'
cmap    = 'viridis'
cbarlbl = r'$[\rm 100 km/s]$'
im      = ax.imshow(VR,extent=extent,cmap=cmap,origin='lower',interpolation='nearest')
cbar    = fig.colorbar(im, format=ticker.FuncFormatter(fmt),shrink=0.8)
cbar.set_label(cbarlbl,fontsize=14)
ax.set_title(title,fontsize=20)
fig.savefig(FIG_DIR / 'rotated_vlos.pdf',bbox_inches='tight',dpi=200)

#### TW Method begins
step_final   = 0.1
RHO_array = gas_binned.copy()
VR_array  = VR.copy()
x_center_array, y_center_array = binned_y.copy(), -binned_x.copy()

#VR_array, RHO_array = f.add_uncertanties(VR_array, RHO_array, err_vr, err_rho * RHO_array)

VR_err = err_vr * np.ones_like(VR_array)
RHO_err = err_rho * RHO_array

# pa_uncertainty = np.random.uniform(-2, 2)
# x_center_array, y_center_array, _, _ = f.rotate_bar(np.deg2rad(pa_uncertainty), x_center_array, y_center_array, 0, 0)



bar = pybar.mybar(Flux=RHO_array, Flux_err=RHO_err,
                  Velocity=VR_array, Velocity_err=VR_err,
                  Yin=y_center_array, Xin=x_center_array,
                  inclin=np.rad2deg(i), PAnodes=0, beta=np.rad2deg(psi_bar), 
                  bar_length=bar_length, if_symmetrize=True)


fig, ax = plt.subplots(figsize=(10, 8))
levels  = np.logspace(16, np.log10(RHO_array.max()),256)
norm    = mc.BoundaryNorm(levels,256)
sc = ax.scatter(bar.X_lon, bar.Y_lon, c=bar.Flux, 
                cmap='seismic', norm=norm, marker='s', s=4)
ax.grid(ls='dashed')
ax.tick_params(labelsize=20, direction='in')
ax.set_xlabel(r'$x\, \rm [kpc]$', fontsize=20)
ax.set_ylabel(r'$y\, \rm [kpc]$', fontsize=20)
ax.set_ylim(-12, 12)
ax.set_aspect('1')
ax.set_title(r'$Density$', fontsize=20)
#plt.colorbar(sc, format='%g', label=r'$\rm 100\, km/s$')
plt.colorbar(sc, format='%g', label=r'$\rm cm^{-3}$')
plt.savefig(FIG_DIR / '3D_galaxy_density_all.png', dpi=200)


bar.tremaine_weinberg()
print(bar.dfx_tw)
print('applied TW method')

omega_bar, omega_bar_err, c, c_err, x_tw, v_tw, x_tw_err, v_tw_err = f.fit(bar)

plasma = cm.get_cmap('plasma', len(x_tw))
colors = plasma.colors

x_fit = np.linspace(np.nanmin(x_tw), np.nanmax(x_tw), 3)
y_GT  = (x_fit * 40) * np.sin(i) + c * 100
y_fit = (x_fit * omega_bar * 100) + c * 100

#print(bar.dfx_tw)
print('fitted')


fig, ax = plt.subplots(1, 2, figsize=(15, 7))

levels = np.linspace(np.nanmin(VR) * 100 - 10, np.nanmax(VR) * 100 + 10, 101)
cbarticks = np.linspace(np.nanmin(VR).round(2) * 100 , np.nanmax(VR).round(2) * 100 , 5)
norm1 = mc.BoundaryNorm(levels, 256)

ax1 = ax[0]
ax2 = ax[1]
sc = ax1.scatter(bar.X_lon, bar.Y_lon, c=bar.Vel_s * 100, 
                 cmap='Greys', marker='s', s=3)
# sc = ax1.scatter(bar.X_lon, bar.Y_lon, c=bar.Flux, 
#                  cmap='gray', norm=norm, s=1.3, alpha=0.4)
for index, y in enumerate(bar.y_slits):
    if index % 4 == 0:
        ax1.plot([bar.X_lon.min(), bar.X_lon.max()], [y, y], c=colors[index])
ax1.set_xlabel(r'$x$ [kpc]', fontsize=15)
ax1.set_ylabel(r'$y$ [kpc]', fontsize=15)
ax1.tick_params(labelsize=12, direction='in')
# ax1.annotate(r'$t=%.1f\, \rm Myr$' % (t * to_Myr), xy=(0.03, 0.94),
#              xycoords='axes fraction', fontsize=16, color='k')
ax1.annotate(r'$i=%.1f ^{\circ}$' % (np.rad2deg(i)), xy=(0.03, 0.89),
             xycoords='axes fraction', fontsize=16, color='k')
ax1.annotate(r'$\psi_{bar}=%.1f ^{\circ}$' % (np.rad2deg(psi_bar)), xy=(0.03, 0.84),
             xycoords='axes fraction', fontsize=16, color='k')

ax1.set_ylim(bar.X_lon.min(), bar.X_lon.max())
ax1.set_aspect("1")
cbar = plt.colorbar(sc, ax=ax1, orientation="horizontal", fraction=0.07, ticks=cbarticks)
cbar.set_label(label=r'$V_{LOS}$ [km/s]', size=15)

print('left panel is ready') 

ax2.plot(x_fit, y_GT,
        c='black', lw=3, label=r'GT: $\Omega_p = 40$ km/s/kpc')
ax2.plot(x_fit, y_fit, 
        c='gray', lw=3, label=fr'fit: $\Omega_p = {(omega_bar * 100 / np.sin(i)).round(1)} \pm {(omega_bar_err * 100 / np.sin(i)).round(1)} $ km/s/kpc')

for x, y, e1, e2, color in zip(x_tw, v_tw * 100, x_tw_err, v_tw_err * 100, colors):
    ax2.errorbar(x, y, xerr=e1, yerr=e2, lw=1, capsize=3, color=color, marker='o')
# ax2.annotate(text='errors: 0.01 km/s, density 10%', xy=(4,10), xytext=(0.5,0.5))

# ax2.annotate(f'errors: \n {err_vr * 100} km/s, \n density {err_rho * 100} \%',
#             xy=(0, 0.5),
#             xycoords='axes fraction',
#             xytext=(0.7, 0.1), fontsize=15)

ax2.set_ylim(-50, 50)
ax2.set_xlim(-1, 1)

ax2.tick_params(labelsize=12,direction='in')
ax2.set_xlabel(r' $\langle x \rangle$ [kpc]', fontsize=15)
ax2.set_ylabel(r' $\langle v \rangle$ [km/s]', fontsize=15)
ax2.legend(fontsize=15, loc='upper left')
ax2.grid(ls='dashed')

print('right panel is ready')

plt.savefig(FIG_DIR / f'TW_method_pybar_all.png', bbox_inches='tight', dpi=200)

