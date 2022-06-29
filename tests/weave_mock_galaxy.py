#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:28:52 2022

@author: pablo
"""
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from pst import SSP
from midas.mock_observation import Observation
from midas.instrument import WEAVE_Instrument
from midas.galaxy import Galaxy
from midas.saveWEAVECube import SaveWEAVECube
import numpy as np
from astropy.io import fits
import os


output = '/home/pablo/WEAVE-Apertiff/mock_cubes/'
# =============================================================================
# Create a Galaxy object
# =============================================================================
gal_id = 188859
# gal_id = 227583
gal_output_path = os.path.join(output, str(gal_id))
if not os.path.isdir(gal_output_path):
    os.mkdir(gal_output_path)

mock_catalogue = fits.getdata(
    '/home/pablo/WEAVE-Apertiff/weave_sample/mock_WA_sample.fits')
cat_pos = np.where(mock_catalogue['ID'] == gal_id)[0][0]
f = h5py.File(
    '/home/pablo/WEAVE-Apertiff/weave_sample/data/sub_{}.hdf5'.format(gal_id),
    'r')
stars = f['stars']
gas = f['gas']
galaxy = Galaxy(
    name=str(gal_id),
    stars=stars,
    gas=gas,
    gal_spin=np.array([mock_catalogue['spin_x'][cat_pos],
                       mock_catalogue['spin_y'][cat_pos],
                       mock_catalogue['spin_z'][cat_pos]]),
    gal_vel=np.array([mock_catalogue['vel_x'][cat_pos],
                      mock_catalogue['vel_y'][cat_pos],
                      mock_catalogue['vel_z'][cat_pos]]),
    gal_pos=np.array([mock_catalogue['pos_x'][cat_pos],
                      mock_catalogue['pos_y'][cat_pos],
                      mock_catalogue['pos_z'][cat_pos]])
    )
galaxy.set_to_galaxy_rest_frame()
galaxy.proyect_galaxy(None)

# =============================================================================
# Create the WEAVE instrument for observing the galaxy
# =============================================================================
weave_instrument = WEAVE_Instrument(z=f['redshift_observation'][()])
# SSP model used for generating stellar spectra
# =============================================================================
# Create the Stellar Population Synthesis model
# =============================================================================
#ssp_model = SSP.PyPopStar(IMF='KRO')
ssp_model = SSP.XSL(IMF='Kroupa', ISO='P00')

# %% # Perform observation
observation = Observation(SSP=ssp_model, Instrument=weave_instrument,
                          Galaxy=galaxy)
observation.compute_los_emission(stellar=True, nebular=False,
                                 kinematics=True,
                                 dust_extinction=True)

# %% Computing physical maps

mean_vel = galaxy.get_stellar_map(
    out=np.zeros((weave_instrument.det_x_n_bins,
                  weave_instrument.det_y_n_bins)),
    stat_val=galaxy.stars['ProjVelocities'][2].copy(),
    statistic='mean')

mean_tau = galaxy.get_stellar_map(
    out=np.zeros((weave_instrument.det_x_n_bins,
                  weave_instrument.det_y_n_bins)),
    stat_val=galaxy.stars['tau_5500'].copy(),
    statistic='mean')

mean_age = galaxy.get_stellar_map(
    out=np.zeros((weave_instrument.det_x_n_bins,
                  weave_instrument.det_y_n_bins)),
    stat_val=galaxy.stars['ages'].copy(),
    statistic='mean')

sigma_vel = galaxy.get_stellar_map(
    out=np.zeros((weave_instrument.det_x_n_bins,
                  weave_instrument.det_y_n_bins)),
    stat_val=galaxy.stars['ProjVelocities'][2].copy(),
    statistic='std')

mass_map = galaxy.get_stellar_map(
    out=np.zeros((weave_instrument.det_x_n_bins,
                  weave_instrument.det_y_n_bins)),
    stat_val=galaxy.stars['Masses'].copy() * 1e10,
    statistic='sum')
# %% Smoothing
sigma_kpc = 0.3
sigma_pixel = sigma_kpc / weave_instrument.pixel_size_kpc.value

observation.cube = observation.gaussian_smooth(
    observation.cube,
    sigma=[0, sigma_pixel, sigma_pixel])

mean_vel = observation.gaussian_smooth(np.nan_to_num(mean_vel),
                                       sigma=sigma_pixel)
np.savetxt(os.path.join(gal_output_path,
                        'stellar_mean_vel_{}.dat'.format(gal_id)),
           mean_vel,
           header='tau_5500')

mean_tau = observation.gaussian_smooth(np.nan_to_num(mean_tau),
                                       sigma=sigma_pixel)
np.savetxt(os.path.join(gal_output_path,
                        'mean_optical_depth_{}.dat'.format(gal_id)),
           mean_tau,
           header='tau_5500')

mean_age = observation.gaussian_smooth(np.nan_to_num(mean_age),
                                       sigma=sigma_pixel)
np.savetxt(os.path.join(gal_output_path,
                        'stellar_mean_age_map_{}.dat'.format(gal_id)),
           mean_age,
           header='Gyr')

sigma_vel = observation.gaussian_smooth(np.nan_to_num(sigma_vel),
                                        sigma=sigma_pixel)
np.savetxt(os.path.join(gal_output_path,
                        'stellar_vel_disp_map_{}.dat'.format(gal_id)),
           sigma_vel,
           header='km/s')

mass_map = observation.gaussian_smooth(np.nan_to_num(mass_map),
                                       sigma=sigma_pixel)
stellar_density = mass_map / weave_instrument.pixel_size_kpc.to('pc').value**2
np.savetxt(os.path.join(gal_output_path, 'stellar_dens_map_{}.dat'.format(gal_id)),
           stellar_density,
           header='Msun/pc**2')
# %%
cube = observation.cube
wave = observation.instrument.wavelength.value
# %%
mag_lim = 24 # mag / arcsec**2
band_wl = 4686
sn_lim = 40
c_ligth_aa = 3e18
f_lim = (10**(-0.4 * (mag_lim + 48.60)) * c_ligth_aa / band_wl**2
         /weave_instrument.pixel_size.to('arcsec').value) # erg/s/AA/cm2

flux = np.nanmedian(
    cube[(wave < band_wl + 300) & (wave > band_wl - 300)], axis=0)

sigma_limit = f_lim / sn_lim

gauss_err = np.random.normal(0, 1, size=cube.size)
gauss_err = gauss_err.reshape(cube.shape)

observation.cube_variance = (np.ones_like(cube)*sigma_limit)**2
observation.cube_variance[observation.cube_variance == 0] = np.nan
observation.cube += gauss_err * observation.cube_variance**0.5
observation.sky = None

cube_err = np.sqrt(observation.cube_variance)
cube = observation.cube
flux = np.nanmedian(
    cube[(wave < band_wl + 300) & (wave > band_wl - 300)], axis=0)
snr = np.nanmedian(
    cube[(wave < band_wl + 300) & (wave > band_wl - 300)]
    / cube_err[(wave < band_wl + 300) & (wave > band_wl - 300)], axis=0)

# %%
SaveWEAVECube(
    observation,
    filename=os.path.join(gal_output_path,
                          'galaxy_ID_{}_xsl'.format(galaxy.name)),
    funit=1e-18,
    data_size=np.float32)

f.close()
# %% Particle level plots
plt.figure()
plt.hist2d(galaxy.stars['ages'].copy(), galaxy.stars['tau_5500'].copy(),
           range=[[0, 13.7], [0, 2]], norm=LogNorm())
plt.savefig(
    os.path.join(gal_output_path, 'hist_age_tau_V_{}.png'.format(gal_id)),
    bbox_inches='tight')
plt.close()

# %%
snr_levels = [1, 10]
snr_color = 'white'
snr_lw = [0.7, 0.7]

plot_x_lims = (
    np.nanmin(
    (np.nanpercentile(galaxy.stars['ProjCoordinates'][0, :], 1),
     observation.instrument.det_x_bins_kpc[0])),
    np.nanmax(
    (np.nanpercentile(galaxy.stars['ProjCoordinates'][0, :], 99),
     observation.instrument.det_x_bins_kpc[-1]))
    )
plot_y_lims = (
    np.nanmin(
    (np.nanpercentile(galaxy.stars['ProjCoordinates'][1, :], 1),
     observation.instrument.det_y_bins_kpc[0])),
    np.nanmax(
    (np.nanpercentile(galaxy.stars['ProjCoordinates'][1, :], 99),
     observation.instrument.det_y_bins_kpc[-1]))
    )

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(20, 20), sharex=True,
                        sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.1))
ax = axs[0, 0]
ax.set_title(r'$z_{obs}=$' + '{:.4f}'.format(observation.instrument.redshift))
mappable = ax.imshow(flux.T, origin='lower', interpolation='none',
                     extent=(observation.instrument.det_x_bins_kpc[0],
                             observation.instrument.det_x_bins_kpc[-1],
                             observation.instrument.det_y_bins_kpc[0],
                             observation.instrument.det_y_bins_kpc[-1]),
                     cmap='nipy_spectral', norm=LogNorm(vmin=f_lim*0.1),
                     aspect='auto',
                     alpha=1)
CS = ax.contour(observation.instrument.det_x_bins_kpc,
                observation.instrument.det_y_bins_kpc,
                snr.T, origin='lower',
                levels=snr_levels, colors=snr_color, alpha=1,
                linestyles=['-', '-'], linewidths=snr_lw)
ax.clabel(CS, CS.levels, inline=True, fontsize=15, colors='r')

plt.colorbar(mappable, ax=ax, label=r'g flux (erg/s/AA/cm2)',
             orientation='horizontal')
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1,
    label='FoV')
ax.add_patch(rect)
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_xlim(plot_x_lims)
ax.set_ylim(plot_y_lims)

ax.legend()
ax = axs[1, 0]
mappable = ax.pcolormesh(
    observation.instrument.det_x_bins_kpc,
    observation.instrument.det_y_bins_kpc,
    mean_vel.T, cmap='seismic',
    vmin=np.nanpercentile(mean_vel, 5),
    vmax=np.nanpercentile(mean_vel, 95))
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1)
ax.contour(observation.instrument.det_x_bins_kpc,
           observation.instrument.det_y_bins_kpc,
           snr.T, origin='lower',
           levels=snr_levels, colors=snr_color, alpha=1,
           linestyles=['-', '-'], linewidths=snr_lw)

ax.add_patch(rect)
cbar = plt.colorbar(mappable, ax=ax, label=r'$v~[km/s]$',
                    orientation='horizontal')
# cbar.set_ticks([-100, -50, 0, 50, 100])
ax.legend()

ax = axs[1, 1]
mappable = ax.pcolormesh(
    observation.instrument.det_x_bins_kpc,
    observation.instrument.det_y_bins_kpc,
    sigma_vel.T, cmap='gnuplot',
    vmin=np.nanpercentile(sigma_vel, 5),
    vmax=np.nanpercentile(sigma_vel, 95))
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1)
ax.contour(observation.instrument.det_x_bins_kpc,
           observation.instrument.det_y_bins_kpc,
           snr.T, origin='lower',
           levels=snr_levels, colors=snr_color, alpha=1,
           linestyles=['-', '-'], linewidths=snr_lw)

ax.add_patch(rect)
cbar = plt.colorbar(mappable, ax=ax, label=r'$\sigma(v)~[km/s]$',
                    orientation='horizontal')
# cbar.set_ticks([-100, -50, 0, 50, 100])

ax = axs[2, 0]
mappable = ax.pcolormesh(
    observation.instrument.det_x_bins_kpc,
    observation.instrument.det_y_bins_kpc,
    mean_age.T, cmap='jet')
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1)
ax.add_patch(rect)
ax.contour(observation.instrument.det_x_bins_kpc,
           observation.instrument.det_y_bins_kpc,
           snr.T, origin='lower',
           levels=snr_levels, colors=snr_color, alpha=1,
           linestyles=['-', '-'], linewidths=snr_lw)

plt.colorbar(mappable, ax=ax, label=r'$<age_*>~[Gyr]$',
             orientation='horizontal')

ax = axs[2, 1]
mappable = ax.pcolormesh(
    observation.instrument.det_x_bins_kpc,
    observation.instrument.det_y_bins_kpc,
    np.log10(stellar_density.T),
    vmin=-1, vmax=3, cmap='terrain')
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1)
ax.add_patch(rect)
ax.contour(observation.instrument.det_x_bins_kpc,
           observation.instrument.det_y_bins_kpc,
           snr.T, origin='lower',
           levels=snr_levels, colors=snr_color, alpha=1,
           linestyles=['-', '-'], linewidths=snr_lw)

plt.colorbar(mappable, ax=ax,
             label=r'$\log_{10}(\Sigma_*/(\rm M_\odot/pc^2))$',
             orientation='horizontal')

ax = axs[2, 2]
mappable = ax.pcolormesh(
    observation.instrument.det_x_bins_kpc,
    observation.instrument.det_y_bins_kpc,
    mean_tau.T, cmap='inferno')
rect = plt.Rectangle(
    (observation.instrument.det_x_bins_kpc[0],
     observation.instrument.det_y_bins_kpc[0]),
    observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
    observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
    fc='none',
    ec='k',
    alpha=1)
ax.add_patch(rect)
ax.contour(observation.instrument.det_x_bins_kpc,
           observation.instrument.det_y_bins_kpc,
           snr.T, origin='lower',
           levels=snr_levels, colors=snr_color, alpha=1,
           linestyles=['-', '-'])

plt.colorbar(mappable, ax=ax, label=r'$<\tau_{5500}>$',
             orientation='horizontal')

axs[0, 1].axis('off')
axs[0, 2].axis('off')
axs[1, 2].axis('off')

fig.savefig(
    os.path.join(gal_output_path, 'maps_galaxy_{}.pdf'.format(gal_id)),
    bbox_inches='tight')


# plt.xlim(30, 550)
# plt.ylim(30, 60)

# Mr Krtxo \(ﾟ▽ﾟ)/