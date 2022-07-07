#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:40:24 2022

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
from midas.cosmology import cosmo
from wetc import signal2noise
import numpy as np
from astropy.io import fits
import os

output = '/Users/pcorcho/WEAVE/mock_cubes/'
input_path = '/Volumes/Elements/WEAVE/weave_sample/data/'

mock_catalogue = fits.getdata(
    '/Volumes/Elements/WEAVE/weave_sample/mock_WA_sample.fits')

# %% Initialise noise model
S = signal2noise.Signal(LIFU=True, offset=0.1,
                        throughput_table=True, sky_model=True)
time = 1200.
n_exp = 3
airmass = 1.2
seeing = signal2noise.totalSeeing(1.0)
# SSP model used for generating stellar spectra

# Default values for g band (plots)
mag_lim = 24  # mag / arcsec**2
band_wl = 4686
sn_lim = 40
c_ligth_aa = 3e18
f_lim = (10 ** (-0.4 * (mag_lim + 48.60)) * c_ligth_aa / band_wl ** 2
         / 0.5**2)  # erg/s/AA/cm2
# =============================================================================
# Create the Stellar Population Synthesis model
# =============================================================================
ssp_model = SSP.PyPopStar(IMF='KRO')
ssp_name = 'pypopstar'
# ssp_model = SSP.XSL(IMF='Kroupa', ISO='P00')
# =============================================================================
# Create the WEAVE instrument for observing the galaxy
# Some properties will be updated during the loop by means of the provided z
# =============================================================================
weave_instrument = WEAVE_Instrument(z=0.1)

# =============================================================================
# Create the template for the observations (this way the SSP is prepared only once)
# Some properties will be updated during the loop (instrument, galaxy)
# =============================================================================
observation = Observation(SSP=ssp_model, Instrument=weave_instrument,
                          Galaxy=None)

for ii, gal_id in enumerate(mock_catalogue['ID']):
    print('Generating cube {}, gal_ID: {}'.format(ii, gal_id))
    # =============================================================================
    # Create a Galaxy object
    # =============================================================================
    if mock_catalogue['n_part_stars'][ii] < 10000:
        continue
    gal_output_path = os.path.join(output, str(gal_id))
    if os.path.isdir(gal_output_path):
        if os.path.isfile(os.path.join(gal_output_path, 'maps_galaxy_{}.pdf'.format(gal_id))):
            continue
    else:
        os.mkdir(gal_output_path)
    cat_pos = ii
    f = h5py.File(os.path.join(input_path, 'sub_{}.hdf5'.format(gal_id)),
                  'r')
    galaxy = Galaxy(
        name=str(gal_id),
        stars=f['stars'],
        gas=f['gas'],
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
    # Update instrument physical parameters
    weave_instrument.redshift = f['redshift_observation'][()]
    weave_instrument.get_pixel_physical_size()
    weave_instrument.detector_bins()
    observation.galaxy = galaxy
    # %% # Perform observation
    observation.compute_los_emission(stellar=True, nebular=False,
                                     kinematics=True,
                                     dust_extinction=True)

    # %% Computing physical maps
    mass_map = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['Masses'].copy(),
        statistic='sum')  # Expressed in 1e10 Msun / h

    mlr_sum_map = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['MLR'].copy(),
        statistic='sum')  # Expressed in 1e10 Msun / h

    # Mass weighted kinematics
    mw_mean_vel = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ProjVelocities'][2].copy() * galaxy.stars['Masses'].copy(),
        statistic='sum')
    mw_mean_vel = mw_mean_vel / (mass_map + 1e-10)

    mw_mean_vel_sq = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ProjVelocities'][2].copy()**2 * galaxy.stars['Masses'].copy(),
        statistic='sum')
    mw_mean_vel_sq = mw_mean_vel_sq / (mass_map + 1e-10)

    mw_sigma_vel = np.sqrt(mw_mean_vel_sq - mw_mean_vel**2)

    # Light weighted kinematics
    lw_mean_vel = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ProjVelocities'][2].copy() * galaxy.stars['MLR'].copy(),
        statistic='sum')
    lw_mean_vel = lw_mean_vel / (mlr_sum_map + 1e-10)

    lw_mean_vel_sq = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ProjVelocities'][2].copy()**2 * galaxy.stars['MLR'].copy(),
        statistic='sum')
    lw_mean_vel_sq = lw_mean_vel_sq / (mlr_sum_map + 1e-10)

    lw_sigma_vel = np.sqrt(lw_mean_vel_sq - lw_mean_vel**2)

    # Mean age
    mw_mean_age = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ages'].copy() * galaxy.stars['Masses'].copy(),
        statistic='sum')
    mw_mean_age = mw_mean_age / (mass_map + 1e-10)

    lw_mean_age = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['ages'].copy() * galaxy.stars['MLR'].copy(),
        statistic='sum')
    lw_mean_age = lw_mean_age / (mlr_sum_map + 1e-10)

    # Mean extinction coefficient
    mean_tau = galaxy.get_stellar_map(
        out=np.zeros((weave_instrument.det_x_n_bins,
                      weave_instrument.det_y_n_bins)),
        stat_val=galaxy.stars['tau_5500'].copy(),
        statistic='mean')

    mass_map *= 1e10 / cosmo.h
    # %% Smoothing
    sigma_kpc = 0.3
    sigma_pixel = np.min(
        (sigma_kpc / weave_instrument.pixel_size_kpc.value, 5))

    observation.cube = observation.gaussian_smooth(
        observation.cube,
        sigma=[0, sigma_pixel, sigma_pixel])

    mw_mean_vel = observation.gaussian_smooth(np.nan_to_num(mw_mean_vel),
                                              sigma=sigma_pixel)
    lw_mean_vel = observation.gaussian_smooth(np.nan_to_num(lw_mean_vel),
                                              sigma=sigma_pixel)
    mw_sigma_vel = observation.gaussian_smooth(np.nan_to_num(mw_sigma_vel),
                                               sigma=sigma_pixel)
    lw_sigma_vel = observation.gaussian_smooth(np.nan_to_num(lw_sigma_vel),
                                               sigma=sigma_pixel)
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_mw_mean_vel_{}.dat'.format(gal_id)),
               mw_mean_vel, header='v km/s')
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_lw_mean_vel_{}.dat'.format(gal_id)),
               lw_mean_vel, header='v km/s')
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_mw_sigma_vel_{}.dat'.format(gal_id)),
               mw_sigma_vel, header='v km/s')
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_lw_sigma_vel_{}.dat'.format(gal_id)),
               lw_sigma_vel, header='v km/s')

    mean_tau = observation.gaussian_smooth(np.nan_to_num(mean_tau),
                                           sigma=sigma_pixel)
    np.savetxt(os.path.join(gal_output_path,
                            'mean_optical_depth_{}.dat'.format(gal_id)),
               mean_tau,
               header='tau_5500')

    mw_mean_age = observation.gaussian_smooth(np.nan_to_num(mw_mean_age),
                                              sigma=sigma_pixel)
    lw_mean_age = observation.gaussian_smooth(lw_mean_age,
                                              sigma=sigma_pixel)
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_mean_age_map_{}.dat'.format(gal_id)),
               mw_mean_age, header='Gyr')
    np.savetxt(os.path.join(gal_output_path,
                            'stellar_mean_age_map_{}.dat'.format(gal_id)),
               lw_mean_age, header='Gyr')

    mass_map = observation.gaussian_smooth(np.nan_to_num(mass_map),
                                           sigma=sigma_pixel)
    stellar_density = mass_map / weave_instrument.pixel_size_kpc.to('pc').value**2
    np.savetxt(os.path.join(gal_output_path, 'stellar_dens_map_{}.dat'.format(gal_id)),
               stellar_density,
               header='Msun/pc**2')
    # %%
    wave = observation.instrument.wavelength.value
    cube_sigma = np.full_like(observation.cube, fill_value=np.nan)

    print('· Estimating SNR for each spaxel')
    for i in range(observation.cube.shape[1]):
        for j in range(observation.cube.shape[2]):
            print("\r --> Spaxel: {}, {}".format(i, j), end='',
                  flush=True)
            f_lambda = observation.cube[:, i, j]
            snr = S.S2N_spectra(time=time, wave=wave, spectra=f_lambda.copy(),
                                airmass=airmass, seeing=seeing,
                                sb=True, n_exposures=n_exp)
            cube_sigma[:, i, j] = f_lambda / snr['SNR']

    # %%
    gauss_err = np.random.normal(0, 1, size=observation.cube.size)
    gauss_err = gauss_err.reshape(observation.cube.shape)

    observation.cube_variance = cube_sigma**2
    observation.cube += gauss_err * cube_sigma
    observation.sky = None

    g_wave_pos = (wave < band_wl + 300) & (wave > band_wl - 300)
    flux = np.nanmedian(
        observation.cube[g_wave_pos], axis=0)
    surface_brightness = flux / weave_instrument.pixel_size.to('arcsec').value**2
    # 24 mag
    pos = (surface_brightness > f_lim*0.9) & (surface_brightness < f_lim*1.1)
    median_spec_24 = np.median(observation.cube[:, pos], axis=1)
    median_spec_err_24 = np.median(cube_sigma[:, pos], axis=1)

    snr = np.nanmedian(observation.cube[g_wave_pos] / cube_sigma[g_wave_pos],
                       axis=0)

    del gauss_err, flux, pos, cube_sigma
    # %%
    SaveWEAVECube(
        observation,
        filename=os.path.join(gal_output_path,
                              'galaxy_ID_{}_{}'.format(galaxy.name, ssp_name)),
        funit=1e-18,
        data_size=np.float32)
    # Close HDF5 file
    f.close()
    # %% PLOTS
    snr_levels = [2, 10, 30]
    snr_color = 'white'
    snr_lw = [1.0] * len(snr_levels)

    plot_x_lims = (observation.instrument.det_x_bins_kpc[0],
                   observation.instrument.det_x_bins_kpc[-1])
    plot_y_lims = (observation.instrument.det_y_bins_kpc[0],
                   observation.instrument.det_y_bins_kpc[-1])

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 20), sharex=False,
                            sharey=False, gridspec_kw=dict(wspace=0.1,
                                                           hspace=0.1))
    ax = axs[0, 3]
    ax.set_title(r'$SNR/\AA$ for median flux at 24 mag')
    ax.semilogy(wave, median_spec_24 / median_spec_err_24)
    ax.set_xlabel(r'$\lambda~(\AA)$')
    ax.tick_params(which='both', left=False, right=True, labelleft=False, labelright=True, direction='in')
    # KINEMATICS
    ax = axs[0, 0]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        lw_mean_vel.T, cmap='seismic',
        vmin=np.nanpercentile(lw_mean_vel, 5),
        vmax=np.nanpercentile(lw_mean_vel, 95))
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    cbar = plt.colorbar(mappable, ax=ax, label=r'$v_{LW}~[km/s]$',
                        orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[1, 0]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        mw_mean_vel.T, cmap='seismic',
        vmin=np.nanpercentile(mw_mean_vel, 5),
        vmax=np.nanpercentile(mw_mean_vel, 95))
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    cbar = plt.colorbar(mappable, ax=ax, label=r'$v_{MW}~[km/s]$',
                        orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[0, 1]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        lw_sigma_vel.T, cmap='gnuplot',
        vmin=np.nanpercentile(lw_sigma_vel, 5),
        vmax=np.nanpercentile(lw_sigma_vel, 95))
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    cbar = plt.colorbar(mappable, ax=ax, label=r'$\sigma(v)_{LW}~[km/s]$',
                        orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[1, 1]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        mw_sigma_vel.T, cmap='gnuplot',
        vmin=np.nanpercentile(mw_sigma_vel, 5),
        vmax=np.nanpercentile(mw_sigma_vel, 95))
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    cbar = plt.colorbar(mappable, ax=ax, label=r'$\sigma(v)_{MW}~[km/s]$',
                        orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    # MEAN STELLAR AGE

    ax = axs[0, 2]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        lw_mean_age.T, cmap='jet')
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    plt.colorbar(mappable, ax=ax, label=r'$<age_*>_{LW}~[Gyr]$',
                 orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[1, 2]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        mw_mean_age.T, cmap='jet')
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    plt.colorbar(mappable, ax=ax, label=r'$<age_*>_{MW}~[Gyr]$',
                 orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    # FLUX AT g BAND
    ax = axs[2, 0]
    ax.set_title(r'$z_{obs}=$' + '{:.4f}'.format(
        observation.instrument.redshift))
    mappable = ax.imshow(surface_brightness.T, origin='lower',
                         interpolation='none',
                         extent=(observation.instrument.det_x_bins_kpc[0],
                                 observation.instrument.det_x_bins_kpc[-1],
                                 observation.instrument.det_y_bins_kpc[0],
                                 observation.instrument.det_y_bins_kpc[-1]),
                         cmap='nipy_spectral', norm=LogNorm(vmin=f_lim*0.1),
                         aspect='auto',
                         alpha=1)
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               surface_brightness.T, levels=[f_lim],
               colors='orange', linestyles='--', linewidths=2)
    CS = ax.contour(observation.instrument.det_x_bins_kpc,
                    observation.instrument.det_y_bins_kpc,
                    snr.T, origin='lower',
                    levels=snr_levels, colors=snr_color, alpha=1,
                    linestyles=['-', '-'], linewidths=snr_lw)
    ax.clabel(CS, CS.levels, inline=True, fontsize=15, colors='r')

    plt.colorbar(mappable, ax=ax,
                 label=r'$g$-Surface Bright. ($\rm erg/s/\AA/cm^2/arcsec^2$)',
                 orientation='horizontal')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[2, 1]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        np.log10(stellar_density.T),
        vmin=-1, vmax=3, cmap='terrain')
    rect = plt.Rectangle(
        (observation.instrument.det_x_bins_kpc[0],
         observation.instrument.det_y_bins_kpc[0]),
        observation.instrument.det_x_bins_kpc[-1]
        - observation.instrument.det_x_bins_kpc[0],
        observation.instrument.det_y_bins_kpc[-1]
        - observation.instrument.det_y_bins_kpc[0],
        fc='none',
        ec='k',
        alpha=1)
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'], linewidths=snr_lw)

    plt.colorbar(mappable, ax=ax,
                 label=r'$\log_{10}(\Sigma_*/(\rm M_\odot/pc^2))$',
                 orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    ax = axs[2, 2]
    mappable = ax.pcolormesh(
        observation.instrument.det_x_bins_kpc,
        observation.instrument.det_y_bins_kpc,
        mean_tau.T, cmap='inferno')
    rect = plt.Rectangle(
        (observation.instrument.det_x_bins_kpc[0],
         observation.instrument.det_y_bins_kpc[0]),
        observation.instrument.det_x_bins_kpc[-1]
        - observation.instrument.det_x_bins_kpc[0],
        observation.instrument.det_y_bins_kpc[-1]
        - observation.instrument.det_y_bins_kpc[0],
        fc='none',
        ec='k',
        alpha=1)
    ax.contour(observation.instrument.det_x_bins_kpc,
               observation.instrument.det_y_bins_kpc,
               snr.T, origin='lower',
               levels=snr_levels, colors=snr_color, alpha=1,
               linestyles=['-', '-'])

    plt.colorbar(mappable, ax=ax, label=r'$<\tau_{5500}>$',
                 orientation='horizontal')
    ax.set_xlim(plot_x_lims)
    ax.set_ylim(plot_y_lims)

    # axs[0, 1].axis('off')
    axs[1, 3].axis('off')
    axs[2, 3].axis('off')

    fig.savefig(
        os.path.join(gal_output_path, 'maps_galaxy_{}.pdf'.format(gal_id)),
        bbox_inches='tight')
    plt.close(fig)
# Mr Krtxo \(ﾟ▽ﾟ)/
