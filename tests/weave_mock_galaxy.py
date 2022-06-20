#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:28:52 2022

@author: pablo
"""

from matplotlib import pyplot as plt
import h5py
from FAENA.measurements.photometry import AB_mags
from pst import SSP
from midas.mock_observation import Observation
from midas.instrument import WEAVE_Instrument
from midas.galaxy import Galaxy
from midas.saveWEAVECube import SaveWEAVECube
from astropy import units as u
from astropy.visualization import make_lupton_rgb
from scipy.stats import binned_statistic_2d
import numpy as np
import pyphot
from astropy.io import fits

lib = pyphot.get_library()
filters = [lib[filter_i] for filter_i in ['SDSS_u', 'SDSS_g', 'SDSS_r']]


id = 643696
mock_catalogue = fits.getdata('/home/pablo/WEAVE-Apertiff/weave_sample/mock_WA_sample.fits')
cat_pos = np.where(mock_catalogue['ID'] == id)[0][0]
f = h5py.File(
    '/home/pablo/WEAVE-Apertiff/weave_sample/data/sub_{}.hdf5'.format(id),
    'r')
stars = f['stars']
# gas = f['PartType0']
galaxy = Galaxy(
    name=str(id),
    stars=stars,
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
# -------------------------------------------------------------------------
galaxy.set_to_galaxy_rest_frame()
galaxy.proyect_galaxy(None)
# projection_vetor = np.array([-1/galaxy.spin[0], 1/galaxy.spin[1], 0])
# galaxy.proyect_galaxy(projection_vetor)
# Create the WEAVE instrument for observing the galaxy
fov = WEAVE_Instrument(z=0.01)
# SSP model used for generating stellar spectra
#ssp_model = SSP.PyPopStar(IMF='KRO')
ssp_model = SSP.XSL(IMF='Kroupa', ISO='P00')

#ssp_model.cut_models(wl_min=3000 * u.angstrom,
#                     wl_max=8000 * u.angstrom)

# Perform observation
# %%
observation = Observation(SSP=ssp_model, Instrument=fov, Galaxy=galaxy)
observation.compute_los_emission()
# observation.los_emission(nworkers=4)
# %%
# cube = observation.cube.value
# cube_ext = observation.cube_extinction.value
# wave = observation.instrument.wavelength.value
# tot_mass_map = observation.phys_data['tot_stellar_mass']
# vel_field = observation.phys_data['stellar_kin_mass_weight']

# plt.figure()
# plt.imshow(np.log10(tot_mass_map), vmax=5)
# plt.colorbar()

# plt.figure()
# plt.imshow(vel_field, cmap='seismic')
# plt.contour(np.log10(tot_mass_map), levels=[5, 6])
# plt.colorbar()

# np.savetxt('/home/pablo/WEAVE-Apertiff/mock_cubes/galaxy_ID_{}_kin'.format(
#             galaxy.name), vel_field)
# # %%
# # RGB IMAGE
# blue_pts = np.where((wave < 5000))[0]
# green_pts = np.where((wave > 5000) & (wave < 6000))[0]
# red_pts = np.where((wave > 6000))[0]
# blue_flux = np.mean(cube[blue_pts, :, :], axis=0)
# green_flux = np.mean(cube[green_pts, :, :], axis=0)
# red_flux = np.mean(cube[red_pts, :, :], axis=0)

# rgb = make_lupton_rgb(red_flux.T/1e-16, green_flux.T/1e-16,
#                       blue_flux.T/1e-16,
#                       Q=10, stretch=0.5,
#                       filename='RGB_galaxy_ID_{}_pypopstar.jpeg'.format(
#                           galaxy.name))
# # %%
# # PHOTOMETRY AND NOISE ESTIMATION
# stacked_spectra = cube.reshape(cube.shape[0],
#                                cube.shape[1] * cube.shape[2])
# _u = np.zeros((stacked_spectra.shape[1], 4))
# _g = np.zeros((stacked_spectra.shape[1], 4))
# _r = np.zeros((stacked_spectra.shape[1], 4))
# for i in range(stacked_spectra.shape[1]):
#     _u[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[0])
#     _g[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[1])
#     _r[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[2])

# u_flux = _u[:, 0].reshape(cube.shape[1], cube.shape[2])
# u_mag = _u[:, 2].reshape(cube.shape[1], cube.shape[2])
# g_flux = _g[:, 0].reshape(cube.shape[1], cube.shape[2])
# g_mag = _g[:, 2].reshape(cube.shape[1], cube.shape[2])
# r_flux = _r[:, 0].reshape(cube.shape[1], cube.shape[2])
# r_mag = _r[:, 2].reshape(cube.shape[1], cube.shape[2])
# # %%
# limit = (g_mag > 23.75) & (g_mag < 24.25)
# flux_limit_SN_10 = np.nanmedian(g_flux[limit])
# sn_24 = 5
# sigma_limit = flux_limit_SN_10 / sn_24
# median_sigma = np.nanmedian(cube[:, limit] / sn_24, axis=(1))
# cube_sigma = np.zeros_like(cube)
# for i in range(cube.shape[1]):
#     for j in range(cube.shape[2]):
#         cube_sigma[:, i, j] = median_sigma

# # observation.cube_variance = cube_sigma**2 * (u.erg/u.s/u.cm**2/u.angstrom)**2
# observation.cube_variance = (np.clip(observation.cube,
#                                      a_min=observation.cube.max() * 1e-5,
#                                      a_max=observation.cube.max())
#                              * 0.01)**2
# observation.cube_variance[observation.cube_variance == 0] = np.nan

# observation.sky = None
# # %%
# SaveWEAVECube(
#     observation,
#     filename='/home/pablo/WEAVE-Apertiff/mock_cubes/galaxy_ID_{}_xsl'.format(
#         galaxy.name),
#     funit=1e-18,
#     data_size=np.float32)

# f.close()
# # %%
# # PLOTS

# fig = plt.figure(figsize=(16, 6))
# ax = fig.add_subplot(141)
# mappable = ax.imshow(u_mag, vmax=24, cmap='gray', aspect='auto',
#                      origin='lower')
# plt.colorbar(mappable, label=r'$\mu_u$', ax=ax, orientation='horizontal')
# CS = ax.contour(u_flux / sigma_limit, levels=[3, 30, 100],
#                 colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
# ax.clabel(CS, inline=1, fontsize=10)
# ax = fig.add_subplot(142)
# mappable = ax.imshow(g_mag, vmax=24, cmap='gray', aspect='auto',
#                      origin='lower')
# plt.colorbar(mappable, label=r'$\mu_g$', ax=ax, orientation='horizontal')
# CS = ax.contour(g_flux / sigma_limit, levels=[3, 30, 100],
#                 colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
# ax.clabel(CS, inline=1, fontsize=10)
# ax = fig.add_subplot(143)
# mappable = ax.imshow(r_mag, vmax=24, cmap='gray', aspect='auto',
#                      origin='lower')
# plt.colorbar(mappable, label=r'$\mu_r$', ax=ax, orientation='horizontal')
# CS = ax.contour(r_flux / sigma_limit, levels=[3, 30, 100],
#                 colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
# ax.clabel(CS, inline=1, fontsize=10)
# ax = fig.add_subplot(144)
# mappable = ax.imshow(g_mag - r_mag, vmin=0.2, vmax=.7, aspect='auto',
#                      origin='lower', cmap='jet',)
# plt.colorbar(mappable, label=r'$g - r$', ax=ax, orientation='horizontal')
# CS = ax.contour(g_mag, levels=[24],
#                 colors=['k'], alpha=1)
# ax.clabel(CS, inline=1, fontsize=10)
# fig.savefig('photometry_galaxy_{}'.format(galaxy.name),
#             bbox_inches='tight')


# stat, xedges, yedges, _ = binned_statistic_2d(
#     galaxy.stars['ProjCoordinates'][0, :],
#     galaxy.stars['ProjCoordinates'][1, :],
#     galaxy.stars['ProjVelocities'][2, :],
#     bins=50)
# np.mean_age, xedges, yedges, _ = binned_statistic_2d(
#     galaxy.stars['ProjCoordinates'][0, :],
#     galaxy.stars['ProjCoordinates'][1, :],
#     galaxy.stars['ages'][:],
#     bins=50)
# xbin = (xedges[:-1] + xedges[1:]) / 2
# ybin = (yedges[:-1] + yedges[1:]) / 2

# # %%
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True,
#                         sharey=True)
# ax = axs[0]
# ax.scatter(galaxy.stars['ProjCoordinates'][0, :],
#            galaxy.stars['ProjCoordinates'][1, :], s=1, alpha=0.1,
#            # c=log10(galaxy.stars['Masses'] * 1e10 / 0.7),
#            c='k',
#            zorder=-1,
#            label='stellar particles')
# ax.set_xlabel('X [kpc]')
# ax.set_ylabel('Y [kpc]s')
# mappable = ax.imshow(g_mag, origin='lower', interpolation='none',
#                      extent=(observation.instrument.det_x_bins_kpc[0],
#                              observation.instrument.det_x_bins_kpc[-1],
#                              observation.instrument.det_y_bins_kpc[0],
#                              observation.instrument.det_y_bins_kpc[-1]),
#                      cmap='nipy_spectral', vmax=24, aspect='auto')
# plt.colorbar(mappable, ax=ax, label=r'g band (mags)',
#              orientation='horizontal')
# rect = plt.Rectangle(
#     (observation.instrument.det_x_bins_kpc[0],
#      observation.instrument.det_y_bins_kpc[0]),
#     observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
#     observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
#     fc='none',
#     ec='k',
#     alpha=1,
#     label='FoV')
# ax.add_patch(rect)
# ax.legend()
# ax = axs[1]
# mappable = ax.contourf(xbin, ybin, stat.T, cmap='seismic', 
#             levels=np.linspace(-150, 150, 50))
# rect = plt.Rectangle((observation.instrument.det_x_bins_kpc[0],
#                 observation.instrument.det_y_bins_kpc[0]),
#               observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
#               observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
#               fc='none',
#               ec='k',
#               alpha=1)
# ax.add_patch(rect)
# cbar = plt.colorbar(mappable, ax=ax, label=r'$v~[km/s]$',
#               orientation='horizontal')
# cbar.set_ticks([-100, -50, 0, 50, 100])
# ax = axs[2]
# mappable = ax.contourf(xbin, ybin, np.mean_age.T, cmap='jet', 
#             levels=20)
# rect = plt.Rectangle((observation.instrument.det_x_bins_kpc[0],
#                 observation.instrument.det_y_bins_kpc[0]),
#               observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
#               observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
#               fc='none',
#               ec='k',
#               alpha=1)
# ax.add_patch(rect)
# plt.colorbar(mappable, ax=ax, label=r'$<age_*>~[Gyr]$',
#               orientation='horizontal')
# fig.savefig('maps_galaxy_{}'.format(galaxy.name), bbox_inches='tight')
# # plt.xlim(30, 550)
# # plt.ylim(30, 60)

# # Mr Krtxo \(ﾟ▽ﾟ)/