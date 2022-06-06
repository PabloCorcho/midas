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
from mock_observation import ObserveGalaxy
from instrument import WEAVE_Instrument
from galaxy import Galaxy
from saveWEAVECube import SaveWEAVECube
from astropy import units as u
from astropy.visualization import make_lupton_rgb
from scipy.stats import binned_statistic_2d
import numpy as np
import pyphot
lib = pyphot.get_library()
filters = [lib[filter_i] for filter_i in ['SDSS_u', 'SDSS_g', 'SDSS_r']]

# Read the data from the simulation
# f = h5py.File('tng100_subhalo_175251.hdf5', 'r')
# stars = f['PartType4']
# gas = f['PartType0']
# galaxy = Galaxy(
#     name='17251',
#     stars=stars,
#     gas=gas,
#     gal_spin=np.array([38.9171, 70.069, -140.988]),
#     gal_vel=np.array([457.701, -275.459, -364.912]),
#     gal_pos=np.array([29338., 1754.63, 73012.9]))
f = h5py.File(
    '/home/pablo/WEAVE-Apertiff/MIDAS/input/simu_data/tng100_subhalo_155.hdf5',
    'r')
stars = f['PartType4']
# gas = f['PartType0']
galaxy = Galaxy(
    name='155',
    stars=stars,
    gal_spin=np.array([35.9405, 3.75965, 14.0041]),
    gal_vel=np.array([-397.813, -1148.97, 986.941]),
    gal_pos=np.array([1039.1, 25973.4, 18764.4]))   
# Small galaxy
# f =  h5py.File('tng100_subhalo_180.hdf5', 'r')
# stars = f['PartType4']
# gas = f['PartType0']
# # Create a galaxy object with stars and gas.
# galaxy = Galaxy(
#     name='180',
#     stars=stars,
#     gas=gas,
#     gal_spin=np.array([-160.155, -29.261, 52.9195]),
#     gal_vel=np.array([-235.995, -304.494, 950.293]),
#     gal_pos=np.array([202.405, 27155.3, 15892.0]))
# -------------------------------------------------------------------------
galaxy.set_to_galaxy_rest_frame()
galaxy.proyect_galaxy(galaxy.spin)
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
observation = ObserveGalaxy(SSP=ssp_model, Instrument=fov, Galaxy=galaxy)
observation.compute_los_emission()
# %%
cube = observation.cube.value
cube_ext = observation.cube_extinction.value
wave = observation.instrument.wavelength.value
sfh = observation.star_formation_history
vel_field = observation.velocity_field
vel_field /= sfh.sum(axis=2).value
np.savetxt('galaxy_popstar_sfh',
        sfh.reshape(sfh.shape[0] * sfh.shape[1], sfh.shape[2]).value)
np.savetxt('galaxy_popstar_vel_field', vel_field)
# %%
# RGB IMAGE
blue_pts = np.where((wave < 5000))[0]
green_pts = np.where((wave > 5000) & (wave < 6000))[0]
red_pts = np.where((wave > 6000))[0]
blue_flux = np.mean(cube[blue_pts, :, :], axis=0)
green_flux = np.mean(cube[green_pts, :, :], axis=0)
red_flux = np.mean(cube[red_pts, :, :], axis=0)

rgb = make_lupton_rgb(red_flux.T/1e-16, green_flux.T/1e-16,
                      blue_flux.T/1e-16,
                      Q=10, stretch=0.5,
                      filename='RGB_galaxy_ID_{}_pypopstar.jpeg'.format(
                          galaxy.name))
plt.imshow(rgb, origin='lower')
# %%
# PHOTOMETRY AND NOISE ESTIMATION
stacked_spectra = cube.reshape(cube.shape[0],
                               cube.shape[1] * cube.shape[2])
_u = np.zeros((stacked_spectra.shape[1], 4))
_g = np.zeros((stacked_spectra.shape[1], 4))
_r = np.zeros((stacked_spectra.shape[1], 4))
for i in range(stacked_spectra.shape[1]):
    _u[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[0])
    _g[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[1])
    _r[i, :] = AB_mags(wave, stacked_spectra[:, i], filter=filters[2])

u_flux = _u[:, 0].reshape(cube.shape[1], cube.shape[2])
u_mag = _u[:, 2].reshape(cube.shape[1], cube.shape[2])
g_flux = _g[:, 0].reshape(cube.shape[1], cube.shape[2])
g_mag = _g[:, 2].reshape(cube.shape[1], cube.shape[2])
r_flux = _r[:, 0].reshape(cube.shape[1], cube.shape[2])
r_mag = _r[:, 2].reshape(cube.shape[1], cube.shape[2])
# %%
limit = (g_mag > 23.75) & (g_mag < 24.25)
flux_limit_SN_10 = np.nanmedian(g_flux[limit])
sigma_limit = flux_limit_SN_10 / 10
median_sigma = np.nanmedian(cube[:, limit] / 10, axis=(1))
cube_sigma = np.zeros_like(cube)
for i in range(cube.shape[1]):
    for j in range(cube.shape[2]):
        cube_sigma[:, i, j] = median_sigma
observation.cube_variance = cube_sigma**2 * (u.erg/u.s/u.cm/u.angstrom)**2

observation.sky = None

SaveWEAVECube(
    observation,
    filename='/home/pablo/WEAVE-Apertiff/mock_cubes/galaxy_ID_{}_pypopstar'.format(galaxy.name))

f.close()
# %%
# PLOTS

fig = plt.figure(figsize=(17, 8))
ax = fig.add_subplot(241)
mappable = ax.imshow(u_mag, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_u$', ax=ax, orientation='horizontal')
CS = ax.contour(u_flux / sigma_limit, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(242)
mappable = ax.imshow(g_mag, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_g$', ax=ax, orientation='horizontal')
CS = ax.contour(g_flux / sigma_limit, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(243)
mappable = ax.imshow(r_mag, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_r$', ax=ax, orientation='horizontal')
CS = ax.contour(r_flux / sigma_limit, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(244)
mappable = ax.imshow(g_mag - r_mag, vmin=0.2, vmax=.7, aspect='auto',
                     origin='lower', cmap='jet',)
plt.colorbar(mappable, label=r'$g - r$', ax=ax, orientation='horizontal')
CS = ax.contour(g_mag, levels=[24],
                colors=['k'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)

ax = fig.add_subplot(245)
mappable = ax.imshow(u_mag_d, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_u$', ax=ax, orientation='horizontal')
CS = ax.contour(u_flux_d / sigma_limit_d, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(246)
mappable = ax.imshow(g_mag_d, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_g$', ax=ax, orientation='horizontal')
CS = ax.contour(g_flux_d / sigma_limit_d, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(247)
mappable = ax.imshow(r_mag_d, vmax=24, cmap='gray', aspect='auto',
                     origin='lower')
plt.colorbar(mappable, label=r'$\mu_r$', ax=ax, orientation='horizontal')
CS = ax.contour(r_flux_d / sigma_limit_d, levels=[3, 30, 100],
                colors=['r', 'limegreen', 'deepskyblue'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
ax = fig.add_subplot(248)
mappable = ax.imshow(g_mag_d - r_mag_d, vmin=0.2, vmax=.7, aspect='auto',
                     origin='lower', cmap='jet',)
plt.colorbar(mappable, label=r'$g - r$', ax=ax, orientation='horizontal')
CS = ax.contour(g_mag, levels=[24],
                colors=['k'], alpha=1)
ax.clabel(CS, inline=1, fontsize=10)
fig.savefig('photometry_galaxy_{}'.format(galaxy.name),
            bbox_inches='tight')


stat, xedges, yedges, _ = binned_statistic_2d(
    galaxy.stars['ProjCoordinates'][0, :],
    galaxy.stars['ProjCoordinates'][1, :],
    galaxy.stars['ProjVelocities'][2, :],
    bins=50)
np.mean_age, xedges, yedges, _ = binned_statistic_2d(
    galaxy.stars['ProjCoordinates'][0, :],
    galaxy.stars['ProjCoordinates'][1, :],
    galaxy.stars['ages'][:],
    bins=50)
xbin = (xedges[:-1] + xedges[1:]) / 2
ybin = (yedges[:-1] + yedges[1:]) / 2

plt.figure()
plt.scatter(galaxy.gas['ProjCoordinates'][0, :],
            galaxy.gas['ProjCoordinates'][1, :], s=1, alpha=0.1,
            # c=log10(galaxy.stars['Masses'] * 1e10 / 0.7),
            c='k'
            # cmap='jet'
            )
plt.xlim(-15, 15)
plt.ylim(-15, 15)

plt.figure()
plt.scatter(galaxy.stars['ProjCoordinates'][0, :],
            galaxy.stars['ProjCoordinates'][1, :], s=1, alpha=0.1,
            # c=log10(galaxy.stars['Masses'] * 1e10 / 0.7),
            c='k'
            # cmap='jet'
            )
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# plt.imshow(stat, extent=(xbin[0], xbin[-1], ybin[0], ybin[-1]),
#            origin='lower', cmap='seismic', vmax=150, vmin=-150)
# plt.colorbar()
plt.contour(xbin, ybin, stat.T, cmap='seismic',
            levels=[-100, -50, -30, -10, 0, 10, 30, 50, 100])
plt.colorbar()

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True,
                        sharey=True)
ax = axs[0]
ax.scatter(galaxy.stars['ProjCoordinates'][0, :],
           galaxy.stars['ProjCoordinates'][1, :], s=1, alpha=0.1,
           # c=log10(galaxy.stars['Masses'] * 1e10 / 0.7),
           c='k',
           zorder=-1,
           label='stellar particles')
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]s')
mappable = ax.imshow(g_mag, origin='lower', interpolation='none',
                     extent=(observation.instrument.det_x_bins_kpc[0],
                             observation.instrument.det_x_bins_kpc[-1],
                             observation.instrument.det_y_bins_kpc[0],
                             observation.instrument.det_y_bins_kpc[-1]),
                     cmap='nipy_spectral', vmax=24, aspect='auto')
plt.colorbar(mappable, ax=ax, label=r'g band (mags)',
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
ax.legend()
ax = axs[1]
mappable = ax.contourf(xbin, ybin, stat.T, cmap='seismic', 
            levels=linspace(-150, 150, 50))
rect = plt.Rectangle((observation.instrument.det_x_bins_kpc[0],
                observation.instrument.det_y_bins_kpc[0]),
              observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
              observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
              fc='none',
              ec='k',
              alpha=1)
ax.add_patch(rect)
cbar = plt.colorbar(mappable, ax=ax, label=r'$v~[km/s]$',
              orientation='horizontal')
cbar.set_ticks([-100, -50, 0, 50, 100])
ax = axs[2]
mappable = ax.contourf(xbin, ybin, np.mean_age.T, cmap='jet', 
            levels=20)
rect = plt.Rectangle((observation.instrument.det_x_bins_kpc[0],
                observation.instrument.det_y_bins_kpc[0]),
              observation.instrument.det_x_bins_kpc[-1] - observation.instrument.det_x_bins_kpc[0],
              observation.instrument.det_y_bins_kpc[-1] - observation.instrument.det_y_bins_kpc[0],
              fc='none',
              ec='k',
              alpha=1)
ax.add_patch(rect)
plt.colorbar(mappable, ax=ax, label=r'$<age_*>~[Gyr]$',
              orientation='horizontal')
fig.savefig('maps_galaxy_{}'.format(galaxy.name), bbox_inches='tight')
# plt.xlim(30, 550)
# plt.ylim(30, 60)
