#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:39:50 2021

@author: pablo
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from generate_grid import Galaxy, WEAVE_Instrument
import h5py

hdul = fits.open('galaxy_pypopstar_eo.fits.gz')
cube = hdul[1].data * 1e-16
wave = hdul[1].header['CRVAL3'] + np.arange(0, cube.shape[2],
                                            hdul[1].header['CDELT3'])
hdul.close()

plt.figure()
plt.imshow(np.log10(cube.sum(axis=2)))

f = h5py.File('tng100_subhalo_175251.hdf5', 'r')
stars = f['PartType4']
gas = f['PartType0']
galaxy = Galaxy(stars=stars,
                gas=gas,
                gal_spin=np.array([38.9171, 70.069, -140.988]),
                gal_vel=np.array([457.701, -275.459, -364.912]),
                gal_pos=np.array([29338., 1754.63, 73012.9]))
galaxy.set_to_galaxy_rest_frame()
# galaxy.proyect_galaxy(galaxy.spin)
projection_vetor = np.array([-1/galaxy.spin[0], 1/galaxy.spin[1], 0])
galaxy.proyect_galaxy(projection_vetor)
f.close()
fov = WEAVE_Instrument(z=0.01,
                           pixel_size=1)
# blue_pts = np.where((wave<5000))[0]
# green_pts = np.where((wave>5000)&(wave<6000))[0]
# red_pts = np.where((wave>6000))[0]
# blue_flux = np.mean(cube[:, :, blue_pts], axis=2)
# green_flux = np.mean(cube[:, :, green_pts], axis=2)
# red_flux = np.mean(cube[:, :, red_pts], axis=2)
# from astropy.visualization import make_lupton_rgb
# rgb = make_lupton_rgb(red_flux.T/1e-16, green_flux.T/1e-16,
#                       blue_flux.T/1e-16,
#                       Q=10, stretch=0.5, filename="rgb.jpeg")
# plt.imshow(rgb, origin='lower')


collapsed = np.nansum(cube, axis=2)

from scipy.stats import binned_statistic_2d
stat, xedges, yedges, _ = binned_statistic_2d(
    galaxy.stars['ProjCoordinates'][0, :],
    galaxy.stars['ProjCoordinates'][1, :],
    galaxy.stars['ProjVelocities'][2, :],
    bins=50)
mean_age, xedges, yedges, _ = binned_statistic_2d(
    galaxy.stars['ProjCoordinates'][0, :],
    galaxy.stars['ProjCoordinates'][1, :],
    galaxy.stars['ages'][:] * galaxy.stars['GFM_InitialMass'][:]*1e10,
    statistic='sum',
    bins=50)

stellar_mass, xedges, yedges, _ = binned_statistic_2d(
    galaxy.stars['ProjCoordinates'][0, :],
    galaxy.stars['ProjCoordinates'][1, :],
    galaxy.stars['GFM_InitialMass'][:]*1e10,
    statistic='sum',
    bins=50)

mean_age = mean_age / stellar_mass

xbin = (xedges[:-1] + xedges[1:]) / 2
ybin = (yedges[:-1] + yedges[1:]) / 2

plt.figure()
plt.scatter(galaxy.gas['ProjCoordinates'][0, :],
            galaxy.gas['ProjCoordinates'][1, :], s=1, alpha=0.1,
            # c=np.log10(galaxy.stars['Masses'] * 1e10 / 0.7), 
            c='k'
            # cmap='jet'
            )
plt.xlim(-15, 15)
plt.ylim(-15, 15)

plt.figure()
plt.scatter(galaxy.stars['ProjCoordinates'][0, :],
            galaxy.stars['ProjCoordinates'][1, :], s=1, alpha=0.1,
            # c=np.log10(galaxy.stars['Masses'] * 1e10 / 0.7), 
            c='k'
            # cmap='jet'
            )
plt.xlim(-15, 15)
plt.ylim(-15, 15)
# plt.imshow(stat, extent=(xbin[0], xbin[-1], ybin[0], ybin[-1]),
#            origin='lower', cmap='seismic', vmax=150, vmin=-150)
# plt.colorbar()
plt.contour(xbin, ybin, stat.T, cmap='seismic', 
            levels=[-100, -50, -30,-10, 0, 10, 30, 50, 100])
plt.colorbar()

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), sharex=True,
                        sharey=True)
ax = axs[0]
ax.scatter(galaxy.stars['ProjCoordinates'][0, :],
           galaxy.stars['ProjCoordinates'][1, :], s=1, alpha=0.1,
           # c=np.log10(galaxy.stars['Masses'] * 1e10 / 0.7), 
           c='k',
           zorder=-1
            )
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
mappable = ax.imshow(np.log10(collapsed.T), origin='lower',
                     interpolation='none',
                     extent=(fov.det_y_bins_kpc[0],
                             fov.det_y_bins_kpc[-1],
                             fov.det_x_bins_kpc[0],
                             fov.det_x_bins_kpc[-1]),
                     cmap='nipy_spectral', vmin=-17,
                     aspect='auto')
plt.colorbar(mappable, ax=ax, label=r'$\log_{10}(F_\lambda)$',
             orientation='horizontal')
rect = plt.Rectangle((fov.det_y_bins_kpc[0],
                      fov.det_x_bins_kpc[0]),
                     fov.det_y_bins_kpc[-1] - fov.det_y_bins_kpc[0],
                     fov.det_x_bins_kpc[-1] - fov.det_x_bins_kpc[0],
                     fc='none',
                     ec='k',
                     alpha=1)
ax.add_patch(rect)
ax = axs[1]
mappable = ax.contourf(xbin, ybin, stat.T, cmap='seismic',
                       levels=np.linspace(-150, 150, 50))
rect = plt.Rectangle((fov.det_y_bins_kpc[0],
                      fov.det_x_bins_kpc[0]),
                     fov.det_y_bins_kpc[-1] - fov.det_y_bins_kpc[0],
                     fov.det_x_bins_kpc[-1] - fov.det_x_bins_kpc[0],
                     fc='none',
                     ec='k',
                     alpha=1)
ax.add_patch(rect)
cbar = plt.colorbar(mappable, ax=ax, label=r'$v~[km/s]$',
                    orientation='horizontal')
cbar.set_ticks([-100, -50, 0, 50, 100])
ax.set_xlim(axs[0].get_xlim())
ax.set_ylim(axs[0].get_ylim())
ax = axs[2]
mappable = ax.contourf(xbin, ybin, mean_age.T, cmap='jet_r',
                       levels=20)
rect = plt.Rectangle((fov.det_y_bins_kpc[0],
                      fov.det_x_bins_kpc[0]),
                     fov.det_y_bins_kpc[-1] - fov.det_y_bins_kpc[0],
                     fov.det_x_bins_kpc[-1] - fov.det_x_bins_kpc[0],
                     fc='none',
                     ec='k',
                     alpha=1)
ax.add_patch(rect)
plt.colorbar(mappable, ax=ax, label=r'$<age_*>_{M_*}~[Gyr]$',
             orientation='horizontal')

fig.subplots_adjust(wspace=0.1)
fig.savefig('galaxy_maps.png', bbox_inches='tight')
# plt.xlim(30, 50)
# plt.ylim(30, 60)