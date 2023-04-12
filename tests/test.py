#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:51:46 2022

@author: pablo
"""

from matplotlib import pyplot as plt
import h5py
#from pst import SSP
from midas.mock_observation import Observation
from midas.instrument import HI_Instrument
from midas.galaxy import Galaxy

import numpy as np
from scipy.stats import binned_statistic_2d
from gas_model import Gas_model


output = './test_output'
# =============================================================================
# Create a Galaxy object
# =============================================================================
# Load galaxy data

f = h5py.File('test_data/sub_20.hdf5', 'r')
# f = h5py.File('/home/pablo/Research/WEAVE-Apertif/weave_sample/data/sub_626893.hdf5')

stars = f['PartType4']
gas = f['PartType0']

# Variables of each component
print("Stellar particles: ", *stars.keys())
print("Gas particles: ", *gas.keys())

# %%

galaxy = Galaxy(
    name='test_galaxy',
    stars=stars,
    gas=gas,
    # gal_spin=np.array([0., 0., 0.]),
    # gal_vel=np.array([gal_vel['vel_x'], gal_vel['vel_y'], gal_vel['vel_z']]),
    # gal_pos=np.array([gal_pos['pos_x'], gal_pos['pos_y'], gal_pos['pos_z']])
    )
galaxy.get_galaxy_pos(stars=True, gas=False)
galaxy.get_galaxy_vel(stars=True, gas=False)
# Remove systemic velocity and recentre spatial coordinates
galaxy.set_to_galaxy_rest_frame()
# Call this function to generate the 2D projected coordinates
galaxy.proyect_galaxy(orthogonal_vector=None)

# Let's see the projected gas mass
mass_map, xedges, yedges, _ = binned_statistic_2d(
    galaxy.gas['ProjCoordinates'][0, :],
    galaxy.gas['ProjCoordinates'][1, :], values=galaxy.gas['Masses'],
    statistic='sum',
    bins=100)
xbins = (xedges[:-1] + xedges[1:]) / 2
ybins = (yedges[:-1] + yedges[1:]) / 2

area = (xedges[1] - xedges[0])**2
area_pc = area * 1e6

plt.figure()
plt.pcolormesh(xbins, ybins, np.log10(mass_map.T / area_pc) + 10, cmap='brg')
plt.xlabel(r'$X$ [kpc]')
plt.ylabel(r'$Y$ [kpc]')
plt.colorbar(label=r'$\log_{10}(\Sigma_{\rm HI}/(\rm M_\odot/ pc^2))$')
# %%
# =============================================================================
# Create the WEAVE instrument for observing the galaxy
# =============================================================================
instrument = HI_Instrument(z=0.02)
# %% # Perform observation

observation = Observation(Instrument=instrument,
                          Galaxy=galaxy, SSP=None)

observation.compute_los_emission(stellar=False, gas=True,
                                 kinematics=True,
                                 dust_extinction=False)


cube = observation.cube

plt.figure()
plt.pcolormesh(instrument.det_x_bins_kpc,
               instrument.det_y_bins_kpc,
               np.log10(np.sum(cube, axis=0).T), cmap='nipy_spectral')
plt.contour(xbins, ybins, np.log10(mass_map.T) + 10, levels=4,
            alpha=0.3, colors='r')
plt.xlabel(r'$X$ [kpc]')
plt.ylabel(r'$Y$ [kpc]')
plt.savefig("hi_map.pdf", bbox_inches='tight')
# plt.close()

plt.figure()
plt.plot(instrument.wavelength / 1e8, cube.sum(axis=(1, 2)))
# plt.xlim(2.1e9, 2.12e9)
plt.xlabel(r'$\lambda~\rm [cm]$')
plt.ylabel(r'$F_\lambda~ \rm [erg/s/\AA/cm^2]$')
plt.savefig("hi_integrated_line_profile.pdf", bbox_inches='tight')
# plt.close()
f.close()

# %%

model = Gas_model(galaxy)
wavelength,gas_emission = model.get_spectra()

plt.figure()
plt.loglog(wavelength, gas_emission)
plt.xlabel('wavelength ($\AA$)')
plt.ylabel('$L_{\u03BB}$' '($erg \ s^{-1}$)')
plt.ylim(1.0e35, 1.0e44)
plt.xlim(10, 1e8)

plt.figure()
plt.loglog(wavelength, gas_emission)
plt.xlabel('wavelength ($\AA$)')
plt.ylabel('$L_{\u03BB}$' '($erg \ s^{-1}$)')
plt.xlim(6.5e3, 6.65e3)
plt.ylim(1.0e37, 1.0e44)