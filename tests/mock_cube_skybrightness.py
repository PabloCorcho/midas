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
from midas.instrument import WEAVEInstrument
from midas.galaxy import Galaxy
from midas.saveWEAVECube import SaveWEAVECube
from midas.cosmology import cosmo
from wetc import signal2noise
import numpy as np
from astropy.io import fits
import os

output = '/home/pablo/Research/WEAVE-Apertif/mock_cubes/'
input_path = '/media/pablo/Elements/WEAVE/weave_sample/data/'
mock_catalogue = fits.getdata(
    '/media/pablo/Elements/WEAVE/weave_sample/mock_WA_sample.fits')

# Default values for g band (plots)
mag_lim = 24  # mag / arcsec**2
band_wl = 4686
sn_lim = 40
c_ligth_aa = 3e18
f_lim = (10 ** (-0.4 * (mag_lim + 48.60)) * c_ligth_aa / band_wl ** 2
         / 0.5**2)  # erg/s/AA/cm2

# Moon brightness
inc_moon = True
moon_phase = 'gibbous'

# LIFU mode
mode = 'HR'
# %% Initialise noise model
S = signal2noise.Signal(LIFU=True, offset=0.1,
                        throughput_table=True,
                        sky_model=True,
                        inc_moon=inc_moon,
                        mode=mode,
                        moon_phase=moon_phase)
time = 1200.
n_exp = 3
airmass = 1.2
seeing = signal2noise.totalSeeing(1.0)

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
weave_instrument = WEAVEInstrument(z=0.1, mode=mode)

# =============================================================================
# Create the template for the observations (this way the SSP is prepared only once)
# Some properties will be updated during the loop (instrument, galaxy)
# =============================================================================
observation = Observation(SSP=ssp_model, Instrument=weave_instrument,
                          Galaxy=None)
# =============================================================================
# Create a Galaxy object
# =============================================================================
gal_id = 362780  # Many particles
# gal_id = 146237  # Few particles
gal_output_path = os.path.join(output, str(gal_id))
cat_pos = np.where(mock_catalogue['ID'] == gal_id)[0][0]
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
# %% Smoothing
sigma_kpc = 0.3
sigma_pixel = np.min(
    (sigma_kpc / weave_instrument.pixel_size_kpc.value, 5))

observation.cube = observation.gaussian_smooth(
    observation.cube,
    sigma=[0, sigma_pixel, sigma_pixel])
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
                          'galaxy_ID_{}_{}_{}_{}'
                          .format(galaxy.name, mode,
                                  ssp_name, moon_phase)),
    funit=1e-18,
    data_size=np.float32)
# Close HDF5 file
f.close()
