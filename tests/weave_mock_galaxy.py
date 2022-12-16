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
from midas.cosmology import cosmo
from wetc import signal2noise
import numpy as np
from astropy.io import fits
import os


output = './test_output'
# =============================================================================
# Create a Galaxy object
# =============================================================================
# Load galaxy data
f = h5py.File('/test_output/sub_239995.hdf5', 'r)

stars = f['stars']
gas = f['gas']

# 
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
ssp_name = 'xsl'
# %% # Perform observation
observation = Observation(SSP=ssp_model, Instrument=weave_instrument,
                          Galaxy=galaxy)
observation.compute_los_emission(stellar=True, nebular=False,
                                 kinematics=True,
                                 dust_extinction=True)



