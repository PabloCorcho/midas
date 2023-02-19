#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:15:14 2023

@author: pablo
"""

from matplotlib import pyplot as plt
import h5py
#from pst import SSP
from midas.mock_observation import Observation
from midas.instrument import HI_Instrument
from midas.galaxy import Galaxy

import numpy as np
from astropy.io import fits
import os

from scipy.stats import binned_statistic_2d

# =============================================================================
# Create a Galaxy object
# =============================================================================

path = '/home/pablo/Develop/MIDAS/tests/test_data/sub_20.hdf5'
f = h5py.File(path)

gal = Galaxy(stars=f['PartType4'], gas=f['PartType0'])
gal.get_galaxy_pos(stars=True, gas=False)
gal.get_galaxy_vel(stars=True, gas=False)
gal.set_to_galaxy_rest_frame()

plt.figure()
plt.hist2d(np.log10(gal.gas['temp']),
           np.log10(gal.gas['Density']),
           bins=30)
plt.xlabel(r'$\log_{10}(\rm T / K)$')
plt.ylabel(r'$\log_{10}(\rm \rho / [10^{10} M_\odot / (ckpc^3/h^3)])$')
#

