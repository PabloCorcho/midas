#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from astropy.io import fits
"""
This class models sky continuum and emission lines for generating 
mock observations
"""


class Sky(object):
    
    pass


class ESO_Sky(Sky):
    path_to_files = '/home/pablo/obs_data/ESO-SKY/UVES_ident'
    fits_files = ['346', '437', '580L', '580U', '800U', '860L', '860U']

    def __init__(self):
        self.load_data()

    def load_data(self):
        """blah.."""
        self.flux = np.zeros(0)
        self.wavelength = np.zeros(0)
        for file in self.fits_files:
            file = 'gident_' + file + '.tfits'
            with fits.open(os.path.join(self.path_to_files, file)) as hdul:
                wl, flux = hdul[1].data['LAMBDA_AIR'], hdul[1].data['FLUX']
                self.wavelength = np.concatenate((self.wavelength, wl))
                self.flux = np.concatenate((self.flux, flux))



if __name__ == '__main__':
    sky = ESO_Sky()