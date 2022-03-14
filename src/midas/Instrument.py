#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:44:36 2022

@author: pablo
"""
from astropy import units as u
import numpy as np


class Fibre(object):
    pass


class Instrument(object):
    pass


class IFU(Instrument):
    """
    This class represents an ideal Integral Field Spectrograph and encapsules
    its fundamental properties.
    """
    # Spatial properties
    fov = (None, None)
    n_fibres = None
    fibre_diameter = None  # expressed in arcsec
    fibre_positions = (None, None)  # relative position to the center in arcsec
    detector_position = (None, None)  # detector position in sky
    # Spectral dimension
    wave_init = None
    wave_end = None
    delta_wave = None

    def build_wavelength_array(self):
        self.wavelength = np.arange(self.wave_init, self.wave_end,
                                    self.delta_wave) * u.angstrom
        self.wavelength_edges = np.arange(self.wave_init - self.delta_wave/2,
                                          self.wave_end + self.delta_wave/2,
                                          self.delta_wave) * u.angstrom

    def observe_source(self, source):
        """
        This function takes as input a 3D grid (spatia and spectral) of the
        object that is going to be observed and returns the corresponding
        spectra obtained.
        """
        for fibre in self.fibre_positions:
            infibre = self.locate_fibre(fibre, source.ra, source.dec)
            source.compute_sed(infibre)
            
    def locate_fibre(self, fibre, ra_coords, dec_coords):
        ra_coverage = (
                fibre[0] + self.detector_position[0] - ra_coords
                < self.fibre_diameter)
        dec_coverage = (
                fibre[1] + self.detector_position[1] - dec_coords
                < self.fibre_diameter)
        return ra_coverage & dec_coverage
            
        
# Mr. Krtxo
