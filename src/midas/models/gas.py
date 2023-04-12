#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:06:40 2022

@author: pablo
"""

from midas.models.emission_HI import emission_HI
import numpy as np


def gaussian(x, a, m, s):
    """Gaussian profile."""
    i = a / np.sqrt(2 * np.pi) / s**2 * np.exp(- (x - m)**2 / s**2 / 2)
    return i


def neutral_21cm_line(wavelength, mass, lsf_fwhm, line_pos=21.12e8):
    """Predict the HI emission."""
    intensity = emission_HI(mass, T=0)
    return gaussian(wavelength, intensity, m=line_pos, s=lsf_fwhm/2)


class Cloudy(object):
    """This class represents a gas model generated by Cloudy

    Attributes
    ----------
    temperatures: (np.ndarray) Temperature range in the model. Units: K
    densities: (np.ndarray) Density range in the model. Units: cm^-3
    metallicities: (np.ndarray) Metallicity range in the model.

    Methods
    -------
    ....

    Example
    -------
    # Predict the spectrum of a particle with 1e5 Msun, T=1e5 K and Z=0.02
    model = Cloudy()
    spectra = model.get_spectra(mass=1e5, temp=1e5, metallicity=0.02)
    """
    temperatures = None
    densities = None
    metallicities = None

    def __init__(self):
        self.load_grid()

    def load_grid(self):
        """Here you should include the method that is able to read the files."""
        pass

    def get_spectra(self, mass, temp, density, metallicity):
        """This function should be able to interpolate the grid a return a spectra for
        a given set of parameters"""
        pass
