# -*- coding: utf-8 -*-


import numpy as np
from astropy import constants as const
from midas.utils import inter_2d
from .emission_model import EmissionGridModel

class Gas_model(EmissionGridModel):
    """This class represents a gas model generated by Cloudy

    Attributes
    ----------
    temperatures: (np.ndarray) Temperature range in the model. Units: K
    densities: (np.ndarray) Density range in the model. Units: cm^-3
    metallicities: (np.ndarray) Metallicity range in the model.

    Methods
    -------
    load_grid:
    get_spectra:    
    """
    h_densities = None
    temperatures = None
    metallicities = None
    emission_coef = None
    wavelength = None

    def __init__(self):
        self.load_grid()

    def load_grid(self):
        """Here you should include the method that is able to read the files."""
        self.h_densities = np.logspace(-3, 3, num=7)
        self.temperatures = np.logspace(3, 7, num=9)
        self.emission_coef = np.loadtxt('emission_coef.txt').reshape(7, 9, 8228)
        self.wavelength = np.loadtxt('wavelength.txt')

    def get_spectra(self, mass, h_density, temperature):
        """This function should be able to interpolate the grid and return a spectra for
        a given set of parameters"""
        gas_emission = inter_2d(self.emission_coef, self.h_densities, self.temperatures,
                                h_density, temperature)
        return gas_emission * mass
