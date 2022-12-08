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