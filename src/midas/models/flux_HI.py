# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:38:15 2022

@author: pablo
"""

"""
Converting power from the HI emission line into flux given a luminosity 
distance
"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from emission_HI import emission_HI

def flux_HI(m_particle,T,D_L):
    FREQ = 1420.405*u.MHz # HI transition frequency [Hz]
    
    L=emission_HI(m_particle,T) # Luminosity
    
    F = (L / (4 * np.pi * np.power(D_L,2))).to(u.W / u.m**2) # Flux in [W/m^2]
    return (3e25 * np.power(FREQ.value,-1) *F.value) * (u.Jy * u.km /u.s) # Flux in [Jy*km/s]

                        
j=flux_HI(10**9.07,200,50.6*u.Mpc)

