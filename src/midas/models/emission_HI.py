# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:07:58 2022

@author: Pablo
"""

"""

Suppose we have a gas cloud made up only of hydrogen particles. Each 
particle has a specific mass in solar masses. We aim to calculate the
emission of this cloud due to the hydrogen hyperfine transition F=1 - F=0
which has a transition rate of 2,8843*10^-15 s^-1. 
Assuming that the temperature in the HI region is low, we expect all 
hydrogen atoms to be in the ground state. In the ground state, (3/4) of all
atoms will be on the F=1 level (triplet) while only (1/4) on the F=0 level 
(singlet).

"""

import numpy as np
import scipy.constants as sc

def emission_HI(m_particles,T):
    """
    This function determines the luminosity in erg/s of an HI cloud. 
    Arguments:
    m_particles:  mass of particles in simulation in solar masses
    T:  gas temperature 
    """
    A_TR = 2.8843e-15 # HI transition rate [s^-1]
    FREQ = 1420.405e6 # HI transition frequency [Hz]
    M_SUN = 2e30 # solar mass [kg]
    L_SUN = 3.826e33 # solar luminosity [erg/s]
    
    N_H = m_particles*M_SUN/sc.m_p # number of H atoms
    
    if T>1e4:
        # ionization 
        return 0.0
    if T<1e4:
        return (3/4) * N_H * A_TR * sc.h * FREQ * 1e7 # [erg/s]
    
"""
j=emission_HI(1e6, 200)
print(format(j,'.1E') + " erg/s")
"""
        
        
    