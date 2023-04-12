import numpy as np
import os
from astropy import constants as const
from astropy.io import fits

from midas.utils import inter_2d
from .emission_model import EmissionGridModel

class SSP_model(EmissionGridModel):
    ages = None
    metallicities = None

    def get_spectra(self, mass, age, metallicity, clip=True):
        if clip:
            metallicity = np.clip(metallicity,
                                a_min=self.metallicities.min(),
                                a_max=self.metallicities.max())
            age = np.clip(age,
                        a_min=self.ages.min(),
                        a_max=self.ages.max())
        s =  inter_2d(self.sed_grid, self.ages,
                      self.metallicities, age, metallicity,
                      log=True)
        return s * mass
    
class PyPopStar_model(SSP_model):
    def __init__(self, IMF, nebular=False):
        
        self.path = self.path = os.path.join('/home/pablo/SSP_TEMPLATES',
                                                'PyPopStar', IMF)
        self.load_grid(IMF=IMF, nebular=nebular)

    def load_grid(self, IMF, nebular):
        self.metallicities = np.array([0.004, 0.008, 0.02, 0.05])
        self.log_ages_yr = np.array([
        5.,  5.48,  5.7 ,  5.85,  6.  ,  6.1 ,  6.18,  6.24,  6.3 ,
        6.35,  6.4 ,  6.44,  6.48,  6.51,  6.54,  6.57,  6.6 ,  6.63,
        6.65,  6.68,  6.7 ,  6.72,  6.74,  6.76,  6.78,  6.81,  6.85,
        6.86,  6.88,  6.89,  6.9 ,  6.92,  6.93,  6.94,  6.95,  6.97,
        6.98,  6.99,  7.  ,  7.04,  7.08,  7.11,  7.15,  7.18,  7.2 ,
        7.23,  7.26,  7.28,  7.3 ,  7.34,  7.38,  7.41,  7.45,  7.48,
        7.51,  7.53,  7.56,  7.58,  7.6 ,  7.62,  7.64,  7.66,  7.68,
        7.7 ,  7.74,  7.78,  7.81,  7.85,  7.87,  7.9 ,  7.93,  7.95,
        7.98,  8.  ,  8.3 ,  8.48,  8.6 ,  8.7 ,  8.78,  8.85,  8.9 ,
        8.95,  9.  ,  9.18,  9.3 ,  9.4 ,  9.48,  9.54,  9.6 ,  9.65,
        9.7 ,  9.74,  9.78,  9.81,  9.85,  9.9 ,  9.95, 10.  , 10.04,
        10.08, 10.11, 10.12, 10.13, 10.14, 10.15, 10.18])
        self.ages = 10**self.log_ages_yr

        header = os.path.join(self.path, 'SSP-{}'.format(IMF))
        if nebular:
            print("> Initialising Popstar models (neb em) (IMF='"
                    + IMF + "')")
            column = 'flux_total'
        else:
            print("> Initialising Popstar models (no neb em) (IMF='"
                    + IMF + "')")
            column = 'flux_stellar'
        with fits.open(header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(
                self.metallicities[0], self.log_ages_yr[0])
                        ) as hdul:
            self.wavelength = hdul[1].data['wavelength']  # Angstrom

        self.sed_grid = self.sed_grid = np.empty(
            shape=(self.wavelength.size,
                   self.log_ages_yr.size,
                   self.metallicities.size))

        for i, Z in enumerate(self.metallicities):
            for j, age in enumerate(self.log_ages_yr):
                filename = header+'_Z{:03.3f}_logt{:05.2f}.fits'.format(Z, age)
                file = os.path.join(self.path, IMF, filename)
                with fits.open(file) as hdul:
                    self.sed_grid[:, j, i] = hdul[1].data[column]  #* const.L_sun.to('erg/s').value # erg/s/AA/Msun
                    hdul.close()
        self.sed_unit = 'Lsun/Angstrom/Msun'
