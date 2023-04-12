import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
import h5py

from midas.models.gas_model import Gas_model
from midas.galaxy import TNG_Galaxy

f = h5py.File('test_data/sub_20.hdf5', 'r')
tng_density = 3.105e-22
galaxy = TNG_Galaxy(f, name='test_galaxy')
gas_model = Gas_model()

spec = np.zeros_like(gas_model.wavelength)

for i in range(galaxy.gas['Masses'].size):
    print("Particle: ", i)
    spec += gas_model.get_spectra(
        mass=galaxy.gas['Masses'][i] * 1e10 / 0.7,
        h_density=(galaxy.gas['Density'][i] * galaxy.gas['GFM_Metals'][i, 0]
                   * tng_density / const.m_p.value / 1e3),
        temperature=galaxy.gas['temp'][i])


plt.figure()
plt.plot(gas_model.wavelength, spec)
plt.xscale('log')
plt.yscale('log')
plt.ylim(spec.max() * np.array([1e-5, 1.1]))
plt.xlim(100, 1e5)
plt.show()
