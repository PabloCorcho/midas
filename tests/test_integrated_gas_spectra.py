import numpy as np
from matplotlib import pyplot as plt
from astropy import constants as const
import h5py

from midas.models.gas_model import Gas_model
from midas.models.ssp_model import PyPopStar_model
from midas.galaxy import Galaxy, TNG_Galaxy

tng_density = 3.105e-22
h_cosmo = 0.7

wavelength = np.logspace(3, 4, 1001)

ssp_model = PyPopStar_model(IMF='KRO')
ssp_model.interpolate_sed(wavelength)

gas_model = Gas_model()
gas_model.interpolate_sed(wavelength)

# %%
f = h5py.File('test_data/sub_20.hdf5', 'r')
galaxy = TNG_Galaxy(f, name='test_galaxy')

gas_spec = np.zeros_like(gas_model.wavelength)

for i in range(galaxy.gas['Masses'].size):
    print("Particle: ", i)
    gas_spec += gas_model.get_spectra(
        mass=galaxy.gas['Masses'][i] * 1e10 / h_cosmo,
        h_density=(galaxy.gas['Density'][i] * galaxy.gas['GFM_Metals'][i, 0]
                   * tng_density / const.m_p.value / 1e3),
        temperature=galaxy.gas['temp'][i])

print("Integrated gas: ", np.trapz(gas_spec, x=gas_model.wavelength))

stellar_spec = np.zeros_like(ssp_model.wavelength)

for i in range(galaxy.stars['GFM_InitialMass'].size):
    mass, age, metals, wind = (
                    galaxy.stars['GFM_InitialMass'][i]
                    * 1e10 / h_cosmo,
                    galaxy.stars['ages'][i] * 1e9,
                    galaxy.stars['GFM_Metallicity'][i],
                    galaxy.stars['wind'][i])
    print("Particle: {} -- Mass: {}, Age: {}, Metals: {}".format(i, mass, age, metals))
    if wind:
        continue
    s = ssp_model.get_spectra(mass, age, metals)
    stellar_spec += s

print(stellar_spec)

print("Integrated stars: ", galaxy.stars['Masses'].sum(),
      "Integrated gas", galaxy.gas["Masses"].sum())

tot_spec = gas_spec + stellar_spec

# %%
plt.figure(figsize=(9, 5))
plt.subplot(211)
plt.plot(gas_model.wavelength,
         gas_spec,
         c='b', label='Gas')
plt.plot(ssp_model.wavelength, stellar_spec, c='r', label='Stellar')
plt.plot(ssp_model.wavelength, tot_spec, c='k', label='Total')
plt.legend(framealpha=0.7)
plt.ylabel(r"$L_\lambda$")
# plt.xscale('log')
plt.yscale('log')
plt.ylim(gas_spec.max() * np.array([1e-5, 1.1]))
plt.xlim(wavelength.min(), wavelength.max())
# plt.xlim(3500, 7000)
plt.subplot(212)
plt.plot(gas_model.wavelength, gas_spec / tot_spec, c='k')
# plt.axhline(1/20, color='r', label='SNR=20', ls='--', alpha=0.2)
# plt.axhline(1/50, color='r', label='SNR=50', ls='--', alpha=0.2)
plt.legend(framealpha=0)
# plt.xscale('log')
plt.yscale('log')
plt.xlim(wavelength.min(), wavelength.max())
plt.xlabel(r"$\lambda~(\AA)$")
plt.ylabel(r"$L_\lambda^{\rm gas} / L_\lambda^{\rm tot}$")
plt.grid(visible=True)
plt.show()